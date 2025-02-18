import logging
import shutil
from typing import Dict, Tuple, Union, Optional, Literal, List
from pathlib import Path
from dataclasses import dataclass
from datasets import Dataset

import torch

from tqdm import tqdm
from deepspeed import comm as dist
from accelerate.utils import gather, pad_across_processes, release_memory
from transformers.trainer_pt_utils import get_model_param_count

from relign.common.deepspeed_utils import prepare_data_loader_for_inference
from relign.common.dataset import EpisodeDataset
from relign.algorithms.base_trainer import BaseTrainer
from relign.utils.trainer import (
    prepare_data_loader_for_training,
    masked_mean,
    masked_rescale_by_std,
    masked_var,
    masked_whiten,
)

from relign.algorithms.ppo.data_collator import (
    PPODataCollator,
    COLUMN_ACTOR_SHIFTED_LOGPS,
    COLUMN_REF_SHIFTED_LOGPS,
    COLUMN_VALUES,
)

from relign.utils.logging import get_logger

logger = get_logger(__name__)

import hashlib
import io
import os


def get_model_hash(model):
    """Small helper function that returns the hash of a models weights"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return hashlib.sha256(buffer.read()).hexdigest()


@dataclass
class PPOHParams:
    """
    Configuration class for PPOTrainer.

    Parameters:
        adap_kl_ctrl (bool):
            Use adaptive KL control, otherwise linear.
        init_kl_coef (Optional[float]):
            Initial KL penalty coefficient (used for adaptive and linear control).
        kl_penalty (Literal["kl", "abs", "mse", "full"]):
            KL penalty options. 'kl': model_logp - ref_logp, 'abs': abs(kl),
            'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution.
        target (Optional[float]):
            Target KL value for adaptive KL control.
        gamma (float):
            Gamma parameter for advantage calculation.
        lam (float):
            Lambda parameter for advantage calculation.
        cliprange (float):
            Range for clipping in PPO policy gradient loss.
        cliprange_value (float):
            Range for clipping values in loss calculation.
        vf_coef (float):
            Scaling factor for value loss.
        early_stopping (bool):
            Whether to stop the PPO optimization loop early if the KL is too high.
        target_kl (float):
            Stop early if we exceed this value by over 50%.
        compare_steps (int):
            Number of steps between comparison of the current reward with the best seen so far.
        ratio_threshold (float):
            Skip mini-batches with high PPO ratios that can cause loss spikes.
        use_score_scaling (bool):
            Use score scaling.
        use_score_norm (bool):
            Use score normalization. Only applicable if use_score_scaling is True.
        score_clip (Optional[float]):
            Score clipping.
        whiten_advantages (bool):
            Whiten the advantages before computing the actor loss.
        grayen_advantages (bool):
            Only change the scale of the advantages to have a std of 1.
        whiten_rewards (bool):
            Whiten the rewards before compute advantages.
        temperature (float):
            The temperature used for sampling.
    """

    adap_kl_ctrl: bool = True
    init_kl_coef: Optional[float] = (0.0001,)
    kl_penalty: Literal["kl", "abs", "mse", "full", "control_variate"] = "kl"
    kl_penalty_loss_type: Optional[Literal["kl", "abs", "mse", "control_variate"]] = (
        "control_variate"
    )
    kl_penalty_loss_clip_max: float = 10
    kl_penalty_loss_clip_min: float = 0
    force_disable_kl_penalty: bool = False
    target: Optional[float] = 6.0
    horizon: Optional[int] = 10000
    gamma: float = 1
    lam: float = 0.96
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    early_stopping: bool = False
    target_kl: float = 1
    compare_steps: int = 1
    ratio_threshold: float = 10.0
    use_score_scaling: bool = False
    use_score_norm: bool = False
    score_clip: Optional[float] = None
    whiten_advantages: bool = True
    grayen_advantages: bool = False
    whiten_rewards: bool = False
    temperature: float = 0.6

    def __post_init__(self):
        assert self.temperature > 0, "Temperature should be positive."
        assert not (
            self.whiten_advantages and self.grayen_advantages
        ), "Either whiten or grayen advantages, not both."


class PPOTrainer(BaseTrainer):
    """
    PPO Trainer.
    Implementation of the PPO update rule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainer_hparams = PPOHParams(**kwargs.get("ppo_hparams", {}))
        self._set_process_log_level(logger)
        # TODO temp actor weights logging
        self.latest_actor_weights_hash = None
        self.latest_critic_weights_hash = None
        # TODO: pass this in combination with a trainer state
        # for checkpoint continuation
        self.checkpoint_path_to_load = None

    def step(self, episodes: EpisodeDataset) -> Path:
        """
        Performs a single update step using the dataset rollout under the current policy.
        Each update step can rum multiple epochs of optimization.
        """

        episodes = self._filter_episodes(episodes)
        episodes = self._hydrate_ref_log_probs(episodes)

        # TODO: Make this one function in the policy
        self.policy.init_actor_engine_if_needed(
            global_batch_size=self.global_batch_size,
            per_device_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            total_num_training_steps=self.total_num_training_steps,
        )

        self.policy.init_critic_engine_if_needed(
            global_batch_size=self.global_batch_size,
            per_device_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            total_num_training_steps=self.total_num_training_steps,
        )

        # Initilaize the hashes if they are none
        if self.latest_actor_weights_hash is None:
            self.latest_actor_weights_hash = get_model_hash(self.policy.actor.module)

        if self.latest_critic_weights_hash is None:
            self.latest_critic_weights_hash = get_model_hash(self.policy.critic.module)

        if (
            not self.policy.cache_ds_engines
            and self.checkpoint_path_to_load is not None
        ):
            # If we dont keep the engines in mem and we have already a checkpoint path from which we can load
            # we load in the engines

            # we also check if we loaded theocrrect engine here
            actor_pre_load_hash = get_model_hash(self.policy.actor.module)
            critic_pre_load_hash = get_model_hash(self.policy.critic.module)

            # engines are not cached, need to laod the latest weights from checkpoin path
            logger.info(f"Loading latest policy from latest policy path")
            self.policy.load_latest_policy_path(self.checkpoint_path_to_load)

            dist.barrier()
            self.distributed_state.wait_for_everyone()

            loaded_actor_weights_hash = get_model_hash(self.policy.actor.module)
            loaded_critic_weights_hash = get_model_hash(self.policy.critic.module)

            # the pre load default model weights should not be the same as the
            # cached weights (on iteration > 1)
            if self.state.iteration != 0:
                logger.info("checking loaded A/C weights")
                assert actor_pre_load_hash != loaded_actor_weights_hash
                assert critic_pre_load_hash != loaded_critic_weights_hash
                # The loaded weights from the cash should be the same
                # as the latest weights hashses
                assert self.latest_actor_weights_hash == loaded_actor_weights_hash
                assert self.latest_critic_weights_hash == loaded_critic_weights_hash
            else:
                logger.info(
                    "Intial state, we skip the A/C cache load check since the cache is empty"
                )

        # change to appropriate input structure
        episodes = self._hydrate_episodes(episodes)
        episodes = self._rescale_and_clip_scores(episodes)

        # TODO: eventually we can use these kls for K dapters
        kls = self._log_episodes_metrics(episodes)

        dataloader = prepare_data_loader_for_training(
            episodes,
            per_device_batch_size=self.per_device_batch_size,
            seed=self.seed,
            drop_last=False,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.dataloader_num_workers,
                "pin_memory": self.dataloader_pin_memory,
            },
        )

        steps_in_epoch = len(dataloader)
        optim_steps_in_epoch = steps_in_epoch // self.gradient_accumulation_steps
        optim_steps_in_epoch = max(optim_steps_in_epoch, 1)

        num_optimization_steps_in_iteration = (
            self.num_epochs_per_iteration * optim_steps_in_epoch
        )

        total_num_optimization_steps = (
            self.num_iterations * num_optimization_steps_in_iteration
        )

        dataloader_iter = iter(dataloader)
        progress_bar = tqdm(
            total=total_num_optimization_steps,
            disable=not self._is_main_process(),
            desc=f"Iteration {self.state.iteration}: Training",
            dynamic_ncols=True,
        )
        progress_bar.update(self.state.global_step)

        logger.info(f"********* PPOtraining step {self.state.iteration}*************")
        logger.info(f"Per device batch size: {self.per_device_batch_size}")
        logger.info(f"Dataloder num workers: {self.dataloader_num_workers}")
        logger.info(f"total_num_optimization_steps: {self.total_num_training_steps}")
        logger.info(
            f"num_optimization_steps_in_iteration:{num_optimization_steps_in_iteration}"
        )
        logger.info(f"current global step: {self.state.global_step}")

        actor_trainable_params = get_model_param_count(
            self.policy.actor, trainable_only=True
        )
        critic_trainable_params = get_model_param_count(
            self.policy.critic, trainable_only=True
        )

        logger.info(f"Actor trainable params = {actor_trainable_params}")
        logger.info(f"Critic trainable params = {critic_trainable_params}")

        running_metrics = {}
        accumulated_metrics = {}
        global_step_last_logged = self.state.global_step

        progress_bar = tqdm(
            total=total_num_optimization_steps,
            disable=not self._is_main_process(),
            desc=f"Iteration {self.state.iteration}: Training",
            dynamic_ncols=True,
        )
        progress_bar.update(self.state.global_step)

        # Set everything in train mode
        self.policy.actor.train()
        self.policy.critic.train()

        dist.barrier()
        for epoch in range(self.num_epochs_per_iteration):
            for step, batch in enumerate(dataloader_iter):
                is_grad_acc_boundary = (
                    self.policy.actor.is_gradient_accumulation_boundary()
                )
                if self.policy.critic is not None:
                    assert (
                        self.policy.critic.is_gradient_accumulation_boundary()
                        == is_grad_acc_boundary
                    ), "Actor and critic should have synchronized optimization steps"

                metrics = self._epoch_step(batch)
                self._update_metrics(running_metrics, accumulated_metrics, metrics)

                if is_grad_acc_boundary:
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    progress_bar.update(1)

                    should_log = self.state.global_step % self.logging_steps == 0
                    if should_log:
                        logger.info("Logging training metrics")
                        self._log_training_metrics(
                            global_step_last_logged,
                            accumulated_metrics,
                            progress_bar,
                        )
                        global_step_last_logged = self.state.global_step

            dataloader_iter = iter(dataloader)
        dist.barrier()

        for key, value in running_metrics.items():
            value = torch.tensor(value, device=self.policy.actor.device)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value = value.cpu().item() / dist.get_world_size()
            running_metrics[key] = value

        self.state.iteration += 1
        progress_bar.close()

        # TODO: delete this if it passes
        # Verify we actually trained
        new_actor_hash = get_model_hash(self.policy.actor.module)
        new_critic_hash = get_model_hash(self.policy.critic.module)
        assert new_actor_hash != self.latest_actor_weights_hash
        assert new_critic_hash != self.latest_critic_weights_hash
        self.latest_actor_weights_hash = new_actor_hash
        self.latest_critic_weights_hash = new_critic_hash

        # TODO: do this conditioned on caching ds engine logic
        # i.e., IF we cache the engines, we dont have to store their full state
        # in memory
        checkpoint_dir = self.project_root_dir / "policy" / "checkpoints"
        current_policy_checkpoint_path = (
            checkpoint_dir / self._get_automatic_checkpoint_name()
        )

        # set the path of the policy we want to load in the next iteration
        if not self.policy.cache_ds_engines:
            logger.info(
                f"setting checkpoint path to load to {current_policy_checkpoint_path}"
            )
            self.checkpoint_path_to_load = current_policy_checkpoint_path
            self.policy.checkpoint_latest_policy_path(current_policy_checkpoint_path)

        # Put up a block here such that other processes dont go delete the critic
        dist.barrier()
        self.distributed_state.wait_for_everyone()

        # Clean the old checkpoint inside the checkpoint dir
        self.policy.clean_old_temp_checkpoints(
            checkpoint_dir, exclude=[current_policy_checkpoint_path]
        )

        # destroy engines and release memory
        self.policy.destroy_ds_engines()
        release_memory()
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Latest policy path: {current_policy_checkpoint_path}")
        # Point this to the actors hf_pretrained for the inference / evaluation class
        return current_policy_checkpoint_path / "hf_pretrained"

    def _hydrate_episodes(self, episodes: Dataset) -> Dataset:
        """
        Takes the collated dataset and hydrates it with the
        logprobs and values under the current policy parameters.
        These will be the baseline logprobs and values i.e., pi_old(a|s)
        """
        with_actor_log_probs_path = (
            self.project_root_dir
            / "trainer"
            / f"episodes__iter{self.state.iteration:04d}"
            / "w_actor_log_probs"
        )
        episodes = self._hydrate_log_probs(episodes)

        if self._is_main_process():
            episodes.save_to_disk(str(with_actor_log_probs_path))
        self.distributed_state.wait_for_everyone()

        del episodes
        release_memory()

        episodes = Dataset.load_from_disk(str(with_actor_log_probs_path))

        with_actor_critic_log_probs_path = (
            self.project_root_dir
            / "trainer"
            / f"episodes__iter{self.state.iteration:04d}"
            / "w_actor_critic_log_probs"
        )

        episodes = self._hydrate_values(episodes)
        if self._is_main_process():
            episodes.save_to_disk(str(with_actor_critic_log_probs_path))
        self.distributed_state.wait_for_everyone()

        del episodes
        release_memory()

        episodes = Dataset.load_from_disk(str(with_actor_critic_log_probs_path))
        return episodes

    def _hydrate_log_probs(
        self, episodes: Dataset, column_name: str = COLUMN_ACTOR_SHIFTED_LOGPS
    ) -> Dataset:
        """
        Compute logprobs under the current (actor) policy and add them to the dataset.
        """
        # Create a distributed data loader such that the order of
        # episodes is preserved when batches are distributed across multiple processes.
        data_loader = prepare_data_loader_for_inference(
            episodes,
            per_device_batch_size=self.per_device_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.dataloader_num_workers,
                "pin_memory": self.dataloader_pin_memory,
            },
        )

        # Switch actor to eval mode before doing inference
        self.policy.actor.eval()
        dist.barrier()

        list_of_log_probs = []
        for inputs in tqdm(
            data_loader, desc="Computing log probs", disable=not self._is_main_process()
        ):
            with torch.no_grad():
                assert torch.all(
                    inputs["attention_mask"][:, 0] == 1
                ), "Expected first token to be unmasked (attention_mask=1)."
                assert inputs["input_ids"].shape[0] == self.per_device_batch_size, (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.per_device_batch_size}."
                )

                # Move inputs to the actor device
                inputs = {k: v.to(self.policy.actor.device) for k, v in inputs.items()}

                seq_lengths = inputs["attention_mask"].sum(dim=1).detach().clone()
                seq_lengths = seq_lengths.unsqueeze(1)  # Shape: (batch_size,

                # Forward pass on actor to get log probabilities for each token
                outputs = self.policy.forward_actor(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    return_all_logp=True,
                    trainer_hparams=self.trainer_hparams,
                )
                logps = outputs.all_logp.detach()
                assert logps.shape[1] == inputs["input_ids"].shape[1] - 1

                # Check that seq_lengths is indeed on CUDA
                seq_lengths = seq_lengths.to(self.policy.actor.device)
                seq_lengths = gather(seq_lengths).cpu()
                logps = pad_across_processes(
                    logps, dim=1, pad_index=0.0, pad_first=False
                )
                logps = gather(logps).cpu()

                assert (
                    logps.shape[0]
                    == inputs["input_ids"].shape[0] * dist.get_world_size()
                )

                # Convert 2D tensors to a list of lists
                logps_seq_lengths = seq_lengths - 1
                for i, seq_len in enumerate(logps_seq_lengths.squeeze().tolist()):
                    assert (
                        seq_len <= logps.shape[1]
                    ), f"seq_len={seq_len} is out of bounds for logps dim={logps.shape[1]}"
                    list_of_log_probs.append(logps[i, :seq_len].tolist())

        # Remove any extra log probs that were added due to global padding
        list_of_log_probs = list_of_log_probs[: len(episodes)]

        # Safely add the new column only once on the main process
        with self.distributed_state.main_process_first():
            episodes = episodes.add_column(name=column_name, column=list_of_log_probs)

        return episodes

    def _hydrate_values(self, episodes: Dataset) -> Dataset:
        """Compute the values and add them to the dataset"""
        # Create a distributed data loader such that the order of
        # episodes is preserved when distributed across multiple processes.
        data_loader = prepare_data_loader_for_inference(
            episodes,
            per_device_batch_size=self.per_device_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.dataloader_num_workers,
                "pin_memory": self.dataloader_pin_memory,
            },
        )

        self.policy.critic.eval()

        dist.barrier()

        list_of_values = []
        for inputs in tqdm(
            data_loader, desc="Computing values", disable=not self._is_main_process()
        ):
            with torch.no_grad():
                # Assume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert inputs["input_ids"].shape[0] == self.per_device_batch_size, (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.per_device_batch_size}."
                )

                critic_device = next(self.policy.critic.parameters()).device
                inputs = {k: v.to(critic_device) for k, v in inputs.items()}

                # Compute the sequence lengths as we need to extract
                # the values of the non-padded tokens
                seq_lengths = inputs["attention_mask"].sum(dim=1).detach().clone()
                seq_lengths = seq_lengths.unsqueeze(1)

                # Compute the values for each token
                outputs = self.policy.forward_critic(
                    attention_mask=inputs["attention_mask"],
                    input_ids=inputs["input_ids"],
                    labels=inputs["labels"],
                )
                values = outputs["values"].detach()
                assert values.shape[1] == inputs["input_ids"].shape[1]

                # Gather across all distributed processes
                seq_lengths = gather(seq_lengths).cpu()
                values = gather(
                    pad_across_processes(values, dim=1, pad_index=0.0, pad_first=False)
                ).cpu()

                # Convert 2d tensors to a list of lists
                for i, seq_len in enumerate(seq_lengths.squeeze().tolist()):
                    assert seq_len <= values.shape[1]
                    list_of_values.append(values[i, :seq_len].tolist())

        # Remove any extra values that were added due to padding
        list_of_values = list_of_values[: len(episodes)]

        with self.distributed_state.main_process_first():
            episodes = episodes.add_column(name=COLUMN_VALUES, column=list_of_values)

        return episodes

    def _hydrate_ref_log_probs(
        self, episodes: Dataset, column_name: str = COLUMN_REF_SHIFTED_LOGPS
    ) -> Dataset:
        logger.info("Computing refrence model logprobs")

        with_ref_log_probs_path = (
            self.project_root_dir
            / "trainer"
            / f"episodes__iter{self.state.iteration:04d}"
            / "w_ref_logp"
        )

        self.policy.init_reference_engine_if_needed(
            self.global_batch_size,
            self.per_device_batch_size,
            self.gradient_accumulation_steps,
            self.total_num_training_steps,
        )

        ## update the episodes
        data_loader = prepare_data_loader_for_inference(
            episodes,
            per_device_batch_size=self.per_device_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.dataloader_num_workers,
                "pin_memory": self.dataloader_pin_memory,
            },
        )

        self.policy.reference.eval()  # put the referenc emodel in non-training mode
        dist.barrier()

        ref_log_probs_list = []
        for inputs in tqdm(
            data_loader,
            desc="Computing reference log probs",
            disable=not self._is_main_process(),
        ):
            with torch.no_grad():
                # Asume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert inputs["input_ids"].shape[0] == self.per_device_batch_size, (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.per_device_batch_size}."
                )

                inputs = {
                    k: v.to(self.policy.reference.device) for k, v in inputs.items()
                }

                # Compute the sequence lengths as we need to extract
                # the log probs of the non-padded tokens
                seq_lengths = inputs["attention_mask"].sum(dim=1).detach().clone()
                seq_lengths = seq_lengths.unsqueeze(1)  # Shape: (batch_size, 1)

                # Compute the log probabilities for each token
                outputs = self.policy.forward_reference(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    return_all_logp=True,
                )

                logps = outputs.all_logp.detach()
                assert logps.shape[1] == inputs["input_ids"].shape[1] - 1

                seq_lengths = seq_lengths.to(self.policy.reference.device)
                seq_lengths = gather(seq_lengths).cpu()
                logps = gather(
                    pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)
                ).cpu()

                assert (
                    logps.shape[0]
                    == inputs["input_ids"].shape[0] * dist.get_world_size()
                )

                # Convert 2d tensors to a list of lists
                logps_seq_lengths = seq_lengths - 1
                for i, seq_len in enumerate(logps_seq_lengths.squeeze().tolist()):
                    assert seq_len <= logps.shape[1]
                    ref_log_probs_list.append(logps[i, :seq_len].tolist())

        ref_log_probs_list = ref_log_probs_list[
            : len(episodes)
        ]  # remove any extra log probs that were added due to global paddingS

        with self.distributed_state.main_process_first():
            episodes = episodes.add_column(name=column_name, column=ref_log_probs_list)
        if self._is_main_process():
            episodes.save_to_disk(str(with_ref_log_probs_path))
        self.distributed_state.wait_for_everyone()

        self.policy.destroy_reference_engine_if_not_cached()
        del episodes
        release_memory()

        episodes = Dataset.load_from_disk(str(with_ref_log_probs_path))
        return episodes

    def _epoch_step(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[float, torch.Tensor]]:
        inputs = {k: v.to(self.policy.actor.device) for k, v in inputs.items()}

        # Turn this into a dataclass
        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
        labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)
        scores = inputs["scores"]  # Shape: (batch_size,)
        logger.info(f"\n\n********************** Scores ********************")
        logger.info(f"scores = {scores}")

        shifted_labels = labels[
            ..., 1:
        ].contiguous()  # Shape: (batch_size, max_seq_len-1)

        shifted_labels_mask = (shifted_labels != -100).to(
            attention_mask.dtype
        )  # Shape: (batch_size, max_seq_len-1)
        logger.info(
            f"\n\n********************** shifted Label Masks ******************8"
        )
        logger.info(f"shifted label masks = {shifted_labels_mask}")

        # Note that this is the log probability of the actor model
        # in the beginning of this iteration (aka the old log probs)
        shifted_actor_logprobs = inputs[
            COLUMN_ACTOR_SHIFTED_LOGPS
        ]  # Shape: (batch_size, max_seq_len-1)
        assert shifted_actor_logprobs.shape == shifted_labels_mask.shape

        ###########################################
        # Compute rewards, retruns and advantages #
        ###########################################
        with torch.no_grad():
            if not self.trainer_hparams.force_disable_kl_penalty:
                shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            else:
                logger.info("KL penalty disabled")
                shifted_ref_logprobs = None

            assert shifted_ref_logprobs != None

            rewards, non_score_rewards, kls = self._compute_rewards(
                scores, shifted_actor_logprobs, shifted_ref_logprobs, attention_mask
            )
            logger.info(f"\n\n********************** Rewards********************")
            logger.info(f"scores = {rewards}")
            # The `advantages` is computed for the actions. That's why they are of shape (batch_size, max_seq_len-1)
            # Shape of `advantages`: (batch_size, max_seq_len-1)
            if "advantages" not in inputs:
                # Note that this is the value of the critic model in the beginning of
                # this iteration (aka the old values)
                values = inputs[COLUMN_VALUES]  # Shape: (batch_size, max_seq_len)
                valid_values = values[:, :-1]  # Shape: (batch_size, max_seq_len-1)
                assert valid_values.shape == shifted_actor_logprobs.shape
                valid_values = valid_values * shifted_labels_mask
                advantages, returns = self._compute_advantages(
                    valid_values, rewards, shifted_labels_mask
                )
                logger.info(
                    "\n\n**********************COMPUTING ADVANTTAGES********************************"
                )
                logger.info(f"Computed advantatges {advantages}")
                logger.info(f"Advantages shape: {advantages.shape}")
                logger.info(f"Advantages mean: {advantages.mean()}")

                logger.info(f"Computed returns {returns}")
                logger.info(f"Returns shape: {returns.shape}")
                logger.info(f"Returns mean: {returns.mean()}")
            else:
                logger.info("precomputed advanatags")
                precomputed_advantages = inputs[
                    "advantages"
                ]  # Shape: (batch_size, max_seq_len)

                # Shift the advantages to left to match the actions
                advantages = precomputed_advantages[
                    :, 1:
                ]  # Shape: (batch_size, max_seq_len-1)
                if self.trainer_hparams.whiten_advantages:
                    advantages = masked_whiten(
                        advantages,
                        shifted_labels_mask,
                        distributed=True,
                        unbiased_variance=True,
                    )
                elif self.trainer_hparams.grayen_advantages:
                    advantages = masked_rescale_by_std(
                        advantages,
                        shifted_labels_mask,
                        distributed=True,
                        unbiased_variance=True,
                    )
                valid_values = None
                returns = None

            assert advantages.shape == shifted_actor_logprobs.shape
            assert rewards.shape == shifted_actor_logprobs.shape

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        #################
        #   Actor Loss  #
        #################
        actor_loss, is_skipped, actor_metrics, approx_ref_kl = self.policy.actor_loss(
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_logprobs=shifted_actor_logprobs,
            ref_logprobs=shifted_ref_logprobs,
            advantages=advantages,
            trainer_hparams=self.trainer_hparams,
        )

        self.policy.actor.backward(actor_loss)
        self.policy.actor.step()

        # Get rid of actor's activations to free up memory
        actor_loss = actor_loss.detach().clone()
        release_memory()

        ###################
        #   Critic Loss   #
        ###################
        critic_loss, critic_metrics = self.policy.critic_loss(
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_valid_values=valid_values,
            returns=returns,
            trainer_params=self.trainer_hparams,
        )

        self.policy.critic.backward(critic_loss)
        self.policy.critic.step()
        # Get rid of critic's activations to free up memory
        critic_loss = critic_loss.detach().clone()

        release_memory()

        #########################
        #  Metrics Bookkeeping  #
        #########################
        metrics = {
            "advantages/mean": masked_mean(advantages, shifted_labels_mask).detach(),
            "advantages/std": masked_var(advantages, shifted_labels_mask)
            .detach()
            .sqrt(),
            "rewards/mean": masked_mean(rewards, shifted_labels_mask).detach(),
            "num_tokens": shifted_labels_mask.sum().detach(),
            "_num_participating_tokens": shifted_labels_mask.sum().detach(),
            **actor_metrics,
            **critic_metrics,
        }

        if returns is not None:
            metrics["returns"] = masked_mean(returns, shifted_labels_mask).detach()

        if non_score_rewards is not None:
            metrics["non_score_rewards"] = masked_mean(
                non_score_rewards, shifted_labels_mask
            )

        if kls is not None or approx_ref_kl is not None:
            if approx_ref_kl is not None:
                kls = approx_ref_kl
            metrics["kls"] = (kls * shifted_labels_mask).sum(dim=1).mean().detach()

        metrics["actor/loss"] = actor_loss
        if critic_loss is not None:
            metrics["critic/loss"] = critic_loss

        return metrics

    def _compute_rewards(
        self,
        scores: torch.FloatTensor,
        shifted_actor_logprobs: torch.FloatTensor,
        shifted_ref_logprobs: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the episodes; one scalar per episode, shape (`batch_size`)
            shifted_actor_logprobs (`torch.FloatTensor`):
                Log probabilities of the actor, shape (`batch_size`, `max_seq_len-1`)
            shifted_ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `max_seq_len-1`)
            attention_mask (`torch.LongTensor`):
                Mask for the input, shape (`batch_size`, `max_seq_len`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: Non-score rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `max_seq_len-1`)
        """
        if (
            shifted_ref_logprobs is not None
            and self.trainer_hparams.kl_penalty_loss_type is None
        ):
            kl = self._compute_kl_penalty(
                shifted_actor_logprobs,
                shifted_ref_logprobs,
                self.trainer_hparams.kl_penalty,
            )
            non_score_rewards = -self.kl_ctl.value * kl
        else:
            # KL penalty is not part of the reward
            logger.info("KL Penalty not part of the reward")
            kl = None
            non_score_rewards = torch.zeros_like(shifted_actor_logprobs)

        # Initialize the rewards with non-score rewards
        rewards = non_score_rewards.clone()

        # Find the last non-masked index for each sample in the batch
        last_non_masked_indices = (
            torch.cumsum(attention_mask, dim=1)[:, -1] - 1
        )  # Shape: (batch_size)
        # Since the length of shifted_actor_log_probs is `max_seq_len - 1`, we need to
        # subtract 1 from the last non-masked index to get the corresponding index
        last_non_masked_label_indices = last_non_masked_indices - 1

        # Reward is score + KL penalty
        batch_size = rewards.size(0)
        rewards[torch.arange(batch_size), last_non_masked_label_indices] += scores

        if kl is not None:
            kl = kl.detach()

        return rewards.detach(), non_score_rewards.detach(), kl

    def _compute_advantages(
        self,
        valid_values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        shifted_labels_mask: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the advantages from the values and rewards.

        Args:
            valid_values (`torch.FloatTensor`):
                The values of the responses, shape (`batch_size`, `max_seq_len-1`)
            rewards (`torch.FloatTensor`):
                The rewards of the responses, shape (`batch_size`, `max_seq_len-1`)
            shifted_labels_mask (`torch.LongTensor`):
                Left Shifted by 1 Mask for the labels (i.e. actions), shape (`batch_size`, `max_seq_len-1`)

        Returns:
            `torch.FloatTensor`: The advantages of the responses, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: The returns of the responses, shape (`batch_size`, `max_seq_len-1`)
        """
        lastgaelam = 0
        advantages_reversed = []
        actions_seq_len = rewards.shape[-1]

        # Make sure invalid rewards are masked
        rewards *= shifted_labels_mask

        if self.trainer_hparams.whiten_rewards:
            rewards = masked_whiten(
                rewards, shifted_labels_mask, shift_mean=False, distributed=True
            )

        for t in reversed(range(actions_seq_len)):
            next_state_values = (
                valid_values[:, t + 1] if t < (actions_seq_len - 1) else 0.0
            )
            delta = (
                rewards[:, t]
                + self.trainer_hparams.gamma * next_state_values
                - valid_values[:, t]
            )
            lastgaelam = (
                delta
                + self.trainer_hparams.gamma * self.trainer_hparams.lam * lastgaelam
            )
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        assert advantages.shape == rewards.shape

        logger.info(f" *************** Valid Values ***************** ")
        logger.info(f"valid values = {valid_values}")

        returns = advantages + valid_values
        if self.trainer_hparams.whiten_advantages:
            advantages = masked_whiten(
                advantages,
                shifted_labels_mask,
                distributed=True,
                unbiased_variance=True,
            )
        elif self.trainer_hparams.grayen_advantages:
            advantages = masked_rescale_by_std(
                advantages,
                shifted_labels_mask,
                distributed=True,
                unbiased_variance=True,
            )

        return advantages.detach(), returns.detach()

    log_keys_to_store_in_running_metrics = [
        "_num_participating_tokens",
    ]

    log_keys_weighed_by_num_participating_tokens = [
        "advantages/mean",
        "advantages/std",
        "rewards/mean",
        "returns",
        "non_score_rewards",
        "actor/loss",
        "actor/logit_entropy",
        "actor/approx_kl",
        "actor/policy_kl",
        "actor/clip_frac",
        "ratio",
        "critic/loss",
        "critic/value",
        "critic/mse",
        "critic/clip_frac",
    ]

    def _log_training_metrics(
        self,
        _globalstep_last_logged: int,
        accumulated_metrics: Dict[str, Union[float, torch.Tensor]],
        progress_bar: tqdm,
    ):
        # Wait for all processes to reach this point
        dist.barrier()

        logs: Dict[str, float] = {}

        # Compute the log values over all processes
        num_steps_since_last_log = (
            self.state.global_step - _globalstep_last_logged
        ) * self.gradient_accumulation_steps

        if "_num_participating_tokens" in accumulated_metrics:
            num_participating_tokens = accumulated_metrics["_num_participating_tokens"]
            dist.all_reduce(num_participating_tokens, op=dist.ReduceOp.SUM)
            num_participating_tokens = num_participating_tokens.item()
        else:
            num_participating_tokens = 1

        for metric_name, metric_value in accumulated_metrics.items():
            if metric_name.startswith("_"):
                continue
            if metric_value is None:
                continue

            is_weighed_by_num_actions = (
                metric_name in self.log_keys_weighed_by_num_participating_tokens
            )

            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.to(self.policy.actor.device)
                dist.all_reduce(metric_value, op=dist.ReduceOp.SUM)
                metric_value = metric_value.item()
                divisor = dist.get_world_size()
            else:
                assert not is_weighed_by_num_actions
                divisor = 1

            if is_weighed_by_num_actions:
                metric_value /= num_participating_tokens
            else:
                metric_value /= divisor * num_steps_since_last_log

            logs[metric_name] = round(metric_value, 8)

        logs["actor/lr"] = self._get_learning_rate(self.policy.actor)
        logs["critic/lr"] = self._get_learning_rate(self.policy.critic)
        logs["epoch"] = round(self.state.epoch, 4)
        logs["step"] = self.state.global_step
        logs["actor/ds_step"] = self.policy.actor.global_steps

        if self.policy.critic is not None:
            logs["critic/ds_step"] = self.policy.critic.global_steps

        # First log the metrics on the progress bar
        progress_bar.set_postfix(logs)

        # Add "train/" prefix for clarity.
        logs = {f"train/{k}": v for k, v in logs.items()}

        logger.info(f"LOGGING TRAINING METRICS {logs}")
        self._cloud_log({**logs, "train/global_step": self.state.global_step})

        # Reset the accumulated metrics
        for key in accumulated_metrics.keys():
            accumulated_metrics[key] -= accumulated_metrics[key]

    def _update_metrics(
        self,
        running_metrics: Dict[str, Union[torch.Tensor, float]],
        accumulated_metrics: Dict[str, Union[torch.Tensor, float]],
        step_metrics: Dict[str, Union[torch.Tensor, float]],
    ):
        dist.barrier()

        def get_initial_value(
            val: Union[float, torch.Tensor],
        ) -> Union[float, torch.Tensor]:
            if isinstance(val, torch.Tensor):
                # Force to float from the start
                return torch.tensor(0.0, dtype=torch.float32, device=val.device)
            return 0.0

        # Initialize running metrics if not already initialized
        for key in step_metrics.keys():
            if key in accumulated_metrics:
                continue
            accumulated_metrics[key] = get_initial_value(step_metrics[key])
            if key in self.log_keys_to_store_in_running_metrics:
                if key not in running_metrics:
                    running_metrics[key] = get_initial_value(step_metrics[key])

        num_tokens = step_metrics["_num_participating_tokens"].item()

        for key, value in step_metrics.items():
            if value is None:
                continue

            # Debug print: If it's a tensor, check dtype
            if isinstance(value, torch.Tensor):
                if value.dtype not in (torch.float32, torch.float64):
                    # This debug line helps identify the offending key
                    logger.warning(
                        f"[DEBUG] Found '{key}' as {value.dtype}; converting to float32."
                    )
                    value = value.to(torch.float32)

            if key in self.log_keys_weighed_by_num_participating_tokens:
                weight = num_tokens
            else:
                weight = 1

            value = value * weight
            accumulated_metrics[key] += value

        running_metrics["_num_participating_tokens"] += num_tokens
