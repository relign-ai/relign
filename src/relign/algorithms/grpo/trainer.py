import io
import hashlib
from typing import Dict, Tuple, Union, Optional, Literal, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from deepspeed import comm as dist
from accelerate.utils import gather, pad_across_processes, release_memory
from datasets import Dataset
from tqdm import tqdm

from transformers.trainer_pt_utils import get_model_param_count
from relign.common.deepspeed_utils import prepare_data_loader_for_inference
from relign.common.dataset import EpisodeDataset
from relign.algorithms.base_trainer import BaseTrainer
from relign.utils.trainer import prepare_data_loader_for_training, masked_mean

from relign.algorithms.grpo.data_collator import (
    GRPODataCollator,
    # GroupedBatchSampler,
    COLUMN_REF_SHIFTED_LOGPS,
)

from relign.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GRPOParams:
    """
    Configuration class for GRPOTrainer.

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
    init_kl_coef: Optional[float] = 0.2
    kl_penalty: Literal["kl", "abs", "mse", "full", "control_variate"] = "kl"
    kl_penalty_loss_type: Optional[Literal["kl", "abs", "mse", "control_variate"]] = (
        None
    )
    kl_penalty_loss_clip_max: float = 10000
    kl_penalty_loss_clip_min: float = 0
    force_disable_kl_penalty: bool = False
    target: Optional[float] = 6.0
    horizon: Optional[int] = 10000
    gamma: float = 1
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    early_stopping: bool = False
    target_kl: float = 1
    compare_steps: int = 1
    ratio_threshold: float = 10.0
    use_score_scaling: bool = True  # Since our scores fall between [0, 2]  we scale
    use_score_norm: bool = False
    score_clip: Optional[float] = None
    whiten_advantages: bool = True
    grayen_advantages: bool = False
    whiten_rewards: bool = False
    temperature: float = 1.0

    def __post_init__(self):
        assert self.temperature > 0, "Temperature should be positive."
        assert not (
            self.whiten_advantages and self.grayen_advantages
        ), "Either whiten or grayen advantages, not both."


def get_model_hash(model):
    """Small helper function that returns the hash of a models weights"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return hashlib.sha256(buffer.read()).hexdigest()


class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer.
    Impelmentation of the GRPO update rule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainer_hparams = GRPOParams(**kwargs.get("grpo_params", {}))
        self._set_process_log_level(logger)
        self.latest_actor_weights_hash = None
        self.checkpoint_path_to_load = None

    def step(self, episodes: EpisodeDataset) -> None:
        """
        Performs a single update step using the dataset rollout under the current policy.
        Each updatestep can rum multiple epochs of optimization.
        """

        episodes = self._filter_episodes(episodes)
        episodes = self._rescale_and_clip_scores(episodes)
        episodes = self._hydrate_ref_log_probs(
            episodes, column_name=COLUMN_REF_SHIFTED_LOGPS
        )

        self.policy.init_actor_engine_if_needed(
            global_batch_size=self.global_batch_size,
            per_device_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            total_num_training_steps=self.total_num_training_steps,
        )

        # Initilaize the hashes if they are none
        if self.latest_actor_weights_hash is None:
            self.latest_actor_weights_hash = get_model_hash(self.policy.actor.module)

        if (
            not self.policy.cache_ds_engines
            and self.checkpoint_path_to_load is not None
        ):
            # we also check if we loaded theocrrect engine here
            actor_pre_load_hash = get_model_hash(self.policy.actor.module)

            # engines are not cached, need to laod the latest weights from checkpoin path
            logger.info(f"Loading latest policy from latest policy path")
            self.policy.load_latest_policy_path(self.checkpoint_path_to_load)

            dist.barrier()
            self.distributed_state.wait_for_everyone()

            loaded_actor_weights_hash = get_model_hash(self.policy.actor.module)

            # the pre load default model weights should not be the same as the
            # cached weights (on iteration > 1)
            if self.state.iteration != 0:
                logger.info("checking loaded A/C weights")
                assert actor_pre_load_hash != loaded_actor_weights_hash
                # The loaded weights from the cash should be the same
                # as the latest weights hashses
                assert self.latest_actor_weights_hash == loaded_actor_weights_hash
            else:
                logger.info(
                    "Intial state, we skip the A/C cache load check since the cache is empty"
                )

        kls = self._log_episodes_metrics(episodes)

        num_groups = len(episodes.unique("group"))
        logger.info(f"Number of groups: {num_groups}")

        dataloader = prepare_data_loader_for_training(
            episodes,
            per_device_batch_size=self.per_device_batch_size,
            seed=self.seed,
            drop_last=False,
            data_loader_kwargs={
                "collate_fn": GRPODataCollator(),
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

        logger.info(f"Per device batch size: {self.per_device_batch_size}")
        logger.info(f"Dataloder num workers: {self.dataloader_num_workers}")
        logger.info(f"total_num_optimization_steps: {self.total_num_training_steps}")
        logger.info(
            f"num_optimization_steps_in_iteration:{num_optimization_steps_in_iteration}"
        )

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

        # Set the actor in train mode
        self.policy.actor.train()

        dist.barrier()
        for epoch in range(self.num_epochs_per_iteration):
            for step, batch in enumerate(dataloader_iter):
                is_grad_acc_boundary = (
                    self.policy.actor.is_gradient_accumulation_boundary()
                )

                metrics = self._epoch_step(batch)
                self._update_metrics(running_metrics, accumulated_metrics, metrics)

                if is_grad_acc_boundary:
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    progress_bar.update(1)

                    should_log = self.state.global_step % self.logging_steps == 0
                    if should_log:
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
        assert new_actor_hash != self.latest_actor_weights_hash
        self.latest_actor_weights_hash = new_actor_hash

        # TODO: do this conditioned on caching ds engine logic
        # i.e., IF we cache the engines, we dont have to store their full state
        # in memory
        checkpoint_dir = self.project_root_dir / "policy" / "checkpoints"
        current_policy_checkpoint_path = (
            checkpoint_dir / self._get_automatic_checkpoint_name()
        )

        # set the path of the policy we want to load in the next iteration
        if not self.policy.cache_ds_engines:
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
                "collate_fn": GRPODataCollator(),
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

    def _epoch_step(self, inputs: dict) -> dict:
        """
        One iteration step of your GRPO training.
        """
        # Move everything to the correct device
        device = self.policy.actor.device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        batch_size = inputs["input_ids"].size(0)

        # There's either a single max_seq_len per sample or we assume they're all identical.
        # If in your collator we appended the same int for every sample,
        # we can just take the first's value:
        max_seq_len = int(inputs["max_seq_length"][0].item())

        # Gather shapes
        input_ids = inputs["input_ids"]  # (B, seq_len)
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        advantages = inputs["advantages"]  # shape (B,)
        groups = inputs["group"]  # shape (B,)
        len_query_token_ids = inputs["len_query_token_ids"]  # shape (B,)
        len_response_token_ids = inputs["len_response_token_ids"]  # shape (B,)

        # Log some basic batch information
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"max_seq_len: {max_seq_len}")
        logger.info(f"input_ids shape: {input_ids.shape}")
        logger.info(f"attention_mask shape: {attention_mask.shape}")
        logger.info(f"labels shape: {labels.shape}")
        logger.info(
            f"advantages shape: {advantages.shape}, sample scores: {advantages[:5]}"
        )
        logger.info(f"groups shape: {groups.shape}, sample groups: {groups[:5]}")
        logger.info(
            f"len_query_token_ids shape: {len_query_token_ids.shape}, "
            f"sample: {len_query_token_ids[:5]}"
        )
        logger.info(
            f"len_response_token_ids shape: {len_response_token_ids.shape}, "
            f"sample: {len_response_token_ids[:5]}"
        )

        shifted_labels = labels[
            ..., 1:
        ].contiguous()  # Shape: (batch_size, max_seq_len-1)

        shifted_labels_mask = (shifted_labels != -100).to(
            attention_mask.dtype
        )  # Shape: (batch_size, max_seq_len-1)

        ##################################################
        #              Compute the actor loss            $
        ##################################################
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        logger.info(f"device {self.distributed_state.process_index} going into loss")

        # Step 2: Compute the actor loss
        actor_loss, is_skipped, actor_metrics, approx_ref_kl, per_token_advantages = (
            self.policy.actor_loss_grpo(
                model_inputs=model_inputs,
                shifted_labels_mask=shifted_labels_mask,
                ref_logprobs=inputs[COLUMN_REF_SHIFTED_LOGPS],
                advantages=advantages,
                trainer_hparams=self.trainer_hparams,
            )
        )

        self.policy.actor.backward(actor_loss)
        self.policy.actor.step()

        # Get rid of actor's activations to free up memory
        actor_loss = actor_loss.detach().clone()
        release_memory()

        #####################
        # Log epoch metrics #
        #####################
        assert per_token_advantages.shape == shifted_labels_mask.shape
        metrics = {
            "advantages/mean": masked_mean(
                per_token_advantages, shifted_labels_mask
            ).detach(),
            "num_tokens": shifted_labels_mask.sum().detach(),
            "_num_participating_tokens": shifted_labels_mask.sum().detach(),
            **actor_metrics,
        }

        metrics["actor/loss"] = actor_loss
        if approx_ref_kl is not None:
            if approx_ref_kl is not None:
                kls = approx_ref_kl
            metrics["kls"] = (kls * shifted_labels_mask).sum(dim=1).mean().detach()

        return metrics

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
                return torch.tensor(0.0, dtype=val.dtype, device=val.device)
            return 0.0

        # Initialize running metrics if not already initialized
        for key in step_metrics.keys():
            if key in accumulated_metrics:
                continue
            log_keys_to_store_in_running_metrics = [
                "_num_participating_tokens",
            ]
            accumulated_metrics[key] = get_initial_value(step_metrics[key])
            if key in log_keys_to_store_in_running_metrics:
                if key not in running_metrics:
                    running_metrics[key] = get_initial_value(step_metrics[key])

        num_tokens = step_metrics["_num_participating_tokens"].item()

        for key, value in step_metrics.items():
            if value is None:
                continue
            if True:
                weight = num_tokens

            value = value * weight
            accumulated_metrics[key] += value

        # Update Running Metrics
        running_metrics["_num_participating_tokens"] += num_tokens

    def _log_training_metrics(
        self,
        _globalstep_last_logged: int,
        accumulated_metrics: Dict[str, Union[float, torch.Tensor]],
        progress_bar: tqdm,
    ):
        # Wait for all processes to reach this point
        logger.info(
            f"\n\n{self.distributed_state.process_index} waiting to log training metrics"
        )
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
        logs["epoch"] = round(self.state.epoch, 4)
        logs["step"] = self.state.global_step
        logs["actor/ds_step"] = self.policy.actor.global_steps

        # First log the metrics on the progress bar
        progress_bar.set_postfix(logs)

        # Add "train/" prefix for clarity.
        logs = {f"train/{k}": v for k, v in logs.items()}
        logger.info(f"logs {logs}")

        self._cloud_log({**logs, "train/global_step": self.state.global_step})

        # Reset the accumulated metrics
        for key in accumulated_metrics.keys():
            accumulated_metrics[key] -= accumulated_metrics[key]

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
