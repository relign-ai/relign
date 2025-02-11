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

from relign.common.deepspeed_utils import prepare_data_loader_for_inference
from relign.common.dataset import EpisodeDataset
from relign.algorithms.base_trainer import BaseTrainer
from relign.utils.trainer import prepare_data_loader_for_training, masked_mean

from relign.algorithms.grpo.data_collator import (
    GRPODataCollator,
    GroupedBatchSampler,
    COLUMN_ACTOR_SHIFTED_LOGPS,
    COLUMN_REF_SHIFTED_LOGPS,
)

from relign.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GRPOParams:
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
    use_score_scaling: bool = False
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


class GRPOTrainer(BaseTrainer):
    """
    PPO Trainer.
    Impelmentation of the PPO update rule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainer_hparams = GRPOParams(**kwargs.get("grpo_params", {}))

    def step(self, episodes: EpisodeDataset) -> None:
        """
        Performs a single update step using the dataset rollout under the current policy.
        Each updatestep can rum multiple epochs of optimization.
        """

        self.policy.init_reference_engine_if_needed(
            self.global_batch_size,
            self.per_device_batch_size,
            self.gradient_accumulation_steps,
            self.total_num_training_steps,
        )

        self.policy.init_actor_engine_if_needed(
            global_batch_size=self.global_batch_size,
            per_device_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            total_num_training_steps=self.total_num_training_steps,
        )

        if not self.policy.cache_ds_engines:
            # engines are not cached, need to laod the latest weights from checkpoin path
            logger.info(f"Loading latest policy from latest policy path")
            self.policy.load_latest_policy_path()
            self.policy._clean_old_temp_checkpoints(
                self.project_root_dir / "policy" / "cache"
            )
        # change to appropriate input structure
        # episodes = self._hydrate_episodes(episodes)

        episodes = self._rescale_and_clip_scores(episodes)
        kls = self._log_episodes_metrics(episodes)
        num_groups = len(episodes.unique("group"))
        logger.info(f"Number of groups: {num_groups}")

        dataloader = DataLoader(
            episodes,
            batch_sampler=GroupedBatchSampler(
                episodes,
                group_column="group",
                groups_per_step=num_groups,
            ),
            collate_fn=GRPODataCollator(),
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
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
        # Set the actor in train mode
        self.policy.actor.train()

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

        dist.barrier()
        for epoch in range(self.num_epochs_per_iteration):
            for step, batch in enumerate(dataloader_iter):
                is_grad_acc_boundary = (
                    self.policy.actor.is_gradient_accumulation_boundary()
                )

                metrics = self._step(batch)
                self._update_metrics(running_metrics, accumulated_metrics, metrics)

                # self._update_metrics(running_metrics, accumulated_metrics, metrics)
                if is_grad_acc_boundary:
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    progress_bar.update(1)

                    should_log = self.state.global_step % self.logging_steps == 0
                    if should_log:
                        logger.info(" logging training metrics")
                        self._log_training_metrics(
                            global_step_last_logged,
                            accumulated_metrics,
                            progress_bar,
                        )
                        global_step_last_logged = self.state.global_step

            dataloader_iter = iter(dataloader)
            dist.barrier()

            self.state.iteration += 0
            progress_bar.close()
            checkpoint_path = self.project_root_dir / "policy" / "cache"
            # checkpoint_name = self._get_automatic_checkpoint_name()

            self.policy.save_latest_policy_path(checkpoint_path)
            dist.barrier()

            # destroy engines and release memory
            self.policy.destroy_ds_engines()
            self.policy.destroy_reference_engine_if_not_cached()

            dist.barrier()
            release_memory()
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            latest_policy_path = checkpoint_path / "actor" / "hf_pretrained"
            logger.info(f"Latest policy path: {latest_policy_path}")

            # return latest policy path
            return latest_policy_path

    def _compute_actor_log_probs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> List[List[float]]:
        """
        Replaces `_hydrate_log_probs` for a single batch.

        Returns a *list of lists* of log probabilities for each example, truncated to
        `seq_len - 1` tokens. The shape can be ragged if your sequences differ in length.

        You can call this inside your training step *instead* of storing them in a Dataset.
        """

        # Make sure the actor model is in eval mode if you don't want dropout, etc.
        # (If you need it in train mode for some reason, you can remove this.)
        self.policy.actor.eval()

        # Move everything to the actor's device
        inputs = {k: v.to(self.policy.actor.device) for k, v in batch.items()}

        # For each sequence, find how many tokens are non-padding
        # shape: (batch_size, 1)
        seq_lengths = inputs["attention_mask"].sum(dim=1, keepdim=True).detach()

        with torch.no_grad():
            # Forward pass on the actor to get token-wise log probabilities
            outputs = self.policy.forward_actor(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],  # or however labels are stored
                return_all_logp=True,  # similar to your existing code
            )
            # outputs.all_logp has shape: [batch_size, seq_len-1]
            logps = outputs.all_logp.detach()

        # Prepare for distributed gather
        seq_lengths = seq_lengths.to(self.policy.actor.device)
        seq_lengths = gather(seq_lengths).cpu()  # shape [global_batch_size, 1]

        # Also gather the logps across processes - first pad them so gather works
        logps = pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)
        logps = gather(logps).cpu()  # shape [global_batch_size, padded_seq_len-1]

        # Convert to a list of lists, trimming each row to its true seq_length-1
        final_log_probs = []
        seq_lengths_minus_one = (
            seq_lengths - 1
        )  # we skip the last token for "shifted" log probs
        for i, seq_len in enumerate(seq_lengths_minus_one.squeeze().tolist()):
            seq_len_int = int(seq_len)
            # Safeguard to avoid out-of-bounds
            assert (
                seq_len_int <= logps.shape[1]
            ), f"seq_len={seq_len_int} exceeds logps dim={logps.shape[1]}"
            # Extract the valid portion for example i
            row_i = logps[i, :seq_len_int].tolist()
            final_log_probs.append(row_i)

        # Return a ragged list-of-lists with shape [global_batch_size, variable_length].
        return final_log_probs

    def _compute_reference_log_probs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> List[List[float]]:
        """
        Replaces `_hydrate_log_probs` for a single batch.

        Returns a *list of lists* of log probabilities for each example, truncated to
        `seq_len - 1` tokens. The shape can be ragged if your sequences differ in length.

        You can call this inside your training step *instead* of storing them in a Dataset.
        """

        # Make sure the referrence model is in eval mode if you don't want dropout, etc.
        # (If you need it in train mode for some reason, you can remove this.)
        self.policy.reference.eval()

        # Move everything to the actor's device
        inputs = {k: v.to(self.policy.actor.device) for k, v in batch.items()}

        # For each sequence, find how many tokens are non-padding
        # shape: (batch_size, 1)
        seq_lengths = inputs["attention_mask"].sum(dim=1, keepdim=True).detach()

        with torch.no_grad():
            # Forward pass on the actor to get token-wise log probabilities
            outputs = self.policy.forward_reference(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],  # or however labels are stored
                return_all_logp=True,  # similar to your existing code
            )
            # outputs.all_logp has shape: [batch_size, seq_len-1]
            logps = outputs.all_logp.detach()

        # Prepare for distributed gather
        seq_lengths = seq_lengths.to(self.policy.reference.device)
        seq_lengths = gather(seq_lengths).cpu()  # shape [global_batch_size, 1]

        # Also gather the logps across processes - first pad them so gather works
        logps = pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)
        logps = gather(logps).cpu()  # shape [global_batch_size, padded_seq_len-1]

        # Convert to a list of lists, trimming each row to its true seq_length-1
        final_log_probs = []
        seq_lengths_minus_one = (
            seq_lengths - 1
        )  # we skip the last token for "shifted" log probs
        for i, seq_len in enumerate(seq_lengths_minus_one.squeeze().tolist()):
            seq_len_int = int(seq_len)
            # Safeguard to avoid out-of-bounds
            assert (
                seq_len_int <= logps.shape[1]
            ), f"seq_len={seq_len_int} exceeds logps dim={logps.shape[1]}"
            # Extract the valid portion for example i
            row_i = logps[i, :seq_len_int].tolist()
            final_log_probs.append(row_i)

        # Return a ragged list-of-lists with shape [global_batch_size, variable_length].
        return final_log_probs

    def _step(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[float, torch.Tensor]]:
        # noinspection DuplicatedCode
        inputs = {k: v.to(self.policy.actor.device) for k, v in inputs.items()}

        shifted_ref_log_probs = self._compute_ref_log_probs(inputs)
        shifted_actor_log_probs = self._compute_actor_log_probs(inputs)

        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        scores = inputs[
            "scores"
        ]  # make sure these are not hussled, on devices because they
        groups = inputs["group"]
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)

        mean_rewards, std_rewards, unique_group_ids, per_token_kl = (
            self._compute_rewards(
                scores=scores,
                groups=groups,
                shifted_actor_logprobs=shifted_ref_log_probs,
                shifted_ref_logprobs=shifted_actor_log_probs,
            )
        )

        advantages = self._compute_advantages(
            rewards=scores,
            unique_ids=unique_group_ids,
            mean_rewards=mean_rewards,
            std_rewards=std_rewards,
            per_token_kl=per_token_kl,
            groups=groups,
            shifted_actor_log_probs=shifted_actor_log_probs,
        )

        shifted_labels = labels[
            ..., 1:
        ].contiguous()  # Shape: (batch_size, max_seq_len-1)
        shifted_labels_mask = (shifted_labels != -100).to(
            attention_mask.dtype
        )  # Shape: (batch_size, max_seq_len-1)

        # Note that this is the log probability of the actor model
        # in the beginning of this iteration (aka the old log probs)
        shifted_actor_logprobs = inputs[
            COLUMN_ACTOR_SHIFTED_LOGPS
        ]  # Shape: (batch_size, max_seq_len-1)
        assert shifted_actor_logprobs.shape == shifted_labels_mask.shape

        #  Compute the rewards, advantages, and returns
        with torch.no_grad():
            # TODO: add KL Penalty Here
            if not self.trainer_hparams.force_disable_kl_penalty:
                shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            else:
                shifted_ref_logprobs = None

            # Shape of rewards: (batch_size, max_seq_len-1)
            mean_rewards, std_rewards, unique_ids, per_token_kl = self._compute_rewards(
                scores=scores,
                groups=groups,
                shifted_actor_logprobs=shifted_actor_logprobs,
                shifted_ref_logprobs=shifted_ref_logprobs,
            )

            # Shape of `advantages`: (batch_size, max_seq_len-1)
            if "advantages" not in inputs:
                advantages = self._compute_advantages(
                    rewards=scores,
                    groups=groups,
                    unique_ids=unique_ids,
                    mean_rewards=mean_rewards,
                    std_rewards=std_rewards,
                    per_token_kl=per_token_kl,
                    shifted_actor_log_probs=shifted_actor_logprobs,
                    shifted_labels_mask=shifted_labels_mask,
                    attention_mask=attention_mask,
                )
            else:
                precomputed_advantages = inputs["advantages"]
                advantages = precomputed_advantages[:, 1:]

            # assert rewards.shape == shifted_actor_logprobs.shape

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        assert advantages.shape == shifted_actor_logprobs.shape
        assert advantages.shape == shifted_ref_logprobs.shape
        # Step 2: Compute the actor loss
        actor_loss, is_skipped, actor_metrics, approx_ref_kl = self.policy.actor_loss(
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_logprobs=shifted_actor_logprobs,
            ref_logprobs=shifted_ref_logprobs,
            advantages=advantages,
        )

        self.policy.actor.backward(actor_loss)
        self.policy.actor.step()

        # Get rid of actor's activations to free up memory
        actor_loss = actor_loss.detach().clone()
        release_memory()

        metrics = {
            "advantages/mean": masked_mean(advantages, shifted_labels_mask).detach(),
            "rewards/mean": mean_rewards,
            # masked_mean(mean_rewards, shifted_labels_mask).detach(),
            "num_tokens": shifted_labels_mask.sum().detach(),
            "_num_participating_tokens": shifted_labels_mask.sum().detach(),
            **actor_metrics,
        }
        # if returns is not None:
        #     metrics["returns"] = masked_mean(returns, shifted_labels_mask).detach()
        metrics["actor/loss"] = actor_loss

        return metrics

        return metrics

    def _compute_rewards(
        self,
        scores,
        groups,
        shifted_ref_logprobs=None,
        shifted_actor_logprobs=None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute per token rewards from scores and KL-penalty.
        Args:
            scores (`torch.FloatTensor`):
                Outcome Scores from the episodes; one scalar per episode, shape (`batch_size`)
            shifted_actor_logprobs (`torch.FloatTensor`):
                Log probabilities of the actor, shape (`batch_size`, `max_seq_len-1`)
            shifted_ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `max_seq_len-1`)
            attention_mask (`torch.LongTensor`):
                Mask for the input, shape (`batch_size`,`max_seq_len`)
            groups (`torch.LongTensor`):
                Group indices, shape `)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: Non-score rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `max_seq_len-1`)
        """
        if (
            shifted_ref_logprobs is not None
            and self.trainer_hparams.kl_penalty_loss_type is None
        ):
            per_token_kl = self._compute_kl_penalty(
                shifted_actor_logprobs,
                shifted_ref_logprobs,
            )
        else:
            # KL penalty is not part of the reward
            per_token_kl = None

        # 1) Identify each unique group and get index mapping
        unique_ids, group_idx = torch.unique(groups, return_inverse=True)
        num_groups = unique_ids.size(0)

        # 2) Prepare accumulators on the same device
        device = scores.device
        sums = torch.zeros(num_groups, device=device, dtype=scores.dtype)
        sum_of_squares = torch.zeros(num_groups, device=device, dtype=scores.dtype)
        counts = torch.zeros(num_groups, device=device, dtype=scores.dtype)

        # 3) Accumulate sums, sum_of_squares, counts in one pass
        sums.index_add_(0, group_idx, scores)
        sum_of_squares.index_add_(0, group_idx, scores**2)
        counts.index_add_(0, group_idx, torch.ones_like(scores))

        # 4) Compute mean & std
        means = sums / counts.clamp_min(1e-7)
        variances = sum_of_squares / counts.clamp_min(1e-7) - means**2
        stds = variances.clamp_min(0).sqrt()

        if per_token_kl is not None:
            per_token_kl = per_token_kl.detach()

        return means, stds, unique_ids, per_token_kl

    def _compute_advantages(
        self,
        rewards: torch.FloatTensor,
        unique_ids: torch.LongTensor,
        mean_rewards: torch.FloatTensor,
        std_rewards: torch.FloatTensor,
        per_token_kl: torch.FloatTensor,
        groups: torch.LongTensor,  # (batch_size,1)
        shifted_actor_log_probs: Optional[torch.FloatTensor] = None,
        shifted_labels_mask: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Compute the advantages from the values and rewards.

        Args:
            mean_rewards (`torch.FloatTensor`): (log_probs)

        Returns:
            `torch.FloatTensor`: The advantages of the group, (`group`, `group_size`, `max_seq_len-1`)
        """
        group_advantages = torch.zeros_like(rewards)
        for i, group in enumerate(groups):
            group_advantages[i] = rewards[i] - mean_rewards[unique_ids == group] / (
                std_rewards[unique_ids == group] + 1e-4
            )

        last_non_masked_indices = torch.cumsum(attention_mask, dim=1)[:, -1] - 1

        last_non_masked_label_indices = (
            last_non_masked_indices - 1
        )  # contains the last indic for each row fow which
        # we have to set the advantees to the group advante . i.e., for each row, assign its row advante from column 0 to the last index
        advantages = torch.zeros_like(per_token_kl)
        advantages[torch.arange(advantages.size(0)), last_non_masked_label_indices] += (
            group_advantages
        )
        advantages -= per_token_kl

        assert advantages.shape == shifted_actor_log_probs.shape
        return advantages.detach()

    def _hydrate_ref_log_probs(
        self, episodes: Dataset, column_name: str = COLUMN_REF_SHIFTED_LOGPS
    ) -> Dataset:
        logger.info("Computing refrence model logprobs")

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

        self.policy.destroy_reference_engine_if_not_cached()
        return episodes

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
