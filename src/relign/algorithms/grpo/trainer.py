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
        from relign.utils.gpu import get_gpu_memory
        logger.info(f"episodes {episodes}")
        logger.info(f"mem pre reference {get_gpu_memory()}")
        self.policy.init_reference_engine_if_needed(
            self.global_batch_size,
            self.per_device_batch_size,
            self.gradient_accumulation_steps,
            self.total_num_training_steps,
        )
        logger.info(f"mem post refernece {get_gpu_memory()}")

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

        episodes = self._rescale_and_clip_scores(episodes)
        # kls = self._log_episodes_metrics(episodes)
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
                # self._update_metrics(running_metrics, accumulated_metrics, metrics)

                if is_grad_acc_boundary:
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    progress_bar.update(1)

                    should_log = self.state.global_step % self.logging_steps == 0
                    if should_log:
                        logger.info(" logging training metrics")
                        # self._log_training_metrics(
                        #     global_step_last_logged,
                        #     accumulated_metrics,
                        #     progress_bar,
                        # )
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

    def shift_and_pad_log_probs_batched(
        actor_log_probs: torch.Tensor,
        query_token_ids: torch.Tensor,
        response_token_ids: torch.Tensor,
        max_seq_length: int,
        pad_logp: float = -1e30,
    ) -> torch.Tensor:
        """
        For each row i in the batch:
        1. figure out how many tokens are in the query (q_len) and response (r_len).
        2. copy actor_log_probs[i, :r_len] into the correct offset in the final array
            (the offset is q_len - 1, since typically the first token is a BOS/PAD).
        3. fill the rest with the pad value.
        Args:
            actor_log_probs: 2D Tensor of shape (batch_size, <=some_seq_len_minus1).
            E.g. the raw unshifted log probs right after computing log_softmax(...).
            query_token_ids: A 2D Tensor of shape (batch_size, <=some_query_len).
            Or any shape that lets us figure out how many query tokens are used.
            response_token_ids: A 2D Tensor of shape (batch_size, <=some_response_len).
            pad_logp: The value to fill for padded spans.
        Returns:
            shifted_logps: A 2D Tensor of shape (batch_size, max_seq_len - 1).
        """
        batch_size = actor_log_probs.size(0)
        # For demonstration, let’s pretend each row can go up to 512 tokens total
        max_seq_len_minus_one = 511

        # Allocate a new tensor for the shifted/padded log probs
        shifted_logps = torch.full(
            (batch_size, max_seq_len_minus_one),
            fill_value=pad_logp,
            dtype=actor_log_probs.dtype,
            device=actor_log_probs.device
        )

        for i in range(batch_size):
            # We assume query_token_ids[i] is not padded or we filter out actual 
            # padding tokens to get the real q_len. 
            # For example:
            q_len = (query_token_ids[i] >= 0).sum().item()  # or len(query_token_ids[i]) if guaranteed
            r_len = (response_token_ids[i] >= 0).sum().item()

            start_pos = q_len - 1
            end_pos = start_pos + r_len

            # Copy actor log probs from [0:r_len] to [start_pos:end_pos]
            # (We also assume actor_log_probs[i, :r_len] holds exactly the log probs
            #  for each token in the response.)
            shifted_logps[i, start_pos:end_pos] = actor_log_probs[i, :r_len]

        return shifted_logps

    def _step(self, inputs: dict) -> dict:
        """
        One iteration step of your GRPO training.
        """
        # Move everything to the correct device
        device = self.policy.actor.device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        batch_size = inputs["input_ids"].size(0)

        # you either get a single max_seq_len per sample or assume they are all identical
        # If in your collator we appended the same int for every sample,
        # we can just take the first's value:
        max_seq_len = int(inputs["max_seq_length"][0].item())

        # Gather shapes
        input_ids = inputs["input_ids"]   # (B, seq_len)
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        scores = inputs["scores"]         # shape (B,)
        groups = inputs["group"]          # shape (B,)
        len_query_token_ids = inputs["len_query_token_ids"]     # shape (B,)
        len_response_token_ids = inputs["len_response_token_ids"] # shape (B,)

        # 1) Compute reference & actor log probs (unshifted)
        ref_log_probs = self._compute_reference_log_probs(inputs)  # list-of-lists, shape [B][seq_len-1]
        actor_log_probs = self._compute_actor_log_probs(inputs)     # list-of-lists, shape [B][seq_len-1]

        # 2) Prepare shifted arrays => so that the portion for query is replaced with pad
        #    and we align the r_len portion to the correct position
        #    Example pad value:
        pad_logp = -1e30

        shifted_ref_log_probs = []
        shifted_actor_log_probs = []

        for i in range(batch_size):
            q_len = int(len_query_token_ids[i].item())
            r_len = int(len_response_token_ids[i].item())

            # unshifted log-prob arrays for sample i
            ref_lp_seq = ref_log_probs[i]
            actor_lp_seq = actor_log_probs[i]

            expected_len = q_len + r_len - 1
            if len(ref_lp_seq) != expected_len:
                raise ValueError(
                    f"Ref log-probs array mismatch. Got {len(ref_lp_seq)}, "
                    f"expected {expected_len} for sample {i}."
                )
            if len(actor_lp_seq) != expected_len:
                raise ValueError(
                    f"Actor log-probs array mismatch. Got {len(actor_lp_seq)}, "
                    f"expected {expected_len} for sample {i}."
                )

            # skip the first (q_len - 1) -- that belongs to query
            ref_resp_part = ref_lp_seq[q_len - 1:]   # length = r_len
            act_resp_part = actor_lp_seq[q_len - 1:] # length = r_len

            # remainder after we place (q_len - 1) + r_len
            remainder = (max_seq_len - 1) - (q_len - 1) - r_len

            # Build final array
            shifted_ref = (
                [pad_logp]*(q_len - 1)
                + list(ref_resp_part)
                + [pad_logp]*remainder
            )
            shifted_actor = (
                [pad_logp]*(q_len - 1)
                + list(act_resp_part)
                + [pad_logp]*remainder
            )

            if len(shifted_ref) != max_seq_len - 1:
                raise ValueError(
                    f"shifted_ref has length {len(shifted_ref)}, expected {max_seq_len - 1}."
                )
            if len(shifted_actor) != max_seq_len - 1:
                raise ValueError(
                    f"shifted_actor has length {len(shifted_actor)}, expected {max_seq_len - 1}."
                )

            shifted_ref_log_probs.append(shifted_ref)
            shifted_actor_log_probs.append(shifted_actor)

        shifted_ref_log_probs = torch.tensor(
            shifted_ref_log_probs,
            dtype=torch.float,
            device=device
        )

        shifted_actor_log_probs = torch.tensor(
            shifted_actor_log_probs,
            dtype=torch.float,
            device=device
        )
 
        logger.info(f"shifted ref log probs {shifted_ref_log_probs}")
        logger.info(f"shifted actor log probs {shifted_actor_log_probs}")

        shifted_labels = labels[
            ..., 1:
        ].contiguous()  # Shape: (batch_size, max_seq_len-1)
        shifted_labels_mask = (shifted_labels != -100).to(
            attention_mask.dtype
        )  # Shape: (batch_size, max_seq_len-1)

        # return
        # Compute the rewards, advantages, and returns
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
            shifted_labels_mask=shifted_labels_mask,
            attention_mask=attention_mask,
        )

        logger.info(f"advantages: {advantages}")
        logger.info(f'mean_rewards {mean_rewards}')
        logger.info(f"std rewads {std_rewards}")
        logger.info(f"unique_goup_ids {unique_group_ids}")
        
        # shifted_actor_logprobs = inputs[
        #     COLUMN_ACTOR_SHIFTED_LOGPS
        # ]  # Shape: (batch_size, max_seq_len-1)
        assert shifted_actor_log_probs.shape == shifted_labels_mask.shape

        #  Compute the rewards, advantages, and returns
        with torch.no_grad():
            # # TODO: add KL Penalty Here
            # if not self.trainer_hparams.force_disable_kl_penalty:
            #     shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            # else:
            #     shifted_ref_logprobs = None

            # Shape of rewards: (batch_size, max_seq_len-1)
            mean_rewards, std_rewards, unique_ids, per_token_kl = self._compute_rewards(
                scores=scores,
                groups=groups,
                shifted_actor_logprobs=shifted_actor_log_probs,
                shifted_ref_logprobs=shifted_ref_log_probs,
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
                    shifted_actor_log_probs=shifted_actor_log_probs,
                    shifted_labels_mask=shifted_labels_mask,
                    attention_mask=attention_mask,
                )
            else:
                precomputed_advantages = inputs["advantages"]
                advantages = precomputed_advantages[:, 1:]

        # assert scores.shape == shifted_actor_logprobs.shape

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        assert advantages.shape == shifted_ref_log_probs.shape
        # Step 2: Compute the actor loss
        actor_loss, is_skipped, actor_metrics, approx_ref_kl = self.policy.actor_loss(
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_logprobs=shifted_actor_log_probs,
            ref_logprobs=shifted_ref_log_probs,
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
        assert advantages.shape == shifted_actor_log_probs.shape

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
