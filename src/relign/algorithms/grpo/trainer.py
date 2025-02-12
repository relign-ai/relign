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
        # compute the
        episodes = self._hydrate_ref_log_probs(
            episodes, column_name=COLUMN_REF_SHIFTED_LOGPS
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

        episodes = self._rescale_and_clip_scores(episodes)
        # kls = self._log_episodes_metrics(episodes)
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
        dist.barrier()

        # Move everything to the actor's device
        inputs = {k: v.to(self.policy.actor.device) for k, v in batch.items()}

        # For each sequence, find how many tokens are non-padding
        # shape: (batch_size, 1)
        seq_lengths = inputs["attention_mask"].sum(dim=1, keepdim=True).detach()
        seq_lengths = seq_lengths.unsqueeze(1)

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
            assert logps.shape[1] == inputs["input_ids"].shape[1] - 1

        # Prepare for distributed gather
        seq_lengths = seq_lengths.to(self.policy.actor.device)
        # seq_lengths = gather(seq_lengths).cpu()  # shape [global_batch_size, 1]

        # Also gather the logps across processes - first pad them so gather works
        logps = pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)
        # logps = gather(logps).cpu()  # shape [global_batch_size, padded_seq_len-1]

        # assert logps.shape[0] == inputs["input_ids"].shape[0] * dist.get_world_size()
        assert logps.shape[0] == inputs["input_ids"].shape[0]

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
        # seq_lengths = gather(seq_lengths).cpu()  # shape [global_batch_size, 1]

        # Also gather the logps across processes - first pad them so gather works
        logps = pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)

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

    def batch_prepare_shifted_logps(
        self,
        shifted_logps_with_query_list: list[torch.Tensor],
        query_lens: torch.LongTensor,
        response_lens: torch.LongTensor,
        max_seq_length: int,
        pad_logp: float,
    ) -> torch.Tensor:
        """
        Mimics the single-example prepare_shifted_logps, but for a batch of examples.
        """
        device = query_lens.device
        batch_size = len(shifted_logps_with_query_list)
        # We'll create a 2D tensor of shape (B, max_seq_length - 1), filled with pad_logp.
        final_shape = (batch_size, max_seq_length - 1)
        final_logps = torch.full(size=final_shape, fill_value=pad_logp, device=device)

        for i in range(batch_size):
            q_len = int(query_lens[i].item())
            r_len = int(response_lens[i].item())
            logps_i = shifted_logps_with_query_list[i].to(device)

            # 1) Check length == q_len + r_len - 1
            expected_length = q_len + r_len - 1
            if logps_i.size(0) != expected_length:
                raise ValueError(
                    f"Expected {expected_length} log-prob values for example {i} "
                    f"(q_len={q_len}, r_len={r_len}), got {logps_i.size(0)}."
                )

            # 2) Slice out just the "response" portion
            #    This is exactly r_len tokens, starting from index (q_len-1)
            #    because the first (q_len-1) tokens belong to the query.
            #    So 'shifted_logps_without_query' should have length == r_len
            shifted_logps_without_query = logps_i[q_len - 1 :]
            if shifted_logps_without_query.size(0) != r_len:
                raise ValueError(
                    f"Example {i}: expected {r_len} tokens in the response portion, "
                    f"got {shifted_logps_without_query.size(0)}."
                )

            # 3) Combine them with pad for the first (q_len - 1) positions
            #    followed by the `shifted_logps_without_query`, then
            #    pad to the right to fill up to (max_seq_length - 1).
            #    We'll do it in python, then convert to Tensor.
            n_pads_at_end = (max_seq_length - 1) - (q_len + r_len - 1)
            # Make sure it's not negative:
            if n_pads_at_end < 0:
                raise ValueError(
                    f"For example {i}, the combined query+response length {q_len + r_len} "
                    f"is bigger than max_seq_length={max_seq_length}, can't pad negative."
                )

            # Build the final array in python
            final_array_list = (
                [pad_logp] * (q_len - 1)
                + shifted_logps_without_query.tolist()
                + [pad_logp] * n_pads_at_end
            )
            # Convert that to a tensor
            final_array_tensor = torch.tensor(
                final_array_list, dtype=torch.float32, device=device
            )

            # Now final_array_tensor should have shape (max_seq_length - 1,).
            if final_array_tensor.size(0) != (max_seq_length - 1):
                raise ValueError(
                    f"Got final_array_tensor of shape {final_array_tensor.size(0)}, "
                    f"expected {max_seq_length - 1}."
                )

            # 4) Assign to final_logps[i, :]
            final_logps[i, :] = final_array_tensor

        return final_logps

    def _step(self, inputs: dict) -> dict:
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
        scores = inputs["scores"]  # shape (B,)
        groups = inputs["group"]  # shape (B,)
        len_query_token_ids = inputs["len_query_token_ids"]  # shape (B,)
        len_response_token_ids = inputs["len_response_token_ids"]  # shape (B,)

        # Log some basic batch information
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"max_seq_len: {max_seq_len}")
        logger.info(f"input_ids shape: {input_ids.shape}")
        logger.info(f"attention_mask shape: {attention_mask.shape}")
        logger.info(f"labels shape: {labels.shape}")
        logger.info(f"scores shape: {scores.shape}, sample scores: {scores[:5]}")
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

        # Step 2: Compute the actor loss
        actor_loss, is_skipped, actor_metrics, approx_ref_kl = (
            self.policy.actor_loss_grpo(
                model_inputs=model_inputs,
                shifted_labels_mask=shifted_labels_mask,
                ref_logprobs=inputs[COLUMN_REF_SHIFTED_LOGPS],
                advantages=scores,
            )
        )

        self.policy.actor.backward(actor_loss)
        self.policy.actor.step()

        # Get rid of actor's activations to free up memory
        actor_loss = actor_loss.detach().clone()
        release_memory()

        # metrics = {
        #     "advantages/mean": masked_mean(advantages, shifted_labels_mask).detach(),
        #     "rewards/mean": mean_rewards,
        #     # masked_mean(mean_rewards, shifted_labels_mask).detach(),
        #     "num_tokens": shifted_labels_mask.sum().detach(),
        #     "_num_participating_tokens": shifted_labels_mask.sum().detach(),
        #     **actor_metrics,
        # }
        # if returns is not None:
        #     metrics["returns"] = masked_mean(returns, shifted_labels_mask).detach()
        # metrics["actor/loss"] = actor_loss
        assert advantages.shape == shifted_actor_log_probs.shape

        # return metrics

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

    def _hydrate_ref_log_probs(
        self, episodes: Dataset, column_name: str = COLUMN_REF_SHIFTED_LOGPS
    ) -> Dataset:
        logger.info("Computing refrence model logprobs")

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

        self.policy.destroy_reference_engine_if_not_cached()
        return episodes
