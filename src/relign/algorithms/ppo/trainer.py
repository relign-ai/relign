from typing import Dict, Tuple, Union, Optional, Literal
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from deepspeed import comm as dist
from accelerate.utils import gather, pad_across_processes, release_memory

from relign.common.deepspeed_utils import prepare_data_loader_for_inference
from relign.common.dataset import EpisodeDataset
from relign.algorithms.base_trainer import BaseTrainer
from relign.utils.trainer import prepare_data_loader_for_training, masked_mean

from relign.algorithms.ppo.data_collator import (
    PPODataCollator,
    COLUMN_ACTOR_SHIFTED_LOGPS,
    COLUMN_REF_SHIFTED_LOGPS,
    COLUMN_VALUES,
)

from relign.utils.logging import get_logger

logger = get_logger(__name__)


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


class PPOTrainer(BaseTrainer):
    """
    PPO Trainer.
    Implementation of the PPO update rule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: make this initialization more robust ( ideally we get it from a config file)
        self.ppo_hparams = PPOHParams(**kwargs.get("ppo_hparams", {}))

    def step(self, episodes: EpisodeDataset) -> None:
        """
        Performs a single update step using the dataset rollout under the current policy.
        Each update step can rum multiple epochs of optimization.
        """
        self.policy.init_actor_engine_if_needed()
        self.policy.init_critic_engine_if_needed()

        # change to appropriate input structure
        episodes = self._hydrate_episodes(episodes)
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

        # Set everything in train mode
        self.policy.actor.train()
        self.policy.critic.train()

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

        self.state.iteration += 1
        progress_bar.close()
        self.policy.destroy_ds_engines()
        release_memory()

        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def _hydrate_episodes(self, episodes: Dataset) -> Dataset:
        """
        Takes the collated dataset and hydrates it with the
        logprobs and values under the current policy parameters.
        These will be the baseline logprobs and values i.e., pi_old(a|s)
        """
        episodes = self._hydrate_log_probs(episodes)
        episodes = self._hydrate_values(episodes)
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

    def _step(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[float, torch.Tensor]]:

        inputs = {k: v.to(self.policy.actor.device) for k, v in inputs.items()}

        # Turn this into a dataclass
        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
        labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)
        scores = inputs["scores"]  # Shape: (batch_size,)

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
            if self._is_kl_penalty_enabled(
                self.ppo_hparams.kl_penalty_loss_type, self.policy.reference
            ):
                shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            else:
                shifted_ref_logprobs = None

            rewards, _, _ = self._compute_rewards(
                scores, shifted_actor_logprobs, shifted_ref_logprobs, attention_mask
            )
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

            assert advantages.shape == shifted_actor_logprobs.shape
            assert rewards.shape == shifted_actor_logprobs.shape

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Step 2: Compute the policy/actor loss
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
        # training step complete

        # Step 3: Compute critic loss
        critic_loss, critic_metrics = self.policy.critic_loss(
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_valid_values=valid_values,
            returns=returns,
            trainer_params=self.ppo_hparams,
        )

        self.policy.critic.backward(critic_loss)
        self.policy.critic.step()
        # Get rid of critic's activations to free up memory
        critic_loss = critic_loss.detach().clone()
        release_memory()

        metrics = {
            "advantages/mean": masked_mean(advantages, shifted_labels_mask).detach(),
            "rewards/mean": masked_mean(rewards, shifted_labels_mask).detach(),
            "num_tokens": shifted_labels_mask.sum().detach(),
            "_num_participating_tokens": shifted_labels_mask.sum().detach(),
            **actor_metrics,
            **critic_metrics,
        }
        if returns is not None:
            metrics["returns"] = masked_mean(returns, shifted_labels_mask).detach()
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
            and self.ppo_params.kl_penalty_loss_type is None
        ):
            kl = self._compute_kl_penalty(
                shifted_actor_logprobs,
                shifted_ref_logprobs,
                trainer_hparams=self.ppo_hparams,
            )
            non_score_rewards = -self.kl_ctl.value * kl
        else:
            # KL penalty is not part of the reward
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

        for t in reversed(range(actions_seq_len)):
            next_state_values = (
                valid_values[:, t + 1] if t < (actions_seq_len - 1) else 0.0
            )
            delta = rewards[:, t] + 1 * next_state_values - valid_values[:, t]
            lastgaelam = delta + 1 * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        assert advantages.shape == rewards.shape

        returns = advantages + valid_values
        return advantages.detach(), returns.detach()

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

            if isinstance(metric_value, torch.Tensor):
                dist.all_reduce(metric_value, op=dist.ReduceOp.SUM)
                metric_value = metric_value.item()
                divisor = dist.get_world_size()
            else:
                metric_value /= divisor * num_steps_since_last_log

            logs[metric_name] = round(metric_value, 8)

        logs["epoch"] = round(self.state.epoch, 4)
        logs["step"] = self.state.global_step
        # First log the metrics on the progress bar
        progress_bar.set_postfix(logs)

        # Add "train/" prefix for clarity.
        logs = {f"train/{k}": v for k, v in logs.items()}

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
            val: Union[float, torch.Tensor]
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
