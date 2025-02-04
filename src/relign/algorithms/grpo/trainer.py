from typing import Dict, Tuple, Union, Optional, Literal
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
from relign.utils.trainer import prepare_data_loader_for_training 

from relign.algorithms.grpo.data_collator import (
    GRPODataCollator, 
    GroupedBatchSampler,
    COLUMN_ACTOR_SHIFTED_LOGPS,
    COLUMN_REF_SHIFTED_LOGPS
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
        # Keep a snapshot of the actor's initial state
        self.policy.init_actor_engine_if_needed()
        self.policy.cache_initial_actor_state()
        self.policy.destroy_actor_engine_if_not_cached()
        self.grpo_params = GRPOParams(**kwargs.get("grpo_params", {}))

    def step(self, episodes: EpisodeDataset) -> None:
        """
        Performs a single update step using the dataset rollout under the current policy.
        Each updatestep can rum multiple epochs of optimization.
        """
        self.policy.init_actor_engine_if_needed()

        # change to appropriate input structure
        episodes = self._hydrate_episodes(episodes)
        dataloader = DataLoader(
            episodes,
            batch_sampler=GroupedBatchSampler(
                episodes, 
                group_column='group',
                groups_per_step=1,
            ), 
            collate_fn=GRPODataCollator(),
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory
        )

        # dataloader = prepare_data_loader_for_training(
        #     episodes, 
        #     per_device_batch_size=self.per_device_batch_size, 
        #     seed=self.seed,
        #     drop_last=False,
        #     data_loader_kwargs={
        #         "collate_fn": GRPODataCollator(),
        #         "num_workers": self.dataloader_num_workers,
        #         "pin_memory": self.dataloader_pin_memory,
        #     }
        # )

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

        # Set the actor in train mode
        self.policy.actor.train()

        for epoch in tqdm(
            range(self.num_epochs_per_iteration), 
            desc="Epoch", 
            disable=not self._is_main_process()
        ):
            for _, batch in enumerate(dataloader_iter):
                self._step(batch)

    def _hydrate_log_probs(
        self, 
        episodes: Dataset, 
        column_name: str = COLUMN_ACTOR_SHIFTED_LOGPS
    )-> Dataset:
        """
        Compute logprobs under the current (actor) policy and add them to the dataset.
        """
        # Create a distributed data loader such that the order of
        # episodes is preserved when batches are distributed across multiple processes.
        data_loader = prepare_data_loader_for_inference(
            episodes,
            per_device_batch_size=self.per_device_batch_size,
            data_loader_kwargs={
                "collate_fn": GRPODataCollator(),
                "num_workers": self.dataloader_num_workers,
                "pin_memory": self.dataloader_pin_memory,
            },
        )

        # Switch actor to eval mode before doing inference
        self.policy.actor.eval()
        dist.barrier()

        list_of_log_probs = []
        for inputs in tqdm(
            data_loader, 
            desc="Computing log probs", 
            disable=not self._is_main_process()
        ):
            with torch.no_grad():
                assert torch.all(inputs["attention_mask"][:, 0] == 1), \
                    "Expected first token to be unmasked (attention_mask=1)."
                assert (
                    inputs["input_ids"].shape[0] == self.per_device_batch_size
                ), (
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
                logps = pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)
                logps = gather(logps).cpu()

                assert (
                    logps.shape[0]
                    == inputs["input_ids"].shape[0] * dist.get_world_size()
                )

                # Convert 2D tensors to a list of lists
                logps_seq_lengths = seq_lengths - 1
                for i, seq_len in enumerate(logps_seq_lengths.squeeze().tolist()):
                    assert seq_len <= logps.shape[1], (
                        f"seq_len={seq_len} is out of bounds for logps dim={logps.shape[1]}"
                    )
                    list_of_log_probs.append(logps[i, :seq_len].tolist())

        # Remove any extra log probs that were added due to global padding
        list_of_log_probs = list_of_log_probs[: len(episodes)]

        # Safely add the new column only once on the main process
        with self.distributed_state.main_process_first():
            episodes = episodes.add_column(name=column_name, column=list_of_log_probs)

        return episodes 

    def _hydrate_reference_log_probs(
        self, 
        episodes: Dataset, 
        column_name: str = COLUMN_REF_SHIFTED_LOGPS 
    ) -> Dataset:
        logger.info('Computing refrence model logprobs')

        self.policy.init_reference_engine_if_needed()

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
        
        self.policy.reference.eval()# put the referenc emodel in non-training mode
        dist.barrier()

        ref_log_probs_list = []
        for inputs in tqdm(
            data_loader, 
            desc="Computing reference log probs", 
            disable=not self._is_main_process()  
        ):
            with torch.no_grad():
                     # Asume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert (
                    inputs["input_ids"].shape[0]
                    == self.per_device_batch_size
                ), (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.per_device_batch_size}."
                )

                inputs = {k: v.to(self.policy.reference.device) for k, v in inputs.items()}

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

        ref_log_probs_list = ref_log_probs_list[: len(episodes)] # remove any extra log probs that were added due to global paddingS

        with self.distributed_state.main_process_first():
                episodes = episodes.add_column(name=column_name, column=ref_log_probs_list)

        return episodes

    def _hydrate_episodes(self, episodes: Dataset) -> Dataset:
        """
        Hydrate the episodes with the log probabilities of the actor and the reference model.
        """
        episodes = self._hydrate_log_probs(episodes)
        episodes = self._hydrate_reference_log_probs(episodes)
        return episodes

    def _step(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[float, torch.Tensor]]:
    
        # noinspection DuplicatedCode
        inputs = {k: v.to(self.policy.actor.device) for k, v in inputs.items()}

        # Turn this into a dataclass
        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
        labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)
        scores = inputs["scores"]  # Shape: (batch_size,)
        groups = inputs["group"] # Shape: (batch_size,) 

        shifted_labels = labels[ ..., 1: ].contiguous()  # Shape: (batch_size, max_seq_len-1)
        shifted_labels_mask = (shifted_labels != -100).to(
            attention_mask.dtype
        )  # Shape: (batch_size, max_seq_len-1)

        # Note that this is the log probability of the actor model
        # in the beginning of this iteration (aka the old log probs)
        shifted_actor_logprobs = inputs[COLUMN_ACTOR_SHIFTED_LOGPS]  # Shape: (batch_size, max_seq_len-1)
        assert shifted_actor_logprobs.shape == shifted_labels_mask.shape

        #  Compute the rewards, advantages, and returns
        with torch.no_grad():
            #TODO: add KL Penalty Here
            if self._is_kl_penalty_enabled():
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
                    attention_mask=attention_mask
                )
            else:
                precomputed_advantages = inputs["advantages"]
                advantages = precomputed_advantages[:, 1:] 

            assert advantages.shape == shifted_actor_logprobs.shape
            # assert rewards.shape == shifted_actor_logprobs.shape


        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

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
            and self.grpo_params.kl_penalty_loss_type is None
        ):
            per_token_kl = self._compute_kl_penalty(shifted_actor_logprobs, shifted_ref_logprobs)
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
        groups: torch.LongTensor, #(batch_size,1)
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
            group_advantages[i] = rewards[i] - mean_rewards[unique_ids == group] /(std_rewards[unique_ids == group] + 1e-4)


        last_non_masked_indices = (
            torch.cumsum(attention_mask, dim=1)[:, -1] -1
        )

        last_non_masked_label_indices = last_non_masked_indices -1 #contains the last indic for each row fow which 
        # we have to set the advantees to the group advante . i.e., for each row, assign its row advante from column 0 to the last index
        advantages = torch.zeros_like(per_token_kl)
        advantages[torch.arange(advantages.size(0)), last_non_masked_label_indices] += group_advantages
        advantages -= per_token_kl
        
        assert advantages.shape == shifted_actor_log_probs.shape 
        return advantages.detach()
