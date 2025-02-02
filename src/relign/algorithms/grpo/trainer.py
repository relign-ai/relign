from typing import Dict, Tuple, Union, Optional

import torch
import torch.nn.functional as F
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
    COLUMN_ACTOR_SHIFTED_LOGPS
)

from relign.utils.logging import get_logger

logger = get_logger(__name__)


class GRPOTrainer(BaseTrainer):
    """
    PPO Trainer.
    Impelmentation of the PPO update rule.
    """

    def __init__(
        self, 
        num_groups: int,
        group_size: int,
        **kwargs
    ):
        self.num_groups =  num_groups
        self.group_size = group_size
        super().__init__(**kwargs)

    def step(self, episodes: EpisodeDataset) -> None:
        """
        Performs a single update step using the dataset rollout under the current policy.
        Each updatestep can rum multiple epochs of optimization.
        """
        self.policy.init_actor_engine_if_needed()

        # change to appropriate input structure
        episodes = self._hydrate_log_probs(episodes)
        dataloader = prepare_data_loader_for_training(
            episodes, 
            per_device_batch_size=self.per_device_batch_size, 
            seed=self.seed,
            drop_last=False,
            data_loader_kwargs={
                "collate_fn": GRPODataCollator(),
                "num_workers": self.dataloader_num_workers,
                "pin_memory": self.dataloader_pin_memory,
            }
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
        scores = inputs["rewards"]  # Shape: (batch_size,)

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
            if self.is_kl_poenalty_enabled():
                shifted_ref_logprobs = None
            else:
                shifted_ref_logprobs = None

            # Shape of rewards: (batch_size, max_seq_len-1)
            rewards, _, _= self._compute_rewards(
                scores, shifted_actor_logprobs, shifted_ref_logprobs, attention_mask
            )

            # Shape of `advantages`: (batch_size, max_seq_len-1)
            if "advantages" not in inputs:
                advantages = self._compute_advantages(rewards)
            else:
                precomputed_advantages = inputs["advantages"]
                advantages = precomputed_advantages[:, 1:] 

            assert advantages.shape == shifted_actor_logprobs.shape
            assert rewards.shape == shifted_actor_logprobs.shape


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
        scores: torch.FloatTensor,
        shifted_actor_logprobs: torch.FloatTensor,
        shifted_ref_logprobs: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute per token rewards from scores and KL-penalty.
        Args:
            scores (`torch.FloatTensor`):
                Outcome Scores from the episodes; one scalar per episode, shape (`group`, `group_size`)
            shifted_actor_logprobs (`torch.FloatTensor`):
                Log probabilities of the actor, shape (`group`, `group_size`, `max_seq_len-1`)
            shifted_ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`group`, `group_size`, `max_seq_len-1`)
            attention_mask (`torch.LongTensor`):
                Mask for the input, shape (`group`, `group_size`, `max_seq_len`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`group`, `group_size`, `max_seq_len-1`)
            `torch.FloatTensor`: Non-score rewards, shape (`group`, `group_size`, `max_seq_len-1`)
            `torch.FloatTensor`: KL penalty, shape (`group`, `group_size`, `max_seq_len-1`)
        """
        if (
            shifted_ref_logprobs is not None
            and self.ppo_hparams.kl_penalty_loss_type is None
        ):
            kl = self._compute_kl_penalty(shifted_actor_logprobs, shifted_ref_logprobs)
            non_score_rewards = -self.kl_ctl.value * kl # can be seen as beta
        else:
            # KL penalty is not part of the reward
            kl = None
            non_score_rewards = torch.zeros_like(shifted_actor_logprobs)

        # Initialize the rewards with non-score rewards
        rewards = scores - non_score_rewards.clone()

        # Find the last non-masked index for each sample in the batch
        last_non_masked_indices = (
            torch.cumsum(attention_mask, dim=1)[:, -1] - 1
        )  # Shape: (batch_size)
        # Since the length of shifted_actor_log_probs is `max_seq_len - 1`, we need to
        # subtract 1 from the last non-masked index to get the corresponding index

        last_non_masked_label_indices = last_non_masked_indices - 1
        #  Reward is score - KL penalty
        batch_size = rewards.size(0)
        rewards[torch.arange(batch_size), last_non_masked_label_indices] -= scores

        if kl is not None:
            kl = kl.detach()

        return rewards.detach(), non_score_rewards.detach(), kl

    def _compute_advantages(
        self,
        rewards: torch.FloatTensor,
        shifted_labels_mask: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor]:
        """
        Compute the advantages from the values and rewards.

        Args:
            rewards (`torch.FloatTensor`):
                The rewards of the responses, shape (`group`, `group_size`)

        Returns:
            `torch.FloatTensor`: The advantages of the group, (`group`, `group_size`, `max_seq_len-1`)
        """

        advantages = []
        for t in reversed(range(actions_seq_len)):
            pass

        assert advantages.shape == rewards.shape
        return advantages.detach()


    def _compute_rewards(
        self, 
        scores: torch.FloatTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the episodes; one scalar per episode, shape (`batch_size`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `max_seq_len-1`)
        """
        # Initialize the rewards with non-score rewards
        rewards = scores.clone()
        non_score_rewards = torch.zeros_like(scores)

        return rewards.detach(), non_score_rewards.detach(), None


    def _compute_kl_penalty(
        self,
        logprob: Union[torch.FloatTensor, np.ndarray],
        ref_logprob: Union[torch.FloatTensor, np.ndarray],
        estimation_type: Optional[str] = None,
    ) -> Union[torch.FloatTensor, np.ndarray]:
        """
        Compute the per-token KL penalty between the log probabilities of the actor and the reference model.

        Args:
            logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the actor, shape (`batch_size`, T)
            ref_logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the reference model, shape (`batch_size`, T)

        Returns:
            `Union[torch.FloatTensor, np.ndarray]`: KL penalty, shape (`batch_size`, `T`)
        """

        if estimation_type is None:
            estimation_type = self.ppo_hparams.kl_penalty

        if estimation_type == "kl":
            return logprob - ref_logprob

        if estimation_type == "abs":
            return (logprob - ref_logprob).abs()

        if estimation_type == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if estimation_type == "control_variate":
            # Compute the per-token approximate KL penalty between the log probabilities of the actor
            # and the reference model as suggested by Schulman in http://joschu.net/blog/kl-approx.html
            #
            # D_KL [π_θ || π_ref] =
            #    π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t) - log(π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t)) - 1
            #

            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                kl = torch.exp(log_ratio) - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                kl = np.exp(log_ratio) - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "seq_control_variate":
            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                prob_ratio = torch.exp(log_ratio.sum(dim=-1, keepdim=True))
                kl = prob_ratio - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                prob_ratio = np.exp(log_ratio.sum(axis=-1, keepdims=True))
                kl = prob_ratio - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(
                ref_logprob, logprob, log_target=True, reduction="none"
            ).sum(-1)

        raise NotImplementedError