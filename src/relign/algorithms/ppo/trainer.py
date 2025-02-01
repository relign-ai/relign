from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import Dataset
from tqdm import tqdm
from deepspeed import comm as dist
from deepspeed import DeepSpeedEngine
from accelerate.utils import gather, pad_across_processes, release_memory

from relign.common.deepspeed_utils import prepare_data_loader_for_inference
from relign.common.dataset import EpisodeDataset
from relign.algorithms.base_trainer import OnPolicyTrainer
from relign.utils.trainer import prepare_data_loader_for_training 

from relign.algorithms.ppo.data_collator import (
    PPODataCollator, 
    COLUMN_ACTOR_SHIFTED_LOGPS, 
    COLUMN_REF_SHIFTED_LOGPS,
    COLUMN_VALUES
)

from relign.utils.logging import get_logger

logger = get_logger(__name__)


class PPOTrainer(OnPolicyTrainer):
    """
    PPO Trainer.
    Implementation of the PPO update rule.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, episodes: EpisodeDataset) -> None:
        """
        Performs a single update step using the dataset rollout under the current policy.
        Each update step can rum multiple epochs of optimization.
        """
        self.policy.init_actor_engine_if_needed()
        self.policy.init_critic_engine_if_needed()

        # change to appropriate input structure
        episodes = self._get_curr_logs_and_values(episodes)
        dataloader = prepare_data_loader_for_training(
            episodes, 
            per_device_batch_size=self.per_device_batch_size, 
            seed=self.seed,
            drop_last=False,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
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

        # Set everything in train mode
        self.policy.actor.train()
        self.policy.critic.train()

        for epoch in tqdm(
            range(self.num_epochs_per_iteration), 
            desc="Epoch", 
            disable=not self._is_main_process()
        ):
            for step, batch in enumerate(dataloader_iter):
                self._step(batch)

    def _get_curr_logs_and_values(
        self, 
        episodes: Dataset
    ) -> Dataset:
        """
        Takes the collated dataset and hydrates it with the
        logprobs and values under the current policy parameters.
        These will be the baseline logprobs and values i.e., pi_old(a|s)
        """
        episodes = self._hydrate_log_probs(episodes)
        episodes = self._hydrate_values(episodes)
        return episodes

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
            data_loader, 
            desc="Computing values", 
            disable=not self._is_main_process()
        ):
            with torch.no_grad():
                # Assume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert (
                    inputs["input_ids"].shape[0]
                    == self.per_device_batch_size
                ), (
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
                    labels=inputs["labels"]
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
            episodes= episodes.add_column(
                name=COLUMN_VALUES,
                column=list_of_values
            )

        return episodes 

    def _step(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[float, torch.Tensor]]:
        # To better understand the alignment of inputs, logits, logps, and labels,
        # we provide a detailed explanation below.
        # -----------------------------------------------------------------------------------------------------------
        # Consider the sequence: "<s> q1 q2 a b c </s>", where "<s> q1 q2" forms the prompt, and "a b c </s>"
        # the response, with `prompt_len = 3` and `response_len = 4`. Additionally, we include a padding token at
        # the end for the sake of generality.
        #
        # Here is the inputs dictionary setup:
        # Inputs:
        # [     <s>           q1           q2           a            b            c           </s>         <p>   ]
        # Attn Mask:
        # [       1            1            1           1            1            1              1           0   ]
        # Labels:
        # [    -100         -100         -100        ID_a         ID_b         ID_c        ID_</s>        -100   ]
        # >>> seq_len = torch.sum(attn_mask) = 7
        #
        # Feeding the Inputs+Attn Mask, the model outputs logits for next tokens in the sequence:
        # Logits:
        # [  p(.|<s>)      p(.|q1)      p(.|q2)      p(.|a)       p(.|b)       p(.|c)      p(.|</s>)    p(.|<p>)]
        #
        # We exclude the nonsensical last logit (predicting beyond </s>). Also, to obtain the logprobs of next
        # ground-truth token, we shift the labels by 1 to the left:
        #
        # Valid Logits:
        # [  p(.|<s>)      p(.|q1)      p(.|q2)      p(.|a)       p(.|b)       p(.|c)      p(.|</s>)  ]
        # Shifted Labels:
        # [     -100         -100         ID_a        ID_b         ID_c        ID_</s>         -100   ]
        # Shifted Labels Mask (aka Action Mask), i.e. shifted_labels != -100, highlighting valid token-preds/actions:
        # [        0            0            1           1            1            1              0   ]
        # Aligning them to obtain logprobs (lp) of predicting the next token (or lp of action):
        # [    lp(q1)       lp(q2)        lp(a)       lp(b)        lp(c)      lp(</s>)       lp(<p>)  ]
        #
        #
        # Applying the labels mask gives us logprobs of predicting the valid response tokens (aka actions):
        # [     -inf          inf         lp(a)       lp(b)        lp(c)      lp(</s>)         -inf   ]
        # Rewriting the original shifted inputs (for clarity). These are the states we care about:
        # [      <s>           q1           q2           a            b            c           </s>   ]
        # In this example, with S=[<s>;q1;q2] as the state and A=a, we compute lp(A|S) = log(p(a|<s>;q1;q2)) = lp(a)
        #
        # Note that the values are also computed for the entire sequence, but similar to inputs, we ignore
        # the last one since it nonsensical (i.e. V(</s>) is not used).
        # Valid Values:
        # [    V(<s>)        V(q1)        V(q2)        V(a)         V(b)         V(c)        V(</s>)  ]
        # Applying the action mask:
        # [     -inf         -inf         V(q2)        V(a)         V(b)         V(c)          -inf   ]
        #
        # >>> logits_seq_len = logps_seq_len = valid_values_len = seq_len - 1 = 6
        # noinspection DuplicatedCode
        inputs = {k: v.to(self.policy.actor.device) for k, v in inputs.items()}

        # Turn this into a dataclass
        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
        labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)
        scores = inputs["rewards"]  # Shape: (batch_size,)

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
            #TODO: add KL Penalty Here
            shifted_ref_logprobs = None

            rewards, non_score_rewards, kls = self._compute_rewards(
                scores, shifted_actor_logprobs, shifted_ref_logprobs, attention_mask
            )
            # The `advantages` is computed for the actions. That's why they are of shape (batch_size, max_seq_len-1)
            # Shape of `advantages`: (batch_size, max_seq_len-1)
            if "advantages" not in inputs:
                # Print all the keys in the inputs
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
            and self.ppo_hparams.kl_penalty_loss_type is None
        ):
            kl = self._compute_kl_penalty(shifted_actor_logprobs, shifted_ref_logprobs)
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
            delta = (
                rewards[:, t]
                +  1 * next_state_values
                - valid_values[:, t]
            )
            lastgaelam = (delta + 1 * self.lam * lastgaelam)
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        assert advantages.shape == rewards.shape

        returns = advantages + valid_values
        return advantages.detach(), returns.detach()