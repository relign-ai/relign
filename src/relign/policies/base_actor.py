from typing import Dict, Optional, Literal

import torch

from relign.policies.base_policy import DeepSpeedPolicy 
from relign.policies.base_policy import ForwardOutput
from relign.utils.trainer import masked_mean, monitor_tensor_anomalies


class ActorPolicy(DeepSpeedPolicy):
    def __init__(
        self, 
        actor_model_fn, 
        actor_config
    ):
        self.actor_model_fn = actor_model_fn
        self.actor_config = actor_config
    
    def _init_actor_model_for_inference(self):
        pass

    def _init_actor_model_for_training(self):
        pass 

    def get_actor_model(self):
        return self.actor_model_fn()

    def get_actor_config(self):
        return self.actor_config

    def forward_actor(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_mean_entropy: bool = False,
        return_logits: bool = True,
        return_sequence_logp: bool = False,
        return_all_logp : bool = False,
        sequence_logp_reduction: Optional[Literal["mean"]] = None
    ) -> ForwardOutput:
        """
        Forward pass of the policy.

        Input ids: shape (batch, seq_length)
        Attention_masks shape (batch, seq_length)
        Labels shape (batch, seq_length)
        """

        outputs = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )

        logits = outputs.logits.float()
        logits /= self.temperature
        # Shift so that tokens < n predict n
        # noinspection DuplicatedCode
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_label_mask = (shift_labels != -100).to(shift_logits.dtype)

        # Make sure all label indices are valid. i.e. convert -100 to 0
        shift_labels[shift_labels == -100] = 0

        log_probs = shift_logits.log_softmax(-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=2, index=shift_labels.unsqueeze(2)
        )
        per_token_log_probs = per_token_log_probs.squeeze(2)

        # Multiply the log probs by the label mask to ignore the padding labels
        per_token_log_probs = per_token_log_probs * shift_label_mask

        output = {}
        if return_logits:
            output["logits"] = logits

        if return_sequence_logp:
            sequence_log_probs = per_token_log_probs.sum(dim=-1)
            if sequence_logp_reduction == "mean":
                sequence_log_probs = sequence_log_probs / shift_label_mask.sum(dim=-1)
            output["sequence_logp"] = sequence_log_probs

        if return_all_logp:
            output["all_logp"] = per_token_log_probs 

        return ForwardOutput(**output)    

    def actor_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: Optional[torch.FloatTensor],
        advantages: torch.FloatTensor,
    ):
        """
        The PPO-style actor loss.
        Assumes self.ppo_hparams, self.kl_ctl, etc. are defined in this policy.
        """
        # Switch to RL terminology
        action_mask = shifted_labels_mask

        # Compute the log probabilities of the actor
        outputs = self.forward_actor(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            return_all_logp=True,
            return_logits=False,
            return_sequence_logp=False,
        )
        logprobs = outputs.all_logp  # shape: (batch_size, seq_len-1)

        # Compute the PPO-clip loss
        log_ratio = (logprobs - old_logprobs) * action_mask
        ratio = torch.exp(log_ratio)

        pg_losses1 = -advantages * ratio
        pg_losses1_anomalies = monitor_tensor_anomalies(pg_losses1.detach(), action_mask)
        clip_range = 0.4
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        pg_losses = torch.max(pg_losses1, pg_losses2)
        pg_loss = masked_mean(pg_losses, action_mask)

        # Possibly apply a KL penalty if self.ppo_hparams.kl_penalty_loss_type is not None
        ref_kl_loss = None
        ref_kl = None
        kl_penalty_loss_type = 0.4
        kl_penalty_loss_clip_max = 1000
        kl_penalty_loss_clip_min  = 0
        kl_clt = 0.2
        if kl_penalty_loss_type is not None:
            # _compute_kl_penalty is below
            ref_kl_tensor = self._compute_kl_penalty(
                logprobs,
                ref_logprobs,
                estimation_type=kl_penalty_loss_type
            )
            # clamp for numerical stability
            ref_kl_tensor = torch.clamp(
                ref_kl_tensor * action_mask,
                min=kl_penalty_loss_clip_min,
                max=kl_penalty_loss_clip_max,
            )
            ref_kl_loss = kl_clt * ref_kl_tensor.sum(dim=1).mean()
            pg_loss = pg_loss + ref_kl_loss
            ref_kl = ref_kl_tensor.detach()

        # Ratio check
        is_skipped = False
        avg_ratio = masked_mean(ratio, action_mask)
        ratio_threshold = 1.5
        if avg_ratio.item() > ratio_threshold:
            logger.warning(
                f"High PPO ratio detected: {avg_ratio.item():.2f}. Skipping this batch."
            )
            pg_loss = pg_loss * 0.0
            is_skipped = True

        pg_clip_frac = masked_mean(
            (pg_losses2 > pg_losses1).float(), action_mask
        )
        approx_kl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, action_mask)
        policy_kl = masked_mean(old_logprobs - logprobs, action_mask)

        metrics = {
            "actor/approx_kl": approx_kl.detach(),
            "actor/policy_kl": policy_kl.detach(),
            "actor/clip_frac": pg_clip_frac.detach(),
            "actor/ratio": avg_ratio.detach(),
        }
        # Include anomaly stats
        for i, v in pg_losses1_anomalies.items():
            metrics[f"actor/pg_losses1_anomalies__{i}"] = v

        return pg_loss, is_skipped, metrics, ref_kl

 