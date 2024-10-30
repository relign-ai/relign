from abc import abstractmethod
from typing import  Tuple, Optional, Dict, Any, Callable, Literal, NamedTuple
from functools import partial

import torch 
from transformers import PreTrainedModel

from common.dataset import EpisodeDataset
from common.logging import get_logger
from policies.base_policy import DeepSpeedPolicy 
from policies.base_critic import PretrainedModelValueHead

logger = get_logger(__name__)

class ActorForwardOutput(NamedTuple):
    logits: Optional[torch.Tensor] = None
    sequence_logp: Optional[torch.Tensor] = None
    all_logp: Optional[torch.Tensor] = None

class CriticForwardOutput(NamedTuple):
    values: Optional[torch.Tensor] = None

class ActorCriticPolicy(DeepSpeedPolicy):
    """
        Base Actor critic type policy. 
        We use deepspeed here because of accelerate's isseus with multiple models

        The actor predicts an action from an observation and a 
        critic predicts the value of an observation. 
    """
    def __init__(
            self,          
            actor_model_fn: Callable[[],PreTrainedModel],
            actor_ds_config: Optional[Dict[str, Any]],
            critic_model_fn: Callable[[], PretrainedModelValueHead],
            critic_ds_config: Optional[Dict[str, Any]],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.actor_model_fn = actor_model_fn
        self.actor_config = actor_ds_config
        self.critic_model_fn = critic_model_fn
        self.critic_config = critic_ds_config

        # initialize the models
        self.actor, self.critic = self._init_models()

    def _init_models(self) -> Tuple[PreTrainedModel, PreTrainedModel]:
        """ Loads the models onto the gpu"""
        actor_model = self._init_actor_model()
        critic_model = self._init_critic_model()
        return actor_model, critic_model

    def _init_actor_model(self) -> PreTrainedModel:
        if hasattr(self, "actor_model"):
            return self.actor
        actor_model = self.actor_model_fn()
        return actor_model
  
    def _init_critic_model(
        self,
    ) -> PreTrainedModel:
        # Check if the model is 'cached'
        if hasattr(self, "critic_model"):
            return self.critic
    
        critic_model = self.critic_model_fn()
        return critic_model 
        
    @abstractmethod
    def predict_actor(self, episodes: EpisodeDataset):
        """
        Response from actor model
        """
        raise NotImplementedError

    @abstractmethod
    def predict_critic(self, episodes: EpisodeDataset) -> float:
        """
        Predict the value of an observation.
        """
        raise NotImplementedError

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
        ) -> ActorForwardOutput:
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
            output["all_logp"]

    def forward_critic(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> CriticForwardOutput:
        """
        Forward pass of the critic.
        """
        outputs = self.critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )

        predicted_values = outputs
        predicted_values = predicted_values.to(torch.float32)

        # mask the query tokens (only interested in value of the response tokens)
        if labels is not None:
            value_mask = (labels != -100).to(attention_mask.dtype)
        else:
            value_mask = attention_mask

        return {"values": predicted_values, "value_mask": value_mask}