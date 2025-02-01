from typing import Tuple, Optional, Dict, Any, Callable, Literal, NamedTuple, Union 

import torch
from transformers import PreTrainedModel 

from relign.policies.base_policy import DeepSpeedPolicy 
from relign.policies.base_policy import ForwardOutput

class ActorPolicy(DeepSpeedPolicy):
    def __init__(self, actor_model_fn, actor_config):
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
    
    def forward_actor() -> ForwardOutput:
        pass

    def actor_loss(
        model_inputs: Dict[str, torch.Tensor],
        shifted_label_maks: torch.Tensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        advantages: torch.FloatTensor,
    ) -> torch.Tensor:
        pass
