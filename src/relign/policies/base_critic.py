from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from relign.utils.logging import get_logger

logger = get_logger(__name__)

class BaseCritic(nn.Module):
    """
    Main critic model. Takes an observation and returns a value estimate of the observation.
    """
    @abstractmethod 
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the value function.
        """
        pass

class PretrainedModelValueHead(BaseCritic):
    def __init__(
            self,
            pretrained_model: PreTrainedModel
    ):  
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config

        hidden_size = self.pretrained_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=True)
        self._init_value_head()


    def _init_value_head(self):
        hidden_size = self.pretrained_model.config.hidden_size
        nn.init.normal_(self.value_head.weight, std=1 / np.sqrt(hidden_size + 1))
        nn.init.constant_(self.value_head.bias, val=0.0)


    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        Arg:
            *args: Variable length input arguments passed to the model.
            **kwargs: Variable length keyword arguments passed to the model.
        Returns:
            value: The value of the input sequence. Shape: (batch_size, sequence_length)
        """
        kwargs["output_hidden_states"] = True

        base_model_output = self.pretrained_model(*args, **kwargs)

        output = base_model_output.hidden_states[-1]

        if output.dtype != self.value_head.weight.dtype:
            output = output.to(self.value_head.weight.dtype)
        
        value = self.value_head(output)
        value = value.squeeze(-1)

        return value


