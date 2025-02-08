from abc import abstractmethod
from typing import Union 
from pathlib import Path
import json

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


    def save_pretrained(self, save_directory: Union[str, Path], safe_serialization: bool = False) -> None:
        """
        Save the model weights and configuration in a Hugging Face compatible format.

        Args:
            save_directory (Union[str, Path]): The directory to which the model will be saved.
            safe_serialization (bool): (Unused for now; maintained for compatibility.)
        """
        # Convert to Path if needed and create the directory
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save the model's state dict
        model_save_path = save_directory / "pytorch_model.bin"
        torch.save(self.state_dict(), model_save_path)
        print(f"Model state dict saved to {model_save_path}")

        # Optionally, save configuration if available
        if hasattr(self, "config"):
            # Support both config classes (with .to_dict()) or simple dictionaries
            if hasattr(self.config, "to_dict"):
                config_dict = self.config.to_dict()
            elif isinstance(self.config, dict):
                config_dict = self.config
            else:
                config_dict = {"config": self.config}

            config_save_path = save_directory / "config.json"
            with config_save_path.open("w") as f:
                json.dump(config_dict, f, indent=2)
            print(f"Model configuration saved to {config_save_path}")


