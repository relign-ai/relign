import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Any, Union
from pathlib import Path
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer, PreTrainedTokenizerFast 
from relign.common.types import JsonDict

from relign.utils.py_utils import is_flash_attention_available
from relign.utils.logging import get_logger

logger = get_logger(__name__)

if is_flash_attention_available():
    from flash_attn.models.gpt import GPTLMHeadModel
    from flash_attn.models.llama import inv_remap_state_dict_hf_llama

    class FlashAttentionModel(GPTLMHeadModel):
        def save_hf_pretrained(self, output_dir: str):
            hf_model_name = self.config._name_or_path

            flash_attn_state_dict = self.state_dict()
            hf_state_dict = inv_remap_state_dict_hf_llama(
                flash_attn_state_dict, self.config
            )

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            config = AutoConfig.from_pretrained(hf_model_name)
            config.save_pretrained(output_dir)

            torch.save(hf_state_dict, output_dir / "pytorch_model.bin")

DROPOUT_CONFIG_KEYS = [
    "dropout",
    "attention_dropout",
    "classifier_dropout",
    "hidden_dropout",
    "activation_dropout",
    "resid_pdrop",
    "embd_pdrop",
    "attn_pdrop",
]

def configure_dropout(hf_model_name: str, dropout_value: float):
    """
    Adjusts dropout settings in the model configuration based on specified keys.

    Args:
        hf_model_name (str): Name of the model in the Hugging Face hub.
        dropout_value (float): Value to set the dropout to.

    Returns:
        dict:  keyword arguments with dropout values set to 0.0 for specified config keys.
    """
    kwargs = {}
    model_config = AutoConfig.from_pretrained(hf_model_name)
    for key in DROPOUT_CONFIG_KEYS:
        if hasattr(model_config, key):
            kwargs[key] = dropout_value
    return kwargs


class PreTrainedModelForCasualLM(PreTrainedModel):
    @classmethod
    def from_di(
        cls,
        hf_model_name: str,
        pretrained_args: Optional[JsonDict] = None,
        lora_config: Optional[JsonDict] = None,
        freeze_config: Optional[JsonDict] = None,
        disable_dropout: bool = False,
        device: Optional[torch.device] = None,
        init_base_model_only: bool = False,
        runtime_hf_model_name: Optional[str] = None,
    ) -> PreTrainedModel:
        from accelerate import PartialState

        is_main_process = PartialState().is_main_process

        if pretrained_args is None:
            pretrained_args = {}

        if runtime_hf_model_name is not None:
            if is_main_process:
                logger.warning(
                    f"Overriding the `hf_model_name` with '{runtime_hf_model_name}' "
                    f"(original='{hf_model_name})'."
                )
            hf_model_name = runtime_hf_model_name

        kwargs = {
            "use_flash_attention_2": pretrained_args.pop(
                "use_flash_attention_2", is_flash_attention_available()
            ),
            "torch_dtype": pretrained_args.pop("torch_dtype", torch.bfloat16),
            "trust_remote_code": True,
        }

        #######################################################
        # disable dropout allows for better training stability
        #######################################################
        if disable_dropout:
            dropout_config = configure_dropout(hf_model_name, 0.0)
            if is_main_process:
                logger.info(f"Disabling dropout for keys: {dropout_config.keys()}")
            kwargs.update(dropout_config)

        if device is not None:
            kwargs["device_map"] = device

        if init_base_model_only:
            model_class = AutoModel
        else:
            model_class = AutoModelForCausalLM

        model = model_class.from_pretrained(
            hf_model_name,
            **pretrained_args,
            **kwargs,
        )
        if disable_dropout and is_main_process:
            logger.info(f"Model config after disabling dropout: {model.config}")
        assert (
            lora_config is None or freeze_config is None
        ), "Only one of lora_config and freeze_config can be specified"

        return model

    @classmethod
    def from_di_flash_attn(
        cls, hf_model_name: str, pretrained_args: Optional[JsonDict] = None
    ) -> PreTrainedModel:
        if not is_flash_attention_available():
            raise ImportError("Please install flash_attn to use this feature")

        if pretrained_args is None:
            pretrained_args = {}

        if "llama" not in hf_model_name.lower():
            raise ValueError("Only llama models are supported for now")

        from flash_attn.utils.pretrained import state_dict_from_pretrained
        from flash_attn.models.llama import (
            llama_config_to_gpt2_config,
            remap_state_dict_hf_llama,
        )

        config = llama_config_to_gpt2_config(
            AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        )
        config.use_flash_attn = True
        config.fused_bias_fc = True
        config.fused_mlp = False
        config.fused_dropout_add_ln = True
        config.residual_in_fp32 = True
        config.hidden_size = config.n_embd
        config._name_or_path = hf_model_name
        logger.info(f"About to load {hf_model_name} into flash_attn model")

        pretrained_state_dict = state_dict_from_pretrained(hf_model_name)
        pretrained_state_dict = {
            key: val
            for key, val in pretrained_state_dict.items()
            if "rotary_emb.inv_freq" not in key
        }
        pretrained_state_dict = remap_state_dict_hf_llama(pretrained_state_dict, config)
        logger.info(f"Loaded {hf_model_name} state dict into CPU")

        # Since flash_attn is only available for A100 and above, we can use bfloat16 safely
        dtype = torch.bfloat16

        model = FlashAttentionModel(config, dtype=dtype, **pretrained_args)
        model.load_state_dict(pretrained_state_dict)
        model.eval()
        logger.info(f"Finished loading {hf_model_name} into flash_attn model")

        return model

class DIPreTrainedTokenizer:
    @classmethod
    def from_di(
        cls, hf_model_name: str, pretrained_args: Optional[JsonDict] = None, **kwargs
    ) -> PreTrainedTokenizerFast:
        if pretrained_args is None:
            pretrained_args = {}

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, use_fast=True, **pretrained_args
        )
        return tokenizer



class PreTrainedModelForValueNetwork(nn.Module):
    def __init__(
        self,
        pretrained_backbone_model: PreTrainedModel,
        value_head_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.pretrained_model = pretrained_backbone_model
        self.config = self.pretrained_model.config

        hidden_size = self.pretrained_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=True)
        self.dropout = (
            nn.Dropout(value_head_dropout)
            if value_head_dropout is not None
            else nn.Identity()
        )

        self._init_value_head()

    def _init_value_head(self):
        hidden_size = self.pretrained_model.config.hidden_size
        nn.init.normal_(self.value_head.weight, std=1 / np.sqrt(hidden_size + 1))
        nn.init.constant_(self.value_head.bias, val=0.0)

    @classmethod
    def from_di(
        cls,
        pretrained_backbone_model: Any ,
        value_head_dropout: Optional[float] = None,
    ) -> nn.Module:
        return cls(pretrained_backbone_model, value_head_dropout=value_head_dropout)

    @property
    def device(self):
        return self.pretrained_model.device

    def gradient_checkpointing_enable(self):
        self.pretrained_model.gradient_checkpointing_enable()

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

        last_hidden_state = base_model_output.hidden_states[-1]

        output = self.dropout(last_hidden_state)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.value_head.weight.dtype:
            output = output.to(self.value_head.weight.dtype)

        value = self.value_head(output)
        value = value.squeeze(-1)

        return value

    def save_pretrained(
        self,
        checkpoint_path: Union[str, Path],
        safe_serialization: Optional[bool] = None,
    ):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True)
        else:
            logger.warning(
                f"Checkpoint path {checkpoint_path} already exists and will be overwritten."
            )

        torch.save(self.state_dict(), checkpoint_path / "pytorch_model.bin")
