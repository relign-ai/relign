import torch
from typing import Optional, Dict, Any
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
        # if disable_dropout:
        #     dropout_config = configure_dropout(hf_model_name, 0.0)
        #     if is_main_process:
        #         logger.info(f"Disabling dropout for keys: {dropout_config.keys()}")
        #     kwargs.update(dropout_config)

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



