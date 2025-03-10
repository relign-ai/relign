import logging
from abc import abstractmethod
from typing import Any, Optional, Union, NamedTuple, Dict, List
from pathlib import Path
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, SchedulerType, get_scheduler, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from accelerate import PartialState
from accelerate.utils import DummyOptim
from transformers.integrations import HfTrainerDeepSpeedConfig
import deepspeed
from deepspeed import DeepSpeedEngine
from wandb.sdk.wandb_run import Run as WandbRun

from relign.common.registry import RegistrableBase
from relign.common.dataset import EpisodeDataset
from relign.utils.logging import get_logger

logger = get_logger(__name__)


class CriticForwardOutput(NamedTuple):
    initial_poilicy_raw_output: CausalLMOutput
    policy_raw_output: CausalLMOutput
    values: torch.Tensor


class ActorForwardOutput(NamedTuple):
    logits: Optional[torch.Tensor] = None
    sequence_logp: Optional[torch.Tensor] = None
    all_logp: Optional[torch.Tensor] = None


class BasePolicy(RegistrableBase):
    """
    A policy takes an observation and returns an action.
    """
    def __init__(
        self,
        seed: int,
        project_root_dir: Path = None,
        gradient_checkpointing: bool = True,
        temperature: float = 0.6,
        weight_decay: float = 0.00,
        learning_rate: float = 1e-6,
        lr_scheduler_type: Optional[Union[SchedulerType, str]] = None,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        bf16: bool = True,
        bf16_full_eval: bool = False,
        warmup_steps: int = 0,
        warmup_ratio : int = 0.03,
    ):
        self.seed = seed
        self.project_root_dir = project_root_dir
        self._init_checkpoint_dir()

        self.gradient_checkpointing = gradient_checkpointing
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.bf16 = bf16
        self.bf16_full_eval = bf16_full_eval
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio

    @abstractmethod
    def predict(self, episodes: EpisodeDataset):  # TODO Define response type
        """
        Predict an action from an observation.
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> ActorForwardOutput:
        """
        Forward pass of the policy.
        """
        pass

    @abstractmethod
    def set_params(self, policy_params):
        self.inference = self.inference.replace(params=policy_params)

    def set_cloud_log(self, cloud_logger: WandbRun):
        self.cloud_logger = cloud_logger

    def _init_checkpoint_dir(self):
        self.checkpoint_dir = self.project_root_dir / "policy"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._validate_checkpoint_format()

    def _validate_checkpoint_format(self, checkpoint_format: Optional[str] = None):
        if checkpoint_format is None:
            checkpoint_format = self.get_checkpoint_format()
        assert checkpoint_format.startswith("ckpt--iter_")
        for state in checkpoint_format.replace("ckpt--", "").split("--"):
            assert len(state.split("_", 1)) == 2
            state_name, state_value = state.split("_", 1)
            assert state_value.startswith("{") and state_value.endswith("}")

    def get_checkpoint_format(self) -> str:
        return "ckpt--iter_{iteration}--epoch_{epoch}--step_{global_step}"

    @abstractmethod
    def checkpoint(self):
        NotImplementedError("checkpoint method is not implemented yet.")


class DeepSpeedPolicy(BasePolicy):
    """
    solely uses DeepSpeed (ds) for training and ditched the Accelerate library. The accelerate library does not support two models in a single
    training loop, which becomes problematic in policies that use multiple models (actor-critic)
    """

    def __init__(
        self, distributed_state: PartialState, cache_ds_engines: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.distributed_state = distributed_state
        self.cache_ds_engines = cache_ds_engines
        self._set_process_log_level(logger)

    def create_optimizer(
        self,
        model: PreTrainedModel,
        weight_decay: float = 0.0,
    ) -> Union[Optimizer, DummyOptim]:
        from accelerate.utils import DummyOptim

        optim_params = get_optimizer_grouped_parameters(model, weight_decay)
        optim = DummyOptim(optim_params)
        optim.param_groups = {}
        return optim

    def create_lr_scheduler(
        self,
        optim: Optimizer,
        name: str,
        warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
    ) -> LRScheduler:
        return get_scheduler(
            name=name,
            optimizer=optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _init_deepspeed_engine_for_inference(
        self,
        model: PreTrainedModel,
        deepspeed_config: Dict[str, Any],
    ) -> DeepSpeedEngine:
        engine, *_ = deepspeed.initialize(
            model=model,
            config=deepspeed_config,
        )
        return engine

    def _init_deepspeed_engine_for_training(
        self,
        model: PreTrainedModel,
        deepspeed_config: Dict[str, Any],
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> DeepSpeedEngine:
        """
        Helper function to convert prertrained model to deepspeed engine.
        """
        kwargs = {
            "model": model,
            "lr_scheduler": lr_scheduler,
            "config": deepspeed_config,
        }

        if isinstance(optimizer, DummyOptim):
            kwargs["model_parameters"] = optimizer.params
        else:
            kwargs["optimizer"] = optimizer
        engine, *_ = deepspeed.initialize(**kwargs)
        return engine

    def _destroy_ds_engine(self, ds_engine: DeepSpeedEngine):
        if self.cache_ds_engines:
            return

        def delete_attr(obj, a_name):
            if hasattr(obj, a_name):
                delattr(obj, a_name)

        # This is a workaround to avoid a memory leak in DeepSpeed
        # This bug exists in the DeepSpeed version 0.14.1
        for name, param in ds_engine.named_parameters():
            delete_attr(param, "get_full_hp_grad")
            delete_attr(param, "set_full_hp_grad")
            delete_attr(param, "load_hp_checkpoint_state")
            delete_attr(param, "_hp_mapping")

        ds_engine.empty_partition_cache()
        ds_engine.destroy()  # todo(milad): why deeospeed has these globals

    def _patch_ds_config_for_optimizer(
        self,
        config: HfTrainerDeepSpeedConfig,
    ):
        config.fill_only("optimizer.params.lr", self.learning_rate, "learning_rate")
        config.fill_only(
            "optimizer.params.betas",
            [self.adam_beta1, self.adam_beta2],
            "adam_beta1+adam_beta2",
        )
        config.fill_only("optimizer.params.eps", self.adam_epsilon, "adam_epsilon")
        config.fill_only(
            "optimizer.params.weight_decay", self.weight_decay, "weight_decay"
        )

    def _patch_ds_config_for_lr_scheduler(
        self,
        config: HfTrainerDeepSpeedConfig,
        total_num_training_steps: int,
        warmup_steps: int,
        learning_rate: float,
    ) -> None:
        config.fill_only(
            "scheduler.params.total_num_steps",
            total_num_training_steps,
            "num_training_steps (calculated)",
        )
        logger.info(
            f" patched scheduler params num train steps to : {total_num_training_steps}"
        )

        config.fill_only(
            "scheduler.params.warmup_num_steps",
            warmup_steps,
            "warmup_steps",
        )

        logger.info(f"patched waruup stpes to {warmup_steps}")
        config.fill_only(
            "scheduler.params.warmup_min_lr",
            0,
            "warmup_min_lr",
        )
        config.fill_only(
            "scheduler.params.warmup_max_lr",
            learning_rate,
            "warmup_max_lr",
        )

    def _patch_ds_config_for_batch_size(
        self,
        config: HfTrainerDeepSpeedConfig,
        global_batch_size: int,
    ) -> None:
        logger.info(f"config pre patched {config}")
        config.fill_only(
            "train_micro_batch_size_per_gpu",
            self.per_device_batch_size,
            "per_device_train_batch_size",
        )
        logger.info(f"patched train micro batch size to {self.per_device_batch_size}")
        config.fill_only(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        logger.info(
            f" patched gradient accumulation steps to {self.gradient_accumulation_steps}"
        )
        config.fill_only(
            "train_batch_size", global_batch_size, "train_batch_size (calculated)"
        )
        logger.info(f"patche train batch size to {global_batch_size}")
        config.fill_only("gradient_clipping", self.max_grad_norm, "max_grad_norm")
        logger.info(f" patched gradient clipping to {self.max_grad_norm}")

    def _patch_ds_config_for_dtype(self, config: HfTrainerDeepSpeedConfig) -> None:
        assert not self.fp16, "FP16 is not supported for now"
        config.fill_only(
            "bf16.enabled", (self.bf16 or self.bf16_full_eval), "bf16|bf16_full_eval"
        )

    def _patch_ds_config_for_bucket_size(
        self, config: HfTrainerDeepSpeedConfig, model_config: PretrainedConfig
    ) -> None:
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if config.is_auto(x)]

        if len(hidden_size_auto_keys) > 0:
            if hasattr(model_config, "hidden_size"):
                hidden_size = model_config.hidden_size
            elif hasattr(model_config, "hidden_sizes"):
                # if there are many hidden sizes pick the largest one
                hidden_size = max(model_config.hidden_sizes)
            else:
                logger.warning(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f"in the DeepSpeed config file: {hidden_size_auto_keys}. We will set them to default values."
                )

                # if hidden size is not available, set the default values
                default_values = {
                    "zero_optimization.reduce_bucket_size": 5e8,
                    "zero_optimization.stage3_prefetch_bucket_size": 5e8,
                    "zero_optimization.stage3_param_persistence_threshold": 1e6,
                }
                for key in hidden_size_auto_keys:
                    if config.is_auto(key):
                        config.fill_only(key, default_values[key])
                return

            config.fill_only(
                "zero_optimization.reduce_bucket_size", hidden_size * hidden_size
            )
            if config.is_zero3():
                # automatically assign the optimal config values based on model config
                config.fill_only(
                    "zero_optimization.stage3_prefetch_bucket_size",
                    0.9 * hidden_size * hidden_size,
                )
                config.fill_only(
                    "zero_optimization.stage3_param_persistence_threshold",
                    10 * hidden_size,
                )

    def _is_main_process(self) -> bool:
        """Deal with the distributed state"""
        return self.distributed_state.is_main_process

    # TODO: move this to deepspeedpolicy class
    def _set_process_log_level(self, logger_obj: logging.Logger):
        if not self.distributed_state.is_local_main_process:
            logger_obj.setLevel(logging.WARNING)


def get_optimizer_grouped_parameters(
    model: PreTrainedModel,
    weight_decay: float,
    lora_lr: Optional[float] = None,
    no_decay_name_list: Optional[List[str]] = None,
    lora_name_list: Optional[List[str]] = None,
):
    # Taken from
    # https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/utils.py#L209
    if lora_name_list is None:
        lora_name_list = ["lora_right_weight", "lora_left_weight"]
    if no_decay_name_list is None:
        no_decay_name_list = [
            "bias",
            "layer_norm.weight",
            "layernorm.weight",
            "norm.weight",
            "ln_f.weight",
        ]

    optimizer_grouped_parameters = [
        # Weight decay, non-lora parameters
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
        },
        # Weight decay, lora parameters
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr,
        },
        # No weight decay, irrespective of lora
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups
