from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Self, JsonDict

from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel
from accelerate import PartialState
from transformers.integrations import HFTrainerDeepSpeedConfig
import deepspeed
from deepspeed import DeepSpeedEngine

from wandb.sdk.wandb_run import Run as WandbRun
from common.deepspeed_utils import get_optimizer_grouped_parameters 

from episode_generation.environment.base_environment import History


class BaseModel():
    """
    Make predictions in response to opservations. 

    In case of Policies, the prediction is an action. 
    In case of Critics, the prediction is the estimated value of the observation.
    """
 
    def __init__(self):
        super().__init__()
        pass

    def save(self, path: str) -> None:
        """ Save the model to a file"""
        raise NotImplementedError( "Not implemented yet" )

    def load(self, path: str, device: torch.device) -> Self:
        """ Load the policy from a file"""
        raise NotImplementedError( "Not implemented yet" )
    
    @property
    def device(self) -> torch.device:
        """ Get the device of the model"""
        pass


class BasePolicy(BaseModel):
    """
    A policy takes an observation and returns an action. 
    """
    @abstractmethod
    def predict(self, observation: History) -> History:
        """
        Predict an action from an observation.
        """
        pass

    @abstractmethod 
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the policy.
        """
        pass

    @abstractmethod
    def set_train(
        self, 
        train: bool,
    ):
        """ Set model in train or inference mode. """

    def get_train(
            self, 
    ) -> bool:
        """ Get the mode of the model"""
        return self.train

    def set_root_dir(self, path: Path):
        self.project_root_dir = path
        self._init_checkpoint_dir()

    def set_distributed_state(self, distributed_state: PartialState):
        self.set_distributed_state = distributed_state

    def set_cloud_log(self, cloud_logger: WandbRun):
        self.cloud_logger = cloud_logger

    def _init_checkpoint_dir(self):
        self.checkpoint_dir = (self.project_root_dir / "policy")
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


class DeepSpeedPolicy(BaseModel):
    """
    solely uses DeepSpeed for training and ditched the Accelerate library. The accelerate library does not support two models in a single
    training loop, which becomes problematic in policies that use multiple models (actor-critic)
    """
    def __init__(
        self,
        cache_ds_engines: bool = False,
        **kwargs
    ):      
        super().__init__(**kwargs) 
        self.cache_ds_engines = cache_ds_engines
        # define default x-axis (for latest wandb versions)
        if self._is_main_process():
            if getattr(self.cloud_logger, "define_metric", None):
                self.cloud_logger.define_metric("train/global_step")
                self.cloud_logger.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )
    
    def _initialize_deepspeed_engine_for_inference(
        self,
        model: PreTrainedModel,
        deepspeed_config: JsonDict,
    ) -> DeepSpeedEngine:
        engine, *_ = deepspeed.initialize(
            model=model,
            config=deepspeed_config,
        )
        return engine

    def _init_deepspeed_model_for_training(
        self,
        model: PreTrainedModel,
        deepspeed_config: JsonDict,
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
            "optimizer": optimizer
        }
        engine, *_ = deepspeed.initialize(**kwargs)
        return engine

    def _destroy_engines(self):
        """ Destorys the engines after use (if not caches)"""
        if self.cache_ds_engines:
            return
        for engine in self.engines:
            self._destroy_ds_engine(engine)

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
        args: TrainingArguments,
    ):
        config.fill_only("optimizer.params.lr", args.learning_rate, "learning_rate")
        config.fill_only(
            "optimizer.params.betas",
            [args.adam_beta1, args.adam_beta2],
            "adam_beta1+adam_beta2",
        )
        config.fill_only("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        config.fill_only(
            "optimizer.params.weight_decay", args.weight_decay, "weight_decay"
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
        config.fill_only(
            "scheduler.params.warmup_num_steps",
            warmup_steps,
            "warmup_steps",
        )
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
        args: TrainingArguments,
        global_batch_size: int,
    ) -> None:
        config.fill_only(
            "train_micro_batch_size_per_gpu",
            args.per_device_train_batch_size,
            "per_device_train_batch_size",
        )
        config.fill_only(
            "gradient_accumulation_steps",
            args.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        config.fill_only(
            "train_batch_size", global_batch_size, "train_batch_size (calculated)"
        )
        config.fill_only("gradient_clipping", args.max_grad_norm, "max_grad_norm")

    def _patch_ds_config_for_dtype(
        self, config: HfTrainerDeepSpeedConfig, args: TrainingArguments
    ) -> None:
        assert not args.fp16, "FP16 is not supported for now"
        config.fill_only(
            "bf16.enabled", (args.bf16 or args.bf16_full_eval), "bf16|bf16_full_eval"
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
        """ Deal with the distributed state"""
        return self.distributed_state.is_main_process
    
    
class BaseCritic(BaseModel):
    """
    Main critic model. Takes an observation and returns a value estimate of the observation.
    """
    @abstractmethod
    def predict(self, observation: History) -> float:
        """
        Predict an value from an observation.
        """
        pass


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


