import shutil
from dataclasses import dataclass
import time
from abc import abstractmethod
from typing import Optional, Dict, Any, Callable, NamedTuple, Union
from pathlib import Path


import torch
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from deepspeed import DeepSpeedEngine
from deepspeed import comm as dist

from relign.common.dataset import EpisodeDataset
from relign.utils.logging import get_logger
from relign.policies.base_policy import CriticForwardOutput
from relign.policies.base_actor import ActorPolicy
from relign.policies.base_critic import PretrainedModelValueHead
from relign.utils.trainer import masked_mean, monitor_tensor_anomalies

logger = get_logger(__name__)


class CriticForwardOutput(NamedTuple):
    values: Optional[torch.Tensor] = None

    def _maybe_init_ds_engine_for_training(
        self,
        model: PreTrainedModel,
        ds_config: Optional[Dict[str, Any]],
        is_actor: bool = True,
    ):
        """
        If ds_config is provided, create optimizer + scheduler
        and wrap the model in a DeepSpeedEngine for training.
        Otherwise return the raw HF model.
        """

        if ds_config is None:
            # user provided no DS config => just keep the raw model
            return model

        # TODO: Fix the optimizer and lrscheduler initialization
        # 1) Create an optimizer
        # optimizer = self.create_optimizer(model, weight_decay=self.weight_decay)
        optimizer = None

        # 2) Create a learning rate scheduler
        self._patch_ds_config_for_lr_scheduler(
            ds_config,
            total_num_training_steps=self.total_num_training_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
        )

        lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config)
        self._patch_ds_config_for_batch_size(ds_config, self.global_batch_size)
        self._patch_ds_config_for_dtype(ds_config)
        self._patch_ds_config_for_bucket_size(ds_config, self.actor_config)

        # 3) Actually init the DS engine.
        #    We rely on self._init_deepspeed_engine_for_training from DeepSpeedPolicy.
        engine = self._init_deepspeed_engine_for_training(
            model=model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        # TODO: engine caching
        return engine


@dataclass
class Checkpoint:
    path: Path
    iteration: int


class ActorCriticPolicy(ActorPolicy):
    """
    Base Actor critic type policy.
    We use deepspeed here because of accelerate's isseus with multiple models

    The actor predicts an action from an observation and a
    critic predicts the value of an observation.
    """

    def __init__(
        self,
        critic_model_fn: Callable[[], PretrainedModelValueHead],
        critic_config: Optional[Dict[str, Any]],
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Check if the actor_ds_config is an instance of HFTrainerDeepspeedconfig, otherwise instantiate it

        self.critic_model_fn = critic_model_fn
        self.critic_config = critic_config

    def _init_critic_model(
        self,
        critic_model_fn: Callable[[], PretrainedModelValueHead],
        only_return_unwrapped_model: bool = False,
        hf_checkpoint_path: Optional[Path] = None,
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_critic_engine"):
            return self._critic_engine

        logger.info("Creating the critic deepspeed engine...")
        critic_model: PreTrainedModel = critic_model_fn()

        if hf_checkpoint_path is not None:
            assert (hf_checkpoint_path / "pytorch_model.bin").exists()
            critic_model.load_state_dict(
                torch.load(hf_checkpoint_path / "pytorch_model.bin")
            )
            critic_model.to(self.distributed_state.device)

        if only_return_unwrapped_model:
            critic_model.to(self.distributed_state.device)
            return critic_model

        if self.gradient_checkpointing:
            critic_model.gradient_checkpointing_enable()

        ds_config = HfTrainerDeepSpeedConfig(self.critic_config)

        # Create the optimizer if DS config has an optimizer section
        has_optimizer = ds_config.get_value("optimizer", None) is not None
        if has_optimizer:
            weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
            if weight_decay == "auto":
                weight_decay = self.weight_decay
            optimizer = self.create_optimizer(critic_model, weight_decay)
        else:
            optimizer = None

        # Create the LR scheduler if DS config has a scheduler
        has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        warmup_steps = self.warmup_steps
        if has_deepspeed_scheduler:
            lr_scheduler = None
            self._patch_ds_config_for_lr_scheduler(
                ds_config,
                total_num_training_steps=self.total_num_training_steps,
                warmup_steps=warmup_steps,
                learning_rate=self.learning_rate,
            )

        elif self.lr_scheduler_type is not None:
            logger.info("Using non-DeepSpeed LR scheduler.")
            lr_scheduler = self.create_lr_scheduler(
                optimizer,
                name=self.lr_scheduler_type,
                warmup_steps=warmup_steps,
                num_training_steps=self.total_num_training_steps,
            )
        else:
            lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config)
        self._patch_ds_config_for_batch_size(ds_config, self.global_batch_size)
        self._patch_ds_config_for_dtype(ds_config)
        self._patch_ds_config_for_bucket_size(ds_config, critic_model.config)

        import json

        logger.info(
            "DS config after batch patching:\n%s",
            json.dumps(ds_config.config, indent=2),
        )

        engine = self._init_deepspeed_engine_for_training(
            critic_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if self.cache_ds_engines:
            self._critic_engine = engine

        self.critic = engine
        return engine

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
            use_cache=False,
        )

        predicted_values = outputs
        predicted_values = predicted_values.to(torch.float32)

        # mask the query tokens (only interested in value of the response tokens)
        if labels is not None:
            value_mask = (labels != -100).to(attention_mask.dtype)
        else:
            value_mask = attention_mask

        return {"values": predicted_values, "value_mask": value_mask}

    def critic_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_valid_values: torch.FloatTensor,
        returns: torch.FloatTensor,
        trainer_params: Any,
    ):
        """
        Critic loss
        """
        # Switch to RL terminology
        action_mask = shifted_labels_mask

        # You can't pass "labels" to the critic, or you can remove them from model_inputs:
        new_inputs = {k: v for k, v in model_inputs.items() if k not in ["labels"]}

        outputs = self.forward_critic(
            input_ids=new_inputs["input_ids"],
            attention_mask=new_inputs["attention_mask"],
            labels=model_inputs.get("labels", None),
        )
        valid_values = outputs["values"][:, :-1]

        # Clipped value function
        values_clipped = torch.clamp(
            valid_values,
            old_valid_values - trainer_params.cliprange_value,
            old_valid_values + trainer_params.cliprange_value,
        )

        vf_losses1 = (valid_values - returns) ** 2
        vf_losses1_anomalies = monitor_tensor_anomalies(
            vf_losses1.detach(), action_mask
        )
        vf_losses2 = (values_clipped - returns) ** 2
        vf_losses = torch.max(vf_losses1, vf_losses2)
        vf_loss = 0.5 * masked_mean(vf_losses, action_mask)

        vf_clip_frac = masked_mean((vf_losses2 > vf_losses1).float(), action_mask)

        metrics = {
            "critic/value": masked_mean(valid_values, action_mask).detach(),
            "critic/mse": masked_mean(
                (valid_values - returns) ** 2, action_mask
            ).detach(),
            "critic/clip_frac": vf_clip_frac.detach(),
        }
        for i, v in vf_losses1_anomalies.items():
            metrics[f"critic/vf_losses1_anomalies__{i}"] = v

        return vf_loss, metrics

    def init_critic_engine_if_needed(
        self,
        global_batch_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        total_num_training_steps: int,
        critic_model_fn: Optional[Callable[[], PretrainedModelValueHead]] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Ensures self.critic (and self._critic_engine) is initialized if not already.
        If 'force_reload' is True, or if no engine is cached, re-initialize.
        """
        logger.info("Initializing critic DeepSpeed engine...")
        self.global_batch_isze = global_batch_size
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_num_training_steps = total_num_training_steps
        # Decide whether to skip if we already have a cached engine
        if (
            (not force_reload)
            and hasattr(self, "_critic_engine")
            and self._critic_engine is not None
        ):
            return  # already loaded

        # If a new callable was passed, update. Otherwise use existing
        if critic_model_fn is not None:
            self.critic_model_fn = critic_model_fn

        logger.info("Initializing critic DeepSpeed engine...")
        self._init_critic_model(
            critic_model_fn=self.critic_model_fn, only_return_unwrapped_model=False
        )
        logger.info("Critic engine init done.")

    def destroy_critic_engine_if_not_cached(self) -> None:
        """
        Destroys the critic engine to free memory if engine caching is disabled.
        """
        if not self.cache_ds_engines:
            if getattr(self, "critic", None) is not None:
                logger.info("Destroying critic engine to free memory.")
                self._destroy_ds_engine(self.critic)
                # If the engine provides a shutdown/cleanup method, call it.
                if hasattr(self.critic, "shutdown") and callable(self.critic.shutdown):
                    try:
                        self.critic.shutdown()
                    except Exception as e:
                        logger.warning(f"Error during critic engine shutdown: {e}")
                # If there is an internal engine attribute, clear it.
                if hasattr(self.critic, "_engine"):
                    del self.critic._engine

                # Remove the critic engine reference.
                self.critic = None

            # Also clear the cached critic engine if it exists.
            if hasattr(self, "_critic_engine"):
                self._critic_engine = None

    def destroy_ds_engines(self) -> None:
        """
        Destroys all cached engines.
        """
        self.destroy_critic_engine_if_not_cached()
        self.destroy_actor_engine_if_not_cached()
        self.destroy_reference_engine_if_not_cached()

    def _load_checkpoint_to_ds_engines(
        self,
        checkpoint_path: Path,
    ) -> None:
        metrics = {}
        if self.actor is not None:
            self.actor.load_checkpoint(str(checkpoint_path / "actor"))
        if self.critic is not None:
            self.critic.load_checkpoint(str(checkpoint_path / "critic"))
        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

    def load_latest_policy_path(self, project_root_dir: Optional[Path] = None) -> None:
        """loads both actor and critic from "policy/cache" folder"""
        if not project_root_dir:
            project_root_dir = self.project_root_dir
        self._load_checkpoint_to_ds_engines(project_root_dir / "policy" / "cache")

    def load_checkpoint(self, checkpoint: Union[Checkpoint, Path]) -> None:
        super().load_checkpoint(checkpoint)
        checkpoint_path = (
            checkpoint if isinstance(checkpoint, Path) else checkpoint.path
        )
        self._load_training_state(checkpoint_path)
        self.checkpoint_path_to_load = checkpoint_path

    def _save_hf_pretrained(self, engine: DeepSpeedEngine, path: Path) -> None:
        """Saves a huggingface model that can be used for inference"""
        if self._is_main_process():
            # Only save on the main process
            assert engine.zero_optimization_stage() < 3
            logger.info(f"Saving HF pretrained weights to {path}")
            unwrapped_model = engine.module
            unwrapped_model.save_pretrained(path, safe_serialization=False)
        dist.barrier()

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """
        saves both a hugginface model of the actor + critic and a
        deepspeed checkpoint which contains all the optimizer states
        saves the actor in `checkpoint_path/actor` and the critic in `checkpoint_path/critic`
        """
        if self._is_main_process():
            if checkpoint_path.exists():
                logger.warning(
                    f"Checkpoint path {checkpoint_path} already exists. Overwriting."
                )
                shutil.rmtree(checkpoint_path)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._save_trainer_state(checkpoint_path)

        if self.actor is not None:
            self._save_hf_pretrained(self.actor, checkpoint_path / "hf_pretrained")
            self.actor.save_checkpoint(str(checkpoint_path / "actor"))

        if self.critic is not None:
            self._save_hf_pretrained(
                self.critic, checkpoint_path / "critic" / "hf_pretrained"
            )
            self.critic.save_checkpoint(str(checkpoint_path / "critic"))

    def save_latest_policy_path(self, checkpoint_path: Path) -> str:
        """
        Saves both the actor and critic engines and returns the path of the actor for inference.

        The caller can specify an arbitrary checkpoint path (for example:
        Path("/some/storage/path/policy/TIMESTAMP") ).
        We will append '/actor/hf_pretrained' and '/critic/hf_pretrained'
        accordingly here.
        """

        # Path to store the HF version of the actor
        actor_hf_pretrained_path = checkpoint_path / "actor" / "hf_pretrained"
        # Path to store the HF version of the critic
        critic_hf_pretrained_path = checkpoint_path / "critic" / "hf_pretrained"

        # Save the HF-pretrained actor weights
        self._save_hf_pretrained(
            self.actor,
            actor_hf_pretrained_path,
        )

        # If the DeepSpeed engines are not cached, we also save the critic HF weights,
        # as well as engine checkpoints for both the actor and critic.
        if not self.cache_ds_engines:
            self._save_hf_pretrained(
                self.critic,
                critic_hf_pretrained_path,
            )
            # Actually save the full engine state for both
            self.actor.save_checkpoint(str(checkpoint_path / "actor"))
            self.critic.save_checkpoint(str(checkpoint_path / "critic"))

        # Return the path of the (actor) HF pretrained directory for inference
        return actor_hf_pretrained_path
