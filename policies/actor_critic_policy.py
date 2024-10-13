from abc import abstractmethod
from typing import  Tuple, Optional, Dict, Any
from pathlib import Path

import torch
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from transformers.utils.logger import get_logger

from common.dataset import EpisodeDataset
from policies.base_policy import DeepSpeedPolicy
from policies.base_critic import PretrainedModelValueHead


get_logger(__name__)

class ActorCriticPolicy(DeepSpeedPolicy):
    """
        Base Actor critic type policy. 
        We use deepspeed here because of accelerate's isseus with multiple models

        The actor predicts an action from an observation and a 
        critic predicts the value of an observation. 
    """
    def __init__(
            self,
            actor_model: PreTrainedModel,
            actor_ds_config: Optional[Dict[str, Any]],
            critic_model: PretrainedModelValueHead,
            critic_ds_config: Optional[Dict[str, Any]],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.actor_model = actor_model
        self.actor_config = actor_ds_config
        self.critic_model = critic_model
        self.critic_config = critic_ds_config

        # initialize the models
        self.actor, self.critic = self._init_models()
    
    def _init_actor_model(self) -> DeepSpeedEngine:
        if hasattr(self, "_actor_engine"):
            return self._actor_engine

        #logger.info(f"Creating the actor deepspeed engine...")

        this_process_device = self.distributed_state.device

        # Load the actor model into GPU
        # noinspection PyTypeChecker
        actor_model: PreTrainedModel = self.actor_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
        )

        if self.args.gradient_checkpointing:
            actor_model.gradient_checkpointing_enable()

        ds_config = HfTrainerDeepSpeedConfig(self.actor_deepspeed_config)

        # Create the optimizer
        has_optimizer = ds_config.get_value("optimizer", None) is not None
        if has_optimizer:
            weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
            if weight_decay == "auto":
                weight_decay = self.args.weight_decay

            optimizer = self.create_optimizer(actor_model, weight_decay)
        else:
            optimizer = None

        # Create the LR scheduler
        # noinspection DuplicatedCode
        has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        warmup_steps = self.args.get_warmup_steps(self.total_num_training_steps)
        if has_deepspeed_scheduler:
            lr_scheduler = None
            self._patch_ds_config_for_lr_scheduler(
                ds_config,
                total_num_training_steps=self.total_num_training_steps,
                warmup_steps=warmup_steps,
                learning_rate=self.args.learning_rate,
            )
        elif self.args.lr_scheduler_type is not None:
            #logger.info("Using non-DeepSpeed LR scheduler.")
            lr_scheduler = self.create_lr_scheduler(
                optimizer,
                name=self.args.lr_scheduler_type,
                warmup_steps=warmup_steps,
                num_training_steps=self.total_num_training_steps,
            )
        else:
            lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config, self.args)
        self._patch_ds_config_for_batch_size(
            ds_config, self.args, self.global_batch_size
        )
        self._patch_ds_config_for_dtype(ds_config, self.args)
        self._patch_ds_config_for_bucket_size(ds_config, actor_model.config)

        engine = self._initialize_deepspeed_engine_for_training(
            actor_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if self.cache_deepspeed_engines:
            self._actor_engine = engine

        return engine
  
    def _init_critic_model(
        self,
        hf_checkpoint_path: Optional[Path] = None,
    ) -> DeepSpeedEngine:
        if hasattr(self, "_critic_engine"):
            return self._critic_engine

        logger.info(f"Creating the critic deepspeed engine...")

        this_process_device = self.distributed_state.device

        ds_config = HFTrainerDeepSpeedConfig(self.critic_config)

        # noinspection PyTypeChecker
        critic_model: PreTrainedModel = self.critic_lazy.construct(
            device=this_process_device,
        )

        if hf_checkpoint_path is not None:
            assert (hf_checkpoint_path / "pytorch_model.bin").exists()
            critic_model.load_state_dict(
                torch.load(hf_checkpoint_path / "pytorch_model.bin")
            )
            critic_model.to(this_process_device)

        # noinspection DuplicatedCode
        if self.args.gradient_checkpointing:
            critic_model.gradient_checkpointing_enable()

        # Create the optimizer
        has_optimizer = ds_config.get_value("optimizer", None) is not None
        if has_optimizer:
            weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
            if weight_decay == "auto":
                weight_decay = self.args.weight_decay

            optimizer = self.create_optimizer(critic_model, weight_decay)
        else:
            optimizer = None

        # Create the LR scheduler
        # noinspection DuplicatedCode
        has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        warmup_steps = self.args.get_warmup_steps(self.total_num_training_steps)
        if has_deepspeed_scheduler:
            lr_scheduler = None
            self._patch_ds_config_for_lr_scheduler(
                ds_config,
                total_num_training_steps=self.total_num_training_steps,
                warmup_steps=warmup_steps,
                learning_rate=self.args.learning_rate,
            )
        elif self.args.lr_scheduler_type is not None:
            #logger.info("Using non-DeepSpeed LR scheduler.")
            lr_scheduler = self.create_lr_scheduler(
                optimizer,
                name=self.args.lr_scheduler_type,
                warmup_steps=warmup_steps,
                num_training_steps=self.total_num_training_steps,
            )
        else:
            lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config, self.args)
        self._patch_ds_config_for_batch_size(
            ds_config, self.args, self.global_batch_size
        )
        self._patch_ds_config_for_dtype(ds_config, self.args)
        self._patch_ds_config_for_bucket_size(ds_config, critic_model.config)

        engine = self._initialize_deepspeed_engine_for_training(
            critic_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if self.cache_deepspeed_engines:
            self._critic_engine = engine

        return engine
    
    def _init_models(self) -> Tuple[DeepSpeedEngine, DeepSpeedEngine]:
        """ Loads the models onto the gpu"""
        self.actor = self._init_critic_model()
        self.critic = self.init_actor_model()

    def destroy_models(self):
        """ Deletes the models if not cashed"""
        self._destroy_ds_engine(self.actor)
        del self.actor
        self._destroy_ds_engine(self.critic)
        del self.critic
        #release_memory()

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

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the policy.
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

    def forward_critic(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
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


