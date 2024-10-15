from abc import abstractmethod
from typing import  Tuple, Optional, Dict, Any, Callable, Literal, Union
from pathlib import Path

import torch
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions


from common.dataset import EpisodeDataset
from common.logging import get_logger
from policies.base_policy import DeepSpeedPolicy
from policies.base_critic import PretrainedModelValueHead

logger = get_logger(__name__)

class ActorCriticPolicy(DeepSpeedPolicy):
    """
        Base Actor critic type policy. 
        We use deepspeed here because of accelerate's isseus with multiple models

        The actor predicts an action from an observation and a 
        critic predicts the value of an observation. 
    """
    def __init__(
            self,          
            actor_model_fn: Callable[[], PreTrainedModel],
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
    

    def _init_models(self) -> Tuple[DeepSpeedEngine, DeepSpeedEngine]:
        """ Loads the models onto the gpu"""
        actor_engine = self._init_actor_model()
        critic_engine = self._init_critic_model()
        return actor_engine, critic_engine

    def _init_actor_model(self) -> DeepSpeedEngine:
        if hasattr(self, "actor"):
            return self.actor

        logger.info(f"Creating the actor deepspeed engine...")

        this_process_device = self.distributed_state.device

        # instantiate the actor model... 
        actor_model: PreTrainedModel = self.actor_model_fn()
        actor_model.to(this_process_device)

        if self.gradient_checkpointing:
            actor_model.gradient_checkpointing_enable()

        ds_config = HfTrainerDeepSpeedConfig(self.actor_config)

        # Create the optimizer
        # has_optimizer = ds_config.get_value("optimizer", None) is not None
        # if has_optimizer:
        #     weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
        #     if weight_decay == "auto":
        #         weight_decay = self.weight_decay

        #     optimizer = self.create_optimizer(actor_model, weight_decay)
        # else:
        #     optimizer = None

        # # Create the LR scheduler
        # # noinspection DuplicatedCode
        # has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        # self.warmup_steps = self.get_warmup_steps(10) #TODO Get the training steps from the trainer?
    
        # if has_deepspeed_scheduler:
        #     lr_scheduler = None
        #     self._patch_ds_config_for_lr_scheduler(
        #         ds_config,
        #         total_num_training_steps=self.total_num_training_steps,
        #         warmup_steps=self.warmup_steps,
        #         learning_rate=self.learning_rate,
        #     )
        # elif self.lr_scheduler_type is not None:
        #     logger.info("Using non-DeepSpeed LR scheduler.")
        #     lr_scheduler = self.create_lr_scheduler(
        #         optimizer,
        #         name=self.lr_scheduler_type,
        #         warmup_steps=self.warmup_steps,
        #         num_training_steps=self.total_num_training_steps,
        #     )
        # else:
        #     lr_scheduler = None

        optimizer=None
        lr_scheduler=None

        self._patch_ds_config_for_optimizer(ds_config)
        self._patch_ds_config_for_batch_size(ds_config, self.global_batch_size)
        self._patch_ds_config_for_dtype(ds_config)
        self._patch_ds_config_for_bucket_size(ds_config, actor_model.config)

        print("ds config", ds_config)

        engine = self._init_deepspeed_engine_for_training(
            actor_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if self.cache_ds_engines:
            self.actor = engine

        return engine
  
    def _init_critic_model(
        self,
        hf_checkpoint_path: Optional[Path] = None,
    ) -> DeepSpeedEngine:
        # Check if the model is 'cached'
        if hasattr(self, "critic"):
            return self.critic

        logger.info(f"Creating the critic deepspeed engine...")

        this_process_device = self.distributed_state.device

        ds_config = HfTrainerDeepSpeedConfig(self.critic_config)

        critic_model: PreTrainedModel = self.critic_model_fn()
        critic_model.to(this_process_device)

        if hf_checkpoint_path is not None:
            assert (hf_checkpoint_path / "pytorch_model.bin").exists()
            critic_model.load_state_dict(
                torch.load(hf_checkpoint_path / "pytorch_model.bin")
            )
            critic_model.to(this_process_device)

        # noinspection DuplicatedCode
        if self.gradient_checkpointing:
            critic_model.gradient_checkpointing_enable()

        # Create the optimizer
        # has_optimizer = ds_config.get_value("optimizer", None) is not None
        # if has_optimizer:
        #     weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
        #     if weight_decay == "auto":
        #         weight_decay = self.weight_decay

        #     optimizer = self.create_optimizer(critic_model, weight_decay)
        # else:
        #     optimizer = None

        # # Create the LR scheduler
        # # noinspection DuplicatedCode
        # has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        # self.warmup_steps = self.get_warmup_steps(10) #TODO Get the training steps from the trainer?

        # if has_deepspeed_scheduler:
        #     lr_scheduler = None
        #     self._patch_ds_config_for_lr_scheduler(
        #         ds_config,
        #         total_num_training_steps=self.total_num_training_steps,
        #         warmup_steps=self.warmup_steps,
        #         learning_rate=self.learning_rate,
        #     )
        # elif self.lr_scheduler_type is not None:
        #     #logger.info("Using non-DeepSpeed LR scheduler.")
        #     lr_scheduler = self.create_lr_scheduler(
        #         optimizer,
        #         name=self.lr_scheduler_type,
        #         warmup_steps=self.warmup_steps,
        #         num_training_steps=self.total_num_training_steps,
        #     )
        # else:
        #     lr_scheduler = None

        optimizer = None
        lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config)
        self._patch_ds_config_for_batch_size(ds_config, self.global_batch_size)
        self._patch_ds_config_for_dtype(ds_config)
        self._patch_ds_config_for_bucket_size(ds_config, critic_model.config)

        engine = self._init_deepspeed_engine_for_training(
            critic_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if self.cache_ds_engines:
            self.critic = engine

        return engine

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
            return_mean_entropy: bool = False,
            return_logits: bool = True,
            return_sequence_logp: bool = False,
            return_all_logp : bool = False,
            sequence_logp_reduction: Optional[Literal["mean"]] = None
        ) -> torch.Tensor:
        """
        Forward pass of the policy.

        Input ids: shape (batch, seq_length)
        Attention_masks shape (batch, seq_length)
        Labels shape (batch, seq_length)
        """

        model_device = next(self.actor.parameters()).device
        assert input_ids.device == model_device
        assert labels.device == model_device
        assert attention_mask.device == model_device

        outputs: Union[CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions ]= self.actor.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )

        assert isinstance(outputs, (CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions)), \
            f"Expected output type to be CausalLMOutputWithPast or CausalLMOutputWithCrossAttentions, but got {type(outputs)}"

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


