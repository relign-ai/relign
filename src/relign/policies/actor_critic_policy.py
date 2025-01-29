from abc import abstractmethod
from typing import  Tuple, Optional, Dict, Any, Callable, Literal, NamedTuple, Union
from functools import partial
from pathlib import Path
import time


import torch 
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from deepspeed import DeepSpeedEngine


from relign.common.dataset import EpisodeDataset
from relign.utils.logging import get_logger
from relign.policies.base_policy import DeepSpeedPolicy 
from relign.policies.base_critic import PretrainedModelValueHead
from relign.utils.trainer import masked_mean, monitor_tensor_anomalies

logger = get_logger(__name__)

class ActorForwardOutput(NamedTuple):
    logits: Optional[torch.Tensor] = None
    sequence_logp: Optional[torch.Tensor] = None
    all_logp: Optional[torch.Tensor] = None


class CriticForwardOutput(NamedTuple):
    values: Optional[torch.Tensor] = None

    def _maybe_init_ds_engine_for_training(
        self,
        model: PreTrainedModel,
        ds_config: Optional[Dict[str, Any]],
        is_actor: bool = True
    ):
        """
        If ds_config is provided, create optimizer + scheduler
        and wrap the model in a DeepSpeedEngine for training.
        Otherwise return the raw HF model.
        """

        if ds_config is None:
            # user provided no DS config => just keep the raw model
            return model

        #TODO: Fix the optimizer and lrscheduler initialization
        # 1) Create an optimizer
        # optimizer = self.create_optimizer(model, weight_decay=self.weight_decay)
        optimizer = None

        # 2) Create a learning rate scheduler
        self._patch_ds_config_for_lr_scheduler(
            ds_config,
            total_num_training_steps=self.total_num_training_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate
        )

        lr_scheduler=None

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

        #TODO: engine caching
        return engine


class ActorCriticPolicy(DeepSpeedPolicy):
    """
        Base Actor critic type policy. 
        We use deepspeed here because of accelerate's isseus with multiple models

        The actor predicts an action from an observation and a 
        critic predicts the value of an observation. 
    """
    def __init__(
        self,          
        actor_model_fn: Callable[[],PreTrainedModel],
        actor_config: Optional[Dict[str, Any]],
        critic_model_fn: Callable[[], PretrainedModelValueHead],
        critic_config: Optional[Dict[str, Any]],
        **kwargs
    ):
        super().__init__(**kwargs)

        # Check if the actor_ds_config is an instance of HFTrainerDeepspeedconfig, otherwise instantiate it

        self.actor_model_fn = actor_model_fn
        self.actor_config = actor_config
        self.critic_model_fn = critic_model_fn
        self.critic_config = critic_config


    def _init_actor_model(
        self,
        actor_model_fn: Callable[[], PreTrainedModel],
        only_return_unwrapped_model: bool = False
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:

        if hasattr(self, "_actor_engine"):
            return self._actor_engine

        logger.info("Creating the actor deepspeed engine...")
        actor_model: PreTrainedModel = actor_model_fn()

        if self.gradient_checkpointing:
            actor_model.gradient_checkpointing_enable()

        ds_config = HfTrainerDeepSpeedConfig(self.actor_config)

        # Create the optimizer if DS config has an optimizer section
        has_optimizer = ds_config.get_value("optimizer", None) is not None
        if has_optimizer:
            weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
            if weight_decay == "auto":
                weight_decay = self.weight_decay
            optimizer = self.create_optimizer(actor_model, weight_decay)
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
        self._patch_ds_config_for_bucket_size(ds_config, actor_model.config)

        logger.info("USED OPTIMZEER {optimizer}")

        engine = self._init_deepspeed_engine_for_training(
            actor_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if self.cache_ds_engines:
            self._actor_engine = engine

        self.actor = engine
        return engine


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
            critic_model.load_state_dict(torch.load(hf_checkpoint_path / "pytorch_model.bin"))
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


    def forward_actor(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_mean_entropy: bool = False,
        return_logits: bool = True,
        return_sequence_logp: bool = False,
        return_all_logp : bool = False,
        sequence_logp_reduction: Optional[Literal["mean"]] = None
    ) -> ActorForwardOutput:
        """
        Forward pass of the policy.

        Input ids: shape (batch, seq_length)
        Attention_masks shape (batch, seq_length)
        Labels shape (batch, seq_length)
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

        output = {}
        if return_logits:
            output["logits"] = logits

        if return_sequence_logp:
            sequence_log_probs = per_token_log_probs.sum(dim=-1)
            if sequence_logp_reduction == "mean":
                sequence_log_probs = sequence_log_probs / shift_label_mask.sum(dim=-1)
            output["sequence_logp"] = sequence_log_probs

        if return_all_logp:
            output["all_logp"] = per_token_log_probs 

        return ActorForwardOutput(**output)


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


    def actor_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: Optional[torch.FloatTensor],
        advantages: torch.FloatTensor,
    ):
        """
        The PPO-style actor loss.
        Assumes self.ppo_hparams, self.kl_ctl, etc. are defined in this policy.
        """
        # Switch to RL terminology
        action_mask = shifted_labels_mask

        # Compute the log probabilities of the actor
        outputs = self.forward_actor(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            return_all_logp=True,
            return_logits=False,
            return_sequence_logp=False,
        )
        logprobs = outputs.all_logp  # shape: (batch_size, seq_len-1)

        # Compute the PPO-clip loss
        log_ratio = (logprobs - old_logprobs) * action_mask
        ratio = torch.exp(log_ratio)

        pg_losses1 = -advantages * ratio
        pg_losses1_anomalies = monitor_tensor_anomalies(pg_losses1.detach(), action_mask)
        clip_range = 0.4
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        pg_losses = torch.max(pg_losses1, pg_losses2)
        pg_loss = masked_mean(pg_losses, action_mask)

        # Possibly apply a KL penalty if self.ppo_hparams.kl_penalty_loss_type is not None
        ref_kl_loss = None
        ref_kl = None
        kl_penalty_loss_type = 0.4
        kl_penalty_loss_clip_max = 1000
        kl_penalty_loss_clip_min  = 0
        kl_clt = 0.2
        if kl_penalty_loss_type is not None:
            # _compute_kl_penalty is below
            ref_kl_tensor = self._compute_kl_penalty(
                logprobs,
                ref_logprobs,
                estimation_type=kl_penalty_loss_type
            )
            # clamp for numerical stability
            ref_kl_tensor = torch.clamp(
                ref_kl_tensor * action_mask,
                min=kl_penalty_loss_clip_min,
                max=kl_penalty_loss_clip_max,
            )
            ref_kl_loss = kl_clt * ref_kl_tensor.sum(dim=1).mean()
            pg_loss = pg_loss + ref_kl_loss
            ref_kl = ref_kl_tensor.detach()

        # Ratio check
        is_skipped = False
        avg_ratio = masked_mean(ratio, action_mask)
        ratio_threshold = 1.5
        if avg_ratio.item() > ratio_threshold:
            logger.warning(
                f"High PPO ratio detected: {avg_ratio.item():.2f}. Skipping this batch."
            )
            pg_loss = pg_loss * 0.0
            is_skipped = True

        pg_clip_frac = masked_mean(
            (pg_losses2 > pg_losses1).float(), action_mask
        )
        approx_kl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, action_mask)
        policy_kl = masked_mean(old_logprobs - logprobs, action_mask)

        metrics = {
            "actor/approx_kl": approx_kl.detach(),
            "actor/policy_kl": policy_kl.detach(),
            "actor/clip_frac": pg_clip_frac.detach(),
            "actor/ratio": avg_ratio.detach(),
        }
        # Include anomaly stats
        for i, v in pg_losses1_anomalies.items():
            metrics[f"actor/pg_losses1_anomalies__{i}"] = v

        return pg_loss, is_skipped, metrics, ref_kl


    def critics_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_valid_values: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        The PPO-style critic loss.
        """
        # Switch to RL terminology
        action_mask = shifted_labels_mask

        # You can't pass "labels" to the critic, or you can remove them from model_inputs:
        new_inputs = {
            k: v for k, v in model_inputs.items() if k not in ["labels"]
        }

        outputs = self.forward_critic(
            input_ids=new_inputs["input_ids"],
            attention_mask=new_inputs["attention_mask"],
            labels=model_inputs.get("labels", None),
        )
        valid_values = outputs["values"][:, :-1]

        # Clipped value function
        values_clipped = torch.clamp(
            valid_values,
            old_valid_values - self.ppo_hparams.cliprange_value,
            old_valid_values + self.ppo_hparams.cliprange_value,
        )

        vf_losses1 = (valid_values - returns) ** 2
        vf_losses1_anomalies = monitor_tensor_anomalies(vf_losses1.detach(), action_mask)
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


    def _compute_kl_penalty(
        self,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        estimation_type: str = "forward_kl",
    ):
        """
        Basic example: forward_kl = (logprobs - ref_logprobs).
        You might have a more sophisticated approach in your PPOTrainer.
        """
        if ref_logprobs is None:
            return torch.zeros_like(logprobs)

        kl = logprobs - ref_logprobs
        # If you want reverse_kl or symmetric_kl, do so here
        if estimation_type == "forward_kl":
            # forward_kl = log p/q = log p - log q
            pass  # already computed
        # elif estimation_type == ...
        return kl


    def init_actor_engine_if_needed(
        self,
        actor_model_fn: Optional[Callable[[], PreTrainedModel]] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Ensures self.actor (and self._actor_engine) is initialized if not already.
        If 'force_reload' is True, or if no engine is cached, re-initialize.
        """
        logger.info("Initializing actor DeepSpeed engine...")

        # Decide whether to skip if we already have a cached engine
        if (not force_reload) and hasattr(self, "_actor_engine") and self._actor_engine is not None:
            return  # already loaded
        
        # If a new callable was passed, update. Otherwise use existing
        if actor_model_fn is not None:
            self.actor_model_fn = actor_model_fn

        logger.info("Initializing actor DeepSpeed engine...")
        self._init_actor_model(actor_model_fn=self.actor_model_fn, only_return_unwrapped_model=False)
        logger.info("Actor engine init done.")


    def init_critic_engine_if_needed(
        self,
        critic_model_fn: Optional[Callable[[], PretrainedModelValueHead]] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Ensures self.critic (and self._critic_engine) is initialized if not already.
        If 'force_reload' is True, or if no engine is cached, re-initialize.
        """
        logger.info("Initializing critic DeepSpeed engine...")

        # Decide whether to skip if we already have a cached engine
        if (not force_reload) and hasattr(self, "_critic_engine") and self._critic_engine is not None:
            return  # already loaded

        # If a new callable was passed, update. Otherwise use existing
        if critic_model_fn is not None:
            self.critic_model_fn = critic_model_fn

        logger.info("Initializing critic DeepSpeed engine...")
        self._init_critic_model(critic_model_fn=self.critic_model_fn, only_return_unwrapped_model=False)
        logger.info("Critic engine init done.")


    def destroy_actor_engine_if_not_cached(self) -> None:
        """
        Destroys the actor engine unless it is cached (self.cache_deepspeed_engines == True).
        Useful for freeing memory between iterations if desired.
        """
        if not self.cache_deepspeed_engines:
            if hasattr(self, "_actor_engine") and self._actor_engine is not None:
                logger.info("Destroying actor engine to free memory.")
                # You might want to call engine.release_workspace() or similar if needed
                del self._actor_engine
                self._actor_engine = None

            # Clear the `self.actor` reference as well, to truly free memory
            self.actor = None


    def destroy_critic_engine_if_not_cached(self) -> None:
        """
        Destroys the critic engine unless it is cached.
        """
        if not self.cache_deepspeed_engines:
            if hasattr(self, "_critic_engine") and self._critic_engine is not None:
                logger.info("Destroying critic engine to free memory.")
                del self._critic_engine
                self._critic_engine = None

            self.critic = None

