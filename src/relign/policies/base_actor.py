import math
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, Optional, Literal, Callable, Union, Any

from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from deepspeed import DeepSpeedEngine

from relign.policies.base_policy import DeepSpeedPolicy
from relign.policies.base_policy import ActorForwardOutput
from relign.utils.trainer import masked_mean, monitor_tensor_anomalies
from relign.utils.logging import get_logger

logger = get_logger(__name__)

def print_deepspeed_config(ds_config: dict) -> None:
    """
    Utility function to print the DeepSpeed configuration in a readable format.
    
    This function prints the entire DeepSpeed config as a formatted JSON string,
    and, if present, separately prints the zero optimization-related settings
    so that you can verify that the desired values (e.g. stage 2) are set.
    
    :param ds_config: DeepSpeed configuration dictionary.
    """
    import json

    print("===== DeepSpeed Config =====")
    print(json.dumps(ds_config, indent=2))
    
    # Check for the new style zero config (sometimes under 'zero_optimization' or "zero_2")
    if "zero_optimization" in ds_config:
        print("\n== 'zero_optimization' Settings ==")
        print(json.dumps(ds_config["zero_optimization"], indent=2))
    elif "zero_2" in ds_config:
        print("\n== 'zero_2' Settings ==")
        print(json.dumps(ds_config["zero_2"], indent=2))
        
    # Also print any legacy DS zero keys that might affect behavior.
    legacy_keys = ["zero_enabled", "zero_force_ds_cpu_optimizer", "zero_optimization_stage"]
    legacy_info = {k: ds_config[k] for k in legacy_keys if k in ds_config}
    if legacy_info:
        print("\n== Legacy Zero Settings ==")
        print(json.dumps(legacy_info, indent=2))
        
class ActorPolicy(DeepSpeedPolicy):
    def __init__(
        self, 
        actor_model_fn, 
        reference_model_fn,
        actor_config, 
        enable_reference: bool = True, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._set_process_log_level(logger)
        self.actor_model_fn = actor_model_fn
        self.reference_model_fn = reference_model_fn
        self.actor_config = actor_config
        self.enable_reference = enable_reference
        self.reference = None

    def _init_actor_model(
        self,
        actor_model_fn: Callable[[], PreTrainedModel],
        only_return_unwrapped_model: bool = False,
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_actor_engine"):
            return self._actor_engine

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
        warmup_steps = self.warmup_steps if self.warmup_steps else self.get_warmup_steps(num_training_steps=self.total_num_training_steps) 
        if has_deepspeed_scheduler:
            logger.info("\n\n*********** Pathing  actor DS config ************")
            lr_scheduler = None
            self._patch_ds_config_for_lr_scheduler(
                ds_config,
                total_num_training_steps=self.total_num_training_steps,
                warmup_steps=warmup_steps,
                learning_rate=self.learning_rate,
            )
        elif self.lr_scheduler_type is not None:
            logger.info("\n\n*********** create  actor non ds scheduler ************")
            lr_scheduler = self.create_lr_scheduler(
                optimizer,
                name=self.lr_scheduler_type,
                warmup_steps=warmup_steps,
                num_training_steps=self.total_num_training_steps,
            )
        else:
            raise("OOPS, you better give a scheduler")

        self._patch_ds_config_for_optimizer(ds_config)
        self._patch_ds_config_for_batch_size(ds_config, self.global_batch_size)
        self._patch_ds_config_for_dtype(ds_config)
        self._patch_ds_config_for_bucket_size(ds_config, actor_model.config)

        print_deepspeed_config(ds_config.config)
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

    def init_actor_engine_if_needed(
        self,
        global_batch_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        total_num_training_steps: int,
        actor_model_fn: Optional[Callable[[], PreTrainedModel]] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Ensures self.actor (and self._actor_engine) is initialized if not already.
        If 'force_reload' is True, or if no engine is cached, re-initialize.
        """
        
        # Set these from the trainer to the policy before engine start
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_num_training_steps = total_num_training_steps
        logger.info("\n\n**************** Initilizing the actor ******************")
        logger.info(f"global_batch_size = {global_batch_size}")
        logger.info(f"total num training steps = {total_num_training_steps}")

        # Decide whether to skip if we already have a cached engine
        if (
            (not force_reload)
            and hasattr(self, "_actor_engine")
            and self._actor_engine is not None
        ):
            return  # already loaded

        # If a new callable was passed, update. Otherwise use existing
        if actor_model_fn is not None:
            self.actor_model_fn = actor_model_fn

        self._init_actor_model(
            actor_model_fn=self.actor_model_fn, only_return_unwrapped_model=False
        )
        logger.info("Actor engine init done.")

    def get_actor_model(self):
        return self.actor_model_fn()

    def get_actor_config(self):
        return self.actor_config

    def forward_actor(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        trainer_hparams: Dict,
        return_mean_entropy: bool = False,
        return_logits: bool = True,
        return_sequence_logp: bool = False,
        return_all_logp: bool = False,
        sequence_logp_reduction: Optional[Literal["mean"]] = None,
        
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
            use_cache=False,
        )

        logits = outputs.logits.float()
        logits /= trainer_hparams.temperature

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

    def actor_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: Optional[torch.FloatTensor],
        advantages: torch.FloatTensor,
        trainer_hparams: Dict[str, Any], #TODO : change this to the right type
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
            trainer_hparams=trainer_hparams
        )

        logprobs = outputs.all_logp  # shape: (batch_size, seq_len-1)
        assert logprobs.shape == old_logprobs.shape
        assert action_mask.shape == logprobs.shape

        # Compute the PPO-clip loss
        log_ratio = (logprobs - old_logprobs) * action_mask
        ratio = torch.exp(log_ratio)

        pg_losses1 = -advantages * ratio
        pg_losses1_anomalies = monitor_tensor_anomalies(
            pg_losses1.detach(), action_mask
        )

        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - trainer_hparams.cliprange, 1.0 + trainer_hparams.cliprange 
        )
        pg_losses = torch.max(pg_losses1, pg_losses2)
        pg_loss = masked_mean(pg_losses, action_mask)

        # Here we want the control variate kl penalty loss type
        kl_clt = 0.0001
        logger.info(
            f"\n\n********************************  pg loss ***************************"
        )
        logger.info(f"pg_loss: {pg_loss}")

        if trainer_hparams.kl_penalty_loss_type is not None:
            # _compute_kl_penalty is below
            ref_kl_tensor = self._compute_kl_penalty(
                logprobs, ref_logprobs, estimation_type=trainer_hparams.kl_penalty_loss_type
            )
            # clamp for numerical stability
            ref_kl_tensor = torch.clamp(
                ref_kl_tensor * action_mask,
                min=trainer_hparams.kl_penalty_loss_clip_min,
                max=trainer_hparams.kl_penalty_loss_clip_max,
            )
            ref_kl_loss = kl_clt * ref_kl_tensor.sum(dim=1).mean()
            pg_loss = pg_loss + ref_kl_loss
            ref_kl = ref_kl_tensor.detach()
        else:
            ref_kl = None
            ref_kl_loss =None 

        # Ratio check
        is_skipped = False
        avg_ratio = masked_mean(ratio, action_mask)
        if avg_ratio.item() > trainer_hparams.ratio_threshold:
            logger.warning(
                f"High ratio detected: {avg_ratio.item():.2f}. Skipping this batch."
            )
            pg_loss = pg_loss * 0.0
            is_skipped = True

        pg_clip_frac = masked_mean((pg_losses2 > pg_losses1).float(), action_mask)
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

    def destroy_actor_engine_if_not_cached(self) -> None:
        """
        Destroys the actor engine to free memory if engine caching is disabled.
        """
        if not self.cache_ds_engines:
            if getattr(self, "actor", None) is not None:
                logger.info("Destroying actor engine to free memory.")
                self._destroy_ds_engine(self.actor)
                # If the engine provides a shutdown/cleanup method, call it.
                if hasattr(self.actor, "shutdown") and callable(self.actor.shutdown):
                    logger.info("Shutting down actor engine.")
                    try:
                        self.actor.shutdown()
                    except Exception as e:
                        logger.warning(f"Error during actor engine shutdown: {e}")
                # If there is an internal engine attribute, clear it.
                if hasattr(self.actor, "_engine"):
                    logger.info("Clearing actor engine internal state.")
                    del self.actor._engine
                # Remove the actor engine reference.
                self.actor = None

            # Also clear the cached actor engine if it exists.
            if hasattr(self, "_actor_engine"):
                self._actor_engine = None

    def cache_initial_actor_state(self) -> None:
        """
        Cache a snapshot of the actor model's state dict.
        This snapshot should be called once (after the actor engine is initialized
        but before training begins) to ensure that the reference model is built
        from a fixed checkpoint.
        """
        if hasattr(self, "actor") and self.actor is not None:
            if isinstance(self.actor, DeepSpeedEngine):
                self.actor_snapshot_state_dict = self.actor.module.state_dict()
            else:
                self.actor_snapshot_state_dict = self.actor.state_dict()
            logger.info("Cached initial actor state for reference model.")
        else:
            logger.warning("Actor engine not initialized; cannot cache initial state.")

    def _init_reference_model(
        self,
        reference_model_fn: Callable[[], PreTrainedModel],
        only_return_unwrapped_model: bool = False,
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_reference_engine"):
            return self._reference_engine

        logger.info("Creating the reference deepspeed engine...")
        ref_model: PreTrainedModel = reference_model_fn()
        
        if self.gradient_checkpointing:
            ref_model.gradient_checkpointing_enable()

        ref_model.eval()
        for param in ref_model.parameters():
            param.required_grad = False

        # use the same settints as the actor for now
        ds_config = HfTrainerDeepSpeedConfig(self.actor_config)
        self._patch_ds_config_for_batch_size(ds_config, self.global_batch_size)
        self._patch_ds_config_for_dtype(ds_config)
        self._patch_ds_config_for_bucket_size(ds_config, ref_model.config)

        import json
        import copy 
        ref_config = copy.deepcopy(ds_config)
        if 'optimizer' in  ref_config.config:
            del ref_config.config["optimizer"]

        engine = self._init_deepspeed_engine_for_inference(
            ref_model,
            deepspeed_config=ref_config.config,
        )

        engine.eval()

        if self.cache_ds_engines:
            self._reference_engine = engine

        self.reference = engine

    def init_reference_engine_if_needed(
        self,
        global_batch_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        total_num_training_steps: int,
        force_reload: bool = False,
    ) -> None:
        """
        Ensures that the reference engine is initialized if enabled.
        """
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_num_training_steps = total_num_training_steps

        if not self.enable_reference:
            return
        if (not force_reload) and (self.reference is not None):
            return
        self._init_reference_model(reference_model_fn=self.reference_model_fn)
        logger.info("Reference engine init done.")

    def forward_reference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_mean_entropy: bool = False,
        return_logits: bool = True,
        return_sequence_logp: bool = False,
        return_all_logp: bool = False,
        sequence_logp_reduction: Optional[Literal["mean"]] = None,
    ) -> ActorForwardOutput:
        """
        Forward pass of the reference model.

        This method is similar to forward_actor but runs the reference model in evaluation mode
        (with torch.no_grad()) and omits temperature scaling.
        """
        if self.enable_reference and self.reference is None:
            self.init_reference_engine_if_needed()

        with torch.no_grad():
            outputs = self.reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
            )

            logits = outputs.logits.float()
            # Note: Skipping temperature scaling for the frozen reference model.

            # Shift logits and labels in the same way as the actor forward
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_label_mask = (shift_labels != -100).to(shift_logits.dtype)
            shift_labels[shift_labels == -100] = 0

            log_probs = shift_logits.log_softmax(-1)
            per_token_log_probs = torch.gather(
                log_probs, dim=2, index=shift_labels.unsqueeze(2)
            )
            per_token_log_probs = per_token_log_probs.squeeze(2)
            per_token_log_probs = per_token_log_probs * shift_label_mask

            output = {}
            if return_logits:
                output["logits"] = logits
            if return_sequence_logp:
                sequence_log_probs = per_token_log_probs.sum(dim=-1)
                if sequence_logp_reduction == "mean":
                    sequence_log_probs = sequence_log_probs / shift_label_mask.sum(
                        dim=-1
                    )
                output["sequence_logp"] = sequence_log_probs
            if return_all_logp:
                output["all_logp"] = per_token_log_probs

            return ActorForwardOutput(**output)

    def destroy_reference_engine_if_not_cached(self) -> None:
        """
        Destroys the reference engine to free memory if engine caching is disabled.
        """
        if not self.cache_ds_engines:
            if getattr(self, "reference", None) is not None:
                logger.info("Destroying reference engine to free memory.")
                self.reference.optimizer = None
                self._destroy_ds_engine(self.reference)
                # Call a shutdown or cleanup method if the engine provides one.
                if hasattr(self.reference, "shutdown") and callable(
                    self.reference.shutdown
                ):
                    logger.info("Shutting down reference engine.")
                    try:
                        self.reference.shutdown()
                    except Exception as e:
                        logger.warning(f"Error during reference engine shutdown: {e}")
                # Clear any internal engine references.
                if hasattr(self.reference, "_engine"):
                    logger.info("Clearing reference engine internal state.")
                    del self.reference._engine
                # Remove the reference engine.
                self.reference = None

            if hasattr(self, "_reference_engine"):
                self._reference_engine = None

    def _compute_kl_penalty(
        self,
        logprob: Union[torch.FloatTensor, np.ndarray],
        ref_logprob: Union[torch.FloatTensor, np.ndarray],
        estimation_type: Optional[str] = None,
    ) -> Union[torch.FloatTensor, np.ndarray]:
        """
        Compute the per-token KL penalty between the log probabilities of the actor and the reference model.

        Args:
            logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the actor, shape (`batch_size`, T)
            ref_logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the reference model, shape (`batch_size`, T)

        Returns:
            `Union[torch.FloatTensor, np.ndarray]`: KL penalty, shape (`batch_size`, `T`)
        """

        if estimation_type is None:
            estimation_type = self.trainer_hparams.kl_penalty

        if estimation_type == "kl":
            return logprob - ref_logprob

        if estimation_type == "abs":
            return (logprob - ref_logprob).abs()

        if estimation_type == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if estimation_type == "control_variate":
            # Compute the per-token approximate KL penalty between the log probabilities of the actor
            # and the reference model as suggested by Schulman in http://joschu.net/blog/kl-approx.html
            #
            # D_KL [π_θ || π_ref] =
            #    π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t) - log(π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t)) - 1
            #

            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                kl = torch.exp(log_ratio) - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                kl = np.exp(log_ratio) - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "seq_control_variate":
            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                prob_ratio = torch.exp(log_ratio.sum(dim=-1, keepdim=True))
                kl = prob_ratio - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                prob_ratio = np.exp(log_ratio.sum(axis=-1, keepdims=True))
                kl = prob_ratio - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(
                ref_logprob, logprob, log_target=True, reduction="none"
            ).sum(-1)

        raise NotImplementedError

    def get_last_checkpoint(self, return_resumable_only: bool = False):
        checkpoints = list(self.checkpoints_dir.iterdir())
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.is_dir() and checkpoint.name.startswith("ckpt--")
        ]
        if return_resumable_only:
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if self.is_checkpoint_resumable(checkpoint)
            ]
        if len(checkpoints) == 0:
            return None

        checkpoints = sorted(
            checkpoints, key=lambda x: self.parse_checkpoint_name(x.name)
        )
        last_checkpoint = checkpoints[-1]
        last_checkpoint_iteration = self.parse_checkpoint_name(last_checkpoint.name)[0]

        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        # grad_acc_kwargs["sync_with_dataloader"] = False
    

        
    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps
            # if self.warmup_steps > 0
            # else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps
