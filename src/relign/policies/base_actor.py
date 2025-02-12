from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, Optional, Literal, Callable, Union, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from deepspeed import DeepSpeedEngine
from deepspeed import comm as dist

from relign.policies.base_policy import DeepSpeedPolicy
from relign.policies.base_policy import ActorForwardOutput
from relign.utils.trainer import masked_mean, monitor_tensor_anomalies
from relign.utils.logging import get_logger

logger = get_logger(__name__)


class ActorPolicy(DeepSpeedPolicy):
    def __init__(
        self, actor_model_fn, actor_config, enable_reference: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        cache_dir = Path(self.project_root_dir) / "policy" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.actor_model_fn = actor_model_fn
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
        import json

        logger.info(
            "DS config after batch patching:\n%s",
            json.dumps(ds_config.config, indent=2),
        )

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
        logger.info("Initializing actor DeepSpeed engine...")
        # Set these from the trainer to the policy before engine start
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_num_training_steps = total_num_training_steps

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
        assert logprobs.shape == old_logprobs.shape
        assert action_mask.shape == logprobs.shape

        # Compute the PPO-clip loss
        log_ratio = (logprobs - old_logprobs) * action_mask
        ratio = torch.exp(log_ratio)

        pg_losses1 = -advantages * ratio
        pg_losses1_anomalies = monitor_tensor_anomalies(
            pg_losses1.detach(), action_mask
        )
        clip_range = 0.4
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        pg_losses = torch.max(pg_losses1, pg_losses2)
        pg_loss = masked_mean(pg_losses, action_mask)

        # Possibly apply a KL penalty if self.ppo_hparams.kl_penalty_loss_type is not None
        ref_kl_loss = None
        ref_kl = None
        kl_penalty_loss_type = "forward_kl"
        kl_penalty_loss_clip_max = 10
        kl_penalty_loss_clip_min = 0
        kl_clt = 0.05
        if kl_penalty_loss_type is not None:
            # _compute_kl_penalty is below
            ref_kl_tensor = self._compute_kl_penalty(
                logprobs, ref_logprobs, estimation_type=kl_penalty_loss_type
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

    def actor_loss_grpo(
        self,
        model_inputs: dict,
        shifted_labels_mask: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        trainer_hparams: dict,
        # optionally old_logprobs if needed for PPO, etc.
    ):
        """
        Re-run forward pass with gradients for your actor network, compute
        the RL objective (policy gradient + KL).
        """
        # 1) Forward pass on the actor model WITH grad
        actor_outputs = self.actor.forward(**model_inputs)
        logits = actor_outputs.logits  # shape (B, seq_len, vocab_size)

        # 2) SHIFT the logits so that it matches the shape of ref_logprobs, etc.
        #    Usually you skip the very last token. We'll do a minimal example:
        shifted_actor_logits = logits[:, :-1].contiguous()

        # 3) Convert logits -> logprobs for each real token
        shifted_actor_logprobs = self._logprobs_from_logits(
            shifted_actor_logits, model_inputs["labels"][:, 1:]
        )
        # shape (B, seq_len-1)

        # 4) Now compute the final objective
        #    a) Policy gradient part
        #    b) KL part vs. ref_logprobs

        # Example code for a typical step:
        # ratio = exp(actor_logp - ref_logp) -- for KL or advantage might differ
        # We'll do a minimal version:
        kl = (
            torch.exp(ref_logprobs - shifted_actor_logprobs)
            - (ref_logprobs - shifted_actor_logprobs)
            - 1
        )
        # shape: (B, seq_len-1)

        # This is basically: advantage * exp(log_pi - STOP_GRAD(log_pi)) => advantage * 1
        # but let's replicate the HF approach:
        adv_term = (
            torch.exp(shifted_actor_logprobs - shifted_actor_logprobs.detach())
            * advantages
        )

        # You can combine kl, or keep them separate:
        # e.g. total_loss = -( adv_term - self.beta * kl )
        # Mask out padding if needed:
        per_token_loss = -(adv_term - trainer_hparams.init_kl_coef * kl)
        per_token_loss = per_token_loss * shifted_labels_mask  # zero out masked tokens

        # Then average for final scalar
        tokens_per_sample = shifted_labels_mask.sum(dim=1)
        loss = (per_token_loss.sum(dim=1) / (tokens_per_sample + 1e-10)).mean()

        # Return
        actor_metrics = {
            "actor_loss": loss.detach().cpu().item(),
            "kl": (kl * shifted_labels_mask).sum().detach().cpu().item(),
        }
        approx_ref_kl = kl.mean().detach().cpu().item()  # an example quantity
        return loss, False, actor_metrics, approx_ref_kl

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
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        """
        Initializes a duplicate/frozen reference DeepSpeed engine.
        Instead of using the current state of the actor engine (which might have
        been updated during training), we load a cached snapshot of the actor model.
        """
        if self.reference is not None:
            return self.reference

        logger.info("Creating the reference deepspeed engine...")

        # Create a fresh model using the same callable.
        ref_model: PreTrainedModel = self.actor_model_fn()

        # Load the cached actor snapshot, if available, to freeze the initial state.
        if (
            hasattr(self, "actor_snapshot_state_dict")
            and self.actor_snapshot_state_dict is not None
        ):
            ref_model.load_state_dict(self.actor_snapshot_state_dict)
        elif hasattr(self, "actor") and self.actor is not None:
            if isinstance(self.actor, DeepSpeedEngine):
                ref_model.load_state_dict(self.actor.module.state_dict())
            else:
                ref_model.load_state_dict(self.actor.state_dict())

        # Freeze the model.
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        # Build the DeepSpeed engine for the reference model (no optimizer or scheduler needed)
        ds_config = HfTrainerDeepSpeedConfig(self.actor_config)
        self._patch_ds_config_for_batch_size(ds_config, self.global_batch_size)
        self._patch_ds_config_for_dtype(ds_config)
        self._patch_ds_config_for_bucket_size(ds_config, ref_model.config)

        import copy

        ref_config = copy.deepcopy(ds_config)
        if "optimizer" in ref_config.config:
            del ref_config.config["optimizer"]

        engine = self._init_deepspeed_engine_for_inference(
            ref_model,
            deepspeed_config=ref_config.config,
        )

        if self.cache_ds_engines:
            self._reference_engine = engine

        self.reference = engine
        return engine

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
        self._init_reference_model()
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

    def destroy_ds_engines(self) -> None:
        """
        Destroys all cached engines.
        """
        self.destroy_actor_engine_if_not_cached()
        self.destroy_reference_engine_if_not_cached()

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

    def _load_checkpoint_to_ds_engines(
        self,
        checkpoint_path: Path,
    ) -> None:
        metrics = {}
        if self.actor is not None:
            self.actor.load_checkpoint(str(checkpoint_path / "actor"))
        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

    def load_latest_policy_path(self, project_root_dir: Optional[Path] = None) -> None:
        """loads both actor and critic from "policy/cache" folder"""
        if not project_root_dir:
            project_root_dir = self.project_root_dir
        self._load_checkpoint_to_ds_engines(project_root_dir / "policy" / "cache")

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

        if self.actor is not None:
            self._save_hf_pretrained(self.actor, checkpoint_path / "hf_pretrained")
            self.actor.save_checkpoint(str(checkpoint_path / "actor"))

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

    def save_latest_policy_path(self, checkpoint_path: Path) -> Path:
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
        # Save the HF-pretrained actor weights
        dist.barrier()
        self._save_hf_pretrained(
            self.actor,
            actor_hf_pretrained_path,
        )
        # TODO: technically one could move this one down
        self.actor.save_checkpoint(str(checkpoint_path / "actor"))
        dist.barrier()

    def get_checkpoint_format(self) -> str:
        return "ckpt--iter_{iteration}--epoch_{epoch}--step_{global_step}"

    def _clean_old_temp_checkpoints(
        self, checkpoint_dir, exclude: Optional[List[Path]] = None
    ) -> None:
        if exclude is None:
            exclude = []

        if self._is_main_process():
            for checkpoint in checkpoint_dir.iterdir():
                if checkpoint.is_dir():
                    logger.info(f"Removing old temp checkpoint {checkpoint}")
                    shutil.rmtree(checkpoint)

        dist.barrier()

    def _logprobs_from_logits(self, logits, labels):
        """
        Gathers the log probability for each 'labels' token from 'logits'.
        """
        # logits: (B, seq_len, vocab_size)
        # labels: (B, seq_len)
        log_probs = F.log_softmax(logits, dim=-1)
        # gather log probs at each label
        token_logprob = torch.gather(
            log_probs, dim=2, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_logprob
