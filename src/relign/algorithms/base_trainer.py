import logging
import json
from typing import Dict, Any, Union, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from accelerate import Accelerator, PartialState
from accelerate.utils import GradientAccumulationPlugin
from deepspeed import DeepSpeedEngine


from datasets import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import PartialState

from relign.policies.base_policy import BasePolicy
from relign.algorithms.ppo.data_collator import (
    COLUMN_REF_SHIFTED_LOGPS,
    COLUMN_ACTOR_SHIFTED_LOGPS,
    COLUMN_VALUES,
)
from relign.utils.logging import get_logger

logger = get_logger(__name__)


class TrainerState:
    """
    Keep track of the global trainer state.
    """

    global_step: int = 0
    epoch: float = 0.0
    iteration: int = 0

    INITIAL_STATE_DICT = {
        "global_step": 0,
        "epoch": 0.0,
        "iteration": 0,
    }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.global_step = state_dict["global_step"]
        self.epoch = state_dict["epoch"]
        self.iteration = state_dict["iteration"]

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "iteration": self.iteration,
        }

    def __repr__(self):
        return f"TrainerState(global_step={self.global_step}, epoch={self.epoch}, iteration={self.iteration})"


class BatchArgs:
    """
    A class to hold the batch arguments.
    """


class BaseTrainer(ABC):
    def __init__(
        self,
        seed: int,
        project_root_dir: Path,
        policy: BasePolicy,
        dataloader_num_workers: int,
        dataloader_pin_memory: bool,
        distributed_state: PartialState,
        gradient_accumulation_steps: int = 1,
        num_epochs_per_iteration: int = 8,
        num_iterations: int = 1,
        num_episodes_per_iteration: int = 1,
        max_seq_length: int = None,
        logging_steps: int = 1,
        per_device_batch_size: Optional[int] = None,
        target_batch_size: Optional[int] = None,
        cloud_log: Optional[Any] = None,
    ):
        """
        Main Trainer object.
        params:
        policy: BasePolicy,
            The policy to be trained.
        per_device_batch_size: int,
            The batch size per device for training.
        seed: int,
            Random seed for reproducibility.
        path: str = "./trainer_state.pth",
            Path to save and load the trainer state.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        dataloader_pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        """
        self.seed = seed
        self.project_root_dir = project_root_dir
        self._init_trainer_dir()
        self.policy = policy
        self.per_device_batch_size = per_device_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.distributed_state = distributed_state
        self.state = TrainerState()

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.num_iterations = num_iterations
        self.target_batch_size = target_batch_size
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.max_seq_length = max_seq_length

        self._compute_batch_size_and_steps()
        self.logging_steps = logging_steps
        self._cloud_log = cloud_log
        # self._create_accelerator_and_postprocess()
        self.deepspeed_plugin = None
        self.trainer_hparams = None  # gets overwritten in trainer implementations

    def _init_trainer_dir(self):
        self.trainer_dir = self.project_root_dir / "trainer"
        self.trainer_dir.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def step(self) -> Path:
        pass

    def save_trainer_state(self, path: str) -> None:
        raise NotImplementedError("save_trainer_state method is not implemented yet.")

    def load_trainer_state(self, path: str) -> None:
        raise NotImplementedError("load_trainer_state method is not implemented yet.")

    def _is_main_process(self):
        return self.distributed_state.is_main_process

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

    def _compute_batch_size_and_steps(self):
        # Log all values that will be used in the computation
        logger.info(f"Starting batch size and step computation with values:")
        logger.info(f"  target_batch_size: {self.target_batch_size}")
        logger.info(f"  per_device_batch_size (initial): {self.per_device_batch_size}")
        logger.info(
            f"  gradient_accumulation_steps (initial): {self.gradient_accumulation_steps}"
        )
        logger.info(f"  num_processes: {self.distributed_state.num_processes}")
        logger.info(f"  num_iterations: {self.num_iterations}")
        logger.info(f"  num_epochs_per_iteration: {self.num_epochs_per_iteration}")
        logger.info(f"  num_episodes_per_iteration: {self.num_episodes_per_iteration}")

        if self.target_batch_size is not None:
            if (
                self.per_device_batch_size is None
                and self.gradient_accumulation_steps is None
            ):
                raise ValueError(
                    "Either per_device_train_batch_size or gradient_accumulation_steps "
                    "should be provided."
                )
            if (
                self.per_device_batch_size is not None
                and self.gradient_accumulation_steps is not None
            ):
                raise ValueError(
                    "Only one of per_device_train_batch_size or gradient_accumulation_steps "
                    "should be provided."
                )

            if self.per_device_batch_size is not None:
                self.gradient_accumulation_steps = (
                    self.target_batch_size
                    // self.per_device_batch_size
                    // self.distributed_state.num_processes
                )
            elif self.gradient_accumulation_steps is not None:
                self.per_device_batch_size = (
                    self.target_batch_size
                    // self.gradient_accumulation_steps
                    // self.distributed_state.num_processes
                )

        self.global_batch_size = (
            self.per_device_batch_size
            * self.gradient_accumulation_steps
            * self.distributed_state.num_processes
        )
        self.total_num_training_steps = (
            self.num_iterations
            * self.num_epochs_per_iteration
            * self.num_episodes_per_iteration
            // self.global_batch_size
        )

        # Finally, log the resulting computations
        logger.info(f"Per device batch size (computed): {self.per_device_batch_size}")
        logger.info(
            f"Gradient accumulation steps (computed): {self.gradient_accumulation_steps}"
        )
        logger.info(
            f"Global batch size (w. parallel, distributed & accumulation): {self.global_batch_size}"
        )
        logger.info(
            f"Total number of training steps (Gradient Updates): {self.total_num_training_steps}"
        )

    def _rescale_and_clip_scores(self, episodes: Dataset) -> Dataset:
        """
        Simplify to just min-max scale the 'scores' column to [0,1].
        """
        if "scores" not in episodes.column_names:
            return episodes  # no-op if column doesn't exist
        
        # Extract all scores
        scores = np.array(episodes["scores"], dtype=np.float32)
        score_min, score_max = scores.min(), scores.max()
        if score_min == score_max:
            # All scores are the same, so set them all to 0 (or 1, your choice)
            logger.warning(
                "All scores are identical; min-max scaling will produce 0.0 for all."
            )
            return episodes.map(
                lambda x: {"scores": 0.0},
                num_proc=self.distributed_state.num_processes,
                desc="Rescaling (degenerate case: identical scores)",
            )

        def transform_reward(example: Dict[str, Any]) -> Dict[str, Any]:
            score = example["scores"]
            scaled_score = (score - score_min) / (score_max - score_min)
            # Ensure final range is strictly within [0,1]
            scaled_score = max(0.0, min(1.0, scaled_score))
            return {"scores": float(scaled_score)}

        # Apply the transform
        episodes = episodes.map(
            transform_reward,
            num_proc=self.distributed_state.num_processes,
            desc="Min-max scaling 'scores' to [0,1]",
        )
        return episodes

    def _log_episodes_metrics(self, episodes: Dataset) -> Optional[float]:
        """
        Log scores, advantages, logprobs, and values of the episodes.

        Args:
            episodes (Dataset): The episodes dataset.

        Returns:
            Optional[float]: The KL from reference policy
        """
        if len(episodes) == 0:
            return

        def compute_seq_logp(
            episode: Dict[str, Any], logprobs_w_query: List[float]
        ) -> float:
            query_len = len(episode["query_token_ids"])
            logprobs = logprobs_w_query[query_len - 1 :]
            seq_logprob = sum(logprobs)
            return seq_logprob

        scores = []
        response_lengths = []
        advantages = []
        ref_logprobs = []
        actor_logprobs = []
        critic_values = []
        kls = []
        control_variate_kls = []
        for e in episodes:
            scores.append(e["scores"])
            response_lengths.append(len(e["response_token_ids"]))
            if "advantages" in e:
                # Check if it's not a float before adding to the list
                if isinstance(e["advantages"], float):
                    pass
                else:
                    advantages += e["advantages"]
            if COLUMN_REF_SHIFTED_LOGPS in e:
                ref_logprobs.append(compute_seq_logp(e, e[COLUMN_REF_SHIFTED_LOGPS]))
            if COLUMN_ACTOR_SHIFTED_LOGPS in e:
                actor_logprobs.append(
                    compute_seq_logp(e, e[COLUMN_ACTOR_SHIFTED_LOGPS])
                )
            if COLUMN_REF_SHIFTED_LOGPS in e and COLUMN_ACTOR_SHIFTED_LOGPS in e:
                actor_lp = np.array(e[COLUMN_ACTOR_SHIFTED_LOGPS])
                ref_lp = np.array(e[COLUMN_REF_SHIFTED_LOGPS])
                kl = self._compute_kl_penalty(actor_lp, ref_lp).sum()
                kls.append(kl)

                # This is unbiased & low variance
                control_variate_kl = self._compute_kl_penalty(
                    actor_lp,
                    ref_lp,
                    estimation_type="control_variate",
                ).sum()
                control_variate_kls.append(control_variate_kl)

            if COLUMN_VALUES in e:
                values = e[COLUMN_VALUES]
                values_without_query = values[
                    len(e["query_token_ids"]) - 1 : -1
                ]  # Skip the last token (</s>)
                if len(values_without_query) == 0:
                    logger.warning(
                        f"Empty values for episode: {json.dumps(e, indent=2)}"
                    )
                critic_values += values_without_query

        scores = np.array(scores)

        logger.info(f"************ SCORES *********************")
        logger.info(f"logged scores {scores}")

        response_lengths = np.array(response_lengths)

        actor_logprobs = np.array(actor_logprobs)
        metrics = {
            "scores/mean": np.mean(scores),
            "scores/std": np.std(scores),
            "scores/dist": scores,
            "response_lengths/mean": np.mean(response_lengths),
            "response_lengths/std": np.std(response_lengths),
            "response_lengths/dist": response_lengths,
            "actor_logprobs/dist": actor_logprobs,
        }

        if len(actor_logprobs) > 0:
            metrics["actor_logprobs/mean"] = np.mean(actor_logprobs)
            metrics["actor_logprobs/normalized_by_response_len"] = np.mean(
                actor_logprobs / response_lengths
            )

        if len(kls) > 0:
            kls = np.array(kls)
            metrics["kls/mean"] = np.mean(kls)
            metrics["kls/dist"] = kls
            kls = float(metrics["kls/mean"])
        else:
            kls = None

        if len(control_variate_kls) > 0:
            control_variate_kls = np.array(control_variate_kls)
            metrics["kls/crtl_var__mean"] = np.mean(control_variate_kls)
            metrics["kls/crtl_var__dist"] = control_variate_kls

        if len(advantages) > 0:
            advantages = np.array(advantages)
            metrics["advantages/mean"] = np.mean(advantages)
            metrics["advantages/std"] = np.std(advantages)
            metrics["advantages/dist"] = advantages

        if len(ref_logprobs) > 0:
            ref_logprobs = np.array(ref_logprobs)
            metrics["ref_logprobs/sum"] = np.mean(ref_logprobs)
            metrics["ref_logprobs/normalized_by_response_len"] = np.mean(
                ref_logprobs / response_lengths
            )
            metrics["ref_logprobs/dist"] = ref_logprobs

        if len(critic_values) > 0:
            critic_values = np.array(critic_values)
            metrics["critic_values/mean"] = np.mean(critic_values)
            metrics["critic_values/std"] = np.std(critic_values)
            metrics["critic_values/dist"] = critic_values

        non_array_metrics = {
            k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
        }
        logger.info(f"Episode Metrics: {non_array_metrics}")

        logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
        self._cloud_log(
            {
                **logs,
                "train/global_step": self.state.global_step,
                "train/global_iteration": self.state.iteration,
            }
        )

        return kls

    def _get_learning_rate(self, engine: DeepSpeedEngine):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            return engine.get_lr()[0]
        except AssertionError as e:
            if "need to call step" in str(e):
                logger.warning(
                    "tried to get lr value before scheduler/optimizer started stepping, returning lr=0"
                )
                last_lr = 0
            else:
                raise

        return last_lr

    def _filter_episodes(self, episodes_dataset: Dataset) -> Dataset:
        """
        Filter out episodes that are too long.
        """
        if self.max_seq_length is not None:
            max_seq_len = self.max_seq_length
            orig_len = len(episodes_dataset)

            def filter_fn(example):
                return (
                    len(example["query_token_ids"]) + len(example["response_token_ids"])
                    <= max_seq_len
                )

            with self.distributed_state.main_process_first():
                episodes_dataset = episodes_dataset.filter(filter_fn, desc="Filtering")

            logger.error(
                f"Filtered out {orig_len - len(episodes_dataset)} episodes "
                f"that are too long. Remaining: {len(episodes_dataset)}"
            )
        return episodes_dataset

    def _set_process_log_level(self, logger_obj: logging.Logger):
        if not self.distributed_state.is_local_main_process:
            logger_obj.setLevel(logging.WARNING)

    def _get_automatic_checkpoint_name(self) -> str:
        checkpoint_format = self.get_checkpoint_format()
        checkpoint_name = checkpoint_format.format(
            iteration=str(self.state.iteration).zfill(4),
            epoch=f"{self.state.epoch:.2f}",
            global_step=str(self.state.global_step).zfill(4),
        )
        return checkpoint_name

    def get_checkpoint_format(self) -> str:
        return "ckpt--iter_{iteration}--epoch_{epoch}--step_{global_step}"

    def set_cloud_logger(self, cloud_log):
        self.cloud_log = cloud_log