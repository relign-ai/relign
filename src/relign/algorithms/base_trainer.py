import json
import os
import shutil
from typing import Dict, Any, Union, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from accelerate import Accelerator, PartialState
from accelerate.utils import GradientAccumulationPlugin


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
        gamma: int = 1,
        lam=0.95,
        logging_steps: int = 5,
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

        self._compute_batch_size_and_steps()

        self.gamma = gamma
        self.lam = lam
        self.logging_steps = logging_steps
        self._cloud_log = cloud_log
        # self._create_accelerator_and_postprocess()
        self.deepspeed_plugin = None
        self.trainer_hparams = None  # gets overwritten in trainer implementations

    def _init_trainer_dir(self):
        self.trainer_dir = self.project_root_dir / "trainer"
        self.trainer_dir.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def step(self) -> None:
        pass

    def save_trainer_state(self, path: str) -> None:
        raise NotImplementedError("save_trainer_state method is not implemented yet.")

    def load_trainer_state(self, path: str) -> None:
        raise NotImplementedError("load_trainer_state method is not implemented yet.")

    def _is_main_process(self):
        return self.distributed_state.is_main_process

    def _is_kl_penalty_enabled(
        self,
        kl_penalty_loss_type: Optional[str] = None,
        policy_reference: Optional[BasePolicy] = None,
    ) -> bool:
        """
        Check if the KL penalty is enabled.
        """
        return not kl_penalty_loss_type is not None and policy_reference is not None

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
        logger.info(f"Per device batch size: {self.per_device_batch_size}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Num of total processes: {self.distributed_state.num_processes}")
        logger.info(
            f"Global batch size (w. parallel, distributed & accumulation): {self.global_batch_size}"
        )
        logger.info(
            f"Total number of training steps (Gradient Updates): {self.total_num_training_steps}"
        )

    def _rescale_and_clip_scores(self, episodes: Dataset) -> Dataset:
        bias_correction = None
        scale_factor = None
        if self.trainer_hparams.use_score_scaling:
            assert "scores" in episodes.column_names, "Scores should be provided."
            scores = torch.tensor(episodes["scores"], dtype=torch.float32)
            scores_mean, scores_std = self.running_scores.update(scores)
            scale_factor = scores_std + torch.finfo(scores.dtype).eps
            if self.trainer_hparams.use_score_norm:  # todo: weird name, right?
                bias_correction = -scores_mean

        clip = self.trainer_hparams.score_clip

        def transform_reward(example: Dict[str, Any]) -> Dict[str, Any]:
            score = example["scores"]
            if bias_correction is not None:
                score = score + bias_correction
            if scale_factor is not None:
                score = score / scale_factor

            if clip is not None:
                score = torch.clip(torch.tensor(score).float(), -clip, clip)

            return {
                "scores": (
                    score.item() if isinstance(score, torch.Tensor) else float(score)
                )
            }

        if "scores" in episodes.column_names and any(
            val is not None for val in [bias_correction, scale_factor, clip]
        ):
            episodes = episodes.map(
                transform_reward,
                num_proc=self.distributed_state.num_processes,
                desc="Rescaling and clipping scores (if needed)",
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
        response_lengths = np.array(response_lengths)
        actor_logprobs = np.array(actor_logprobs)
        metrics = {
            "scores/mean": np.mean(scores),
            "scores/std": np.std(scores),
            "scores/dist": scores,
            "response_lengths/mean": np.mean(response_lengths),
            "response_lengths/std": np.std(response_lengths),
            "response_lengths/dist": response_lengths,
            "actor_logprobs/sum": np.mean(actor_logprobs),
            "actor_logprobs/normalized_by_response_len": np.mean(
                actor_logprobs / response_lengths
            ),
            "actor_logprobs/dist": actor_logprobs,
        }

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
