from typing import Dict, Any, Union, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from accelerate import PartialState

from relign.policies.base_policy import BasePolicy


@dataclass
class Checkpoint:
    path: Path
    iteration: int


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


class BaseTrainer(ABC):
    def __init__(
        self,
        seed: int,
        project_root_dir: Path,
        policy: BasePolicy,
        per_device_batch_size: int,
        dataloader_num_workers: int,
        dataloader_pin_memory: bool,
        distributed_state: PartialState,
        gradient_accumulation_steps: int = 1,
        num_epochs_per_iteration: int = 1,
        num_iterations: int = 1,
        num_epochs: int = 1,
        gamma: int = 1,
        lam=0.95,
        logging_steps: int = 5,
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
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.lam = lam
        self.logging_steps = logging_steps

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
        trainer_hparams: Optional[Dict[str, Any]] = None,
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
            estimation_type = trainer_hparams.kl_penalty

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
