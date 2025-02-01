from typing import Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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
        lam = 0.95,
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



