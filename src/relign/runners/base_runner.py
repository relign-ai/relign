import os

from typing import Type, Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path  # Corrected import

from relign.algorithms.train_loop import TrainLoop 
from relign.policies.base_policy import BasePolicy 
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator
from relign.algorithms.base_trainer import BaseTrainer

class BaseRunner(ABC):
    def __init__(
        self,
        experiment_name: str,
        directory: str,
        algorithm_cls: Type[TrainLoop],
        policy_cls: Type[BasePolicy],
        trainer_cls: Type[BaseTrainer],
        episode_generator_cls: Type[BaseEpisodeGenerator],
        policy_kwargs: Dict[str, Any],
        trainer_kwargs: Dict[str, Any],
        episode_generator_kwargs: Dict[str, Any],
        algorithm_kwargs: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        # Initialize the seed
        if seed is None:
            import time
            import hashlib

            seed_str = f"{time.time()}_{os.getpid()}"
            # Ensure the seed is within the valid range for NumPy
            self.seed = int(
                hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16
            ) % (2**32 - 1)
        else:
            self.seed = seed
        self._init_random_seed()

        # classes
        # TODO: change naming of algorithm here to train loop to avoid confusion
        self.algorithm_cls: Type[TrainLoop] = algorithm_cls
        self.policy_cls: Type[BasePolicy] = policy_cls
        self.trainer_cls: Type[BaseTrainer] = trainer_cls
        self.episode_generator_cls: Type[BaseEpisodeGenerator] = episode_generator_cls

        # Keyword arguments
        self.policy_kwargs: Dict[str, Any] = policy_kwargs
        self.trainer_kwargs: Dict[str, Any] = trainer_kwargs
        self.episode_generator_kwargs: Dict[str, Any] = episode_generator_kwargs
        self.algorithm_kwargs: Dict[str, Any] = algorithm_kwargs

        self.experiment_name = experiment_name
        self.directory = directory

        self.exp_root = self._init_experiment_dir()
        self.log_dir = self._init_log_dir()

    def _init_experiment_dir(self) -> Path:
        """
        Initialize a directory at directory/experiment_name.
        Returns the experiment root path.
        """
        # Create the experiment root directory
        exp_root = Path(self.directory) / self.experiment_name
        exp_root.mkdir(parents=True, exist_ok=True)
        return exp_root

    def _init_log_dir(self) -> Path:
        """
        Creates a log directory in the experiment root directory.
        """
        # Create the log directory inside the experiment root
        log_dir = self.exp_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _init_random_seed(self):
        import random
        import numpy as np
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @abstractmethod
    def _init_policy(self):
        pass

    @abstractmethod
    def _init_trainer(self):
        pass

    @abstractmethod
    def _init_episode_generator(self):
        pass

    @abstractmethod
    def _init_algorithm(self):
        pass

    def run(self):
        """
        Run the algorithm
        """
        # Run the learn method of the algorithm
        self.algorithm.learn()


