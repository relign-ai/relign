import os

from typing import Type, Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path  # Corrected import

from relign.common.registry import RegistrableBase, Lazy
from relign.algorithms.train_loop import TrainLoop
from relign.policies.base_policy import BasePolicy
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator
from relign.algorithms.base_trainer import BaseTrainer
from relign.episode_generators.with_reward_function import BaseRewardFunction

from relign.utils.logging import get_logger

logger = get_logger(__name__)


class BaseRunner(ABC, RegistrableBase):
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        wandb_project: str,
        directory: str,

        algorithm_cls: Lazy[Type[TrainLoop]],
        policy_cls: Lazy[Type[BasePolicy]],
        trainer_cls: Lazy[Type[BaseTrainer]],
        episode_generator_cls: Lazy[Type[BaseEpisodeGenerator]],

        seed: Optional[int] = None,
        debug_mode: bool = False,
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

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.wandb_project = wandb_project 
        self.directory = directory

        self.exp_root = self._init_experiment_dir()
        self.log_dir = self._init_log_dir()
        self.debug_mode = debug_mode

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
    
    def test(self):
        """
        Run the algorithm in test mode
        """
        # Run the test method of the algorithm
        self.algorithm.test()

    def _create_cloud_logger(self):
        try:
            import wandb
        except ImportError:
            logger.warning(
                "Wandb is not installed. Please install it using `pip install wandb`"
            )
            return None

        if wandb.run is None:
            if self.debug_mode:
                mode = "disabled"
            else:
                mode = None

            settings = wandb.Settings()
            wandb.init(
                config={"seed": self.seed},
                project="relign-02",
                name=self.experiment_name,
                resume="allow",
                mode=mode,
                force=True,
            )

        return wandb.run


@BaseRunner.register("test_runner")
class TestIntegrationRunner(BaseRunner):
    """ 
    This is a simple tesst runner we can call in our integration test  
    """
    def __init__(
        self, 
        reward_function=Lazy[BaseRewardFunction], 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reward_function = reward_function
        self.run_called = False
    
    def _init_policy(self):
        # Minimal implementation for testing
        self.policy = None
        
    def _init_trainer(self):
        # Minimal implementation for testing
        self.trainer = None
        
    def _init_episode_generator(self):
        # For testing purposes, use the actual reward function if provided
        self.episode_generator = self.reward_function
        
    def _init_algorithm(self):
        # Minimal implementation for testing
        self.algorithm = None
        
    def run(self):
        self.run_called = True
        # Access the GSM8K task through the reward function to verify it worked
        if hasattr(self, 'episode_generator') and hasattr(self.episode_generator, 'math_task'):
            self.math_task = self.episode_generator.math_task
        return True
    
