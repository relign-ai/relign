import os

from abc import ABC
from pathlib import Path  # Corrected import
from datetime import timedelta

import torch

from algorithms.base_algorithm import BaseAlgorithm
from policies.base_policy import BasePolicy
from episode_generation.base_episode_generator import BaseEpisodeGenerator
from algorithms.base_trainer import BaseTrainer

class BaseRunner(ABC):
    def __init__(
            self,
            experiment_name: str,
            directory: str,
            algorithm: BaseAlgorithm,
            policy: BasePolicy,
            trainer: BaseTrainer,
            episode_generator: BaseEpisodeGenerator,
    ):
        # the modules        
        self.algorithm = algorithm
        self.policy = policy
        self.trainer = trainer
        self.episode_generator = episode_generator

        self.experiment_name = experiment_name
        self.directory = directory  # Corrected spelling

        self.exp_root = self._initialize_experiment_directory()  # Corrected method name
        self.log_dir = self._initialize_log_directory()
        self._initialize_sub_dirs()

    
    def _initialize_experiment_directory(self) -> Path:
        """ 
        Initialize a directory at directory/experiment_name.
        Returns the experiment root path.
        """
        # Create the experiment root directory
        exp_root = Path(self.directory) / self.experiment_name
        exp_root.mkdir(parents=True, exist_ok=True)
        return exp_root


    def _initialize_log_directory(self) -> Path:
        """
        Creates a log directory in the experiment root directory.
        """
        # Create the log directory inside the experiment root
        log_dir = self.exp_root / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    

    def _initialize_sub_dirs(self):
        """
        Set the root directories and trigger the direcotry creation process. 
        """
        self.episode_generator.set_root_dir(self.exp_root)
        self.algorithm.set_root_dir(self.exp_root)
        self.trainer.set_root_dir(self.exp_root)
        self.policy.set_root_dir(self.exp_root)
    
    def run(self):
        """
            Run the algorithm
        """
        # Run the learn method of the algorithm
        self.algorithm.learn()




class DistributedRunner(BaseRunner):
    def __init__(
        self,
        experiment_name: str,
        directory: Path,
        use_deepspeed: bool,
        algorithm: BaseAlgorithm,
        policy: BasePolicy,
        trainer: BaseTrainer,
        episode_generator: BaseEpisodeGenerator
    ):
        super().__init__(BaseRunner, self)
        self.experiment_name = experiment_name
        self.directory = directory  # Corrected spelling
        self.use_deepspeed = use_deepspeed
        
        self.algorithm = algorithm
        self.policy = policy
        self.trainer = trainer
        self.episode_generator = episode_generator

        self._initialize_distributed_setup()
        self.exp_root = self._initialize_experiment_directory()  # Corrected method name
        self.log_dir = self._initialize_log_directory()
        self._initialize_sub_dirs()

    def _initialize_distributed_setup(self):
        from accelerate import PartialState

        use_cpu = not torch.cuda.is_available()
        ddp_timeout = 10000

        if self.use_deepspeed:
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(
                timeout=timedelta(seconds=ddp_timeout)
            )
            del os.environ["ACCELERATE_USE_DEEPSPEED"]
        else:
            kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
            if not use_cpu:
                kwargs["backend"] = "nccl"
            self.distributed_state = PartialState(use_cpu, **kwargs)

        # passdown the state to all the modules. 
        self.algorithm.set_deepspeed(self.distributed_state)
        self.policy.set_deepspeed(self.distributed_state)
        self.trainer.set_deepspeed(self.distributed_state)
        self.episode_generator.set_deepspeed(self.distributed_state)