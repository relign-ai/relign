import os
from typing import Type, TypeVar, Generic, Dict, Any, Optional 
from abc import ABC, abstractmethod
from pathlib import Path  # Corrected import
from datetime import timedelta

import torch

from algorithms.base_algorithm import BaseAlgorithm
from policies.base_policy import BasePolicy, DeepSpeedPolicy
from episode_generators.base_episode_generator import BaseEpisodeGenerator
from algorithms.base_trainer import BaseTrainer

# Define TypeVars for Generics
P = TypeVar('P', bound=BasePolicy)
Pds = TypeVar('Pds', bound=DeepSpeedPolicy)
T = TypeVar('T', bound=BaseTrainer)
E = TypeVar('E', bound=BaseEpisodeGenerator)
A = TypeVar('A', bound=BaseAlgorithm)

class BaseRunner(ABC, Generic[P, T, E, A]):
    def __init__(
            self,
            experiment_name: str,
            directory: str,
            algorithm_cls: Type[A],
            policy_cls: Type[P],
            trainer_cls: Type[T],
            episode_generator_cls: Type[E],
            policy_kwargs: Dict[str, Any],
            trainer_kwargs: Dict[str, Any],
            episode_generator_kwargs: Dict[str, Any],
            algorithm_kwargs: Dict[str, Any],
            seed: Optional[int] = None
    ):
        # Initialize the seed 
        if seed is None:
            import time
            import hashlib
            seed_str = f"{time.time()}_{os.getpid()}"
            # Ensure the seed is within the valid range for NumPy
            self.seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
        else:
            self.seed = seed
        self._init_random_seed()

        # classes        
        self.algorithm_cls = algorithm_cls
        self.policy_cls = policy_cls
        self.trainer_cls = trainer_cls
        self.episode_generator_cls = episode_generator_cls

        # Keyword arguments
        self.policy_kwargs = policy_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.episode_generator_kwargs = episode_generator_kwargs
        self.algorithm_kwargs = algorithm_kwargs

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
        log_dir = self.exp_root / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _init_random_seed(self):
        import random
        import numpy as np
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        #if self.ditributed_state.num_processes > 0:
        #    torch.cuda.manual_seed_all(self.seed)

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


#TODO: Sharpen the typing here. 
# The policy should be a deepspeedpolicy, 
# The trainer has to be a ...
# The episode gnerator does not nessecerrely have to be distributed
# The is no distinguishment in Distirbuted/Undistirbued algorithms so this is good. 
class DistributedRunner(BaseRunner, Generic[Pds, T, E, A]):
    """
        Base runner but with a distirbuted state.
    """
    def __init__(
        self,
        experiment_name: str, #These should be going to the base class
        directory: Path, # This should be going to the base class
        use_deepspeed: bool,
        policy_cls: Type[Pds],
        trainer_cls: Type[T],
        episode_generator_cls: Type[E],
        algorithm_cls: Type[A],
        policy_kwargs: Dict[str, Any],
        trainer_kwargs: Dict[str, Any],
        episode_generator_kwargs: Dict[str, Any],
        algorithm_kwargs: Dict[str, Any],
    ):
        super().__init__(
            experiment_name=experiment_name,
            directory=directory,
            algorithm_cls=algorithm_cls,
            policy_cls=policy_cls,  # Type[Pds], acceptable as Pds is bound to DeepSpeedPolicy
            trainer_cls=trainer_cls,
            episode_generator_cls=episode_generator_cls,
            policy_kwargs=policy_kwargs,
            trainer_kwargs=trainer_kwargs,
            episode_generator_kwargs=episode_generator_kwargs,
            algorithm_kwargs=algorithm_kwargs,
        )

        self.use_deepspeed = use_deepspeed

        self._init_distributed_setup()
        self.exp_root = self._init_experiment_dir()  # Corrected method name
        self.log_dir = self._init_log_dir()
       
        #Init PTEA
        self._init_policy()
        self._init_trainer()
        self._init_episode_generator()
        self._init_algorithm()
    
    def _init_distributed_setup(self):
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

    def _init_policy(self):
        self.policy: Pds = self.policy_cls(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            **self.policy_kwargs
        )

    def _init_trainer(self):
        self.trainer : T = self.trainer_cls(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            policy=self.policy,
            **self.trainer_kwargs
        )

    def _init_episode_generator(self):
        self.episode_generator : E = self.episode_generator_cls(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            **self.episode_generator_kwargs
        )

    def _init_algorithm(self):
        self.algorithm : A = self.algorithm_cls(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            policy=self.policy,
            trainer=self.trainer,
            episode_generator=self.episode_generator,
            **self.algorithm_kwargs
        )
