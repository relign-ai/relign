from abc import ABC, abstractmethod
from pathlib import Path
from logging import Logger

from accelerate import PartialState

from policies.base_policy import BasePolicy 
from episode_generators.base_episode_generator import BaseEpisodeGenerator
from algorithms.base_trainer import BaseTrainer

class BaseAlgorithm(ABC):
    def __init__(
            self, 
            seed: int,
            project_root_dir: Path,
            policy: BasePolicy, 
            trainer: BaseTrainer,
            episode_generator: BaseEpisodeGenerator, 
            distributed_state: PartialState,
            verbose: int = 0,
            num_iterations: int = 100,
            num_episodes_per_iteration: int = 100,
            evaluation_freq: int = 10,
            checkpoint_freq: int = 10,
    ):
        """
            Base of Reinforcement Learning Algorithms.

            :param policy: The policy to use (PPO, ARCHER, etc)
            :param trainer: The trainer to use. ( Here we write the update rule and such)
            :param episode_generator: Returns episodes to train on. 

            :param device: The device to use.
            :param verbose: The verbosity level.
        """
        self.seed = seed
        self.project_root_dir = project_root_dir,

        self.policy = policy
        self.trainer = trainer
        self.episode_generator = episode_generator
        self.verbose = verbose
        self.num_iterations = num_iterations
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.evaluation_freq = evaluation_freq
        self.checkpoint_freq = checkpoint_freq
        self.distributed_state = distributed_state
    
    @abstractmethod
    def learn(self, *args, **kwargs) -> None:
        raise NotImplementedError("learn method is not implemented yet.")

    @property
    def logger(self) -> Logger:
        raise NotImplementedError("logger method is not implemented yet.")
