import os
from pathlib import Path
from typing import Type, Generic, Dict, Any, TypeVar
from datetime import timedelta

import torch

from relign.runners.base_runner import BaseRunner
from relign.policies.base_policy import BasePolicy
from relign.algorithms.base_trainer import BaseTrainer
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator
from relign.algorithms.base_algorithm import BaseAlgorithm
from relign.policies.actor_critic_policy import DeepSpeedPolicy


# Define TypeVars for Generics
P = TypeVar("P", bound=BasePolicy)
Pds = TypeVar("Pds", bound=DeepSpeedPolicy)
T = TypeVar("T", bound=BaseTrainer)
E = TypeVar("E", bound=BaseEpisodeGenerator)
A = TypeVar("A", bound=BaseAlgorithm)

# TODO: Sharpen the typing here.
# The policy should be a deepspeedpolicy,
# The episode gnerator does not nessecerrely have to be distributed
# The is no distinguishment in Distirbuted/Undistirbued algorithms so this is good.


class DistributedRunner(BaseRunner, Generic[Pds, T, E, A]):
    """
    Base runner but with a distirbuted state.
    """

    def __init__(
        self,
        experiment_name: str,  # These should be going to the base class
        directory: Path,  # This should be going to the base class
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

        # Init PTEA
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
            **self.policy_kwargs,
        )

    def _init_trainer(self):
        self.trainer: T = self.trainer_cls(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            policy=self.policy,
            **self.trainer_kwargs,
        )

    def _init_episode_generator(self):
        self.episode_generator: E = self.episode_generator_cls(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            **self.episode_generator_kwargs,
        )

    def _init_algorithm(self):
        self.algorithm: A = self.algorithm_cls(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            policy=self.policy,
            trainer=self.trainer,
            episode_generator=self.episode_generator,
            **self.algorithm_kwargs,
        )
