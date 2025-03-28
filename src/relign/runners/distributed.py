import os
from pathlib import Path
from typing import Type, Generic, Dict, Any, TypeVar
from datetime import timedelta

import torch
from deepspeed import comm as dist

from relign.common.registry import Lazy
from relign.runners.base_runner import BaseRunner
from relign.policies.base_policy import BasePolicy
from relign.algorithms.base_trainer import BaseTrainer
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator
from relign.algorithms.train_loop import TrainLoop
from relign.policies.base_actor import DeepSpeedPolicy

from relign.utils.logging import get_logger

logger = get_logger(__name__)


# Define TypeVars for Generics
P = TypeVar("P", bound=BasePolicy)
Pds = TypeVar("Pds", bound=DeepSpeedPolicy)
T = TypeVar("T", bound=BaseTrainer)
E = TypeVar("E", bound=BaseEpisodeGenerator)
A = TypeVar("A", bound=TrainLoop)

# TODO: Sharpen the typing here.
# The policy should be a deepspeedpolicy,
# The episode gnerator does not nessecerrely have to be distributed
# The is no distinguishment in Distirbuted/Undistirbued algorithms so this is good.

@BaseRunner.register("distributed_runner")
class DistributedRunner(BaseRunner, Generic[Pds, T, E, A]):
    """
    Base runner but with a distirbuted state.
    """
    def __init__(
        self,
        experiment_name: str,  # These should be going to the base class
        directory: Path,  # This should be going to the base class
        use_deepspeed: bool,
        run_name: str,
        wandb_project: str,
        policy_cls: Lazy[Type[Pds]],
        trainer_cls: Lazy[Type[T]],
        episode_generator_cls: Lazy[Type[E]],
        algorithm_cls: Lazy[Type[A]],
        cloud_logger=None,
        mode: str ="train",
    ):
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            directory=directory,
            wandb_project=wandb_project,
            algorithm_cls=algorithm_cls,
            policy_cls=policy_cls,  # Type[Pds], acceptable as Pds is bound to DeepSpeedPolicy
            trainer_cls=trainer_cls,
            episode_generator_cls=episode_generator_cls,
        )

        self.use_deepspeed = use_deepspeed

        logger.info(f"Initializing runner class")

        self._init_distributed_setup()
        self.exp_root = self._init_experiment_dir()  # Corrected method name
        self.log_dir = self._init_log_dir()
        self.mode = mode

        if self.distributed_state.is_main_process:
            cloud_logger = self._create_cloud_logger()
            if cloud_logger is not None:
                from wandb.sdk.wandb_run import Run

                self.cloud_logger: Run = self._create_cloud_logger()

        if dist.is_initialized():
            logger.info("Distributed is initialized")

        if self.distributed_state.use_distributed:
            self.distributed_state.wait_for_everyone()

        if self.mode == "train":
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
        self.policy: Pds = self.policy_cls.from_config(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
        )

    def _init_trainer(self):
        self.trainer: T = self.trainer_cls.from_config(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            policy=self.policy,
            cloud_log=self._cloud_log,
        )

    def _init_episode_generator(self):
        self.episode_generator: E = self.episode_generator_cls.from_config(
            seed=self.seed,
            project_root_dir=self.exp_root,
            distributed_state=self.distributed_state,
            cloud_log=self._cloud_log,
            cloud_save=self._cloud_save,
        )

    def _init_algorithm(self):
        if self.mode == "train":
            self.algorithm: A = self.algorithm_cls.from_config(
                seed=self.seed,
                project_root_dir=self.exp_root,
                distributed_state=self.distributed_state,
                policy=self.policy,
                trainer=self.trainer,
                episode_generator=self.episode_generator,
                cloud_logger=self._cloud_log,
                cloud_updater=self._cloud_update,
            )
        else: 
            self.algorithm: A = self.algorithm_cls.from_config(
                seed=self.seed,
                project_root_dir=self.exp_root,
                distributed_state=self.distributed_state,
                episode_generator=self.episode_generator,
                cloud_logger=self._cloud_log,
                cloud_updater=self._cloud_update,
            )


    # Cloud logging etcetera 
    def _cloud_log(self, *args, **kwargs):
        if self.distributed_state.is_main_process and self.cloud_logger is not None:
            self.cloud_logger.log(*args, **kwargs)

    def _cloud_save(self, *args, **kwargs):
        if self.distributed_state.is_main_process and self.cloud_logger is not None:
            self.cloud_logger.save(*args, **kwargs)

    def _cloud_update(self, *args, **kwargs):
        if self.distributed_state.is_main_process and self.cloud_logger is not None:
            self.cloud_logger.summary.update(*args, **kwargs)
