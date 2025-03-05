import shutil
from typing import Union, Dict, List
from pathlib import Path
from tqdm import tqdm
from logging import Logger
from datasets import Dataset

from accelerate import PartialState
from relign.algorithms.base_trainer import BaseTrainer
from relign.policies.base_policy import BasePolicy
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator
from relign.common.dataset import EpisodeDataset
from relign.utils.dataset import remove_null_columns
from relign.eval.evaluator import Evaluator
from relign.eval.analyzer import TaskPerformanceAnalyzer
from relign.inference.inference_pipeline import InferencePipeline
from deepspeed import comm as dist
from relign.utils.logging import get_logger

logger = get_logger(__name__)


class TestLoop:
    def __init__(
        self,
        seed: int,
        project_root_dir: Path,
        episode_generator: BaseEpisodeGenerator,
        evaluator_cls: Evaluator,
        evaluator_kwargs: Dict,
        distributed_state: PartialState,
        model_path: Path ,
        verbose: int = 0,
        num_iterations: int = 100,
        cloud_logger = None,
        cloud_updater = None,
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
        self.project_root_dir = project_root_dir
        self.episode_generator = episode_generator
        self.evaluator_cls = evaluator_cls
        self.evaluator_kwargs = evaluator_kwargs
        self.verbose = verbose
        self.model_path = model_path
        self.num_iterations = num_iterations
        self.distributed_state = distributed_state
        self.cloud_logger = cloud_logger
        self.cloud_updater = cloud_updater

        # Intitialize the evaluator:
        logger.info(f"Initializing evaluator")
        self.evaluator = self.evaluator_cls(
            **evaluator_kwargs,
            project_root_dir=project_root_dir,
            distributed_state=distributed_state,
            cloud_logger=self.cloud_logger,
            cloud_updater=self.cloud_updater,
            seed=self.seed,
        )

    def test(self):
        """
        Main training loop. Trains the policy for 'num_iterations' rounds.
        Evaluates every 'evaluation_freq' rounds.
        Checkpoints every 'checkpoint_freq' rounds.
        """
        is_local_main_process = self.distributed_state.is_local_main_process
        current_policy_path = None

        for iteration in tqdm(range(self.num_iterations)):
            if is_local_main_process:
                logger.info("*" * 80)
                logger.info(f"Running test iteration {iteration}")

            #################
            #   Evaluate    #
            #################
            logger.info(
                f"Evaluating current policy... on rank {self.distributed_state.process_index}"
            )
            self._evaluate(
                iteration=iteration, 
                global_step=iteration,
                current_policy_path=self.model_path
            )

            self.distributed_state.wait_for_everyone()
            dist.barrier()

            if is_local_main_process:
                logger.info(f"Iteration {iteration} complete")

    def _evaluate(
        self, 
        iteration: int, 
        global_step:int ,
        current_policy_path: Path
    ):
        # Evaluate the current policy on the validation and test sets.
        self.evaluator.evaluate(
            iteration=iteration,
            global_step=global_step,
            latest_policy_path=current_policy_path,
        )
