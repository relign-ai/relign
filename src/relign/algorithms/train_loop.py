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


class TrainLoop:
    def __init__(
        self,
        seed: int,
        project_root_dir: Path,
        policy: BasePolicy,
        trainer: BaseTrainer,
        episode_generator: BaseEpisodeGenerator,
        evaluator_cls: Evaluator,
        evaluator_kwargs: Dict,
        distributed_state: PartialState,
        verbose: int = 0,
        num_iterations: int = 100,
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
        self.project_root_dir = project_root_dir
        self.policy = policy
        self.trainer = trainer
        self.episode_generator = episode_generator
        self.evaluator_cls = evaluator_cls
        self.evaluator_kwargs = evaluator_kwargs
        self.verbose = verbose
        self.num_iterations = num_iterations
        self.evaluation_freq = evaluation_freq
        self.checkpoint_freq = checkpoint_freq
        self.distributed_state = distributed_state

        # Intitialize the evaluator:
        self.evaluator = self.evaluator_cls(
            **evaluator_kwargs,
            project_root_dir=project_root_dir,
            distributed_state=distributed_state,
            cloud_logger=None,
            seed=self.seed,
        )

    def learn(self):
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
                logger.info(f"Running iteration {iteration}")
                logger.info("*" * 80)

            ####################
            # Generate episodes#
            ####################
            logger.info(
                f"Rank {self.distributed_state.process_index}: About to _generate_episodes() for iteration {iteration}"
            )
            episodes = self._generate_episodes(
                iteration=iteration, current_policy_path=current_policy_path
            )
            logger.info(
                f"Rank {self.distributed_state.process_index} done with episode generation."
            )

            self.distributed_state.wait_for_everyone()
            dist.barrier()

            ##################
            # Trainer step   #
            ##################
            logger.info(f"Rank {self.distributed_state.process_index}: About to step.")
            current_policy_path = self.trainer.step(episodes=episodes)
            logger.info(
                f"Rank {self.distributed_state.process_index} done with trainer step."
            )

            self.distributed_state.wait_for_everyone()
            dist.barrier()

            #################
            #   Evaluate    #
            #################
            if iteration % self.evaluation_freq == 0:
                logger.info(
                    f"Evaluating current policy... on rank {self.distributed_state.process_index}"
                )
                self._evaluate(
                    iteration=iteration, current_policy_path=current_policy_path
                )

            self.distributed_state.wait_for_everyone()
            dist.barrier()

            #################
            #   Checkpoint  #
            #################
            logger.info(
                f"Rank {self.distributed_state.process_index} done with evaluation."
            )
            # Checkpointing (e.g. saving tokenizer) -- only on main process.
            logger.info(
                f"Rank {self.distributed_state.process_index} about to checkpoint."
            )
            if (
                current_policy_path is not None
                and self.episode_generator.tokenizer is not None
                and is_local_main_process
            ):
                logger.info(f"Saving the tokenizer at {current_policy_path}")
                self.episode_generator.tokenizer.save_pretrained(current_policy_path)

            # Synchronize after checkpointing.
            self.distributed_state.wait_for_everyone()
            dist.barrier()

            #################
            #  HouseCleaning #
            ##################
            self._clean_episodes()
            self.distributed_state.wait_for_everyone()
            dist.barrier()

            if is_local_main_process:
                logger.info(f"Iteration {iteration} complete")

    def _generate_episodes(
        self,
        iteration: int,
        current_policy_path: Union[str | None],
        # TODO allow_from_cache: bool = True,
    ) -> EpisodeDataset:
        """
        Generate episodes under the current policy and save them to disk.
        Params:
            iteration_id:
            current_policy_path: path to the weights of the current policy.
            #TODO allow_from_cache: bool = True,
        """
        # Wait for all processes to reach this point
        if self.distributed_state.use_distributed:
            self.distributed_state.wait_for_everyone()
            dist.barrier()

        # TODO: handle distributed environments differently
        # for now we just generate it in the main process
        # Feth the epiode path on all the devices
        episode_path = self.episode_generator.get_episode_checkpoint_path(iteration)

        # compute the epiosdes on the main process in non-distirbuted envirnoments
        if not self.episode_generator.supports_distributed:
            if self.distributed_state.is_main_process:
                episode_path = self.episode_generator.generate(
                    iteration=iteration, latest_policy_path=current_policy_path
                )
        else:
            logger.info(
                f"rank {self.distributed_state.process_index} entering generating gen."
            )
            episodes = self.episode_generator.generate(
                iteration=iteration, latest_policy_path=current_policy_path
            )
            assert isinstance(episodes, Dataset)
            if self.distributed_state.is_main_process:
                remove_null_columns(episodes).save_to_disk(str(episode_path))

        self.distributed_state.wait_for_everyone()
        dist.barrier()

        episode_dataset = Dataset.load_from_disk(str(episode_path))
        self.episode_generator.log_episodes(
            episode_dataset,
            iteration,
            num_examples=3,
            seed=self.seed,
            log_to_cloud=True,
        )

        return episode_dataset

    def _evaluate(self, iteration: int, current_policy_path: Path):
        self.evaluator.evaluate(
            iteration=iteration,
            latest_policy_path=current_policy_path,
        )

    def _checkpoint(self, iteration: int):
        logger.info("Checkpointing models...")
        self.trainer.policy.checkpoint(
            self.project_root_dir / "policy" / "checkpoint" / f"policy_{iteration}.pt"
        )
        # TODO: checkpoint other parts of the train state here

    def _clean_episodes(self) -> None:
        if self.distributed_state.is_main_process:
            keep_iterations = []  # TODO: add checkpoint iterations here
            keep_iterations += [0]  # Always keep the initial iteration
            keep_iterations = set(keep_iterations)

            # Remove unnecessary episodes insided experiment_root
            for episode in self.project_root_dir.glob("episodes/episodes_*"):
                if not episode.is_dir():
                    continue

                # episode_iter = int(episode.name.split("_")[1])
                # if episode_iter in keep_iterations:
                #     continue

                logger.info(
                    f"Removing exp_root/episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)

            # Remove unnecessary temp_episodes
            for episode in self.project_root_dir.glob("temp_episodes/iteration__*"):
                if not episode.is_dir():
                    continue

                episode_iter = int(episode.name.split("__")[1])
                if episode_iter in keep_iterations:
                    continue

                logger.info(
                    f"Removing temp episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)
        dist.barrier()
