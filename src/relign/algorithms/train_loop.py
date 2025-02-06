from typing import Union
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

from relign.utils.logging import get_logger

logger = get_logger(__name__)

class TrainLoop():
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
        self.project_root_dir = (project_root_dir,)
        self.policy = policy
        self.trainer = trainer
        self.episode_generator = episode_generator
        self.verbose = verbose
        self.num_iterations = num_iterations
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.evaluation_freq = evaluation_freq
        self.checkpoint_freq = checkpoint_freq
        self.distributed_state = distributed_state

    def learn(self):
        """
        Main training loop. Trains the policy for 'num rounds' rounds.
        Evaluates every 'eval_freq' rounds.
        Checkpoints every 'checkpoint_freq' rounds.
        """
        current_policy_path = None
        for iteration in tqdm(range(self.num_iterations)):
            # Collect rollouts under the current policy.
            episodes = self._generate_episodes(
                iteration=iteration, 
                current_policy_path=current_policy_path
            )

            current_policy_path = self.trainer.step(episodes=episodes)

        # Evalutate
        if iteration % self.evaluation_freq == 0:
            self._evaluate()

        # Checkpoint
        if iteration % self.checkpoint_freq == 0:
            self._checkpoint()

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
        # TODO: handle distributed environments differently
        # for now we just generate it in the main process
        # Feth the epiode path on all the devices
        episode_path = self.episode_generator.get_episode_checkpoint_path(iteration)

        # compute the epiosdes on the main process in non-distirbuted envirnoments
        if not self.episode_generator.supports_distributed:
            if self.distributed_state.is_main_process:
                episode_path = self.episode_generator.generate(
                    iteration=iteration,
                    altest_policy_path=current_policy_path
                )
        else:
            episodes = self.episode_generator.generate(
                iteration=iteration, 
                latest_policy_path=current_policy_path 
            )
            assert isinstance(episodes, Dataset)
            if self.distributed_state.is_main_process:
                remove_null_columns(episodes).save_to_disk(episode_path)

        self.distributed_state.wait_for_everyone()
        episode_dataset = Dataset.load_from_disk(episode_path)
        return episode_dataset

    def _evaluate(self):
        ...

    def _checkpoint(self, iteration: int):
        logger.info("Checkpointing models...")
        self.trainer.policy.checkpoint(self.project_root_dir / "policy"/ "checkpoint"/ f"policy_{iteration}.pt")
        #TODO: checkpoint other parts of the train state here

    @property
    def logger(self) -> Logger:
        raise NotImplementedError("logger method is not implemented yet.")
