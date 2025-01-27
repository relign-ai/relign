from tqdm import tqdm
from datasets import Dataset

from relign.common.dataset import EpisodeDataset
from relign.utils.logging import get_logger
from relign.algorithms.base_algorithm import BaseAlgorithm
from relign.policies.base_policy import BasePolicy
from relign.episode_generators.on_policy_episode_generator import(
    OnPolicyEpisodeGenerator,
)
from relign.algorithms.base_trainer import OnPolicyTrainer
from relign.utils.dataset import remove_null_columns

logger = get_logger(__name__)

class OnPolicyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        policy: BasePolicy,  # Reinforce, Actor Critic, Policy Gradient...(so base for now)
        trainer: OnPolicyTrainer,
        episode_generator: OnPolicyEpisodeGenerator,
        verbose: int = 0,
        num_iterations: int = 100,
        num_episodes_per_iteration: int = 100,
        evaluation_freq: int = 10,
        checkpoint_freq: int = 10,
        **kwargs,
    ):
        """
        On-Policy Algorithm Base Class.

        :param policy: The policy to use (must be a BaseActorCritic instance)
        :param trainer: The trainer to use.
        :param episode_generator: Generates episodes to train on.
        :param verbose: The verbosity level.
        :param num_iterations: Total number of iterations to train for.
        :param num_episodes_per_iteration: Number of episodes per training iteration.
        :param evaluation_freq: Frequency of evaluation.
        :param checkpoint_freq: Frequency of checkpointing.
        :param **kwargs: Additional algorithm-specific arguments.
        """
        super().__init__(
            policy=policy,
            trainer=trainer,
            episode_generator=episode_generator,
            verbose=verbose,
            num_iterations=num_iterations,
            num_episodes_per_iteration=num_episodes_per_iteration,
            evaluation_freq=evaluation_freq,
            checkpoint_freq=checkpoint_freq,
            **kwargs,
        )

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

            self.trainer.step(episodes=episodes)

            # Onpolicy paramater update for next iteration
            # self.policy.set_params(self.trainer.policy_train_state.params)

        # Evalutate
        if iteration % self.evaluation_freq == 0:
            self._evaluate()

        # Checkpoint
        if iteration % self.checkpoint_freq == 0:
            self._checkpoint()

    def _generate_episodes(
        self,
        iteration: int,
        current_policy_path: str,
        # TODO allow_from_cache: bool = True,
    ) -> EpisodeDataset:
        """
        Generate episodes under the current policy and save them to disk.
        Params:
            iteration_id:
            current_policy_path: path to the weights of the current policy.
            #TODO allow_from_cache: bool = True,
        """
        # Generate epiosdes data set
        # TODO: is this the cleanest way? do we pass a path, or do we simply pass the policy module? idk?
        # do we run into trouble when we use distributed learning?
        # or distributed generation in the  episode generator?
        self.episode_generator.set_policy_path(current_policy_path)

        # TODO: handle distributed environments differently
        # for now we just generate it in the main process

        # Feth the epiode path on all the devices
        episode_path = self.episode_generator.get_episode_checkpoint_path(iteration)

        # compute the epiosdes on the main process in non-distirbuted envirnoments
        if not self.episode_generator.supports_distributed:
            if self.distributed_state.is_main_process:
                episode_path = self.episode_generator.generate(
                    self.num_episodes_per_iteration, iteration, return_path=True
                )
        else:
            episodes = self.episode_generator.generate(iteration=iteration)
            assert isinstance(episodes, Dataset)
            if self.distributed_state.is_main_process:
                remove_null_columns(episodes).save_to_disk(episode_path)

        self.distributed_state.wait_for_everyone()
        episode_dataset = Dataset.load_from_disk(episode_path)
        return episode_dataset

    def _evaluate(self):
        ...

    def _checkpoint(self):
        ...

