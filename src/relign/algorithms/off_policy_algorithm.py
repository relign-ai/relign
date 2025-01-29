from tqdm import tqdm

from common.buffer import ReplayBuffer

from episode_generation.base_episode_generator import BaseEpisodeGenerator
from algorithms.base_algorithm import BaseAlgorithm

from algorithms.base_trainer import OffPolicyTrainer
from policies.base_policy import BasePolicy

class OffPolicyAlgorithm(BaseAlgorithm):
    def __init__(
            self, 
            policy: BasePolicy,  # Reinforce, Actor Critic, Policy Gradient...(so base for now)
            trainer: OffPolicyTrainer, 
            episode_generator: BaseEpisodeGenerator, 
            verbose: int = 0,
            num_iterations: int = 100,
            num_episodes_per_iteration: int = 100,
            evaluation_freq: int = 10,
            checkpoint_freq: int = 10,
            **kwargs,
        ):
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
        replay_buffer = ReplayBuffer()
        for round in tqdm(range(self.num_iterations)):
            # TODO: Implement this
            self.trainer.step()