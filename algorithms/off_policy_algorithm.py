from tqdm import tqdm

from common.buffer import ReplayBuffer

from episode_generation.base_episode_generator import BaseEpisodeGenerator
from algorithms.base_algorithm import BaseAlgorithm

from algorithms.base_trainer import TrainerOffPolicy
from policy.base_policy import BasePolicy

class OffPolicyAlgorithm(BaseAlgorithm):
    def __init__(
            self, 
            policy: BasePolicy, 
            trainer: TrainerOffPolicy, 
            episode_generator: BaseEpisodeGenerator, 
        ):
        super().__init__(policy, episode_generator, trainer)

    def learn():
        replay_buffer = ReplayBuffer()
        for round in tqdm(range(self.num_rounds)):
            
            # Maybe do some evaluation here
            trajectories = collect_rollouts( # what should we name this function
                self.policy,
                self.environment,
                num_rollouts=num_rollouts,
                rollout_batch_size=rollout_batch_size,
            )

            replay_buffer.add(trajectories)