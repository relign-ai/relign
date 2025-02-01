import pytest

from relign.policies.base_policy import BasePolicy
from relign.algorithms.base_trainer import BaseTrainer 
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator 


class TestTrainers:
    def test_trainer_init(
        self,
        ppo_trainer: BaseTrainer,
        grpo_trainer: BaseTrainer
    ):
        assert ppo_trainer is not None
        assert grpo_trainer is not None

    @pytest.mark.parametrize(
        "trainer_fixture, policy_fixture", 
        # ['ppo_trainer', 'actor_critic_policy', 'episode_generator'],
        ['grpo_trainer', 'actor_policy', 'episode_generator']
    )
    def test_trainer_train(
        self,
        trainer_fixture: BaseTrainer,
        policy_fixture: BasePolicy,
        episode_generator: BaseEpisodeGenerator
    ):
        episodes = episode_generator.generate(5, 1)
        trainer_fixture.step(episodes)
