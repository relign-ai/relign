import pytest
from transformers import AutoModelForCausalLM, AutoModel

from relign.policies.base_policy import BasePolicy, ForwardOutput
from relign.policies.base_critic import BaseCritic, PretrainedModelValueHead
from relign.policies.actor_critic_policy import ActorCriticPolicy
from relign.episode_generators.base_episode_generator import DebugEpisodeGenerator 

@pytest.fixture
def debug_episode_generator():
    return DebugEpisodeGenerator(file_path="tests/mock_data/debug_data.json")

@pytest.fixture
def actor_model_fn():
    def _actor_model_fn():
        return AutoModelForCausalLM.from_pretrained("gpt-2")
    return _actor_model_fn

@pytest.fixture
def critic_model_fn():
    def _critic_model_fn():
        critic_backbone = AutoModel.from_pretrained("gpt-2")
        return PretrainedModelValueHead(pretrained_model=critic_backbone)
    return _critic_model_fn

@pytest.fixture
def actor_critic_policy(actor_model_fn, critic_model_fn):
    return ActorCriticPolicy(
        actor_model_fn=actor_model_fn, 
        critic_model_fn=critic_model_fn
    )

policies = [actor_critic_policy]

class TestPolicies:
    @pytest.mark.parametrize("policy", [actor_critic_policy])
    def test_policy_forward(
        self, 
        policy: BasePolicy, 
        debug_episode_generator: DebugEpisodeGenerator,
    ):
        episodes = debug_episode_generator.generate(5, 1) 
        results = policy.forward(episodes)

        assert isinstance(results, ForwardOutput)
        assert results.initial_poilicy_raw_output is not None
        assert results.policy_raw_output is not None
        assert results.values is not None 

        
    def test_policy_backward(self, pollicy: BasePolicy):
        pass