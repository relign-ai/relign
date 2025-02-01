# conftest.py
import pytest
from transformers import AutoModelForCausalLM, AutoModel

# Import all the modules/classes you need here
from relign.policies.base_policy import ActorPolicy
from relign.policies.actor_critic_policy import ActorCriticPolicy
from relign.policies.base_critic import PretrainedModelValueHead
from relign.algorithms.ppo.trainer import PPOTrainer
from relign.algorithms.base_trainer import BaseTrainer
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator
from relign.episode_generators.grouped_episode_generator import GroupedEpisodeGenerator
from relign.inference.cot_inference_strategy import COTInferenceStrategy
from relign.inference.tree_inference.expansion import EfficientIIDExpander
from relign.guidance.llms._mock import Mock


@pytest.fixture
def actor_model_fn():
    """
    Returns an actor model function for testing, e.g. GPT-2.
    """
    def _actor_model_fn():
        return AutoModelForCausalLM.from_pretrained("gpt-2")
    return _actor_model_fn



@pytest.fixture
def critic_model_fn():
    """
    Returns a critic model function. Could be the same base as actor or something else.
    """
    def _critic_model_fn():
        critic_backbone = AutoModel.from_pretrained("gpt-2")
        return PretrainedModelValueHead(pretrained_model=critic_backbone)
    return _critic_model_fn



@pytest.fixture
def actor_critic_policy(actor_model_fn, critic_model_fn):
    """
    Creates a standard Actor-Critic policy for tests (used by PPO, for example).
    """
    return ActorCriticPolicy(
        actor_model_fn=actor_model_fn,
        critic_model_fn=critic_model_fn
    )


@pytest.fixture
def actor_policy(actor_model_fn):
    return ActorPolicy(
        actor_model_fn=actor_model_fn
    )


@pytest.fixture
def base_episode_generator():
    """
    A simple base episode generator for unit or integration tests.
    """
    return BaseEpisodeGenerator(
        # fill in arguments if required
    )


@pytest.fixture
def grouped_episode_generator():
    """
    A specialized grouped episode generator for testing grouped sampling logic.
    """
    return GroupedEpisodeGenerator(
        # fill in arguments for grouped episodes
    )


@pytest.fixture
def ppo_trainer() -> BaseTrainer:
    """
    Returns a mock or real PPO trainer for integration tests.
    """
    return PPOTrainer(
        per_device_batch_size=4,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


@pytest.fixture
def cot_inference_strategy():
    """
    A fixture for chain-of-thought inference strategy (if needed).
    """
    return COTInferenceStrategy(
        guidance_llm=Mock(),
        node_expander=EfficientIIDExpander(
            # fill in program, branch strategies, etc.
        ),
        max_depth=2,
        result_dir="test_output"
    )