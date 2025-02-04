import pytest
from transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer

from relign.policies.base_critic import PretrainedModelValueHead
from relign.policies.base_actor  import ActorPolicy
from relign.policies.actor_critic_policy import ActorCriticPolicy


# --- Tokenizer Fixture ---
# Parameterizes between a dummy and a different tokenizer
@pytest.fixture
def mock_tokenizer():
    ...


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def actor_model_fn():
    """
    Returns an actor model function for testing, e.g. GPT-2.
    """
    def _actor_model_fn():
        return AutoModelForCausalLM.from_pretrained("gpt2")
    return _actor_model_fn

@pytest.fixture
def critic_model_fn():
    """
    Returns a critic model function. Could be the same base as actor or something else.
    """
    def _critic_model_fn():
        critic_backbone = AutoModel.from_pretrained("gpt2")
        return PretrainedModelValueHead(pretrained_model=critic_backbone)
    return _critic_model_fn

@pytest.fixture
def actor_critic_policy(
    actor_model_fn, 
    critic_model_fn, 
    deepspeed_config,
    distributed_single_gpu,
    experiment_dir
):
    """
    Creates a standard Actor-Critic policy for tests (used by PPO, for example).
    """
    return ActorCriticPolicy(
        actor_model_fn=actor_model_fn,
        critic_model_fn=critic_model_fn,
        actor_config=deepspeed_config,
        critic_config=deepspeed_config,
        distributed_single_gpu=distributed_single_gpu,
        project_root_dir=experiment_dir,
        seed=69
    )

@pytest.fixture
def actor_policy(
    actor_model_fn, 
    deepspeed_config,
    distributed_single_gpu,
    experiment_dir,
):
    return ActorPolicy(
        actor_model_fn=actor_model_fn,
        actor_config=deepspeed_config,
        distributed_state=distributed_single_gpu,
        project_root_dir=experiment_dir,
        seed=69
    )
