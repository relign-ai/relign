# conftest.py
import pytest
from transformers import AutoModelForCausalLM, AutoModel
from pathlib import Path
from unittest.mock import Mock
from accelerate import PartialState

# Import all the modules/classes you need here
from relign.policies.actor_critic_policy import ActorCriticPolicy
from relign.policies.base_actor import ActorPolicy
from relign.policies.base_critic import PretrainedModelValueHead
from relign.algorithms.ppo.trainer import PPOTrainer
from relign.algorithms.base_trainer import BaseTrainer
from relign.episode_generators.base_episode_generator import BaseEpisodeGenerator
from relign.inference.cot_inference_strategy import COTInferenceStrategy
from relign.inference.tree_inference.expansion import EfficientIIDExpander
# from relign.guidance.llms._mock import Mock
from relign.tasks.gsm8k import GSM8K


from relign.episode_generators.envs.math_episode_generator import (
    MathEpisodeGenerator,
    MATHRewardFunction,
)

from relign.inference.tree_inference.branch_factor_strategy import ListBranchFactor
from relign.inference.tree_inference.answer_extraction import IdentityAnswerExtractor
from relign.common.vllm_server import VLLMServer


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
    # return GroupedEpisodeGenerator(
    #     # fill in arguments for grouped episodes
    # )
    ...


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
def answer_extractor():
    """
    Returns an answer extractor for testing.
    """
    return IdentityAnswerExtractor(node_key_name="text")



@pytest.fixture
def cot_inference_strategy(node_expander):
    """
    A fixture for chain-of-thought inference strategy (if needed).
    """
    return COTInferenceStrategy(
        guidance_llm=Mock(),
        node_expander=node_expander,
        max_depth=2,
        result_dir="test_output"
    )


# --- Tokenizer Fixture ---
# Parameterizes between a dummy and a different tokenizer
@pytest.fixture(params=["dummy_tokenizer", "different_tokenizer"], ids=["dummy", "different"])
def tokenizer(request):
    # You can replace these strings with actual tokenizer instances if needed.
    if request.param == "dummy_tokenizer":
        return "dummy_tokenizer"
    elif request.param == "different_tokenizer":
        return "different_tokenizer"
    else:
        raise ValueError("Unexpected tokenizer parameter")


# --- Core Fixtures ---
@pytest.fixture
def experiment_dir(tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


@pytest.fixture
def task(tokenizer):
    return GSM8K(
        answer_prefix=None,
        load_dataset_dict=True,
        dataset_dict_path="data/gsm8k",
        remove_calculator_expressions=True,
    )


@pytest.fixture
def reward_function(tokenizer, task):
    return MATHRewardFunction(
        tokenizer=tokenizer,
        math_task=task,
        penalize_unfinished_response=True,
        unfinished_response_penalty=0.0,
        timeout=1,
    )


@pytest.fixture
def node_expander(tokenizer):
    branch_factors = [{"depth": 0, "branch_factor": 2}]
    program = """{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}"""
    return EfficientIIDExpander(
        branch_factor_strategy=ListBranchFactor(branch_factors=branch_factors),
        program=program,
        program_kwargs={
            "temperature": 0.8,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stop": '"\n\n\nProblem:"',
        },
        node_text_template="{chain_of_thought}",
        tokenizer=tokenizer,
        model_context_size=2047,
    )


@pytest.fixture
def question_template():
    return """
    [MATH_TASK] Problem:
    {query}

    Solution:
    """


@pytest.fixture
def inference_strategy(answer_extractor, node_expander, question_template, experiment_dir):
    n_rollouts_per_sample = 1
    max_concurrent_generations = 1
    max_concurrent_programs = 1
    mock_guidance = Mock()
    return COTInferenceStrategy(
        samples=n_rollouts_per_sample,
        question_field='query',
        question_template=question_template,
        max_concurrent_generations=max_concurrent_generations,
        max_concurrent_programs=max_concurrent_programs,
        answer_extractor=answer_extractor,
        node_expander=node_expander,
        guidance_llm=mock_guidance,
        max_depth=4,
        result_dir=Path(experiment_dir) / "chain_of_thoughts",
    )


@pytest.fixture
def vllm_server():
    return VLLMServer()


# --- Parameterized Episode Generator Configurations ---
episode_gen_configs = [
    {
        "num_episodes_per_iteration": 1,
        "reasoning_step_delimiter": '',
        "wait_until_memory_release": True,
        "answer_prefix": "\n\n # Answer\n",
        "max_sequence_length": 2048,
        "max_question_length": 1512,
        "initial_model_name_or_path": "realtreetune/rho-1b-sft-GSM8K",
    },
    {
        "num_episodes_per_iteration": 2,
        "reasoning_step_delimiter": "###",
        "wait_until_memory_release": False,
        "answer_prefix": "\n\n## Answer:\n",
        "max_sequence_length": 1024,
        "max_question_length": 800,
        "initial_model_name_or_path": "realtreetune/another-model",
    },
]

@pytest.fixture(params=episode_gen_configs, ids=["config1", "config2"])
def episode_gen_config(request):
    return request.param


@pytest.fixture
def distributed_state_cpu():
    from datetime import timedelta
    ddp_timeout = 10000
    kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
    return PartialState(True, **kwargs)


# --- Centralized Episode Generator Fixture ---
@pytest.fixture
def episode_generator(
    tokenizer,
    task,
    reward_function,
    inference_strategy,
    vllm_server,
    question_template,
    episode_gen_config,
    distributed_state_cpu,
):
    return MathEpisodeGenerator(
        distributed_state=distributed_state_cpu,
        tokenizer=tokenizer,
        num_episodes_per_iteration=episode_gen_config["num_episodes_per_iteration"],
        reasoning_step_delimiter=episode_gen_config["reasoning_step_delimiter"],
        wait_until_memory_release=episode_gen_config["wait_until_memory_release"],
        answer_prefix=episode_gen_config["answer_prefix"],
        max_sequence_length=episode_gen_config["max_sequence_length"],
        max_question_length=episode_gen_config["max_question_length"],
        reward_function=reward_function,
        inference_strategy=inference_strategy,
        vllm_server=vllm_server,
        task=task,
        initial_model_name_or_path=episode_gen_config["initial_model_name_or_path"],
        question_template=question_template,
        seed=569,
        project_root_dir=Path("tests"),
    )