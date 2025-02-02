import pytest
from pathlib import Path

from relign.episode_generators.envs.math_episode_generator import (
    MathEpisodeGenerator,
    MATHRewardFunction,
)

@pytest.fixture
def math_reward_function(tokenizer, gsm8k):
    return MATHRewardFunction(
        tokenizer=tokenizer,
        math_task=gsm8k,
        penalize_unfinished_response=True,
        unfinished_response_penalty=0.0,
        timeout=1,
    )

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
def math_episode_generator(
    tokenizer,
    gsm8k,
    math_reward_function,
    cot_inference_strategy,
    vllm_server,
    episode_gen_config,
    distributed_single_gpu,
):
    question_template = """
        [MATH_TASK] Problem:
        {query}

        Solution:
    """

    return MathEpisodeGenerator(
        distributed_state=distributed_single_gpu,
        tokenizer=tokenizer,
        num_episodes_per_iteration=episode_gen_config["num_episodes_per_iteration"],
        reasoning_step_delimiter=episode_gen_config["reasoning_step_delimiter"],
        wait_until_memory_release=episode_gen_config["wait_until_memory_release"],
        answer_prefix=episode_gen_config["answer_prefix"],
        max_sequence_length=episode_gen_config["max_sequence_length"],
        max_question_length=episode_gen_config["max_question_length"],
        reward_function=math_reward_function,
        inference_strategy=cot_inference_strategy,
        vllm_server=vllm_server,
        task=gsm8k,
        initial_model_name_or_path=episode_gen_config["initial_model_name_or_path"],
        question_template=question_template,
        seed=569,
        project_root_dir=Path("tests/output/math_episode_gen"),
    )


@pytest.fixture
def grouped_episode_generator():
    """
    A specialized grouped episode generator for testing grouped sampling logic.
    """
    ...