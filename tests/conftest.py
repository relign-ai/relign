# conftest.py
import pytest

from .fixtures.task_fixtures import gsm8k

from .fixtures.episode_generation_fixtures import (
    math_reward_function,
    episode_gen_config,
    math_episode_generator,
    math_grouped_episode_generator,
)

from .fixtures.inference_fixtures import (
    math_question_template,
    identity_answer_extractor,
    list_branch_factor,
    efficient_iid_node_expander,
    cot_inference_strategy,
    cot_inference_strategy_cls_kwargs,
    inference_pipeline,
)

from .fixtures.policies_fixtures import (
    tokenizer,
    actor_model_fn,
    critic_model_fn,
    actor_policy,
    actor_critic_policy,
)

from .fixtures.trainer_fixtures import ppo_trainer, grpo_trainer, train_loop

from .fixtures.common_fixtures import (
    experiment_dir,
    vllm_server,
    distributed_state_cpu_deepspeed,
    distributed_state_cpu_no_deepspeed,
    distributed_single_gpu,
    deepspeed_config,
)

from .fixtures.guidance_fixtures import mock_guidance, math_mock_with_think_tags


from .fixtures.eval_fixtures import evaluator, task_performance_analyzer


def pytest_addoption(parser):
    """
    Add a command line option for specifying the base output directory.
    """
    parser.addoption(
        "--output-dir",
        action="store",
        default="tests/output",
        help="Base directory for storing test outputs",
    )
