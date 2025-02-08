import hydra
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

from relign.tasks import GSM8K
from relign.policies.base_actor import ActorPolicy
from relign.algorithms.train_loop import TrainLoop
from relign.algorithms.grpo.trainer import GRPOTrainer
from relign.episode_generators.envs.math_episode_generator import (
    MathEpisodeGenerator,
    MATHRewardFunction,
)
from relign.inference.tree_inference.branch_factor_strategy import ListBranchFactor
from relign.inference.cot_inference_strategy import COTInferenceStrategy
from relign.inference.tree_inference.expansion import EfficientIIDExpander
from relign.inference.tree_inference.answer_extraction import (
    IdentityAnswerExtractor,
)

# For actor or actor-critic methods, we can use a Distributed Runner
from relign.runners.distributed_runner import DistributedRunner
from relign.common.vllm_server import VLLMServer
from relign.guidance.llms import OpenAIVLLM

from relign.inference.inference_pipeline import VLLMInferencePipeline
from relign.eval.evaluator import Evaluator
from relign.eval.analyzer import TaskPerformanceAnalyzer


def grpo_gsm(cfg, local_rank: int = -1):
    # ------ Deepspeed Config ------ #
    ds_config = cfg.deepspeed
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    experiment_name = "grpo-cot-rho1b-gsm"
    experiment_dir = "experiment"

    # --------- Tokenizer --------------- #
    tokenizer = AutoTokenizer.from_pretrained("realtreetune/rho-1b-sft-GSM8K")

    # --------- Model Definition --------- #
    def actor_model_fn():
        # GRPO uses a single actor model (no aux critic).
        return AutoModelForCausalLM.from_pretrained("realtreetune/rho-1b-sft-GSM8K")

    # --------- Task Definition ---------- #
    task = GSM8K(
        answer_prefix=None,
        load_dataset_dict=True,
        dataset_dict_path="data/gsm8k",
        intermetdiate_step_tags=["<think>", "</think>"],
        remove_calculator_expressions=True,
    )

    # ------- Task Reward Function -------- #
    reward_function = MATHRewardFunction(
        tokenizer=tokenizer,
        math_task=task,
        penalize_unfinished_response=True,
        unfinished_response_penalty=0.0,
        timeout=1,
    )

    # --------- Inference (Chain-of-thought) Strategy --------- #
    n_episodes_per_iteration = 50
    n_rollouts_per_sample = 10  # group size in GRPO
    max_concurrent_programs = 1
    max_concurrent_generations = 1

    guidance_llm_cls = OpenAIVLLM
    guidance_llm_kwargs = {
        "api_key": "EMPTY",
        "max_calls_per_min": 1e6,
        "caching": False,
        "max_retries": 10,
    }

    # ---------- Node Expansion ---------- #
    answer_extractor = IdentityAnswerExtractor(node_key_name="text")
    program = """{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}"""
    branch_factors = [{"depth": 0, "branch_factor": 2}]
    node_expander = EfficientIIDExpander(
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

    question_template = """
    [MATH_TASK] Problem:
    {query}

    answr format:
    <think>reasoning</rink>
    Solution:
    """

    # ---- Chain of Thought Strategy Instance --- #
    cot_inference_strategy = COTInferenceStrategy(
        samples=n_rollouts_per_sample,
        question_field="query",
        question_template=question_template,
        max_concurrent_generations=max_concurrent_generations,
        max_concurrent_programs=max_concurrent_programs,
        answer_extractor=answer_extractor,
        node_expander=node_expander,
        guidance_llm_cls=guidance_llm_cls,
        guidance_llm_kwargs=guidance_llm_kwargs,
        max_depth=2,
        result_dir=Path(experiment_dir) / "chain_of_thoughts",
    )

    # ----------- Episode Generator ------------ #
    vllm_server = VLLMServer()
    episode_generator = MathEpisodeGenerator
    episode_generator_kwargs = {
        "tokenizer": tokenizer,
        "num_episodes_per_iteration": n_episodes_per_iteration,
        "reasoning_step_delimiter": "",
        "answer_prefix": "\n\n # Answer\n",
        "max_sequence_length": 2048,
        "max_question_length": 1512,
        "reward_function": reward_function,
        "inference_strategy": cot_inference_strategy,
        "vllm_server": vllm_server,
        "initial_model_name_or_path": "realtreetune/rho-1b-sft-GSM8K",
        "question_template": question_template,
    }

    # ----------- Policy --------------- #
    actor_policy = ActorPolicy
    actor_kwargs = {
        "actor_model_fn": actor_model_fn,
        "actor_config": ds_config,
    }

    # ----------- Trainer --------------- #
    grpo_trainer_class = GRPOTrainer
    grpo_trainer_kwargs = {
        "target_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False,
    }

    # ----------- (Optional) Evaluation Pipeline --------------- #
    # If you'd like to evaluate similarly to the PPO script, add an inference pipeline + evaluator:
    inference_pipeline_kwargs = {
        "inference_strategy_cls": COTInferenceStrategy,
        "inference_strategy_kwargs": {
            "samples": 1,
            "max_depth": 2,
            "question_field": "query",
            "question_template": question_template,
            "answer_extractor": answer_extractor,
            "node_expander": node_expander,
            "guidance_llm_cls": guidance_llm_cls,
            "guidance_llm_kwargs": guidance_llm_kwargs,
            "max_concurrent_generations": 1,
            "max_concurrent_programs": 1,
        },
        "task": task,
        "dataset_split": "validation",
    }
    evaluator_cls = Evaluator
    evaluator_kwargs = {
        "task": task,
        "tokenizer": tokenizer,
        "inference_pipeline_cls": VLLMInferencePipeline,
        "inference_pipeline_kwargs": inference_pipeline_kwargs,
        "vllm_server": vllm_server,
        "vllm_gpu_memory_utilization": "auto",
        "force_rerun": False,
        "every_n_checkpoints": 1,
        "analyzers": [
            TaskPerformanceAnalyzer(
                task=task,
                metrics_prefix="task_performance",
            )
        ],
    }

    # ----------- Algorithm -------------- #
    num_iterations = 500
    algorithm_cls = TrainLoop
    algorithm_kwargs = {
        "num_iterations": num_iterations,
        "verbose": 1,
        "evaluation_freq": 10,
        "checkpoint_freq": 10,
        "evaluator_cls": evaluator_cls,
        "evaluator_kwargs": evaluator_kwargs,
    }

    # ----------- Runner -------------- #
    runner = DistributedRunner(
        experiment_name=experiment_name,
        directory=experiment_dir,
        use_deepspeed=True,
        policy_cls=actor_policy,
        trainer_cls=grpo_trainer_class,
        episode_generator_cls=episode_generator,
        algorithm_cls=algorithm_cls,
        policy_kwargs=actor_kwargs,
        trainer_kwargs=grpo_trainer_kwargs,
        episode_generator_kwargs=episode_generator_kwargs,
        algorithm_kwargs=algorithm_kwargs,
    )

    # Start training
    runner.run()


def main():
    parser = argparse.ArgumentParser(description="Deepspeed training")
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown = parser.parse_known_args()

    hydra.initialize(config_path="../configs", version_base=None)
    cfg = hydra.compose(config_name="config")
    grpo_gsm(cfg=cfg, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
