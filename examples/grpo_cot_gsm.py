import hydra
from textwrap import dedent
import argparse
from omegaconf import OmegaConf

from relign.tasks import GSM8K
from relign.policies.base_actor import ActorPolicy
from relign.algorithms.train_loop import TrainLoop
from relign.algorithms.grpo.trainer import GRPOTrainer
from relign.episode_generators.envs.math_episode_generator import (
    MathEpisodeGeneratorGroupedRewards,
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

from relign.models.base_model import DIPreTrainedTokenizer, PreTrainedModelForCasualLM


def grpo_gsm(cfg, local_rank=None):
    # ------ Deepspeed Config ------ #
    ds_config = cfg.deepspeed
    ds_config = OmegaConf.to_container(ds_config, resolve=True)
    initial_model_name = "realtreetune/rho-1b-sft-GSM8K"
    experiment_name = "grpo-cot-rho1b-gsm"
    experiment_dir = "experiment"

    # --------- Tokenizer --------------- #
    tokenizer = DIPreTrainedTokenizer.from_di(
        hf_model_name=initial_model_name,
    )

    # ---------Model Defenition ---------#
    def actor_model_fn():
        # Load the base actor model from pretrained weights.
        return PreTrainedModelForCasualLM.from_di(
            hf_model_name=initial_model_name,
            pretrained_args={
                "use_flash_attention_2": True,
            },
        )

    def reference_model_fn():
        # Load the base actor model from pretrained weights.
        return PreTrainedModelForCasualLM.from_di(
            hf_model_name=initial_model_name,
            disable_dropout=True,
            pretrained_args={
                "use_flash_attention_2": True,
            },
        )

    # --------- Task Definition ---------- #
    answer_prefix = "\n####"
    task = GSM8K(
        answer_prefix=answer_prefix,
        load_dataset_dict=True,
        dataset_dict_path="data/gsm8k",
        intermetdiate_step_tags=["<think>", "</think>"],
        remove_calculator_expressions=True,
        use_original_format=True,
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
    max_seq_length = 2048 
    num_episodes_per_iteration = 512 
    num_rollouts_per_sample = 16 # group size

    # num groups
    num_dataset_samples_per_iteration = (
        num_episodes_per_iteration / num_rollouts_per_sample
    )
    target_batch_size = 64 
    num_iterations = 650
    sampling_temperature = 0.9
    num_epoch_per_iterations = 2
    gradient_accumulation_steps = 1
    max_concurrent_programs = 512 
    max_concurrent_generations = 256 
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
    branch_factors = [{"depth": 0, "branch_factor": num_rollouts_per_sample}]
    node_expander = EfficientIIDExpander(
        branch_factor_strategy=ListBranchFactor(branch_factors=branch_factors),
        program=program,
        program_kwargs={
            "temperature": sampling_temperature,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stop": '"\n\n\nProblem:"',
        },
        node_text_template="{chain_of_thought}",
        tokenizer=tokenizer,
        model_context_size=2047,
    )

    question_template = dedent("""\
    [MATH_TASK] Problem:
    {query}
    first thinks about the reasoning process in and provides the the answer. The reasoning
    process and answer are enclosed within <think> </think> followed by an answer tag: \n####, respectively, i.e.,
    <think> reasoning process here </think> \n#### final answer 
    """)

    # ---- Chain of Thought Strategy Instance --- #
    cot_inference_strategy_cls = COTInferenceStrategy
    cot_inference_strategy_kwargs = {
        "samples": num_rollouts_per_sample,
        "question_field": "query",
        "question_template": question_template,
        "max_concurrent_generations": max_concurrent_generations,
        "max_concurrent_programs": max_concurrent_programs,
        "answer_extractor": answer_extractor,
        "node_expander": node_expander,
        "guidance_llm_cls": guidance_llm_cls,
        "guidance_llm_kwargs": guidance_llm_kwargs,
        "max_depth": 2,
    }

    # ----------- Episode Generator ------------ #
    vllm_server = VLLMServer
    episode_generator = MathEpisodeGeneratorGroupedRewards
    episode_generator_kwargs = {
        "tokenizer": tokenizer,
        "num_episodes_per_iteration": num_episodes_per_iteration,
        "dataset_num_samples_per_iteration": int(num_dataset_samples_per_iteration),
        "reasoning_step_delimiter": "",
        "answer_prefix": "\n\n# Answer\n",
        "append_bos_to_query": True,
        "append_eos_to_response": True,
        "dataset_shuffle_on_each_iteration": True,
        "dataset_sample_with_replacement": True,
        "max_sequence_length": max_seq_length,
        "max_question_length": 1512,
        "reward_function": reward_function,
        "fill_missing_episodes": True,
        "inference_strategy_cls": cot_inference_strategy_cls,
        "inference_strategy_kwargs": cot_inference_strategy_kwargs,
        "vllm_server_cls": vllm_server,
        "vllm_gpu_memory_utilization": "auto",
        "wait_until_memory_release": True,
        "task": task,
        "save_generations_every_n_iteration": 1,
        "initial_model_name_or_path": initial_model_name,
        "question_template": question_template,
    }

    # ----------- Policy --------------- #
    actor_policy = ActorPolicy
    actor_kwargs = {
        "actor_model_fn": actor_model_fn,
        "reference_model_fn": reference_model_fn,
        "actor_config": ds_config,
        "temperature": sampling_temperature,
        "warmup_steps": 0,
    }

    # ----------- Trainer --------------- #
    grpo_trainer_class = GRPOTrainer
    grpo_trainer_kwargs = {
        "target_batch_size": target_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs_per_iteration": num_epoch_per_iterations,
        "num_episodes_per_iteration": num_episodes_per_iteration,
        "num_iterations": num_iterations,
        "max_seq_length": max_seq_length,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False,
    }

    # ----------- Algorithm -------------- #
    from relign.eval.evaluator import Evaluator
    from relign.eval.analyzer import TaskPerformanceAnalyzer

    analysers_cls = [TaskPerformanceAnalyzer]
    analysers_kwargs = [{"task": task, "metrics_prefix": "task_performance"}]

    evaluator_cls = Evaluator
    evaluator_kwargs = {
        "task": task,
        "dataset_split": "validation",
        "tokenizer": tokenizer,
        "inference_strategy_cls": cot_inference_strategy_cls,
        "inference_strategy_kwargs": cot_inference_strategy_kwargs,
        "vllm_server_cls": vllm_server,
        "vllm_gpu_memory_utilization": "auto",
        "wait_until_memory_release": True,
        "force_rerun": False,
        "every_n_checkpoints": 1,
        "analysers_cls": analysers_cls,
        "analysers_kwargs": analysers_kwargs,
    }

    algorithm_cls = TrainLoop
    algorithm_kwargs = {
        "num_iterations": num_iterations,
        "verbose": 1,
        "evaluation_freq": 4,
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

    hydra.initialize(config_path="../configs", version_base=None)
    cfg = hydra.compose(config_name="config")
    grpo_gsm(cfg=cfg)


if __name__ == "__main__":
    main()
