import hydra
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from omegaconf import OmegaConf

from relign.tasks import GSM8K
from relign.policies.actor_critic_policy import ActorCriticPolicy
from relign.policies.base_critic import PretrainedModelValueHead
from relign.algorithms.train_loop import TrainLoop
from relign.algorithms.ppo.trainer import PPOTrainer
from relign.episode_generators.envs.math_episode_generator import (
    MathEpisodeGenerator,
    MATHRewardFunction,
)

from relign.inference.cot_inference_strategy import COTInferenceStrategy
from relign.inference.tree_inference.expansion import EfficientIIDExpander
from relign.inference.tree_inference.answer_extraction import IdentityAnswerExtractor

# For actor critic methods, we need a Distributed Runner
from relign.runners.distributed_runner import DistributedRunner
from relign.common.vllm_server import VLLMServer
from relign.guidance.llms import OpenAIVLLM 
from relign.inference.tree_inference.branch_factor_strategy import ListBranchFactor


def ppo_gsm(cfg, local_rank: int = -1):
    ds_config = cfg.deepspeed
    ds_config = OmegaConf.to_container(ds_config, resolve=True)
    experiment_name = "ppo-cot-rho1b-gsm"
    experiment_dir = "experiment"

    # --------- Tokenizer --------------- #
    tokenizer = AutoTokenizer.from_pretrained("realtreetune/rho-1b-sft-GSM8K")

    # ---------Model Defenition ---------#
    def actor_model_fn():
        ## load gp2 as actor
        return AutoModelForCausalLM.from_pretrained("realtreetune/rho-1b-sft-GSM8K")

    def critic_model_fn():
        # Wrap the critic with the value head model.
        critic_backbone = AutoModel.from_pretrained("realtreetune/rho-1b-sft-GSM8K")
        return PretrainedModelValueHead(
            pretrained_model=critic_backbone
        )  # critics need to be wrapped in a pretrained value head

    # --------- Task Definition ----------#
    task = GSM8K(
        answer_prefix=None,
        load_dataset_dict=True,
        dataset_dict_path="data/gsm8k",
        remove_calculator_expressions=True,
    )

    # ------- Task Reward Function --------#
    reward_function = MATHRewardFunction(
        tokenizer=tokenizer,
        math_task=task,
        penalize_unfinished_response=True,
        unfinished_response_penalty=0.0,
        timeout=1,
    )

    n_episodes_per_iteration = 50 
    n_rollouts_per_sample = 1
    max_concurrent_programs = 30 
    max_concurrent_generations = 40 

    guidance_llm_cls = OpenAIVLLM
    guidance_llm_kwargs = {
        "api_key": 'EMPTY',
        "max_calls_per_min": 1e6,
        "caching": False,
        "max_retries": 10,
    }

    # ---------- Node Expanders---------- #
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

    Solution:
    """

    # ---- Chain of thought Strategy --- #
    cot_inference_strategy_cls = COTInferenceStrategy 
    cot_inference_strategy_kwargs = {
        "samples": n_rollouts_per_sample,
        "question_field": "query",
        "question_template": question_template,
        "max_concurrent_generations": max_concurrent_generations,
        "max_concurrent_programs": max_concurrent_programs,
        "answer_extractor": answer_extractor,
        "node_expander": node_expander,
        "guidance_llm_cls": guidance_llm_cls,
        "guidance_llm_kwargs": guidance_llm_kwargs,
        "max_depth": 2,
        "result_dir": Path(experiment_dir) / "chain_of_thoughts",
    }

    # ----------- Episode Generator ------------#
    vllm_server = VLLMServer()
    episode_generator = MathEpisodeGenerator
    episode_generator_kwargs = {
        "tokenizer": tokenizer,
        "num_episodes_per_iteration": n_episodes_per_iteration,
        "reasoning_step_delimiter": "",
        "wait_until_memory_release": True,
        "answer_prefix": "\n\n # Answer\n",
        "max_sequence_length": 2048,
        "max_question_length": 1512,
        "reward_function": reward_function,
        "inference_strategy_cls": cot_inference_strategy_cls,
        "inference_strategy_kwargs": cot_inference_strategy_kwargs,
        "vllm_server": vllm_server,
        "task": task,
        "initial_model_name_or_path": "realtreetune/rho-1b-sft-GSM8K",
        "question_template": question_template,
    }

    # ----------- Policy ---------------#
    actor_critic_policy = ActorCriticPolicy
    actor_critic_kwargs = {
        "actor_model_fn": actor_model_fn,
        "critic_model_fn": critic_model_fn,
        "actor_config": ds_config,
        "critic_config": ds_config,
    }

    # ----------- Trainer ---------------#
    ppo_trainer_class = PPOTrainer
    ppo_trainer_kwargs = {
        "target_batch_size": 64, 
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": False,
    }

    # ----------- Algorithm--------------#
    algorithm_cls = TrainLoop
    algorithm_kwargs = {
        "num_iterations": 100,
        "num_episodes_per_iteration": n_episodes_per_iteration,
        "verbose": 1,
        "evaluation_freq": 10,
        "checkpoint_freq": 10,
    }

    # The main runner object
    runner = DistributedRunner(
        experiment_name=experiment_name,
        directory=experiment_dir,
        use_deepspeed=True,
        policy_cls=actor_critic_policy,
        trainer_cls=ppo_trainer_class,
        episode_generator_cls=episode_generator,
        algorithm_cls=algorithm_cls,
        policy_kwargs=actor_critic_kwargs,
        trainer_kwargs=ppo_trainer_kwargs,
        episode_generator_kwargs=episode_generator_kwargs,
        algorithm_kwargs=algorithm_kwargs,
    )

    # Start train run
    runner.run()


def main():
    parser = argparse.ArgumentParser(description="Deepspeed training")
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown = parser.parse_known_args()

    hydra.initialize(config_path="../configs", version_base=None)
    cfg = hydra.compose(config_name="config")
    ppo_gsm(cfg=cfg, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
