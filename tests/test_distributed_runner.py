import argparse
import hydra
from omegaconf import OmegaConf, dictconfig
from transformers import AutoModelForCausalLM, AutoModel

from relign.algorithms.train_loop import TrainLoop
from relign.policies.actor_critic_policy import ActorCriticPolicy
from relign.policies.base_critic import PretrainedModelValueHead
from relign.algorithms.ppo.trainer import PPOTrainer
from relign.episode_generators.base_episode_generator import DebugEpisodeGenerator
from relign.runners.distributed_runner import DistributedRunner


def test_drunner_ppo_policy(cfg, local_rank: int =-1):
    ds_config = cfg.deepspeed
    print(type(ds_config))
    ds_config = OmegaConf.to_container(ds_config, resolve=True)
    # Wrap this inside a model dir with actor models and critic models definitions?
    # Most likely a good idea. 
    # Define the models here 
    def actor_model_fn():
        ## load gp2 as actor
        return AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")


    def critic_model_fn():
        # Wrap the critic with the value head model.
        critic_backbone = AutoModel.from_pretrained("gpt2")
        return PretrainedModelValueHead(pretrained_model=critic_backbone).to("cuda")

    actor_critic_policy = ActorCriticPolicy
    actor_critic_kwargs = {
        "actor_model_fn": actor_model_fn,
        "critic_model_fn": critic_model_fn,
        "actor_config" : ds_config,
        "critic_config" : ds_config,
    }

    # Train algorithm ( a single update step, multiple epochs)
    ppo_trainer_class = PPOTrainer
    ppo_trainer_kwargs = {
        'per_device_batch_size': 10,
        'dataloader_num_workers': 1,
        'dataloader_pin_memory': False,
    }

    episode_generator_class = DebugEpisodeGenerator
    episode_generator_kwargs = {
        'file_path': "./tests/mock_data/gpt2_imdb_ppo_iter50_samples.json",
    }
    
    # Define an algorithm (takes a policy and a trainer)
    algorithm_cls = TrainLoop
    algorithm_kwargs = {
        "num_iterations": 2,
        "num_episodes_per_iteration":5,
        "verbose":1,
        "evaluation_freq":10,
        "checkpoint_freq":10  
    }
    
    # The main runner object
    runner = DistributedRunner(
        experiment_name="runner_test",
        directory="experiment",
        use_deepspeed=True,

        policy_cls=actor_critic_policy,
        trainer_cls=ppo_trainer_class,
        episode_generator_cls=episode_generator_class,
        algorithm_cls=algorithm_cls,

        policy_kwargs=actor_critic_kwargs,
        trainer_kwargs=ppo_trainer_kwargs,
        episode_generator_kwargs=episode_generator_kwargs,
        algorithm_kwargs=algorithm_kwargs
    )
    runner.run()


def main():
    parser = argparse.ArgumentParser(description='Deepspeed training')
    parser.add_argument('--local_rank', type=int, default=0)
    args, unknown = parser.parse_known_args()

    hydra.initialize(config_path="../configs", version_base=None)
    cfg = hydra.compose(config_name="config")
    test_drunner_ppo_policy(cfg=cfg, local_rank=args.local_rank)

if __name__ == "__main__":
    main()