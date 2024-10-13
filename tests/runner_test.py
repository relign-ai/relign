from algorithms.on_policy_algorithm import OnPolicyAlgorithm

from policies.actor_critic_policy import ActorCriticPolicy
from algorithms.ppo.trainer import PPOTrainer
from episode_generation.base_episode_generator import DebugEpisodeGenerator

from runners.base_runner import BaseRunner

def test_ppo_policy_algo():
    # The policy
    actor_critic_policy = ActorCriticPolicy()


    # Train algorithm ( a single update step, multiple epochs)
    ppo_trainer = PPOTrainer(
        policy=actor_critic_policy,
        per_device_batch_size = 10,
        seed = 10,
    )

    # Episode generator
    episode_generator = DebugEpisodeGenerator(
        file_path="./test_data/gpt2_imdb_ppo_iter50_samples.json",
        policy_path="dummy_gpt2_path.th",
    )
    
    # Define an algorithm
    algo = OnPolicyAlgorithm(
        policy=actor_critic_policy,
        trainer=ppo_trainer,
        episode_generator=episode_generator,
        num_iterations=1,
        num_episodes_per_iteration=5,
        verbose=1,
        evaluation_freq=10,
        checkpoint_freq=10  
    )
    

    # The main runner object
    runner = BaseRunner(
        experiment_name="runner_test",
        directory="experiment",
        algorithm=algo,
        episode_generator=episode_generator,
        trainer=ppo_trainer,
        policy=actor_critic_policy,
    )

    # Start the algorithm
    runner.run()


if __name__ == "__main__":
    test_ppo_policy_algo()