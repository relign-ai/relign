{
  // Base experiment template that can be extended for specific experiments
  experiment_name: error "experiment_name must be specified",
  directory: "experiment",
  
  // Runner configuration
  type: "distributed_runner",
  use_deepspeed: true,
  
  // These should be overridden by specific experiment configs
  tokenizer: error "tokenizer must be specified",
  policy_cls: error "policy_cls must be specified",
  policy_kwargs: error "policy_kwargs must be specified",
  task: error "task must be specified",
  reward_function: error "reward_function must be specified",
  inference_strategy: error "inference_strategy must be specified",
  episode_generator_cls: error "episode_generator_cls must be specified",
  episode_generator_kwargs: error "episode_generator_kwargs must be specified",
  trainer_cls: error "trainer_cls must be specified",
  trainer_kwargs: error "trainer_kwargs must be specified",
  algorithm_cls: error "algorithm_cls must be specified",
  algorithm_kwargs: error "algorithm_kwargs must be specified",
}