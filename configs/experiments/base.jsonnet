{
  // Base experiment template that can be extended for specific experiments
  experiment_name: error "experiment_name must be specified",
  directory: "experiment",
  
  // Runner configuration
  type: "distributed_runner",
  use_deepspeed: true,
  
  // These should be overridden by specific experiment configs
  policy_cls: error "policy_cls must be specified",
  episode_generator_cls: error "episode_generator_cls must be specified",
  trainer_cls: error "trainer_cls must be specified",
  algorithm_cls: error "algorithm_cls must be specified",
}