// PPO experiment for GSM8K with Rho-1B model
local base = import '../experiments/base.jsonnet';
local deepspeed = import '../deepspeed/zero_2.jsonnet';
local cot_strategy = import '../inference/strategies/cot.jsonnet';
local efficient_iid = import '../inference/node_expanders/efficient_iid.jsonnet';
local episode_generator = import '../episode_generators/math.jsonnet';
local ppo_trainer = import '../trainers/ppo.jsonnet';
local actor_critic = import '../policies/actor_critic.jsonnet';
local train_loop = import '../train_loop/train_loop.jsonnet';
local evaluator = import '../evaluators/gsm8k.jsonnet';
local prompts = import '../prompts/gsm8k_standard.jsonnet';

base + {
  experiment_name: "relign-ppo-gsm8k",

  // Policy configuration
  policy_cls: actor_critic,
  
  // Episode generator
  episode_generator_cls: episode_generator,
  
  // Trainer config
  trainer_cls: ppo_trainer,
  
  // Algorithm config
  algorithm_cls: train_loop,
}