// PPO experiment for GSM8K with Rho-1B model
local base = import '../experiments/base.jsonnet';
local deepspeed = import '../deepspeed/zero_2.jsonnet';
local model = import '../models/rho_1b.jsonnet';
local gsm8k = import '../tasks/gsm8k.jsonnet';
local math_reward = import '../reward_functions/math_reward.jsonnet';
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
  
  // Models and tokenizers
  tokenizer: model.tokenizer,
  
  // Policy configuration
  policy_cls: actor_critic,
  policy_kwargs: {
    actor_model_fn: model.actor,
    critic_model_fn: model.critic,
    reference_model_fn: model.reference,
    actor_config: deepspeed,
    critic_config: deepspeed,
  },
  
  // Task and reward
  task: gsm8k + {
    answer_prefix: "\n####",
  },

  reward_function: math_reward,
  
  // Inference configuration
  inference_strategy: cot_strategy + {
    question_template: prompts.standard,
    node_expander: efficient_iid,
  },
  
  // Episode generator
  episode_generator_cls: episode_generator,
  episode_generator_kwargs: episode_generator + {
    question_template: prompts.standard,
  },
  
  // Trainer config
  trainer_cls: ppo_trainer,
  trainer_kwargs: ppo_trainer,
  
  // Algorithm config
  algorithm_cls: train_loop,
  algorithm_kwargs: train_loop + {
    evaluator: evaluator,
  },
}