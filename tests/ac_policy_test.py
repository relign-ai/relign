from datetime import timedelta

import torch
from transformers import AutoModelForCausalLM, AutoModel
from accelerate import PartialState

from policies.actor_critic_policy import ActorCriticPolicy
from policies.base_critic import PretrainedModelValueHead


def test_actor_critic():
    # distrited state
    

    use_cpu = not torch.cuda.is_available()
    kwargs = {"timeout": timedelta(seconds=10000)}
    if not use_cpu:
        kwargs["backend"] = "nccl"
    distributed_state = PartialState(use_cpu, **kwargs)

    ## load gp2 as actor
    actor = AutoModelForCausalLM.from_pretrained("gpt2")
    
    critic_backbone = AutoModel.from_pretrained("gpt2")
    critic = PretrainedModelValueHead(pretrained_model=critic_backbone)
    
    ## Critic model?
    ac_policy = ActorCriticPolicy(
        distributed_state=distributed_state,
        actor_model=actor,
        critic_model=critic,
        actor_ds_config=None,
        critic_ds_config=None,
    )

if __name__ ==  "__main__":
    test_actor_critic()