import json

from datetime import timedelta

import hydra
from omegaconf import dictconfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoModel
from accelerate import PartialState

from policies.actor_critic_policy import ActorCriticPolicy
from policies.base_critic import PretrainedModelValueHead

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def test_actor_critic(cfg: dictconfig):

    ds_config = cfg.deepspeed
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    use_cpu = not torch.cuda.is_available()
    kwargs = {"timeout": timedelta(seconds=10000)}
    if not use_cpu:
        kwargs["backend"] = "nccl"
    distributed_state = PartialState(use_cpu, **kwargs)

    print("DS CONFIG", ds_config)

    # We want to do this more neatly
    def actor_model_fn():
        ## load gp2 as actor
        return AutoModelForCausalLM.from_pretrained("gpt2")

    def critic_model_fn():
        # Wrap the critic with the value head model.
        critic_backbone = AutoModel.from_pretrained("gpt2")
        return PretrainedModelValueHead(pretrained_model=critic_backbone)

    ac_policy = ActorCriticPolicy(
        distributed_state=distributed_state,
        actor_model_fn=actor_model_fn,
        critic_model_fn=critic_model_fn,
        actor_ds_config=ds_config,
        critic_ds_config=ds_config,
    )
    
    batch_size = 1
    sequence_length = 10
    vocab_size = 50257  # GPT-2 voc

    device = next(ac_policy.actor.parameters()).device

    # Generate random input_ids within the vocab size
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length)).to(device)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long).to(device)
    labels = input_ids.clone().to(device)

    # **Move inputs to the same device as the model**
    # Get the device from the actor model

    actor_output = ac_policy.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    critic_output = ac_policy.forward_critic(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    print("Actor output", actor_output)
    print("Critic output", critic_output)

if __name__ ==  "__main__":
    test_actor_critic()