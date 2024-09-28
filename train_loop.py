import time
import os
from tqdm import tqdm
import torch
import accelerate

from environment.base import Environment


from agent.base import Agent


def train_loop(
    agent: Agent,
    agent_type: str,
    env: Environment,
    accelerator: accelerate.Accelerator,
    tokenizer: AutoTokenizer,
    iterations: int,
    eval_freq: int,
    save_path: str,
    critic_lr: float,
    lm_lr: float,
    gamma: float,
    tau: float,
    epochs: int,
    actor_epochs: int,
    grad_accum_steps: int,
    max_grad_norm: float,
):

    trainer = Trainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=tokenizer,
        critic_lr=critic_lr,
        lm_lr=lm_lr,
        gamma=gamma,
        tau=tau,
        epochs=epochs,
        actor_epochs=actor_epochs,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
    )

    replay_buffer = ReplayBuffer()
    all_trajectories = []

    for i in tqdm(range(iterations)):
        if accelerator.is_main_process:
            trajectories = None  # rollout trajectories

            # do eval every eval_freq
            all_trajectories += trajectories

            print(">>> saving replay buffer & trajectories")
            torch.save(all_trajectories, os.path.join(save_path, "trajectories.pt"))
            torch.save(replay_buffer, os.path.join(save_path, "replay_buffer.pt"))

            time.sleep(15)
        else:
            info = {}

        accelerator.wait_for_everyone()
        all_trajectories = torch.load
        replay_buffer = torch.load()

        if "filtered" in agent_type.lower():
            # do some filtering
            filtered_replay_buffer = None
            info.update(trainer.update(filtered_replay_buffer))
        else:
            trainer.update(replay_buffer)
