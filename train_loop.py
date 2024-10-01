import time
import os
from tqdm import tqdm
import torch
import accelerate
from transformers import AutoTokenizer
from environment.environment_base import BaseEnvironment
from agent.agent_base import BaseAgent
from algorithms.archer.trainer import ArcherTrainer


def train_loop(
    agent: BaseAgent,
    agent_type: str,
    environment: BaseEnvironment,
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

    trainer = ArcherTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=tokenizer,
        critic_lr=critic_lr,
        lm_lr=lm_lr,
        grad_accum_steps=grad_accum_steps,
        gamma=gamma,
        tau=tau,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        actor_epochs=actor_epochs,
    )

    # replay_buffer = ReplayBuffer()
    all_trajectories = []

    for i in tqdm(range(iterations)):
        if accelerator.is_main_process:
            trajectories = None  # rollout trajectories

            # do eval every eval_freq
            all_trajectories += trajectories

            agent.get_action(observation=)
            environment.step(action=)
            

            # environment rollout
            trajectories = environment.rollout()

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
