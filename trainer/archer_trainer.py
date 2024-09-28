import accelerate
from agent.base import Agent
from transformers import AutoTokenizer


class ArcherTrainer:
    def __init__(
        self,
        agent: Agent,
        accelerator: accelerate.Accelerator,
        tokenizer: AutoTokenizer,
        critic_lr: float,
        lm_lr: float,
        gamma: float,
        tau: float,
        epochs: int,
        actor_epochs: int,
        grad_accum_steps: int,
        max_grad_norm: float,
    ):
        self.agent = agent
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.critic_lr = critic_lr
        self.lm_lr = lm_lr
        self.gamma = gamma
        self.tau = tau
        self.epochs = epochs
