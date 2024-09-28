from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate


class AgentBase(nn.Module, ABC):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        critic,
        double_critic,
        device: torch.device,
        accelerator: accelerate.Accelerator,
        policy_lm: str = "gpt2",
        cache_dir: str = "~/.cache",
        dropout: float = 0.5,
        TEMPLATE: Optional[str] = None,
        use_lora: bool = False,
        do_sample: bool = True,
        temperature: float = 1.0,
        max_new_tokens: int = 32,
        use_bfloat16: bool = False,
        eos_str: str = "\n",
    ):
        super(AgentBase, self).__init__()
        self.device = device
        self.accelerator = accelerator
        self.policy_lm = policy_lm
        self.cache_dir = cache_dir
        self.dropout = dropout
        self.template = TEMPLATE
        self.use_lora = use_lora
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_bfloat16 = use_bfloat16
        self.eos_str = eos_str

        self.model = self._load_model()
        self._initialize_tokenizer()
        self.critic, self.target_critic = self._initialize_critics()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def _load_model(self) -> AutoModelForCausalLM:
        if self.use_bfloat16:
            model = AutoModelForCausalLM.from_pretrained(
                self.policy_lm, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.policy_lm, cache_dir=self.cache_dir
            )
        model.to(self.device)

        if self.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model

    def _initialize_tokenizer(self) -> AutoTokenizer:
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @abstractmethod
    def prepare(self):
        """Prepare the model and critics for training/inference."""
        self.model, self.critic, self.target_critic = self.accelerator.prepare(
            self.model, self.critic, self.target_critic
        )

    @abstractmethod
    def get_action(self, observation: List[str]) -> List[str]:
        """Generate action based on observation."""
        pass

    @abstractmethod
    def get_log_prob(self, observation: List[str], action: List[str]) -> torch.Tensor:
        """Calculate log probability of the action given the observation."""
        pass

    def soft_update_target_critic(self, tau: float):
        """Soft update the target critic parameters."""
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train_mode(self):
        """Set the agent to training mode."""
        self.train()
        self.critic.train()
        self.target_critic.train()

    def eval_mode(self):
        """Set the agent to evaluation mode."""
        self.eval()
        self.critic.eval()
        self.target_critic.eval()
