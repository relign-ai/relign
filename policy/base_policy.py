from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Self, Union

from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate
from accelerate import PartialState
from accelerate.utils import DummyOptim
from transformers import PreTrainedModel

from wandb.sdk.wandb_run import Run as WandbRun

from common.double_critic import DoubleCritic
from common.deepspeed_utils import get_optimizer_grouped_parameters 

from episode_generation.environment.base_environment import History


class BaseModel(nn.Module):
    """
    Make predictions in response to opservations. 

    In case of Policies, the prediction is an action. 
    In case of Critics, the prediction is the estimated value of the observation.
    """
    
    def __init__(self):
        super().__init__()
        pass

    def save(self, path: str) -> None:
        """ Save the model to a file"""
        raise NotImplementedError( "Not implemented yet" )

    def load(self, path: str, device: torch.device) -> Self:
        """ Load the policy from a file"""
        raise NotImplementedError( "Not implemented yet" )
    
    @property
    def device(self) -> torch.device:
        """ Get the device of the model"""
        pass


class BasePolicy(BaseModel):
    """
    A policy takes an observation and returns an action. 
    """

    @abstractmethod
    def predict(self, observation: History) -> History:
        """
        Predict an action from an observation.
        """
        pass


    @abstractmethod 
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the policy.
        """
        pass


    def set_root_dir(self, path: Path):
        self.project_root_dir = path
        self._init_checkpoint_dir()

    def set_cloud_log(self, cloud_logger: WandbRun):
        self.cloud_logger = cloud_logger

    def _init_checkpoint_dir(self):
        self.checkpoint_dir = (self.project_root_dir / "policy")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._validate_checkpoint_format()


    def _validate_checkpoint_format(self, checkpoint_format: Optional[str] = None):
        if checkpoint_format is None:
            checkpoint_format = self.get_checkpoint_format()
        assert checkpoint_format.startswith("ckpt--iter_")
        for state in checkpoint_format.replace("ckpt--", "").split("--"):
            assert len(state.split("_", 1)) == 2
            state_name, state_value = state.split("_", 1)
            assert state_value.startswith("{") and state_value.endswith("}")


    def get_checkpoint_format(self) -> str:
        return "ckpt--iter_{iteration}--epoch_{epoch}--step_{global_step}"


#TODO: Do we need a deepspeed policy? Or will deepspeed be the default?
# When we do not use deepspeed will we just set the main process on the cpu? 
# Or will deepspeed be a hard requirement?
class DeepSpeedMixin(BasePolicy):
    """
    The difference between this and a normal `Basepolicy` is that this class solely uses DeepSpeed for training and
    ditched the Accelerate library. This is because the Accelerate library does not support two models in a single
    training loop, which becomes problematic in actor-critic training.
    """
    def __init__(
        self,
        experiment_root: Path,
    ):
        # define default x-axis (for latest wandb versions)
        if self._is_main_process():
            if getattr(self.cloud_logger, "define_metric", None):
                self.cloud_logger.define_metric("train/global_step")
                self.cloud_logger.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

    def set_deepspeed(self, distributed_state: PartialState):
        self.distirbuted_state = distributed_state


    def _is_main_process(self) -> bool:
        """ Deal with the distributed state"""
        return self.distributed_state.is_main_process



class BaseActorCritic(DeepSpeedMixin, BasePolicy):
    """
        Base Actor critic type policy. 

        The actor predicts an action from an observation and a 
        critic predicts the value of an observation. 
    """
    
    @abstractmethod
    def predict_actor(self, episodes: History) -> History:
        """
        Predict an action from an observation.
        """
        pass
    
    @abstractmethod
    def predict_critic(self, episodes: History) -> float:
        """
        Predict the value of an observation.
        """
        pass

    @abstractmethod
    def forward_actor(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the policy.
        """
        pass

    @abstractmethod
    def forward_critic(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the critic.
        """
        pass


class BaseCritic(BaseModel):
    """
    Main critic model. Takes an observation and returns an estimate of the value of the observation.
    """
    @abstractmethod
    def predict(self, observation: History) -> float:
        pass

# TODO: Delete soon.. decrepated
class BaseAgent(nn.Module, ABC):
    def __init__(
        self,
        device: torch.device,
        accelerator: accelerate.Accelerator,
        policy_lm: str = "gpt2",
        critic_lm: str = "roberta-base",
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
        super(BaseAgent, self).__init__()
        self.device = device
        self.accelerator = accelerator
        self.policy_lm = policy_lm
        self.critic_lm = critic_lm
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
        self.tokenizer, self.truncation_side = self._load_tokenizer()
        self.critic, self.target_critic = self._load_critics()

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

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.policy_lm,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            padding_side="left",
        )
        truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, truncation_side

    def _load_critics(self) -> Tuple[nn.Module, nn.Module]:
        critic = DoubleCritic(
            device=self.device,
            accelerator=self.accelerator,
            critic_lm=self.critic_lm,
            cache_dir=self.cache_dir,
            in_dim=768,
            out_dim=1,
        )
        target_critic = DoubleCritic(
            device=self.device,
            accelerator=self.accelerator,
            critic_lm=self.critic_lm,
            cache_dir=self.cache_dir,
            in_dim=768,
            out_dim=1,
        )
        return critic, target_critic

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


