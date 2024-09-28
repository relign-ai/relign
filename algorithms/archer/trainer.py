import torch
from agent.base import Agent
import accelerate
from transformers import AutoTokenizer
import copy


class ArcherTrainer:
    def __init__(
        self,
        agent: Agent,
        accelerator: accelerate.ccelerator,
        tokenizer: AutoTokenizer,
        criterion: torch.nn.Module = torch.nn.MSELoss(), # loss function
        critic_lr: float = 1e-3,
        lm_lr: float = 1e-3,
        grad_accum_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        epochs: int = 100,
        max_grad_norm: float = 1.0,
        actor_epochs: int = 100,
    ):
        self.agent = agent
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.critic_lr = critic_lr
        self.lm_lr = lm_lr

        self.grad_accum_steps = grad_accum_steps
        self.gamma = gamma
        self.tau = tau
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

        # Prepare the optimzzers
        self.lm_optimizer = torch.optim.AdamW(
            self.agent.base_lm.parameters(), lr=self.lm_lr
        )

        self.critic_optimizer = torch.optim.AdamW(
            self.agent.critic.parameters(), lr=self.critic_lr
        )

        self.step = 0


    def critic_loss(
            self, 
            observation,
            action,
            reward,
            next_observation,
            done,
            mc_return,
            **kwargs,
    ): 
        
        reward = (
            torch.Tensor(reward)
            .to(
                self.accelerator.unwrap_model(self.agent.model).device
                dtyp=self.accelerator.unwrap_model(self.agent.model).dtype
            )
            .flatten()
        )

        done = (
            torch.Tensor(done)
            .to(
                self.accelerator.unwrap_model(self.agent.model).device,
                dtype=self.accelerator.unwrap_model(self.agent.model).dtype,
            )
            .flatten()
        )

        q1, q2, v1, v2 = self.agent.critic(observation, action, detach_model=False)

        with torch.no_grad():
            # Get action under the current policy?
            pi_action = self.agent.get_action(copy.deepcopy(observation))

            #
            target_q1, target_q2, _, _ = self.agent.critic(
                observation, pi_action, detach_model=True 
            )

        q1 = q1.flatten()
        q2 = q2.flatten()
        v1 = v1.flatten()
        v2 = v2.flatten()
        target_q1 = target_q1.flatten()
        target_q2 = target_q2.flatten()

        with torch.no_grad():
            # action is dummy here since we are not using it
            _, _, target_v1, target_v2 = self.agent.target_critic(
                next_observation, copy.deepcopy(action)
            )
            target_v1 = reward + (1 - done) * target_v1.flatten() * self.gamma
            target_v2 = reward + (1 - done) * target_v2.flatten() * self.gamma
            
        q1_loss = self.criterion(q1, target_v1)
        q2_loss = self.criterion(q2, target_v2)

        v1_loss = self.criterion(v1, target_q1)
        v2_loss = self.criterion(v2, target_q2)

        # Perform gradient descent on the losses. 
        self.accelerator.backward((q1_loss + q2_loss + v1_loss + v2_loss))
        
        q1_loss, q2_loss, v1_loss, v2_loss = (
            q1_loss.detach().cpu(),
            q2_loss.detach().cpu(),
            v1_loss.detach().cpu(),
            v2_loss.detach().cpu(),
        )
        q1, q2, v1, v2, target_q1, target_q2 = (
            q1.detach().cpu(),
            q2.detach().cpu(),
            v1.detach().cpu(),
            v2.detach().cpu(),
            target_q1.detach().cpu(),
            target_q2.detach().cpu(),
        )
    
        return {
            "q1.loss": q1_loss,
            "q2.loss": q2_loss,
            "v1.loss": v1_loss,
            "v2.loss": v2_loss,
            "q1.mean": torch.mean(q1),
            "q1.min": torch.min(q1),
            "q1.max": torch.max(q1),
            "q1.std": torch.std(q1),
            "q2.mean": torch.mean(q2),
            "q2.max": torch.max(q2),
            "q2.min": torch.min(q2),
            "q2.std": torch.std(q2),
            "v1.mean": torch.mean(v1),
            "v1.min": torch.min(v1),
            "v1.max": torch.max(v1),
            "v1.std": torch.std(v1),
            "v2.mean": torch.mean(v2),
            "v2.max": torch.max(v2),
            "v2.min": torch.min(v2),
            "v2.std": torch.std(v2),
            "target_q1.mean": torch.mean(target_q1),
            "target_q1.min": torch.min(target_q1),
            "target_q1.max": torch.max(target_q1),
            "target_q1.std": torch.std(target_q1),
            "target_q2.mean": torch.mean(target_q2),
            "target_q2.max": torch.max(target_q2),
            "target_q2.min": torch.min(target_q2),
            "target_q2.std": torch.std(target_q2),
        }


            
