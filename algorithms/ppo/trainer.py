from typing import Any
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from deepspeed import comm as dist

from common.deepspeed_utils import prepare_data_loader_for_training, prepare_data_loader_for_inference
from common.dataset import EpisodeDataset
from common.logging import get_logger
from policies.actor_critic_policy import ActorForwardOutput, CriticForwardOutput
from algorithms.base_trainer import OnPolicyTrainer
from algorithms.ppo.data_collator import PPODataCollator


logger = get_logger(__name__)

class PPOTrainer(OnPolicyTrainer):
    """
        PPO Trainer. 
        Impelmentation of the PPO update rule. 
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def step(self, episodes: EpisodeDataset) -> None:
        """
            Performs a single update step using the dataset rollout under the current policy. 
            Each updatestep can rum multiple epochs of optimization. 
        """         
        # change to appropriate input structure
        episodes = self._get_curr_logs_and_values(episodes)

        dataloader = DataLoader(
            episodes, 
            per_device_batch_size=self.per_device_batch_size,
            shuffle=True
        )

        steps_in_epoch = len(dataloader)
        optim_steps_in_epoch = steps_in_epoch // self.args.gradient_accumulation_steps
        optim_steps_in_epoch = max(optim_steps_in_epoch, 1)

        num_optimization_steps_in_iteration = (
            self.num_epochs_per_iteration * optim_steps_in_epoch
        )

        total_num_optimization_steps = (
            self.num_iterations * num_optimization_steps_in_iteration
        )

        dataloader_iter = iter(dataloader)

        progress_bar = tqdm(
            total=total_num_optimization_steps,
            disable=not self._is_main_process(),
            desc=f"Iteration {self.state.iteration}: Training",
            dynamic_ncols=True,
        )
        progress_bar.update(self.state.global_step)

        # Set everything in train mode
        self.policy.actor.train()
        self.policy.critic.train()

        for epoch in range(self.num_epochs):
            for step, inputs in enumerate(dataloader_iter):
                # Prepare data in batches
                for batch in tqdm(dataloader):
                    self._step(batch, step)
             
    def _get_curr_logs_and_values(self, episodes: Dataset) -> Dataset:
        """
            Takes the collated dataset and hydrates it with the
            logprobs and values under the current policy parameters. 
            These will be the baseline logprobs and values i.e., pi_old(a|s)
        """
        episodes = self._hydrate_log_probs(episodes)
        episodes = self._hydrate_values(episodes)
        return episodes

    def _hydrate_log_probs(self, episodes: Dataset, column_name: str = "log_probs") -> Dataset:
        """ Compute the logprobs and add them to dataset"""
        data_loader = prepare_data_loader_for_inference(
            episodes, 
            per_device_batch_size=self.per_device_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.dataloader_num_workers,
                "pin_memory": self.dataloader_pin_memory,
            }
        )
        self.policy.actor.eval()
        
        # iterate through the dataset. 
        list_of_log_probs = []
        for inputs in tqdm(
            data_loader, desc="Computing log probs...", disable=not self._is_main_process()
        ):  
            with torch.no_grad():
                # Dont call forward actor but forward_actor
                output: ActorForwardOutput = self.policy.forward_actor(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                    return_all_logp=True,
                )
            print("actor forward output", output.keys())
            log_probs = output["all_logp"].detach()
            assert log_probs.shape[1] == inputs["input_ids"].shape[1] -1

            assert log_probs.shape[0] == inputs["input_ids"].shape[0] * dist.get_world_size()
            list_of_log_probs.append(log_probs)

        # Add to dataset, convince yourself thi shas to be in .main_process_first?
        with self.distributed_state.main_process_first():
            print("Type episodes", type(episodes))
            logger.info(f"type epiosdes {type(episodes)}")
            episodes = episodes.add_column(name=column_name, column=list_of_log_probs)

        return episodes
    
    def _hydrate_values(self, episodes: Dataset) -> Dataset:
        """ Compute the values and add them to the dataset"""

    def _step(batch: Dataset) -> None:
        """
            Process a batch.
        """
        # 1. compute rewards
        #  rewards = self._compute_rewards(inputs)
        #2. compute advantages
        #3. compute returns
        #4. Compute actor loss
        #5. Compute critic loss
        
    def _compute_actor_loss(self) -> Any:
        pass

    def _compute_critic_loss(self) -> Any:
        pass


    

