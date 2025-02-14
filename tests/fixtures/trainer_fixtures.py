import pytest

from relign.algorithms.base_trainer import BaseTrainer
from relign.algorithms.ppo.trainer import PPOTrainer
from relign.algorithms.grpo.trainer import GRPOTrainer
from relign.algorithms.train_loop import TrainLoop


@pytest.fixture
def ppo_trainer(
    experiment_dir,
    actor_policy,
    distributed_state_cpu_no_deepspeed,
) -> BaseTrainer:
    """
    Returns a mock or real PPO trainer for integration tests.
    """
    return PPOTrainer(
        project_root_dir=experiment_dir,
        seed=69,
        policy=actor_policy,
        distributed_state=distributed_state_cpu_no_deepspeed,
        per_device_batch_size=4,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


@pytest.fixture
def grpo_trainer(experiment_dir, actor_policy, distributed_single_gpu) -> BaseTrainer:
    """
    Returns a mock or real GRPO trainer for integration tests.
    """
    return GRPOTrainer(
        project_root_dir=experiment_dir,
        seed=69,
        policy=actor_policy,
        distributed_state=distributed_single_gpu,
        per_device_batch_size=4,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


@pytest.fixture
def train_loop():
    """
    Returns a train loop for integration tests.
    """
    return TrainLoop(
        num_epochs=1,
        num_batches=1,
        num_steps=1,
        num_episodes=1,
    )
