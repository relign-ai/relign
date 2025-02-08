from datasets import load_from_disk


class TestGRPOTrainer:
    """Trainer class of the GRPO algorithm"""

    def test_grpo_train_step(self, grpo_trainer):
        """
        Test a single training step of the GRPO methodw.
        """
        episodes = load_from_disk("./tests/mock_data/mock_math_group_episodes_ds")
        grpo_trainer.step(episodes)

