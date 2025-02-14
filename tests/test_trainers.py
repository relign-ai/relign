from datasets import load_from_disk


def test_grpo_train_step(grpo_trainer):
    """
    Test a single training step of the GRPO methodw.
    """
    episodes = load_from_disk("./tests/mock_data/mock_math_group_episodes_ds")
    print("episodes", episodes)
    grpo_trainer.step(episodes)


def test_test():
    print("hello world")
