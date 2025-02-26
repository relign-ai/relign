import pytest

from relign.tasks.math.gsm8k import GSM8K

@pytest.fixture
def gsm8k():
    return GSM8K(
        answer_prefix="\n####",
        load_dataset_dict=True,
        dataset_dict_path="data/gsm8k",
        remove_calculator_expressions=True,
        use_original_format=True,  # TODO: is this really nessecary? and why?
        intermetdiate_step_tags=["<think>", "</think>"],
        reward_on_last_token=True,
    )
