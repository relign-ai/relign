import pytest

from relign.tasks import GSM8K

@pytest.fixture
def gsm8k():
    return GSM8K(
        answer_prefix=None,
        load_dataset_dict=True,
        dataset_dict_path="data/gsm8k",
        remove_calculator_expressions=True,
    )
