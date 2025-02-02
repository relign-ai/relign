import pytest
from pathlib import Path

import torch
from accelerate import PartialState

from relign.common.vllm_server import VLLMServer


@pytest.fixture
def vllm_server():
    return VLLMServer()


@pytest.fixture
def distributed_state_cpu():
    from datetime import timedelta
    ddp_timeout = 10000
    kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
    return PartialState(True, **kwargs)

@pytest.fixture
def distributed_single_gpu():
    """ Fixture for a deepspeed distributed state with a single gpu"""
    # Check gpu availability
    if not torch.cuda.is_available():
        pytest.skip("Skipping test as no GPU is available")

    from datetime import timedelta
    ddp_timeout = 10000
    kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
    return PartialState(False, **kwargs, device=torch.device("cuda:0"))

@pytest.fixture
def distributed_multi_gpu():
    """ Fixture for a deepspeed distributed state with multiple gpus"""
    # Check gpu availability
    if not torch.cuda.is_available():
        pytest.skip("Skipping test as no GPU is available")

    from datetime import timedelta
    ddp_timeout = 10000
    kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
    kwargs["backend"] = "nccl"
    return PartialState(False, **kwargs, device=torch.device("cuda:0"))

@pytest.fixture
def deepspeed_distributed():
    """ Fixture for a deepspeed distributed state with multiple gpus"""
    # Check gpu availability
    if not torch.cuda.is_available():
        pytest.skip("Skipping test as no GPU is available")
    ...

@pytest.fixture
def experiment_dir(request, pytestconfig):
    """
    Creates a test-specific experiment directory based on the test's node id and the
    provided --output-dir option.
    Example structure: test/output/tests_test_episode_generator_py__test_generate_episode_generation
    """
    base_output_dir = Path(pytestconfig.getoption("output_dir"))
    # Use the test node id, replacing characters that could be problematic in file names.
    test_nodeid = request.node.nodeid.replace("/", "_").replace("::", "_").replace(" ", "_")
    exp_dir = base_output_dir / test_nodeid
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir