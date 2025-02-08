import pytest
from pathlib import Path
from datetime import timedelta

import torch
from accelerate import PartialState
import torch.distributed as dist

from relign.common.vllm_server import VLLMServer
# from relign.utils.config import load_deepspeed_config


@pytest.fixture
def vllm_server():
    return VLLMServer()


@pytest.fixture
def distributed_state_cpu_deepspeed():
    from datetime import timedelta

    ddp_timeout = 10000
    kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
    return PartialState(True, **kwargs)


@pytest.fixture
def distributed_state_cpu_no_deepspeed():
    import os
    from datetime import timedelta
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    ddp_timeout = 10000
    kwargs = {"backend": "gloo", "timeout": timedelta(seconds=ddp_timeout)}
    dist.init_process_group(init_method="env://", **kwargs)

    return PartialState(True, **kwargs)


@pytest.fixture
def distributed_single_gpu():
    """Fixture for a deepspeed distributed state with a single gpu"""
    # Check gpu availability
    if not torch.cuda.is_available():
        pytest.skip("Skipping test as no GPU is available")

    from datetime import timedelta

    ddp_timeout = 10000
    kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
    return PartialState(False, **kwargs, device=torch.device("cuda:0"))


@pytest.fixture
def distributed_multi_gpu():
    """Fixture for a deepspeed distributed state with multiple gpus"""
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
    """Fixture for a deepspeed distributed state with multiple gpus"""
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
    test_nodeid = (
        request.node.nodeid.replace("/", "_").replace("::", "_").replace(" ", "_")
    )
    exp_dir = base_output_dir / test_nodeid
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


@pytest.fixture
def deepspeed_config():
    return {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {},
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 16,
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": True},
        "zero_0": {
            "stage": 0,
            "allgather_partitions": True,
            "allgather_bucket_size": 500000000,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 50000000,
            "contiguous_gradients": True,
        },
    }
