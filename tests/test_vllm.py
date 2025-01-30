import pytest

from relign.common.vllm_server import VLLMServer
from relign.utils.gpu import get_gpu_memory
from relign.episode_generators.on_policy_episode_generator import OnPolicyEpisodeGenerator

@pytest.fixture
def vllm_server():
    return VLLMServer()


@pytest.fixture
def epiosde_generator():
    return OnPolicyEpisodeGenerator()


class TestVLLMServer:
    def test_start_and_stop_vllm(
        self, 
        vllm_server: VLLMServer, 
        tolarace=1.1
    ):
        pre_gpu_mem_usage = get_gpu_memory()

        vllm_server.start_server(hf_ckpt_path_or_model="realtreetune/rho-1b-sft-GSM8K")
        processes = vllm_server.process
        print(processes)
        vllm_server.stop_server()

        post_gpu_mem_usage = get_gpu_memory()

        # Assert that memory usage is released and processec are killd
        assert post_gpu_mem_usage < pre_gpu_mem_usage * tolarace 