from relign.common.vllm_server import VLLMServer
from relign.utils.gpu import get_gpu_memory

def are_elements_less(a, b):
    return all([a_i < b_i for a_i, b_i in zip(a, b)])

class TestVLLMServer:
    def test_start_and_stop_vllm(
        self, 
        vllm_server: VLLMServer, 
        tol: float = 1.1
    ):

        pre_gpu_mem_usage = get_gpu_memory()

        vllm_server.start_server(hf_ckpt_path_or_model="realtreetune/rho-1b-sft-GSM8K")
        vllm_server.stop_server()

        post_gpu_mem_usage = get_gpu_memory()


        # Assert that memory usage is released and processec are killd
        pre_gpu_mem_usage_tol = [gpu_mem_usage * tol for gpu_mem_usage in pre_gpu_mem_usage]

        assert are_elements_less(post_gpu_mem_usage, pre_gpu_mem_usage_tol)