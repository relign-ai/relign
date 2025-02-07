from pathlib import Path
import random
from typing import Optional, Callable, Dict, Tuple, Dict, Any, Union, List

import torch
from transformers import PreTrainedTokenizer
from accelerate import PartialState
from accelerate.utils import release_memory

from relign.inference.inference_pipeline import InferencePipeline
from relign.tasks import Task
from relign.eval.analyzer import Analyzer
from relign.common.vllm_server import VLLMServer
from relign.utils.logging import get_logger
from relign.utils.gpu import get_gpu_memory
from relign.utils.py_utils import find_n_free_ports


logger = get_logger(__name__)


class Evaluator:
    def __init__(
        self,
        tokenizer,
        inference_pipeline_cls: InferencePipeline,
        inference_pipeline_kwargs: Optional[Dict],
        vllm_server: VLLMServer,
        distributed_state: PartialState,
        project_root_dir: Path,
        task: Task,
        vllm_gpu_memory_utilization: Union[float, str] = "auto",
        force_rerun: bool = False,
        every_n_checkpoints: int = 1,  # TODO: do we still need this?
        cloud_logger: Optional[Callable] = None,
        analyzers=Optional[List[Analyzer]],
        seed: int = 69,
    ):
        """
        Base Evaluator class of reinfrocement learning algorithms.
        Runs an evaluation on the model located in the latest policy path,
        TODO: or on all checkpoints if on checkpoint is true
        """
        self.project_root_dir = project_root_dir
        # if self.analyzers is not None:
        self._init_evaluation_dir()
        self.force_rerun = force_rerun
        self.every_n_checkpoints = every_n_checkpoints
        self.analyzers = analyzers
        self.cloud_logger = cloud_logger
        self.inference_pipeline_cls = inference_pipeline_cls
        self.inference_pipeline_kwargs = inference_pipeline_kwargs
        self.distributed_state = distributed_state
        self.vllm_server = vllm_server
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.tokenizer = tokenizer
        self.task = task
        self.seed = seed
        self._port_generator_rng = random.Random(self.seed)

    def _init_evaluation_dir(self):
        evaluation_dir = self.project_root_dir / "evals"
        evaluation_dir.mkdir(exist_ok=True, parents=True)

    def evaluate(
        self,
        tokenizer: PreTrainedTokenizer,
        iteration: int,
        latest_policy_path: Optional[Path] = None,
        from_checkpoints: bool = False,
    ):
        """
        Main evaluation loop. Evaluates the latest policy path.
        """

        # TODO: Do batch evaluations on all the checkpints
        if not from_checkpoints:
            model_cpt_path = latest_policy_path
        else:
            raise NotImplementedError(
                "Evaluation from checkpoints is not yet implemented"
            )

        # eval_dir_path = Path(self.evaluation_dir / model_cpt)
        eval_dir = Path(self.project_root_dir / "evals")
        eval_dir.mkdir(exist_ok=True)
        inference_result_dir = Path(eval_dir / f"it_{iteration}")
        inference_result_dir.mkdir(exist_ok=True)

        process_index = self.distributed_state.process_index

        vllm_init_fn = self._get_vllm_init_fn(
            results_dir=inference_result_dir,
            hf_ckpt_path_or_model=latest_policy_path,
            process_index=process_index,
            seed=self.seed,
        )

        # Starts thhe vllm server and retuirns the server kwargs
        vllm_server, server_kwargs = vllm_init_fn()

        try:
            # instantiate the infernce pipeline, with the correct inference
            infer_pipeline = InferencePipeline(
                **self.inference_pipeline_kwrags,
                **server_kwargs,
                tokenizer=tokenizer,
                use_cache=True,
            )

            # Run the inference pipeline
            results = infer_pipeline.generate()
            logger.info(f"Evaluation results: {results}")

        finally:
            # Stop the server
            vllm_server.stop_server()

        analysis_results = []

        # Run analysis on inference results
        # if self.analyzers is not None:
        #     for analyzer in self.analyzers:
        #         analysis = analyzer.analyze(results)
        #         analysis_results.append(analysis)
        #         self._log_analysis_metrics(analysis)

        return analysis_results

    def _log_analysis_metrics(self, analysis): ...

    def _get_vllm_init_fn(
        self,
        results_dir: Path,
        hf_ckpt_path_or_model: str,
        process_index: int,
        seed: int,
    ) -> Callable[[], Tuple[VLLMServer, Dict[str, Any]]]:
        logger.info("setup up vllm server")
        vllm_gpu_memory_utilization = self.vllm_gpu_memory_utilization
        self._set_vllm_ports(seed=seed)
        vllm_port = self._vllm_port
        if vllm_gpu_memory_utilization == "auto":
            # Compute the GPU utilization based on amount of remaining memory
            allocated_mem_mb = get_gpu_memory()[process_index]
            total_mem_mb = (
                torch.cuda.get_device_properties(process_index).total_memory / 1024**2
            )

            remaining_mem_mb = (
                total_mem_mb - allocated_mem_mb
            ) * 0.9  # Allow for 10% tolerance
            vllm_gpu_memory_utilization = round(remaining_mem_mb / total_mem_mb, 2)

            logger.info(
                f"GPU #{process_index} Auto-computed vLLM GPU memory utilization: {vllm_gpu_memory_utilization}. "
                f"Currently Allocated: {allocated_mem_mb} MB, "
                f"Total: {total_mem_mb} MB, "
                f"Remaining: {remaining_mem_mb} MB."
            )

        def _init() -> Tuple[VLLMServer, Dict[str, Any]]:
            vllm_log_path = results_dir / "vllm_server.log"

            logger.info(
                f"Rank #{process_index} starting vLLM: "
                f"model={hf_ckpt_path_or_model}   port={vllm_port}   seed={seed}"
            )
            server_url = self.vllm_server.start_server(
                hf_ckpt_path_or_model=hf_ckpt_path_or_model,
                gpu_idx=process_index,
                wait_for_response=True,
                log_path=vllm_log_path,
                timeout=800,
            )

            logger.info(f"SERVER URL {server_url}")
            return self.vllm_server, {
                "api_base": server_url,
                "model": hf_ckpt_path_or_model,
            }

        return _init

    # TODO: dont need this for the current impelmentation
    # def _prepare_for_vllm(self, ckpt_dir: Path):
    #     """Prepare the checkpoint directory for evaluation."""
    #     # Use current working directory to create temporary ckpt path
    #     output_dir = Path.cwd() / f"tmp_ckpt__{ckpt_dir.name}"
    #     if output_dir.exists():
    #         shutil.rmtree(output_dir)
    #     output_dir.mkdir(exist_ok=True, parents=True)

    #     # Check if it already has hf_pretrained directory
    #     hf_pretrained_dir = ckpt_dir / "hf_pretrained"
    #     if hf_pretrained_dir.exists() and hf_pretrained_dir.is_dir():
    #         for file in hf_pretrained_dir.iterdir():
    #             (output_dir / file.name).symlink_to(file.absolute())

    #         if not (output_dir / "config.json").exists():
    #             config = AutoConfig.from_pretrained(self.model.gg_params["hf_model_name"])
    #             config.save_pretrained(output_dir)

    #         hf_tokenizer_files = [f for f in output_dir.glob("tokenizer*")]
    #         if len(hf_tokenizer_files) == 0:
    #             self.tokenizer.save_pretrained(output_dir)

    #         return output_dir

    def _set_vllm_ports(self, seed: Optional[int] = None):
        """
        The main process searches for self.distributed_state.num_processes's free ports.
        and then broadcasts the ports to all processes.
        """
        if self.distributed_state.process_index == 0:
            ports = find_n_free_ports(
                self.distributed_state.num_processes, generator=self._port_generator_rng
            )
            logger.info(f"Found free ports: {ports}")
        else:
            ports = [0] * self.distributed_state.num_processes

        from accelerate.utils import broadcast_object_list

        ports = broadcast_object_list(ports, from_process=0)
        release_memory()

        self._vllm_port = ports[self.distributed_state.process_index]
        logger.info(
            f"Rank {self.distributed_state.process_index} using vLLM port {self._vllm_port}"
        )
