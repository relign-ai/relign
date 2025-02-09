import logging
import json
from pathlib import Path
import random
from typing import Optional, Callable, Dict, Tuple, Dict, Any, Union, List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from accelerate import PartialState
from accelerate.utils import release_memory

from relign.inference.base_inference_strategy import InferenceStrategy
from relign.tasks import Task
from relign.eval.analyzer import Analyzer
from relign.common.vllm_server import VLLMServer
from relign.utils.logging import get_logger
from relign.utils.py_utils import find_n_free_ports
from relign.utils.gpu import get_gpu_memory, wait_for_memory_release


logger = get_logger(__name__)


class Evaluator:
    def __init__(
        self,
        tokenizer,
        inference_strategy_cls: InferenceStrategy,
        inference_strategy_kwargs: Optional[Dict],
        analysers_cls: List[Analyzer],
        analysers_kwargs: List[Dict],
        vllm_server_cls: VLLMServer,
        wait_until_memory_release: bool,
        distributed_state: PartialState,
        project_root_dir: Path,
        task: Task,
        dataset_split: str,
        vllm_gpu_memory_utilization: Union[float, str] = "auto",
        force_rerun: bool = False,
        every_n_checkpoints: int = 1,  # TODO: do we still need this?
        cloud_logger: Optional[Callable] = None,
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
        self.analysers_cls = analysers_cls
        self.analysers_kwargs = analysers_kwargs
        self.cloud_logger = cloud_logger
        self.inference_strategy_cls = inference_strategy_cls
        self.inference_strategy_kwargs = inference_strategy_kwargs
        self.distributed_state = distributed_state
        self.vllm_server_cls = vllm_server_cls
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.wait_until_memory_release = wait_until_memory_release
        self.tokenizer = tokenizer
        self.task = task
        self.dataset_split = dataset_split
        self.seed = seed
        self._port_generator_rng = random.Random(self.seed)

        # Initialize the analyzers
        self.analysers = []
        for analyser, kwrgs in zip(self.analysers_cls, self.analysers_kwargs):
            kwrgs["cloud_logger"] = self.cloud_logger
            kwrgs["project_root_dir"] = self.project_root_dir
            self.analysers.append(analyser(**kwrgs))

    def _init_evaluation_dir(self):
        evaluation_dir = self.project_root_dir / "evals"
        evaluation_dir.mkdir(exist_ok=True, parents=True)

    def evaluate(
        self,
        iteration: int,
        latest_policy_path: Optional[Path] = None,
    ):
        """
        Main evaluation loop. Evaluates the latest policy path.
        """
        # First, we'll log that we're entering the generate method with our rank (process_index)
        process_index = self.distributed_state.process_index
        logger.info(f"[RANK={process_index}] Entering generate method...")

        this_process_device = self.distributed_state.device
        release_memory()
        logger.info(
            f"[RANK={process_index}] Finished release_memory() after entering generate."
        )

        this_process_device = self.distributed_state.device
        logger.info(f"Process {this_process_device} is evaluating")
        release_memory()

        assert latest_policy_path is not None, "latest_policy_path cannot be None."
        latest_policy_path = self._prepare_ckpt(latest_policy_path)

        ##################
        # Create eval dir
        ##################
        eval_root_dir = Path(self.project_root_dir / "evals")
        eval_root_dir.mkdir(exist_ok=True)

        inference_result_dir = Path(eval_root_dir / f"it_{iteration}")
        inference_result_dir.mkdir(exist_ok=True)

        process_index = self.distributed_state.process_index
        target_device_index = self._get_device_index()
        gpu_mem_usage_before_mb = get_gpu_memory()[target_device_index]

        ###################
        # Prepare the eval dataset
        ####################
        dataset = self.task.get_datasets(self.dataset_split)
        logger.info(f"Original Dataset size: {len(dataset)}")

        logger.info(
            f"Dataset Examples: "
            f"{json.dumps([dataset[i] for i in range(min(5, len(dataset)))], indent=2, sort_keys=True)}"
        )

        inp_ds_path = inference_result_dir / f"input_dataset_{process_index}"
        dataset.save_to_disk(str(inp_ds_path))
        del dataset

        dataset = Dataset.load_from_disk(str(inp_ds_path))
        dataset = dataset.shard(
            num_shards=self.distributed_state.num_processes,
            index=process_index,
            contiguous=True,
        )

        results_dir = inference_result_dir / "eval_results" / f"results_{process_index}"
        results_dir.mkdir(parents=True, exist_ok=True)

        ######################################
        # Prepare the server and run inference
        ######################################
        seed = self.seed + process_index * 100 + iteration
        vllm_init_fn = self._get_vllm_init_fn(
            results_dir=inference_result_dir,
            hf_ckpt_path_or_model=str(latest_policy_path),
            process_index=process_index,
            seed=seed,
        )

        results = self._run_inference(
            dataset_shard=dataset,
            vllm_init_fn=vllm_init_fn,
            results_root_dir=inference_result_dir,
            iteration=iteration,
            gpu_memory_usage_before_mb=gpu_mem_usage_before_mb,
            seed=seed,
        )

        logger.info(f"Evaluation results: {results}")

        analysis_results = []
        if self.analysers is not None:
            for analyzer in self.analysers:
                logger.info(f"Running analyser: {analyzer.__class__.__name__}")
                analysis = analyzer.analyze(results)
                analysis_results.append(analysis)
                self._log_analysis_metrics(analysis)

        return analysis_results

    def _run_inference(
        self,
        dataset_shard: Dataset,
        vllm_init_fn: Callable[[], Tuple[VLLMServer, Dict[str, Any]]],
        results_root_dir: Path,
        iteration: int,
        gpu_memory_usage_before_mb: int,
        seed: int,
    ) -> Dataset:
        """
        Potentially start a vLLM server and run inference to generate results needed for episode generation.

        Args:
            dataset_shard (Dataset):
                The shard of the prompt dataset to run inference on.
            vllm_init_fn (Callable[[], Tuple[VLLMServer, Dict[str, Any]]]):
                A function that initializes the vLLM server and returns the server object and the server URL.
            results_root_dir (Path):
                The directory to save the results to (this is unique for ea
            seed (int):
                The seed for this process to use for inference.
        """

        infer_result_path = results_root_dir / "results_ds"
        vllm_server, guidance_llm_kwargs = vllm_init_fn()
        target_device_index = self._get_device_index()

        # Try to run infernece, if the code fails in here, clean up
        results = None
        try:
            # initialize the inference strategy class with updates result dir
            self.inference_strategy = self.inference_strategy_cls(
                **self.inference_strategy_kwargs,
                result_dir=results_root_dir,
                seed=seed,
                cloud_logger=None,
                log_level=(
                    logging.WARNING
                    if not self.distributed_state.is_local_main_process
                    else None
                ),
            )

            # initialize the guidance_llm with the right server settings
            self.inference_strategy._init_guidance_llm(**guidance_llm_kwargs)
            results = self.inference_strategy.generate(dataset_shard)

            logging.info(f"obtained {len(results)} from inference strategy")

            # Convert the results to a list before saving
            episodes = []
            for row in results:
                episode = dict(row)  # copy all fields from row
                episode["iteration"] = iteration
                episodes.append(episode)

            episode_ds = Dataset.from_list(episodes)
            episode_ds.save_to_disk(str(infer_result_path))
        finally:
            logger.info(f"Cleaning up vllm server.. Device:{target_device_index}")
            vllm_server.stop_server()
            if results is not None:
                del results
            del vllm_server
            release_memory()
            self.vllm_cleanup(
                target_gpu_index=target_device_index,  # or can i just use self.distributed_state.device.index here? even if there are multiple ?
                gpu_memory_usage_before_mb=gpu_memory_usage_before_mb,
            )
            release_memory()

        results = Dataset.load_from_disk(str(results_root_dir / "results_ds"))
        return results

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
        logger.info(f"Rank {process_index}: about to call _set_vllm_ports() …")

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

            self.vllm_server = VLLMServer(
                port=vllm_port,
                seed=self.seed,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
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

    def _prepare_ckpt(self, latest_policy_path: Path) -> Path:
        """
        Make sure the tokenizer is included in the latest_policy_path.

        This method checks if latest_policy_path is already the correct directory
        containing tokenizer files. If not, it creates or uses a subdirectory
        called 'hf_pretrained' inside latest_policy_path and saves the tokenizer there.
        This ensures that downstream processes (like the vLLM engine) can correctly load
        the GPT2TokenizerFast tokenizer.
        """
        hf_pretrained_dir = latest_policy_path

        # Look for existing tokenizer files (like tokenizer.json, vocab files, etc.)
        tokenizer_files = list(hf_pretrained_dir.glob("tokenizer*"))
        if not tokenizer_files:
            logger.info(
                f"No tokenizer files found in {hf_pretrained_dir}. Saving tokenizer there."
            )
            self.tokenizer.save_pretrained(hf_pretrained_dir)
        else:
            logger.info(
                f"Tokenizer files already present in {hf_pretrained_dir}: {[f.name for f in tokenizer_files]}"
            )

        return latest_policy_path

    def _set_vllm_ports(self, seed: Optional[int] = None):
        """
        The main process searches for self.distributed_state.num_processes's free ports.
        and then broadcasts the ports to all processes.
        """
        if self.distributed_state.process_index == 0:
            logger.info("finding free ports")
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

    def vllm_cleanup(
        self,
        gpu_memory_usage_before_mb: int,
        target_gpu_index: int,
    ):
        if self.wait_until_memory_release:
            threshold_mb = gpu_memory_usage_before_mb * 1.1  # Allow for 10% tolerance
            wait_for_memory_release(
                target_gpu_index=target_gpu_index,
                threshold_mb=threshold_mb,
            )
            logger.info("GPU memory usage is below threshold. Continuing.")

    def _get_device_index(self) -> int:
        """Returns a valid device index either from self.distributed_state.device.index or using torch.cuda.current_device()."""
        device = self.distributed_state.device
        return device.index if device.index is not None else torch.cuda.current_device()
