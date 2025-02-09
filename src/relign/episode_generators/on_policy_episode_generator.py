import json
import random
import shutil
import tempfile
import time
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple, Callable

import torch.cuda
from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets

from relign.utils.gpu import get_gpu_memory, wait_for_memory_release
from relign.utils.py_utils import find_n_free_ports
from relign.common.vllm_server import VLLMServer, compute_vllm_stats

from relign.episode_generators.base_episode_generator import (
    BaseEpisodeGenerator,
    Episode,
)
from relign.inference.base_inference_strategy import InferenceStrategy
from relign.tasks.base_task import Task

from relign.utils.logging import get_logger


logger = get_logger(__name__)


class OnPolicyEpisodeGenerator(BaseEpisodeGenerator):
    can_precompute_episodes: bool = False
    supports_distributed: bool = True

    def __init__(
        self,
        inference_strategy_cls: InferenceStrategy,
        inference_strategy_kwargs: Dict[str, Any],
        vllm_server_cls: VLLMServer,
        task: Task,
        seed: int,
        initial_model_name_or_path: str,
        vllm_gpu_memory_utilization: Union[float, str] = 0.9,
        vllm_min_available_gpu_memory_mb: Optional[int] = None,
        wait_until_memory_release: bool = False,
        dataset_shuffle_on_each_iteration: bool = True,
        dataset_shuffle_before_portion: bool = True,
        dataset_split: str = "train",
        dataset_portion: Optional[float] = None,
        dataset_num_samples_per_iteration: Optional[int] = None,
        dataset_sample_with_replacement: bool = True,
        dataset_initial_size: Optional[int] = None,
        total_num_iterations: Optional[int] = None,
        temp_dir_root: Optional[str] = None,
        fill_missing_episodes: bool = False,
        max_question_length: Optional[int] = None,
        question_template: Optional[str] = None,
        save_generations_every_n_iteration: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        The base class for episode generators that generate episodes by sampling from the model.
        It supports distributed environments.
        """
        super().__init__(**kwargs)
        self._logger = logger

        self.inference_strategy_cls = inference_strategy_cls
        self.inference_strategy_kwargs = inference_strategy_kwargs
        self.vllm_server_cls = vllm_server_cls
        self.task = task
        self.dataset_split = dataset_split
        self.seed = seed
        self.initial_model_name_or_path = initial_model_name_or_path
        self.debug = debug
        self.dataset_portion = dataset_portion
        self.dataset_num_samples_per_iteration = dataset_num_samples_per_iteration
        self.dataset_shuffle_before_portion = dataset_shuffle_before_portion
        self.dataset_shuffle_on_each_iteration = dataset_shuffle_on_each_iteration
        self.dataset_sample_with_replacement = dataset_sample_with_replacement
        self.dataset_initial_size = dataset_initial_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_min_available_gpu_memory_mb = vllm_min_available_gpu_memory_mb
        self.wait_until_memory_release = wait_until_memory_release
        self.total_num_iterations = total_num_iterations
        self.fill_missing_episodes = fill_missing_episodes
        self.max_question_length = max_question_length
        self.question_template = question_template
        self.save_generations_every_n_iteration = save_generations_every_n_iteration

        if (
            self.dataset_portion is not None
            and self.dataset_num_samples_per_iteration is not None
        ):
            raise ValueError(
                "Only one of `dataset_portion` and `dataset_num_samples_per_iteration` can be set."
            )
        if (
            self.dataset_portion is None
            and self.dataset_num_samples_per_iteration is None
        ):
            self.dataset_portion = 1.0

        if temp_dir_root is None:
            self.temp_dir_root = self.project_root_dir / "temp_episodes"
            self._log_on_main(f"Using default temp_dir_root: {self.temp_dir_root}")
        else:
            self.temp_dir_root = Path(temp_dir_root)
        self.temp_dir_root.mkdir(parents=True, exist_ok=True)

        self._orig_ds = None

        self._vllm_port = None
        self._port_generator_rng = random.Random(self.seed)
        self._set_vllm_ports()

    def _set_vllm_ports(self, seed: Optional[int] = None):
        """
        The main process searches for self.distributed_state.num_processes's free ports.
        and then broadcasts the ports to all processes.
        """
        # if self.distributed_state.process_index == 0:
        ports = find_n_free_ports(
            self.distributed_state.num_processes, generator=self._port_generator_rng
        )
        logger.info(f"Found free ports: {ports}")

        self._vllm_port = ports[self.distributed_state.process_index]
        logger.info(
            f"Rank {self.distributed_state.process_index} using vLLM port {self._vllm_port}"
        )

    def _log_on_main(self, msg, level="info"):
        if self.is_main_process() and self._logger is not None:
            getattr(self._logger, level)(msg)

    def _init_orig_ds(self):
        ds = self.task.get_datasets(self.dataset_split)
        self._log_on_main(f"Initial Dataset Size: {len(ds)}")
        ds = self._filter_init_dataset(ds)

        self.initial_ds_after_filter_size = len(ds)

        self._orig_ds = ds
        if self.dataset_initial_size is not None:
            self._orig_ds = self._orig_ds.shuffle(self.seed).select(
                range(self.dataset_initial_size)
            )
            self._log_on_main(
                f"Dataset Size after initial selection: {len(self._orig_ds)}"
            )

        if not self.dataset_sample_with_replacement:
            # Create the dataset once on all processes
            self._log_on_main(
                f"Creating and caching dataset for once on all processes."
            )
            if self.total_num_iterations is None:
                if self.dataset_shuffle_on_each_iteration:
                    self._orig_ds = self._orig_ds.shuffle(seed=self.seed)
            else:
                # Compute the number of dataset repeats needed to cover all iterations
                dataset_size = len(self._orig_ds)
                samples_per_iteration = (
                    self.dataset_num_samples_per_iteration
                    if self.dataset_num_samples_per_iteration is not None
                    else int(dataset_size * self.dataset_portion)
                )

                num_repeats = (
                    self.total_num_iterations * samples_per_iteration // dataset_size
                )
                num_repeats += 1
                if num_repeats > 1:
                    self._log_on_main(
                        f"Repeating the dataset {num_repeats} times to cover all iterations."
                    )
                    if self.distributed_state.is_main_process:
                        new_ds = concatenate_datasets(
                            [
                                self._orig_ds.shuffle(seed=self.seed + i)
                                for i in range(num_repeats)
                            ]
                        )
                        new_ds.save_to_disk(self.temp_dir_root / "cached_dataset")
                        del new_ds
                        release_memory()
                    self._orig_ds = Dataset.load_from_disk(
                        str(self.temp_dir_root / "cached_dataset")
                    )
                else:
                    if self.dataset_shuffle_on_each_iteration:
                        self._orig_ds = self._orig_ds.shuffle(seed=self.seed)

    def _get_device_index(self) -> int:
        """Returns a valid device index either from self.distributed_state.device.index or using torch.cuda.current_device()."""
        device = self.distributed_state.device
        return device.index if device.index is not None else torch.cuda.current_device()

    def generate(
        self, iteration: Optional[int] = None, latest_policy_path: Optional[Path] = None
    ):
        """
        Generate episodes by sampling from the model.
        """
        # First, we'll log that we're entering the generate method with our rank (process_index)
        process_index = self.distributed_state.process_index
        logger.info(f"[RANK={process_index}] Entering generate method...")

        this_process_device = self.distributed_state.device
        release_memory()
        logger.info(
            f"[RANK={process_index}] Finished release_memory() after entering generate."
        )

        if self.vllm_min_available_gpu_memory_mb is not None:
            total_mem_mb = (
                torch.cuda.get_device_properties(this_process_device.index).total_memory
                / 1024**2
            )
            used_threshold_mb = total_mem_mb - self.vllm_min_available_gpu_memory_mb
            logger.info(
                f"[RANK={process_index}] Need at least {self.vllm_min_available_gpu_memory_mb} MB free. "
                f"Waiting for GPU{this_process_device.index} used memory to be below {used_threshold_mb} MB. "
                f"Total GPU memory: {total_mem_mb} MB."
            )
            wait_for_memory_release(
                this_process_device.index,
                threshold_mb=used_threshold_mb,
            )
            logger.info(f"[RANK={process_index}] Finished waiting for memory release.")

        device_index = self._get_device_index()
        gpu_memory_usage_before_mb = get_gpu_memory()[device_index]

        from deepspeed.runtime.utils import see_memory_usage

        see_memory_usage(
            f"[RANK={process_index}] Before generating episodes", force=True
        )
        if iteration is None:
            self._log_on_main(
                "Iteration is None. Using 0 as the iteration.", level="warning"
            )
            iteration = 0

        logger.info(f"[RANK={process_index}] iteration is set to {iteration}.")

        # Prepare the dataset on all processes
        logger.info(f"[RANK={process_index}] Checking if _orig_ds is None.")
        if self._orig_ds is None:
            logger.info(
                f"[RANK={process_index}] _orig_ds is None, entering main_process_first..."
            )
            with self.distributed_state.main_process_first():
                self._init_orig_ds()

        dataset = self._orig_ds
        if self.dataset_num_samples_per_iteration is not None:
            num_samples = self.dataset_num_samples_per_iteration
        else:
            num_samples = int(self.initial_ds_after_filter_size * self.dataset_portion)
            self._log_on_main(
                f"Using {num_samples} samples for each iteration based on the dataset portion."
            )
        assert num_samples <= len(dataset)

        if not self.dataset_sample_with_replacement:
            # Split the dataset into portions and select one portion based on the iteration
            samples_per_iteration = (
                self.dataset_num_samples_per_iteration
                if self.dataset_num_samples_per_iteration is not None
                else int(self.initial_ds_after_filter_size * self.dataset_portion)
            )
            start_idx = samples_per_iteration * iteration
            end_idx = samples_per_iteration * iteration + num_samples
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            # Shuffle the dataset so that the same dataset is not used in every iteration
            do_shuffle = (
                self.dataset_shuffle_on_each_iteration
                or self.dataset_shuffle_before_portion
            )
            if do_shuffle:
                logger.info(msg=f" SEED AND ITERATION, {self.seed, iteration}")
                dataset = dataset.shuffle(seed=self.seed + iteration)

            dataset = dataset.select(range(num_samples))

        self._log_on_main(
            f"Dataset Size(portion={self.dataset_portion}): {len(dataset)}"
        )
        self._log_on_main(
            f"Dataset Examples: "
            f"{json.dumps([dataset[i] for i in range(min(2, len(dataset)))], indent=2, sort_keys=True)}"
        )

        temp_dir = self.temp_dir_root / f"iteration__{iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save to disk so that it's memory efficient. Note that this is done on all processes.
        # to avoid any issues with distributed environment and funkiness of HF Datasets.
        inp_ds_path = temp_dir / f"input_dataset__{process_index}"
        dataset.save_to_disk(str(inp_ds_path))
        del dataset

        # The same dataset is loaded on all processes
        dataset = Dataset.load_from_disk(str(inp_ds_path))

        # Shard the dataset based on the number of processes
        dataset = dataset.shard(
            num_shards=self.distributed_state.num_processes,
            index=process_index,
            contiguous=True,
        )

        results_dir = temp_dir / "infer_results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create a seed based on self.seed, self.dist_state.process_index, and iteration
        seed = self.seed + process_index * 100 + iteration

        if latest_policy_path is None:
            hf_ckpt_path_or_model = self.initial_model_name_or_path
        else:
            hf_ckpt_path_or_model = str(latest_policy_path)

        vllm_init_fn = self._get_vllm_init_fn(
            results_dir=results_dir,
            hf_ckpt_path_or_model=hf_ckpt_path_or_model,
            process_index=process_index,
            seed=seed,
        )

        metrics = {}
        t0 = time.time()

        infer_results = self._run_inference(
            dataset_shard=dataset,
            vllm_init_fn=vllm_init_fn,
            results_root_dir=results_dir,
            iteration=iteration,
            gpu_memory_usage_before_mb=gpu_memory_usage_before_mb,
            seed=seed,
        )

        metrics["timing/episode_generation/inference"] = time.time() - t0
        logger.info(f"Process {process_index} finished inference.")

        t0 = time.time()
        # Generate episodes from inference results. Each process generates its own episodes.
        episodes = self._generate_episodes(infer_results, iteration)
        episodes_lst = [self._convert_to_dict(e) for e in episodes]

        episodes_ds_shard = Dataset.from_list(episodes_lst)
        episodes_ds_shard.save_to_disk(
            str(temp_dir / f"episodes" / f"shard_{process_index:02d}")
        )

        del episodes_ds_shard
        release_memory()
        metrics["timing/episode_generation/inferResult_to_episodes"] = time.time() - t0

        # Log the vLLM stats
        if self.distributed_state.is_main_process:
            try:
                vllm_stats = compute_vllm_stats(results_dir / "vllm_server.log")
            except Exception as e:
                logger.error(f"Error while computing vLLM stats: {e}")
                vllm_stats = {}

            if "avg_generation_throughput" in vllm_stats:
                vllm_stats["total_approx_generation_throughput"] = (
                    vllm_stats["avg_generation_throughput"]
                    * self.distributed_state.num_processes
                )

            vllm_stats = {f"vllm_stats/{k}": round(v, 2) for k, v in vllm_stats.items()}
            logger.info(f"vLLM Stats: {vllm_stats}")
            metrics.update(vllm_stats)

        self.cloud_log(metrics)

        # Concatenate all episodes shards
        self.distributed_state.wait_for_everyone()
        if self.is_main_process():
            shard_paths = list((temp_dir / f"episodes").glob("shard_*"))
            shard_paths.sort(key=lambda x: int(x.name.split("shard_")[-1]))

            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            if self.num_episodes_per_iteration is None:
                pass
            elif len(merged) > self.num_episodes_per_iteration:
                merged = merged.shuffle(seed=self.seed + iteration)
                merged = merged.select(range(self.num_episodes_per_iteration))
            elif len(merged) < self.num_episodes_per_iteration:
                if self.fill_missing_episodes:
                    # Fill the missing episodes by repeating the existing ones
                    logger.warning(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration}). "
                        f"Repeating the existing episodes."
                    )
                    num_repeats = self.num_episodes_per_iteration // len(merged) + 1
                    merged = concatenate_datasets([merged] * num_repeats)
                    merged = merged.shuffle(seed=self.seed + iteration)
                    merged = merged.select(range(self.num_episodes_per_iteration))
                    logs = {f"episodes_metric/fill_missing_episodes": num_repeats}
                    self.cloud_log({**logs, "train/global_iteration": iteration})
                else:
                    raise ValueError(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration})"
                    )

            merged.save_to_disk(str(temp_dir / "episodes" / "merged"))
            del merged
            release_memory()

        self.distributed_state.wait_for_everyone()
        episodes = Dataset.load_from_disk(str(temp_dir / "episodes" / "merged"))

        see_memory_usage("After generating episodes", force=True)

        self._save_generations_to_cloud(temp_dir, iteration)
        self._clean_up_temp_dir(temp_dir)

        self.distributed_state.wait_for_everyone()

        return episodes

    def _run_inference(
        self,
        dataset_shard: Dataset,
        vllm_init_fn: Callable[[], Tuple[VLLMServer, Dict[str, Any]]],
        results_root_dir: Path,
        iteration: int,
        gpu_memory_usage_before_mb: int,
        seed: int,
    ):
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

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        raise NotImplementedError

    def _get_vllm_init_fn(
        self,
        results_dir: Path,
        hf_ckpt_path_or_model: str,
        process_index: int,
        seed: int,
    ) -> Callable[[], Tuple[VLLMServer, Dict[str, Any]]]:
        vllm_gpu_memory_utilization = self.vllm_gpu_memory_utilization
        logger.info(f"Rank #{process_index} looking for vLLM ports with seed {seed}")

        self._set_vllm_ports(seed=seed)

        logger.info(
            f"Rank #{process_index} after _set_vllm_ports, using port {self._vllm_port}"
        )
        vllm_port = self._vllm_port

        if vllm_gpu_memory_utilization == "auto":
            # Compute the GPU utilization based on amount of remaining memory
            allocated_mem_mb = get_gpu_memory()[process_index]
            total_mem_mb = (
                torch.cuda.get_device_properties(process_index).total_memory / 1024**2
            )
            remaining_mem_mb = (
                total_mem_mb - allocated_mem_mb
            ) * 0.8  # Allow for tolerance
            vllm_gpu_memory_utilization = round(remaining_mem_mb / total_mem_mb, 2)
            logger.info(
                f"GPU #{process_index} Auto-computed vLLM GPU memory utilization: {vllm_gpu_memory_utilization}. "
                f"Allocated: {allocated_mem_mb} MB, Total: {total_mem_mb} MB, Remaining: {remaining_mem_mb} MB."
            )

        def _init() -> Tuple[VLLMServer, Dict[str, Any]]:
            vllm_log_path = results_dir / "vllm_server.log"
            logger.info(
                f"Rank #{process_index} starting vLLM: model={hf_ckpt_path_or_model} port={vllm_port} seed={seed}"
            )
            t0 = time.time()
            self.vllm_server = VLLMServer(
                seed=self.seed,
                port=vllm_port,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
            )
            server_url = self.vllm_server.start_server(
                hf_ckpt_path_or_model=hf_ckpt_path_or_model,
                gpu_idx=process_index,
                wait_for_response=True,
                log_path=vllm_log_path,
                timeout=800,
            )
            self.cloud_log(
                {
                    "timing/episode_generation/vllm_start": time.time() - t0,
                }
            )
            logger.info(f"SERVER URL {server_url}")
            return self.vllm_server, {
                "api_base": server_url,
                "model": hf_ckpt_path_or_model,
            }

        return _init

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

    def _save_generations_to_cloud(self, generations_dir: Path, iteration: int):
        if self.cloud_log is None or not self.is_main_process():
            return

        if self.save_generations_every_n_iteration is None:
            # Saving generations is disabled
            return

        # if iteration != 0 and iteration % self.save_generations_every_n_iteration != 0:
        #     # We only save generations every n iterations and the first iteration
        #     return

        temp_dir = Path(tempfile.mkdtemp())

        generations = temp_dir / f"iteration__{iteration:04d}.zip"
        shutil.make_archive(
            str(generations.with_suffix("")),
            format="zip",
            root_dir=generations_dir,
        )
        self.cloud_save(str(generations.absolute()), policy="now")

    def _filter_init_dataset(self, dataset: Dataset) -> Dataset:
        if self.max_question_length is None:
            return dataset

        tokenizer = self.tokenizer
        question_template = self.question_template
        max_question_length = self.max_question_length
        question_format_keys = []
        for column in dataset.column_names:
            if f"{{{column}}}" in self.question_template:
                question_format_keys.append(column)

        if len(question_format_keys) == 0:
            raise ValueError(
                "No columns found in the question template. "
                "Please add the column names in the question template."
            )

        def filter_out_long_questions(example):
            format_kwargs = {key: example[key] for key in question_format_keys}
            prompt = question_template.format(**format_kwargs)
            tokens = tokenizer(prompt).input_ids
            return len(tokens) <= max_question_length

        length_before = len(dataset)
        dataset = dataset.filter(
            filter_out_long_questions, num_proc=4, desc="Filtering long questions"
        )
        self._log_on_main(
            f"Filtered out {length_before - len(dataset)} long questions from {length_before} questions."
        )
        return dataset

    def _clean_up_temp_dir(self, temp_dir: Path) -> None:
        if not self.is_main_process():
            return

        try:
            # Remove all input_dataset__* directories
            for p in temp_dir.glob("input_dataset__*"):
                shutil.rmtree(p, ignore_errors=True)

            # Remove all episodes shards
            for p in (temp_dir / "episodes").glob("shard_*"):
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error while cleaning up temp dir: {e}")

    def _convert_to_dict(self, episode_obj) -> Dict[str, Any]:
        if isinstance(episode_obj, dict):
            return episode_obj

        return asdict(episode_obj)
