import random
import shutil
import tempfile
import torch
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
from abc import ABC
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod

import torch.cuda
import wandb
from accelerate.utils import release_memory
from accelerate import PartialState
from datasets import Dataset, concatenate_datasets

from relign.inference.base_inference_strategy import InferenceStrategy
from relign.tasks.base_task import BaseTask
from relign.utils.gpu import get_gpu_memory, wait_for_memory_release
from relign.utils.py_utils import find_n_free_ports
from relign.common.vllm_server import VLLMServer
from relign.inference.base_inference_strategy import InferenceStrategy
from relign.common.vllm_server import VLLMServer
from relign.utils.py_utils import find_n_free_ports
from relign.tasks.base_task import BaseTask
from relign.common.dataset import EpisodeDataset
from relign.utils import logging
from relign.tokenization.base_tokenizer import Tokenizer

logger = logging.get_logger(__name__)


@dataclass
class Episode:
    """
    A single episode.
    """
    query_token_ids: List[int]
    response_token_ids: List[int]
    reward: float # Final reward of the episode 
    process_rewards: Optional[List[float]] = None # Process rewards for each token in the response
    advantages: Optional[List[float]] = None # Advantages for each token in the response
    group: Optional[int] = None # GRPO for grouped advantages/rewards

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.response_token_ids) > 0
        assert self.reward is not None

        if self.advantages is not None:
            assert len(self.advantages) == len(
                self.response_token_ids
            ), "advantages have to be the same length as the response token ids"
        
        
        if self.process_rewards is not None:
            if len(self.process_rewards) != len(self.response_token_ids):
                print(f"[DEBUG] Token count mismatch: process_rewards length = {len(self.process_rewards)}; response_token_ids length = {len(self.response_token_ids)}")
            assert len(self.process_rewards) == len(
                self.response_token_ids
            ), "process_rewards have to be the same length as the response token ids" 


@dataclass
class BaseEpisodeGeneratorArgs:
    """
    TODO: Document the use and effect of each argument
    """
    task: BaseTask
    tokenizer: Tokenizer
    vllm_server: VLLMServer
    inference_strategy: InferenceStrategy

    policy_path: Optional[str] = None 
    vllm_gpu_memory_utilization: Union[float, str] = 0.9
    vllm_min_available_gpu_memory_mb: Optional[int] = None
    wait_until_memory_release: bool = False
    dataset_shuffle_on_each_iteration: bool = True
    dataset_shuffle_before_portion: bool = True
    dataset_split: str = "train"
    dataset_portion: Optional[float] = None
    dataset_num_samples_per_iteration: Optional[int] = None
    dataset_sample_with_replacement: bool = True
    dataset_initial_size: Optional[int] = None
    total_num_iterations: Optional[int] = None
    temp_dir_root: Optional[str] = None
    fill_missing_episodes: bool = False
    max_question_length: Optional[int] = None
    question_template: Optional[str] = None
    save_generations_every_n_iteration: Optional[int] = None
    debug: bool = False
    append_bos_to_query: Union[str, bool] = "auto",
    append_eos_to_response: Union[str, bool] = "auto",
    distributed_state: PartialState = None
    num_episodes_per_iteration: int = None
    project_root_dir: Optional[Path] = None
    cloud_logger: Optional[Any] = None
    seed: int = 42
    

class BaseEpisodeGenerator(ABC):
    can_precompute_episodes: bool = False
    supports_distributed: bool = False

    def __init__(
        self,
        args: BaseEpisodeGeneratorArgs,
    ):

        self._logger = logger

        for k, v in asdict(args).items():
            setattr(self, k, v)

        if self.project_root_dir is not None:
            self._init_episode_dir()

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

        if self.temp_dir_root is None:
            self.temp_dir_root = self.project_root_dir / "temp_episodes"
            self._log_on_main(f"Using default temp_dir_root: {self.temp_dir_root}")
        else:
            self.temp_dir_root = Path(self.temp_dir_root)
        self.temp_dir_root.mkdir(parents=True, exist_ok=True)

        self._orig_ds = None

        self._vllm_port = None
        self._port_generator_rng = random.Random(self.seed)
        self._set_vllm_ports()


    # ----------------- Abstract Functions -----------------#

    @abstractmethod
    def generate(
        self, 
        iteration: Optional[int] = None,
        latest_policy_path: Optional[Path] = None
    ) -> Union[List[Episode], Dataset]:
        """
        TODO: Document this function
        """
        ...


    # ----------------- Public Functions -----------------#
            
    def set_policy_path(self, policy_path: str) -> None:
        self.policy_path = policy_path


    def get_episode_checkpoint_path(self, iteration: int) -> Path:
        """
        Sets the checkpoint directory based on the iteration 
        """
        if self.episodes_checkpoint_dir is None:
            raise ValueError("episodes_checkpoint_dir is not set")

        return (self.episodes_checkpoint_dir / f"episodes_{str(iteration).zfill(4)}.json")


    def checkpoint_episodes(self, episodes: EpisodeDataset, iteration: int) -> Path:
        """
        Saves the episodes to disk
        """
        assert isinstance(episodes, EpisodeDataset), f"episodes have to be of type EpisodeDataset, not {type(episodes)}"
        checkpoint_path = self.get_episode_checkpoint_path(iteration)
        episodes.save_to_disk(checkpoint_path) 
        return checkpoint_path


    def is_main_process(self) -> bool:
        return self.distributed_state.is_main_process
    

    def log_episodes(
        self,
        episodes: Union[List[Episode], Dataset],
        iteration_idx: int,
        num_examples: int = 100,
        num_examples_for_wandb: int = 128,
        seed: int = 42,
        log_to_cloud: bool = True,
    ):
        if not self.is_main_process():
            return

        table = wandb.Table(
            columns=[
                "idx",
                "query",
                "response",
                "query_tokens",
                "response_tokens",
                "advantages",
                "reward",
                "instance_length",
            ]
        )

        logger.info(f"Logging {num_examples} examples:")
        rng = random.Random(seed)

        num_console_logs = min(num_examples, len(episodes))
        num_wandb_logs = min(num_examples_for_wandb, len(episodes))
        indices = rng.sample(range(len(episodes)), num_wandb_logs)

        for idx in indices:
            episode = episodes[idx]
            if not isinstance(episode, dict):
                episode = asdict(episode)

            query_token_ids = episode["query_token_ids"]
            response_token_ids = episode["response_token_ids"]
            reward = episode["reward"]

            query_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in query_token_ids
            ]
            query = self.tokenizer.decode(query_token_ids)

            response_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in response_token_ids
            ]
            response = self.tokenizer.decode(response_token_ids)

            advantages = episode.get("advantages")
            instance_length = len(query_token_ids) + len(response_token_ids)

            table.add_data(
                idx,
                query,
                response,
                ", ".join(query_tokens),
                ", ".join(response_tokens),
                ", ".join(
                    [str(a) for a in advantages] if advantages is not None else []
                ),
                reward,
                instance_length,
            )

            if len(table.data) >= num_console_logs:
                continue

            logger.info(f"Example {idx}")
            for k, v in episode.items():
                logger.info(f"{k}: `{v}`")
            logger.info(f"Query: `{query}`")
            logger.info(f"Response: `{response}`")
            logger.info(f"Instance Length: {instance_length}")
            logger.info(f"Reward = Scores: {reward}")

            if advantages is not None:
                # Log aligned advantages with response tokens
                logger.info("Advantages:")
                for i, (adv, tok) in enumerate(zip(advantages, response_tokens)):
                    logger.info(f"{str(i).zfill(4)}: {tok:<20} -> {adv}")

            logger.info("-" * 100)

        if log_to_cloud and self.cloud_logger is not None:
            self.cloud_logger.log({f"episodes/iteration_{iteration_idx:04}": table})


    #----------------- Internal Helper Functions -----------------# 

    def _init_episode_dir(self):
        """
        Instantiate the working directory for the episodes checkpoints
        """
        self.episodes_checkpoint_dir = (self.project_root_dir / "episodes")
        print("episode generator root dir", self.episodes_checkpoint_dir)
        self.episodes_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    
    def _should_append_bos_to_query(self) -> bool:
        """
        Determine whether to append BOS to the query based on the tokenizer
        """

        if self.append_bos_to_query == "auto":
            return self.append_bos_to_query

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.bos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append BOS for tokenizer {self.tokenizer.name_or_path}"
            )
    

    def _should_append_eos_to_response(self) -> bool:
        """
        Determine whether to append EOS to the response based on the tokenizer
        """

        if self.append_eos_to_response == "auto":
            return self.append_eos_to_response

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.eos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append EOS for tokenizer {self.tokenizer.name_or_path}"
            )
    
    
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
        if not torch.cuda.is_available():
            return None

        device = self.distributed_state.device
        return device.index if device.index is not None else torch.cuda.current_device()

    
    def _run_inference(
        self,
        dataset_shard: Dataset,
        vllm_init_fn: Callable[[], Tuple[VLLMServer, Dict[str, Any]]],
        results_root_dir: Path,
        iteration: int,
        gpu_memory_usage_before_mb: int,
    ):
        """
        Potentially start a vLLM server and run inference to generate results needed for episode generation.

        Args:
            dataset_shard (Dataset):
                The shard of the prompt dataset to run inference on.
            vllm_init_fn (Callable[[], Tuple[VLLMServer, Dict[str, Any]]]):
                A function that initializes the vLLM server and returns the server object and the server URL.
            results_root_dir (Path):
                The directory to save the results to (this is unique for each process).
            seed (int):
                The seed for this process to use for inference.
        """
        infer_result_path = results_root_dir / "results_ds"
        vllm_server, guidance_llm_kwargs = vllm_init_fn()
        target_device_index = self._get_device_index()

        results = self.inference_strategy.generate(dataset_shard)
         
        # Convert the results to a list before saving 
        episodes = []
        for row in results:
            episode = dict(row)  # copy all fields from row
            episode["iteration"] = iteration
            episodes.append(episode)

        episode_ds = Dataset.from_list(episodes)
        episode_ds.save_to_disk(str(infer_result_path))

        vllm_server.stop_server()
        del results
        del vllm_server
        release_memory()

        self.vllm_cleanup(
            target_gpu_index=target_device_index, # or can i just use self.distributed_state.device.index here? even if there are multiple ?
            gpu_memory_usage_before_mb=gpu_memory_usage_before_mb,
        )
        release_memory()

        results = Dataset.load_from_disk(str(results_root_dir / "results_ds"))
        return results

    
    def _get_vllm_init_fn(
        self,
        results_dir: Path,
        hf_ckpt_path_or_model: str,
        process_index: int,
        seed: int,
    ) -> Callable[[], Tuple[VLLMServer, Dict[str, Any]]]:
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
    
    
    def _save_generations_to_cloud(self, generations_dir: Path, iteration: int):
        if self.cloud_logger is None or not self.is_main_process():
            return

        if self.save_generations_every_n_iteration is None:
            # Saving generations is disabled
            return

        if iteration != 0 and iteration % self.save_generations_every_n_iteration != 0:
            # We only save generations every n iterations and the first iteration
            return

        temp_dir = Path(tempfile.mkdtemp())

        generations = temp_dir / f"iteration__{iteration:04d}.zip"
        shutil.make_archive(
            str(generations.with_suffix("")),
            format="zip",
            root_dir=generations_dir,
        )
        self.cloud_logger.save(str(generations.absolute()), policy="now")

        
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


    def _cloud_log(self, *args, **kwargs):
        if self.is_main_process() and self.cloud_logger is not None:
            self.cloud_logger.log(*args, **kwargs)


    def _vllm_cleanup(
        self,
        gpu_memory_usage_before_mb: int,
        target_gpu_index: int,
    ):
        if self.wait_until_memory_release:
            threshold_mb = ( gpu_memory_usage_before_mb * 1.1)  # Allow for 10% tolerance
            wait_for_memory_release(
                target_gpu_index=target_gpu_index,
                threshold_mb=threshold_mb,
            )
    
    