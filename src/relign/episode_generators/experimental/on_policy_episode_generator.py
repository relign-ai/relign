import json
import time
import uuid
import evaluate
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple

import numpy as np
import torch
import torch.cuda
from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets

from relign.episode_generators.experimental import BaseEpisodeGenerator, BaseEpisodeGeneratorArgs, Episode
from relign.utils.gpu import get_gpu_memory, wait_for_memory_release
from relign.common.vllm_server import compute_vllm_stats
from relign.utils.logging import get_logger
from relign.episode_generators.tree_episode_generator import TreeEpisodeUtils
from relign.utils.logging import get_logger 

logger = get_logger(__name__)


class OnPolicyEpisodeGenerator(BaseEpisodeGenerator, TreeEpisodeUtils):
    """
    Allow for a policy path (model weights) or
    pass down the current policy to infer from
    during episode generation.
    """

    def __init__(
        self,
        base_args: BaseEpisodeGeneratorArgs = None,
        initial_model_name_or_path: Optional[str] = None,
        reasoning_step_delimiter: Optional[str] = None,
        answer_prefix: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        tokenization_check_query_reconstruction: bool = True,
        tokenization_check_response_reconstruction: bool = True,
        **kwargs,
    ):

        # Ideally the base_args should be passed to the super class, but for backwards compatibility
        # if it is not provided we try to get the values from the kwargs
        if base_args is None:
            try:
                base_args_fields = {field.name for field in BaseEpisodeGeneratorArgs.__dataclass_fields__.values()}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_args_fields}
                base_args = BaseEpisodeGeneratorArgs(**filtered_kwargs)
            except Exception as e:
                raise ValueError(
                    f"Failed initializing a '{self.__class__.__name__}' object. Either provide a BaseEpisodeGeneratorArgs 
                    object or all the required arguments: {e}"
                )

        super().__init__(base_args)

        self.initial_model_name_or_path = initial_model_name_or_path
        self.max_sequence_length = max_sequence_length
        self.reasoning_step_delimiter = reasoning_step_delimiter
        self.answer_prefix = answer_prefix

        self.tokenization_check_query_reconst = tokenization_check_query_reconstruction
        self.tokenization_check_response_reconst = (
            tokenization_check_response_reconstruction
        )

        try:
            self._bleu_metric = evaluate.load(
                "bleu",
                experiment_id=uuid.uuid4().hex,
            )
        except Exception as e:
            self._bleu_metric = None


    #----------------- Abstract Function Implementations -----------------#

    def generate(
        self, 
        iteration: Optional[int] = None, 
        latest_policy_path: Optional[Path] = None
    ):
        """
        Generate episodes by sampling from the model.
        """
        this_process_device = self.distributed_state.device
        release_memory()

        if self.vllm_min_available_gpu_memory_mb is not None:
            total_mem_mb = (
                torch.cuda.get_device_properties(this_process_device.index).total_memory
                / 1024**2
            )
            used_threshold_mb = total_mem_mb - self.vllm_min_available_gpu_memory_mb
            logger.info(
                f"Need at least {self.vllm_min_available_gpu_memory_mb}. "
                f"Waiting for GPU{this_process_device.index} used memory to be below {used_threshold_mb} MB. "
                f"Total GPU memory: {total_mem_mb} MB."
            )
            wait_for_memory_release(
                this_process_device.index,
                threshold_mb=used_threshold_mb,
            )

        device_index = self._get_device_index()

        if device_index is not None:
            gpu_memory_usage_before_mb = get_gpu_memory()[device_index] 
            from deepspeed.runtime.utils import see_memory_usage
            see_memory_usage("Before generating episodes", force=True)
        else:
            gpu_memory_usage_before_mb = -1

        if iteration is None:
            self._log_on_main(
                "Iteration is None. Using 0 as the iteration.", level="warning"
            )
            iteration = 0

        process_index = self.distributed_state.process_index

        # Prepare the dataset on all processes
        if self._orig_ds is None:
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
        dataset.save_to_disk(inp_ds_path)
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
        
        if hf_ckpt_path_or_model is None:
            raise ValueError(f"({self.__class__.__name__}) Either initial_model_name_or_path or latest_policy_path must be provided.")

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
        )

        for result in infer_results:
            logger.info(f"INFER RESUTLS {result}")

        metrics["timing/episode_generation/inference"] = time.time() - t0
        logger.info(f"Process {process_index} finished inference.")

        t0 = time.time()
        # Generate episodes from inference results. Each process generates its own episodes.
        episodes = self._generate_episodes(infer_results, iteration)
        episodes_lst = [self._convert_to_dict(e) for e in episodes]

        print("episodes_lst", episodes_lst)
        episodes_ds_shard = Dataset.from_list(episodes_lst)
        episodes_ds_shard.save_to_disk(
            temp_dir / f"episodes" / f"shard_{process_index:02d}"
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

        self._cloud_log(metrics)

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
                    self._cloud_log({**logs, "train/global_iteration": iteration})
                else:
                    raise ValueError(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration})"
                    )

            merged.save_to_disk(temp_dir / "episodes" / "merged")
            del merged
            release_memory()

        self.distributed_state.wait_for_everyone()
        episodes = Dataset.load_from_disk(str(temp_dir / "episodes" / "merged"))

        see_memory_usage("After generating episodes", force=True)

        self._save_generations_to_cloud(temp_dir, iteration)
        self._clean_up_temp_dir(temp_dir)

        self.distributed_state.wait_for_everyone()

        return episodes


    #----------------- Helper Functions -----------------#

    def _generate_episodes(
        self, 
        inference_results: Dataset, 
        iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        episodes = []
        metrics = {}
        for instance in inference_results:
            tree = json.loads(instance["_treetune__reasoning_tree"])
            paths = self.extract_paths_from_tree(tree)

            all_rewards = []
            all_responses = []
            for path in paths:

                assert len(path["node_chain"]) == 2, "Does not support multi-hop paths."

                finish_reason = path["node_chain"][-1]["finish_reason"]
                query_text = path["node_chain"][0]["text"]
                full_text = path["node_chain"][-1]["full_text"]
                response_text = full_text[len(query_text) :]

                # Compute the number of reasoning steps in response 
                try:
                    num_reasoning_steps = self._compute_number_of_reasoning_steps(
                        response_text
                    )
                    metrics.setdefault("num_reasoning_steps", []).append(
                        num_reasoning_steps
                    )
                    metrics.setdefault("parse_failed", []).append(False)
                except Exception as e:
                    metrics.setdefault("parse_failed", []).append(True)

                if finish_reason != "length":
                    # Generation stopped because the model hit <eos>
                    reward, is_unfinished_response = self.task.reward(
                        query_text, response_text, instance
                    )
                else:
                    # Generation stopped because the model hit the `max_tokens` limit
                    reward = self.task.get_unfinished_response_penalty()
                    is_unfinished_response = True

                try:
                    query_token_ids, response_token_ids, offsets = (
                        self._tokenize_query_and_response(
                            query_text,
                            response_text,
                            # Only append EOS token if the response is complete
                            allow_append_eos=not is_unfinished_response
                        )
                    )
                except Exception as e:
                    logger.info(f"Failed to tokenize query and response: {e}")
                    metrics.setdefault("empty_response", []).append(True)
                    continue

                all_responses.append(response_text)

                if self.max_sequence_length is not None:
                    seq_len = len(query_token_ids) + len(response_token_ids)
                    if seq_len > self.max_sequence_length:
                        # Truncate the response
                        response_token_ids = response_token_ids[
                            : self.max_sequence_length - len(query_token_ids)
                        ]
                        reward = self.task.get_unfinished_response_penalty()
                        is_unfinished_response = True

                if len(response_token_ids) == 0:
                    logger.info('empty response')
                    metrics.setdefault("empty_response", []).append(True)
                    continue

                metrics.setdefault("empty_response", []).append(False)
                metrics.setdefault("is_unfinished_response", []).append(
                    is_unfinished_response
                )

                # Generate an episode
                episode = Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    reward=float(reward),
                )

                episodes.append(episode)
                all_rewards.append(float(reward))

            if len(all_rewards) > 0:
                once_hit = any([r == 1.0 for r in all_rewards])
                metrics.setdefault("once_hit", []).append(float(once_hit))

            if len(all_responses) > 1:
                metrics.setdefault("num_unique_responses", []).append(
                    len(set(all_responses))
                )
                if self._bleu_metric is not None:
                    bleu = self._avg_bleu_of_pairs_of_response(all_responses)
                    metrics.setdefault("trajectory_bleu", []).append(bleu)


        self._update_metrics(metrics, iteration)

        return episodes
    

    def _update_metrics(self, metrics: Dict, iteration: int): 

        if "is_unfinished_response" in metrics:
            metrics["is_unfinished_response"] = sum(
                metrics["is_unfinished_response"]
            ) / len(metrics["is_unfinished_response"])

        if "empty_response" in metrics:
            metrics["empty_response"] = sum(metrics["empty_response"]) / len(
                metrics["empty_response"]
            )

        if "num_reasoning_steps" in metrics:
            num_reasoning_steps = np.array(metrics.pop("num_reasoning_steps"))
            metrics["num_reasoning_steps/dist"] = num_reasoning_steps
            metrics["num_reasoning_steps/mean"] = np.mean(num_reasoning_steps)

        if "parse_failed" in metrics:
            metrics["parse_failed"] = sum(metrics["parse_failed"]) / len(
                metrics["parse_failed"]
            )

        if "once_hit" in metrics:
            metrics["once_hit"] = sum(metrics["once_hit"]) / len(metrics["once_hit"])

        if "trajectory_bleu" in metrics:
            metrics["trajectory_bleu"] = sum(metrics["trajectory_bleu"]) / len(
                metrics["trajectory_bleu"]
            )

        if len(metrics) > 0:
            logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
            self._cloud_log({**logs, "train/global_iteration": iteration})


    def _compute_number_of_reasoning_steps(self, response_text: str) -> int:

        """
        TODO: This used to call the task's split_solution_into_intermediate_steps method,
        but the number of reasoning steps depends on the algorithm (i.e., it is different for
        PPO and GRPO), so it should be determined differently. How to do this is not clear yet.
        """  
        return -1
        # indices = self.task.split_solution_into_intermediate_steps(response_text)
        # return len(indices) - 1, indices


    def _tokenize_query_and_response(
        self, 
        query: str, 
        response: str, 
        allow_append_eos: bool = True,
        return_offsets: bool = False
    ) -> Tuple[List[int], List[int]]:

        trajectory = {"query_text": query, "response_text": response}

        safety_check_kwargs = safety_check_kwargs or {}
        query_text = trajectory["query_text"]
        response_text = trajectory["response_text"]

        episode_text = f"{query_text}{response_text}"
        episode_encoding = self.tokenizer(
            episode_text,
            add_special_tokens=False,  # We will add BOS and EOS tokens at the end
            return_offsets_mapping=True,
        )

        token_ids = episode_encoding["input_ids"]
        offsets = episode_encoding["offset_mapping"]

        response_start_index = next(
            i for i, (start, end) in enumerate(offsets) if start >= len(query_text)
        )
        query_token_ids = token_ids[:response_start_index]
        response_token_ids = token_ids[response_start_index:]

        self._safety_check_tokenization(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            query=query_text,
            response=response_text,
            episode_text=episode_text,
            **safety_check_kwargs,
        )

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids

        if allow_append_eos and self._should_append_eos_to_response():
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]

        if return_offsets:
            return query_token_ids, response_token_ids, offsets
        else:
            return query_token_ids, response_token_ids

    
    def _safety_check_tokenization(
        self,
        query_token_ids: List[str],
        response_token_ids: List[str],
        query: str,
        response: str,
        episode_text: str,
        check_query_reconstruction: bool = True,
        check_response_reconstruction: bool = True,
    ):
        """
        TODO: Not sure what the use of this function is, maybe to check if the tokenizer 
        throws an error? Maybe consider removing this or making is more useful. 
        """

        decoding_kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }

        decoded_instance = self.tokenizer.decode(
            query_token_ids + response_token_ids, **decoding_kwargs
        )

        
    def _avg_bleu_of_pairs_of_response(self, response: List[str]) -> float:
        preds = []
        refs = []
        for i in range(len(response)):
            for j in range(i + 1, len(response)):
                sen_1 = response[i]
                sen_2 = response[j]
                preds.append(sen_1)
                refs.append(sen_2)
        bleu_full_stats = self._bleu_metric.compute(predictions=preds, references=refs)
        bleu = bleu_full_stats["bleu"]
        return bleu
    