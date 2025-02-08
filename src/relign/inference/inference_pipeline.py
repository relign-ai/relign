import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

from datasets import Dataset
from wandb.sdk.wandb_run import Run

from relign.common.types import JsonDict

# from relign.common.py_utils import need_to_minimize_stored_files
from relign.inference.base_inference_strategy import InferenceStrategy
from relign.tasks.base_task import Task

from relign.utils.logging import get_logger


logger = get_logger(__name__)


class InferencePipeline:
    pass


class VLLMInferencePipeline(InferencePipeline):
    def __init__(
        self,
        inference_strategy_cls: InferenceStrategy,
        inference_strategy_kwargs: List[JsonDict],
        task: Task,
        dataset_split: str,
        inference_name: Optional[str] = None,
        exp_root_dir: Optional[Path] = None,
        dataset_portion: float = 1.0,
        dataset_shuffle_before_portion: bool = False,
        dataset_num_shards: int = 1,
        dataset_shard_index: int = 0,
        debug_mode: bool = True,
        cloud_logger: Optional[Run] = None,
        use_cache: Optional[bool] = None,
        enable_cloud_logging_during_inference: bool = True,
        seed: int = 42,
        metrics_prefix: str = "",
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        checkpoint_global_step: Optional[int] = None,
    ):
        """
        Params:
            exp_root (Path):
                The root directory of the experiment.
            inference_strategy (InferenceStrategy):
                The inference strategy to use
            task (Task):
                The task to use
            dataset_split (str):
                The dataset split from task which will be used for inference.
                It should match the splits provided by the task.
            dataset_portion (float):
                The portion of the dataset to use for inference.
                Useful for debugging.
            dataset_shuffle_before_portion (bool):
                Whether to shuffle the dataset before taking the portion.
            dataset_num_shards (int):
                The number of shards to split the dataset into.
                This is useful for parallelizing inference across multiple jobs.
            dataset_shard_index (int):
                The index of the shard to use for this job.
                This is useful for parallelizing inference across multiple jobs.
            analyzers (List[JsonDict]):
                A list of analyzers to use.
            debug_mode (bool):
                Whether to run in debug mode.
            use_cache (bool):
                Whether to use cache.
            enable_cloud_logging_during_inference (bool):
                Whether to enable cloud logging during inference.
            seed (int):
                The seed to use.
            metrics_prefix (str):
                The prefix to use for metrics.
        """
        self.task = task
        self.exp_root_dir = exp_root_dir
        self.dataset_split = dataset_split
        self.dataset_portion = dataset_portion
        self.dataset_shuffle_before_portion = dataset_shuffle_before_portion
        self.dataset_num_shards = dataset_num_shards
        self.dataset_shard_index = dataset_shard_index
        self.debug_mode = debug_mode
        self.cloud_logger = cloud_logger
        self.seed = seed
        self.metrics_prefix = metrics_prefix
        self.inference_name = inference_name
        self.checkpoint_global_step = checkpoint_global_step
        self.model = model
        self.inference_strategy_cls = inference_strategy_cls
        self.inference_strategy_kwrags = inference_strategy_kwargs
        self.api_base = api_base

        if self.inference_name is not None:
            self.metrics_prefix = f"{self.inference_name}/{self.metrics_prefix}"

        if self.exp_root_dir is None:
            # Create a tmp directory
            self.exp_root_dir = Path("/tmp") / next(tempfile._get_candidate_names())
            self.exp_root_dir.mkdir(parents=True, exist_ok=True)

        if self.debug_mode:
            logger.info("Debug mode is on. Using 10 examples from the dataset.")
            dataset_len = len(self.task.get_datasets(self.dataset_split))
            self.dataset_portion = 10 / dataset_len

        # Unique identifier for this inference job.
        self.inference_job_id = f"{self.dataset_split}__{self.dataset_shard_index}__{self.dataset_num_shards}"
        logger.info(
            f"Inference on {task.name} (split__shard__num_shards): {self.inference_job_id}"
        )

        inference_strategy_kwargs = {"result_dir": self.get_result_dir()}
        if enable_cloud_logging_during_inference:
            inference_strategy_kwargs["cloud_logger"] = self.cloud_logger

        # Point the llm_guidance to the right serveri
        logger.info(f"Using inference strategy: {inference_strategy_kwargs}")
        self.inference_strategy = inference_strategy_cls(**inference_strategy_kwargs)

        if self.api_base is not None:
            self.inference_strategy.init_guidance_llm(
                **{
                    "api_base": self.api_base,
                    "model": self.model,
                }
            )

    def get_result_dir(self) -> Path:
        if hasattr(self, "_cached_results_dir"):
            return self._cached_results_dir

        result_dir = (
            self.exp_root_dir / "eval_inference_results" / self.inference_job_id
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def _get_result_dir(self) -> Path:
        return self.get_result_dir()

    def generate(self):
        dataset = self.task.get_datasets(self.dataset_split)
        logger.info(f"Original Dataset size: {len(dataset)}")

        if self.dataset_portion < 1.0:
            if self.dataset_shuffle_before_portion:
                dataset = dataset.shuffle(seed=42)
            dataset = dataset.select(range(int(len(dataset) * self.dataset_portion)))
            logger.info(
                f"Portion {self.dataset_portion} of Dataset size: {len(dataset)}"
            )

        logger.info(
            f"Dataset Examples: "
            f"{json.dumps([dataset[i] for i in range(min(5, len(dataset)))], indent=2, sort_keys=True)}"
        )

        dataset = dataset.shard(
            self.dataset_num_shards, self.dataset_shard_index, contiguous=True
        )
        logger.info(
            f"Sharded Dataset size: {len(dataset)} "
            f"(shard {self.dataset_shard_index + 1} / {self.dataset_num_shards})"
        )

        results = self.inference_strategy.generate(dataset)
        return results

    def save_results_to_cloud(self, results: Dataset):
        output_dir = self._get_result_dir()
        results.save_to_disk(output_dir)

        # First, create a zip file of the inference results into output_dir/inference_results.zip
        # This is because the cloud logger only accepts files.
        temp_dir = Path(tempfile.mkdtemp())
        inference_results_zip = (
            temp_dir / f"{self.metrics_prefix.replace('/', '__')}inference_results.zip"
        )
        logger.info(f"Creating zip file {inference_results_zip}")
        shutil.make_archive(
            str(inference_results_zip.with_suffix("")), "zip", output_dir
        )
        # Then, upload the zip file to the cloud.
        self.cloud_logger.save(str(inference_results_zip.absolute()), policy="now")


InferencePipeline.default_implementation = "vllm"
