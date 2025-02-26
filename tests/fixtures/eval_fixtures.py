import pytest

from relign.eval.analyzer import BaseAnalyzer, TaskPerformanceAnalyzer
from relign.eval.evaluator import Evaluator


@pytest.fixture
def task_performance_analyzer(
    gsm8k,
    distributed_single_gpu,
):
    return TaskPerformanceAnalyzer(
        task=gsm8k, cloud_logger=None, distributed_state=distributed_single_gpu
    )


@pytest.fixture
def evaluator(
    task_performance_analyzer: BaseAnalyzer,
    experiment_dir,
    inference_pipeline,
    tokenizer,
    vllm_server,
    distributed_state_cpu_no_deepspeed,
):
    inference_pipeline_cls, inference_pipeline_kwrags = inference_pipeline
    return Evaluator(
        project_root_dir=experiment_dir,
        distributed_state=distributed_state_cpu_no_deepspeed,
        tokenizer=tokenizer,
        vllm_server=vllm_server,
        inference_pipeline_cls=inference_pipeline_cls,
        inference_pipeline_kwargs=inference_pipeline_kwrags,
        analyzers=[task_performance_analyzer],
        force_rerun=False,
        cloud_logger=None,
        every_n_checkpoints=1,
        task=task_performance_analyzer,
    )
