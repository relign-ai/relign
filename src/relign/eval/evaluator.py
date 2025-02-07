from pathlib import Path
from typing import Optional, Callable, List, Dict
from relign.inference.inference_pipeline import InferencePipeline 
from relign.eval.analyzer import Analyzer

from relign.utils.logging import get_logger

logger = get_logger(__name__)

class Evaluator:
    def  __init__(
        self,
        inference_pipeline_cls: InferencePipeline,
        inference_pipeline_kwargs : Optional[Dict],
        project_root_dir: Path,
        force_rerun: bool = False,
        every_n_checkpoints: int = 1,
        cloud_logger: Optional[Callable] = None,
        analyzers = Optional[List[Analyzer]]
    ):
        """ 
        Base Evaluator class of reinfrocement learning algorithms. 
        Runs an evaluation on the model located in the latest policy path, 
        TODO: or on all checkpoints if on checkpoint is true
        """
        self.project_root_dir = project_root_dir
        self._init_evaluation_dir()
        self.force_rerun = force_rerun
        self.every_n_checkpoints = every_n_checkpoints
        self.analyzers = analyzers
        self.cloud_logger = cloud_logger
        self.inference_pipeline_cls = inference_pipeline_cls
        self.inference_pipeline_kwargs = inference_pipeline_kwargs

    def _init_evaluation_dir(self):
        evaluation_dir = self.project_root_dir / "evals"
        evaluation_dir.mkdir(exist_ok=True, parents=True)

    def evaluate(
        self,
        tokenizer,
        seed: int = 69,
        latest_policy_path: Optional[Path] = None,
        from_checkpoints: bool = False,
    ):
        """
        Main evaluation loop. Evaluates the latest policy path.
        """
        vllm_server = self._get_evaluation_vllm_server()
        if not from_checkpoints:
            model_cpt = latest_policy_path 
        else:
            raise NotImplementedError("Evaluation from checkpoints is not yet implemented")

        eval_dir_path = Path(self.evaluation_dir  / model_cpt)
        vllm_ckpt_dir = self.prepare_for_vllm(latest_policy_path)

        try: 
            server_url = vllm_server.start_server(
                hf_ckpt_path_or_model=vllm_ckpt_dir,
                wait_for_response=True,
                timeout=800,
            )

            infer_pipeline = InferencePipeline(
                self.inference_strategy,
                task=self.task,
                tokenizer=tokenizer,
                seed=seed,
                api_base_url=server_url,
                use_cache=True,
                cloud_logger=self.cloud_logger,
                exp_root=self.project_root_dir,
            )

            results = infer_pipeline.generate()
            logger.info(f"Evaluation results: {results}")

        finally:
            # Stop the server
            vllm_server.stop_server()

        # if self.analyzers is not None:
        #     for analyzer in self.analyzers:
        #         analysis = analyzer.analyze(results)
        #         self._log_analysis_metrics(analysis)


    def _log_analysis_metrics(self, analysis):
        pass


    def _get_evaluation_vllm_server():
        pass