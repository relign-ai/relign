from relign.utils.logging import get_logger


logging = get_logger(__name__)


class TestEval:
    def test_evaluation_pipeline(
        self,
        tokenizer,
        evaluator,
        actor_critic_policy,
    ):
        actor_critic_policy.init_actor_engine_if_needed()
        actor_critic_policy.init_critic_engine_if_needed()
        # latest_policy_path = actor_critic_policy.save_latest_policy_path()
        latest_policy_path = "tests/output/tests_test_eval.py_TestEval_test_evaluation_pipeline/policy/cache/actor/hf_pretrained"
        logging.info("start test evaluation")
        eval_results = evaluator.evaluate(
            iteration=1,
            tokenizer=tokenizer,
            latest_policy_path=latest_policy_path,
        )

        assert eval_results is not None

    def test_evaluation_analyzer(self): ...
