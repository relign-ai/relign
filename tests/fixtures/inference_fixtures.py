import pytest
from pathlib import Path

from relign.inference.cot_inference_strategy import COTInferenceStrategy
from relign.inference.tree_inference.expansion import EfficientIIDExpander
from relign.inference.tree_inference.answer_extraction import IdentityAnswerExtractor
from relign.inference.tree_inference.branch_factor_strategy import ListBranchFactor


@pytest.fixture
def identity_answer_extractor():
    """
    Returns an answer extractor for testing.
    """
    return IdentityAnswerExtractor(node_key_name="text")


@pytest.fixture
def list_branch_factor():
    """
    Returns a branch factor strategy for testing.
    """
    return ListBranchFactor(branch_factors=[{"depth": 0, "branch_factor": 1}])
    

# --- Node Expanders ---- #
@pytest.fixture
def efficient_iid_node_expander(tokenizer, list_branch_factor):
    program = """{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}"""
    return EfficientIIDExpander(
        branch_factor_strategy=list_branch_factor,
        program=program,
        program_kwargs={
            "temperature": 0.8,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stop": '"\n\n\nProblem:"',
        },
        node_text_template="{chain_of_thought}",
        tokenizer=tokenizer,
        model_context_size=2047,
    )


@pytest.fixture
def math_question_template():
    """
    Returns a question template for testing.
    """
    return """
        [MATH_TASK] Problem:
        {query}

        Solution:
    """ 


@pytest.fixture
def cot_inference_strategy(
    efficient_iid_node_expander,
    identity_answer_extractor,
    math_question_template,
    experiment_dir,
    math_mock_with_think_tags,
):
    """
    A fixture for chain-of-thought inference strategy (if needed).
    """
    guidance_llm = math_mock_with_think_tags
    
    return COTInferenceStrategy(
        guidance_llm=guidance_llm,
        node_expander=efficient_iid_node_expander,
        answer_extractor=identity_answer_extractor,
        question_template=math_question_template,
        question_field='query',
        samples=4, # Number of thought chains we rollout from a single node
        max_depth=2,
        result_dir=Path(experiment_dir) / "chain_of_thoughts",
        max_concurrent_generations=1,
        max_concurrent_programs=1,
    )