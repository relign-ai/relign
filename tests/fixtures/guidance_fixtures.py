import pytest

from relign.guidance.llms._mock import Mock


@pytest.fixture
def mock_guidance():
    return Mock()


@pytest.fixture
def math_mock_with_think_tags():
    # Your example solution strings with reasoning steps and final answers.
    example_solutions = [
        "<think> First, we consider the problem 15 + 27. </think>\n"
        "<think> Adding 15 and 27 gives us 42. </think>\n"
        "#### 42",
        "<think> Compute 8 multiplied by 7. </think>\n"
        "<think> The product of 8 and 7 is 56. </think>\n"
        "#### 56",
        "<think> Evaluate the expression 100 minus 45. </think>\n"
        "<think> Subtracting 45 from 100 yields 55. </think>\n"
        "#### 55",
        "<think> First, divide 9 by 3 to get 3. </think>\n"
        "<think> Then, multiply the result by 5. </think>\n"
        "<think> Multiplying 3 by 5 gives 15. </think>\n"
        "#### 15",
        "<think> First, add 3.5 and 2.5 to obtain 6.0. </think>\n"
        "<think> Then, multiply 6.0 by 2 to arrive at the final result. </think>\n"
        "#### 12.0",
    ]

    # Instantiate the Mock LLM.
    return Mock(output={"": example_solutions})
