import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

from relign.utils import logging
from relign.tasks import Task

logger = logging.get_logger(__name__)

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)


def remove_text_between_symbols(text, start_symbol, end_symbol):
    pattern = f"{re.escape(start_symbol)}.*?{re.escape(end_symbol)}"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text


class GSM8K(Task):
    def __init__(
        self,
        use_original_format: bool = False,
        remove_calculator_expressions: bool = True,
        intermediate_step_delimiter: Optional[str] = "\n",
        intermetdiate_step_tags: Optional[Tuple[str, str]] = None,
        match_single_think_tag: bool = True,
        answer_prefix: Optional[str] = "\n#### ",
        reward_on_last_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.remove_calculator_expressions = remove_calculator_expressions
        self.use_original_format = use_original_format
        self.answer_prefix = answer_prefix
        self.intermediate_step_delimiter = intermediate_step_delimiter
        self.intermediate_step_tags = intermetdiate_step_tags
        self.match_single_think_tag = match_single_think_tag
        self.reward_on_last_token = reward_on_last_token

    def extract_predicted_answer_from_text(
        self, text: str, problem: Optional[str] = None
    ) -> Optional[str]:
        if self.use_original_format:
            # Extract the final answer based on ####
            if "####" not in text:
                return None
            parts = text.split("####")
            assert len(parts) >= 2
            return parts[-1].strip()

        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip()
            return pred_answer

    def extract_gold_answer_from_text(self, text: str) -> str:
        return text.split("####")[1].strip()

    def get_format_rewards(self, solution: str) -> int:
        """
        Check for proper tag-based formatting and return 1 if the format criteria are met, else 0.
        A "well-formatted" answer must:
          1) Contain exactly 1 opening <think> and 1 closing </think>.
          2) The closing </think> must come *after* the opening <think>.
          3) After the </think> (plus optional whitespace/newlines), the first non-whitespace
             characters must be the answer prefix (e.g. "#### ").
        """
        if self.intermediate_step_tags is not None and self.match_single_think_tag:
            import re

            open_tag = self.intermediate_step_tags[0]
            close_tag = self.intermediate_step_tags[1]

            # 1) Ensure exactly one <think> and one </think>.
            if solution.count(open_tag) != 1 or solution.count(close_tag) != 1:
                return 0

            # 2) Ensure the single </think> comes *after* the single <think>.
            open_idx = solution.index(open_tag)
            close_idx = solution.index(close_tag)
            if close_idx < open_idx:
                return 0

            # 3) Check that the content after </think> has only optional whitespace/newlines
            #    before the expected answer prefix (e.g. #### ).
            suffix = solution[close_idx + len(close_tag) :]
            if not re.match(r'^[\s]*#### ', suffix.lstrip()):
                return 0

            # If the criteria are met, return the reward.
            return 1

        return 0

    def split_solution_into_intermediate_steps(self, solution: str) -> List[int]:
        """
        Split the solution into reasoning steps.

        Args:
            solution: The full solution text.

        Returns:
            A list of indices where each index corresponds to the start of a reasoning step.
        """
        assert self.use_original_format, "This method is only for original format"
        assert self.intermediate_step_delimiter is not None

        delimiter = self.intermediate_step_delimiter
        answer_prefix = self.answer_prefix

        # Attempt to split out the final answer if answer_prefix is provided
        if answer_prefix is None:
            sol_without_answer, answer = solution, None
        else:
            solution_parts = solution.split(answer_prefix)
            if len(solution_parts) < 2:
                sol_without_answer, answer = solution, None
            else:
                sol_without_answer, answer = solution_parts

        # ========== Tag-Based Extraction ========== #
        # Now handled by get_format_rewards
        format_reward = self.get_format_rewards(solution)
        if format_reward == 1:
            # If properly formatted with one <think> block, return a single reward index
            return [1]

        # ===== Delimiter-based splitting (existing logic) =====
        steps = sol_without_answer.split(delimiter)

        # Merge leading empty steps into the first non-empty step.
        first_non_empty_step_idx = None
        for i, step in enumerate(steps):
            if step.strip() != "":
                first_non_empty_step_idx = i
                break

        if first_non_empty_step_idx is not None and first_non_empty_step_idx > 0:
            new_first_step = delimiter.join(steps[: first_non_empty_step_idx + 1])
            steps = [new_first_step] + steps[first_non_empty_step_idx + 1 :]

        if answer is not None:
            # Merge the final answer with the last non-empty step.
            last_non_empty_step_idx = None
            for i in range(len(steps) - 1, -1, -1):
                if steps[i].strip() != "":
                    last_non_empty_step_idx = i
                    break
            if last_non_empty_step_idx is None:
                last_non_empty_step_idx = 0
            new_last_step = delimiter.join(steps[last_non_empty_step_idx:])
            new_last_step = f"{new_last_step}{answer_prefix}{answer}"
            steps = steps[:last_non_empty_step_idx] + [new_last_step]

        reconstructed_solution = delimiter.join(steps)
        assert (
            reconstructed_solution == solution
        ), f"{reconstructed_solution} != {solution}"

        indices = [0]
        for i, step in enumerate(steps):
            if i == 0:
                indices.append(indices[-1] + len(step))
            else:
                indices.append(indices[-1] + len(step) + len(delimiter))

        assert indices[-1] == len(solution), f"{indices[-1]} != {len(solution)}"
        return indices

    def grade_answer(
        self,
        *,
        given_answer: Optional[str] = None,
        ground_truth: str = None,
        item: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        if given_answer is None:
            return False

        assert ground_truth is not None
        return (
            given_answer.strip().replace(",", "").lower()
            == ground_truth.strip().lower()
        )

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        once_hit_acc = []
        correct_frac = []
        majority_vote_acc = []
        unique_answer_count = []
        none_answer_extracted = []

        for solution_candidates, ref in zip(predictions, references):
            gold_answer = ref["answer"]

            assert len(solution_candidates) > 0
            answer_candidates = [
                self.extract_predicted_answer_from_text(sol)
                for sol in solution_candidates
            ]
            none_answer_extracted.append(
                sum([1 for ans in answer_candidates if ans is None])
                / len(answer_candidates)
            )

            grading_results = [
                self.grade_answer(given_answer=ans, ground_truth=gold_answer, item=ref)
                for ans in answer_candidates
            ]
            once_hit_acc.append(float(any(grading_results)))
            correct_frac.append(sum(grading_results) / len(grading_results))

            answer_candidates = [
                tuple(ans) if isinstance(ans, list) else ans
                for ans in answer_candidates
            ]

            majority_answer, _ = Counter(answer_candidates).most_common(n=1)[0]
            assert len(answer_candidates) == len(grading_results)
            majority_answer_index = answer_candidates.index(majority_answer)
            majority_answer_is_correct = grading_results[majority_answer_index]
            majority_vote_acc.append(majority_answer_is_correct)

            unique_answer_count.append(len(set(answer_candidates)))

        once_hit = sum(once_hit_acc) / len(once_hit_acc)
        correct_frac = sum(correct_frac) / len(correct_frac)

        return {
            "once_hit": once_hit,
            "exact_match": once_hit,  # for backwards compatibility
            "correct_frac": correct_frac,
            "exact_match_frac": correct_frac,  # for backwards compatibility
            "majority_vote_acc": sum(majority_vote_acc) / len(majority_vote_acc),
            "unique_answer_count": sum(unique_answer_count) / len(unique_answer_count),
            "none_answer_extracted_frac_per_problem": (
                sum(none_answer_extracted) / len(none_answer_extracted)
            ),
        }

    def build_dataset(
        self,
    ) -> DatasetDict:
        datasets = super().build_dataset()
        datasets = datasets.map(
            self._preprocess_example, num_proc=4, desc="Preprocessing examples"
        )
        return datasets

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question = example["question"].strip()
        answer = example["answer"]
        solution, final_answer = answer.split("####")
        output = {}
        if self.remove_calculator_expressions:
            solution = remove_text_between_symbols(solution, "<<", ">>")
            answer_without_calculator = remove_text_between_symbols(answer, "<<", ">>")
            answer = answer_without_calculator
            output["answer_without_calculator"] = answer_without_calculator
        final_answer = final_answer.strip()
        solution = solution.strip()
        output.update(
            {
                "problem": question,
                "solution": answer,
                "answer": final_answer,
                "query": question,
                "_solution": solution,
                "_final_answer": final_answer,
            }
        )
        return output
