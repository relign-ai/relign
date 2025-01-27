from typing import Optional

from relign.common.types import JsonDict
from relign.utils.py_utils import format_string
from relign.utils.guidance import run_program
from relign.inference.tree_inference import Node


class AnswerExtractor():
    def __init__(self, seed: Optional[int] = None, **kwargs):
        self.seed = seed
        self._run_program = run_program 

    def set_run_program(self, run_program):
        self._run_program = run_program

    def set_seed(self, seed: int):
        self.seed = seed

    async def extract_from_node(self, node: Node) -> str:
        return await self.extract(node["full_text"])

    async def extract(self, full_text: str) -> str:
        raise NotImplementedError()


class NextTurnAnswerExtractor(AnswerExtractor):
    def __init__(self, program: str, program_kwargs: JsonDict):
        super().__init__()
        self.program_template = format_string(program, **program_kwargs)
        assert (
            '"final_answer"' in self.program_template
        ), "Program_template must contain 'final_answer'"

    async def extract(self, full_text: str) -> str:
        result = await self._run_program(self.program_template, prefix=full_text)
        variables = result.variables()
        final_answer = variables["final_answer"]

        return final_answer


class NextTurnCodeAnswerExtractor(NextTurnAnswerExtractor):
    async def extract(self, full_text: str) -> str:
        final_answer = await super().extract(full_text)
        final_answer = "```\ndef " + final_answer
        return final_answer


class NextTurnABCDAnswerExtractor(AnswerExtractor):
    def __init__(self, program: str, program_kwargs: JsonDict):
        super().__init__()
        self.program_template = format_string(program, **program_kwargs)
        assert (
            '"final_answer"' in self.program_template
        ), "Program_template must contain 'final_answer'"

    async def extract(self, full_text: str) -> str:
        result = await self._run_program(self.program_template, prefix=full_text)
        variables = result.variables()
        final_answer = variables["final_answer"]
        # make sure just one answer is in the final answer
        count = 0
        for ch in ["A", "B", "C", "D"]:
            if ch in final_answer:
                count += 1
        if count != 1:
            return "no-answer"

        for ch in ["A", "B", "C", "D"]:
            if ch in final_answer:
                return ch


class IdentityAnswerExtractor(AnswerExtractor):
    def __init__(self, node_key_name: str = "full_text", **kwargs):
        super().__init__(**kwargs)
        self.node_key_name = node_key_name

    async def extract_from_node(self, node: Node) -> str:
        return node[self.node_key_name]


class IdentityWithSolutionPrefix(IdentityAnswerExtractor):
    def __init__(
        self, solution_prefix: str, end_of_turn_token: Optional[str] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.solution_prefix = solution_prefix
        self.end_of_turn_token = end_of_turn_token

    async def extract_from_node(self, node: Node) -> str:
        answer = await super().extract_from_node(node)
        parts = answer.split(self.solution_prefix)
        assert (
            len(parts) >= 2
        ), f"Expected '{self.solution_prefix}' in answer. Got:\n{answer}"
        # Remove the first part
        parts = parts[1:]
        out = self.solution_prefix.join(parts)

        if self.end_of_turn_token is not None:
            finish_reason = node["finish_reason"]
            if finish_reason != "length":
                out += self.end_of_turn_token
                node["full_text"] += self.end_of_turn_token

        return out
