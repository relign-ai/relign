from __future__ import annotations
from abc import ABC, abstractmethod
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .data import INVALID_QUESTION


class GuessCityOracle(ABC):
    @abstractmethod
    def generate_answer(self, word: str, question: str, return_full: bool=False) -> str:
        pass


def get_oracle_prompt(word: str, question: str):
    prompt = (
    f"""Answer the question about the city truthfully.
    object: {word}
    question: {question}
    answer: """
    )
    return prompt


class SimpleOracle(GuessCityOracle):
    def __init__(
            self, 
            device:torch.device,  
            cache_dir: str = "./cache",
            env_load_path: str = "./env_load_path.pth"
        ):
        self.tokenizer = T5Tokenizer.from_pretrained(
            "google/flan-t5-small", cache_dir=cache_dir
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-small", cache_dir=cache_dir
        ).to(device)
        self.model.load_state_dict(torch.load(env_load_path)["model_state_dict"])
        self.answer_re_pattern = re.compile(r"(yes|no)")

    def generate_answer(self, word: str, question: str, return_full: bool=False) -> str:
        if question == INVALID_QUESTION:
            if return_full:
                return "No.", "No."
            return "No."

        prompt = get_oracle_prompt(word, question)
        encoder_inputs = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate output token IDs
        generated_ids = self.model.generate(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs["attention_mask"],
            max_length=64,
            do_sample=False,
        )

        # Decode the generated token IDs to a string
        answer_full = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        answer_match = self.answer_re_pattern.match(answer_full)
        if answer_match is not None:
            answer = answer_match[0].capitalize() + "."
        
        if answer_match is None:
            answer = "No."
        
        if return_full:
            return answer, answer_full
        return answer
