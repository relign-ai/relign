import random

from typing import Tuple, List, Optional, Dict
from episode_generation.environment.base_environment import Env, TextHistory
from episode_generation.environment.guess_city.oracle import GuessCityOracle
from episode_generation.environment.guess_city.data import create_trajectory_from_history, INITIAL_STR
from episode_generation.environment.environment_factory import EnvironmentFactory

from common.types import Reward, Done, Text

class GuessCityPolicyEnvironment(Env):
    def __init__(self, 
                 oracle: GuessCityOracle,
                 word_list: List[str],
                 max_conversation_length: int = 20,
                ):
        self.oracle = oracle
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length
        self.history = ""
        self.done = True

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, Reward, Done]:
        assert text_history[-1].is_action
        assert self.curr_word is not None, "call env.reset() first."

        question = text_history[-1].text.strip()
        answer = self.oracle.generate_answer(self.curr_word, question)
        # print(f"step: question={question}, answer={answer}")
        answer_text = Text(answer + "\n", is_action=False)
        trajectory = create_trajectory_from_history(self.curr_word, text_history + (answer_text,), self.max_conversation_length)

        return trajectory

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        if seed is not None:
            self.random = random.Random(seed)
        if options is None:
            options = {}
        deterministic = options.get("deterministic", False)
        if deterministic:
            assert seed is not None, "In deterministic mode, the seed specifies which word to use."
            word_ind = seed % len(self.word_list)
            self.curr_word = self.word_list[word_ind]
        else:
            self.curr_word = self.random.choice(self.word_list)

        # print(f"reset: word={self.curr_word}")
        return (Text(INITIAL_STR, is_action=False),)\
        
    
    def copy(self):
        return GuessCityPolicyEnvironment(
            oracle=self.oracle,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
        )


EnvironmentFactory.register_environment("guess-city", GuessCityPolicyEnvironment)