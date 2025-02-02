
import pytest
from typing import List

from relign.episode_generators.base_episode_generator import Episode 

def test_math_episode_generator(math_episode_generator):
    """
    Test that the generate_episode method produces a result in a dictionary format.
    You might want to expand on this test based on what a valid episode is supposed to
    look like in your application.
    """
    episode = math_episode_generator.generate()
    assert isinstance(episode, List[Episode]), "Expected the generated episode to be a dict"
    assert isinstance(episode, dict), "Expected the generated episode to be a dict"
    assert isinstance(episode, dict), "Expected the generated episode to be a dict"


def test_math_grouped_episod_generator(math_grouped_episode_generator):
    """
    Test that the generate_episode method produces a result in a dictionary format.
    You might want to expand on this test based on what a valid episode is supposed to
    look like in your application.
    """
    episode = math_grouped_episode_generator.generate()

    assert isinstance(episode, dict), "Expected the generated episode to be a dict"