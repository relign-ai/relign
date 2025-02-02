
import pytest


def test_generate_episode_returns_dict(episode_generator):
    """
    Test that the generate_episode method produces a result in a dictionary format.
    You might want to expand on this test based on what a valid episode is supposed to
    look like in your application.
    """
    episode = episode_generator.generate()
    assert isinstance(episode, dict), "Expected the generated episode to be a dict"