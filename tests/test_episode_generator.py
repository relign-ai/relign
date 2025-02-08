from dataclasses import asdict
from datasets import load_from_disk, Dataset


class TestMathEpisodeGenerator:
    def test_math_episode_generator(self, math_episode_generator):
        """
        Test that the generate_episode method produces a result in a dictionary format.
        You might want to expand on this test based on what a valid episode is supposed to
        look like in your application.
        """
        episode = math_episode_generator.generate()
        assert isinstance(episode, dict), "Expected the generated episode to be a dict"


class TestMathGroupedEpisodeGenerator:
    def test_math_grouped_episod_generator(self, math_grouped_episode_generator):
        """
        Test that the generate_episode method produces a result in a dictionary format.
        You might want to expand on this test based on what a valid episode is supposed to
        look like in your application.
        """
        episode = math_grouped_episode_generator.generate()

        assert isinstance(episode, dict), "Expected the generated episode to be a dict"

    def test_episode_gen_function(self, math_grouped_episode_generator):
        """
        Test that the generate_episode method produces a result in a dictionary format.
        You might want to expand on this test based on what a valid episode is supposed to
        look like in your application.
        """
        # 5 samples of branch 4 COT from the math mock test
        inference_results_ds = load_from_disk('./tests/mock_data/cot_mock_math_inference_results')
        episodes = math_grouped_episode_generator._generate_episodes(
            inference_results=inference_results_ds, iteration = 1
        )
        episodes_lst = [asdict(e) for e in episodes]

        print("episodes_lst", episodes_lst)
        episodes_ds = Dataset.from_list(episodes_lst)
        episodes_ds.save_to_disk('./tests/mock_data/mock_math_group_episodes_ds')