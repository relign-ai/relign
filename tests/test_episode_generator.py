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
    def test_math_task(self, gsm8k):
        example_solutions = [
            """<think> First, we consider the problem 15 + 27.
            Adding 15 and 27 gives us 42. </think>
            #### 42""",  # gets a reard of 1

            """<think> Compute 8 multiplied by 7.
            <think> The product of 8 and 7 is 56.
            #### 56""",  # gets a reward of 0, no </think>

            """<think> Evaluate the expression 100 minus 45. </think>
            <think> Subtracting 45 from 100 yields 55. </think>
            #### 55""",  # gets a reward of 0 double think tags

            """<think> First, divide 9 by 3 to get 3.
            Then, multiply the result by 5.
            Multiplying 3 by 5 gives 15. </think>
            #### 15""",  # gets a reward of 1

            """<think> First, add 3.5 and 2.5 to obtain 6.0.
            Then, multiply 6.0 by 2 to arrive at the final result. </think>
            ## 12.0""",  # gets a reward of 0. \n## is not \####

            """
            <think> we think a little bit </think> some random text
            #### 20
            """, # gets a reward of zero due  to some text outisde think tags thaat is not the answer
        ]
        collected_indices = []
        for example_solution in example_solutions:
            indices = gsm8k.split_solution_into_intermediate_steps(example_solution)
            collected_indices.append(indices)

        print("collected indices", collected_indices)
        assert collected_indices == [[1], [], [], [1], [], []]

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
        inference_results_ds = load_from_disk(
            "./tests/mock_data/cot_mock_math_inference_results"
        )
        episodes = math_grouped_episode_generator._generate_episodes(
            inference_results=inference_results_ds, iteration=1
        )
        episodes_lst = [asdict(e) for e in episodes]

        print("episodes_lst", episodes_lst)
        episodes_ds = Dataset.from_list(episodes_lst)
        episodes_ds.save_to_disk("./tests/mock_data/mock_math_group_episodes_ds")
