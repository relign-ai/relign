import json
from typing import List, Union, Dict, Any, Tuple, Optional

from datasets import Dataset

from relign.common.registry import RegistrableBase
from relign.episode_generators.base_episode_generator import Episode
from relign.episode_generators.on_policy import (
    OnPolicyEpisodeGenerator
)
from relign.episode_generators.tree import TreeEpisodeUtils

from relign.utils.logging import get_logger

logger = get_logger(__name__)

class RewardFunction(RegistrableBase):
    def get_unfinished_response_penalty(self) -> float:
        raise NotImplementedError

    def __call__(
        self, query: str, response: str, dataset_instance: Dict[str, Any]
    ) -> Tuple[float, bool]:
        raise NotImplementedError

    def is_unfinished_response(
        self, response: str, dataset_instance: Dict[str, Any]
    ) -> bool:
        raise NotImplementedError


class EpisodeGeneratorWithRewardFunction(OnPolicyEpisodeGenerator, TreeEpisodeUtils):
    def __init__(
        self,
        reward_function: RewardFunction,
        append_bos_to_query: Union[str, bool] = "auto",
        append_eos_to_response: Union[str, bool] = "auto",
        tokenization_check_query_reconstruction: bool = True,
        tokenization_check_response_reconstruction: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_function = reward_function 
        self.append_bos_to_query = append_bos_to_query
        self.append_eos_to_response = append_eos_to_response
        logger.info(f"append bos to query={self.append_bos_to_query}")
        logger.info(f"append eos to resp= {self.append_eos_to_response}")
        self.tokenization_check_query_reconst = tokenization_check_query_reconstruction
        self.tokenization_check_response_reconst = (
            tokenization_check_response_reconstruction
        )

    def _generate_episodes(
        self, 
        inference_results: Dataset, 
        iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        episodes = []
        for instance in inference_results:
            tree = json.loads(instance["_treetune__reasoning_tree"])
            paths = self.extract_paths_from_tree(tree)
            for path in paths:
                assert len(path["node_chain"]) == 2, "Does not support multi-hop paths."

                query_text = path["node_chain"][0]["text"]
                full_text = path["node_chain"][-1]["full_text"]
                response_text = full_text[len(query_text) :]

                reward, is_unfinished_response = self.reward_function(
                    query_text, response_text, instance
                )

                try:
                    query_token_ids, response_token_ids = (
                        self._tokenize_query_and_response(
                            query_text,
                            response_text,
                            allow_append_eos=not is_unfinished_response,
                        )
                    )
                except Exception as e:
                    return []

                episode = Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    scores=reward,
                )
                episodes.append(episode)
        return episodes

    def _tokenize_query_and_response(
        self, 
        query: str, 
        response: str, 
        allow_append_eos: bool = True,
        return_offsets: bool = False
    ) -> Tuple[List[int], List[int]]:
        # This a legacy method that is not used anymore. It is kept here for reference.
        return self._tokenize_trajectory(
            {"query_text": query, "response_text": response},
            is_unfinished_response=not allow_append_eos,
            return_offsets=return_offsets,
        )

    def _tokenize_trajectory(
        self,
        trajectory: Dict[str, Any],
        is_unfinished_response: bool = False,
        return_offsets: bool = False,
        return_word_ids: bool = False,
        safety_check_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Tuple[List[int], List[int]],
        Tuple[List[int], List[int], List[Tuple[int, int]]],
    ]:
        safety_check_kwargs = safety_check_kwargs or {}
        query_text = trajectory["query_text"]
        response_text = trajectory["response_text"]

        episode_text = f"{query_text}{response_text}"
        episode_encoding = self.tokenizer(
            episode_text,
            add_special_tokens=False,  # We will add BOS and EOS tokens at the end
            return_offsets_mapping=True,
        )

        token_ids = episode_encoding["input_ids"]
        offsets = episode_encoding["offset_mapping"]

        response_start_index = next(
            i for i, (start, end) in enumerate(offsets) if start >= len(query_text)
        )
        query_token_ids = token_ids[:response_start_index]
        response_token_ids = token_ids[response_start_index:]

        self._safety_check_tokenization(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            query=query_text,
            response=response_text,
            episode_text=episode_text,
            **safety_check_kwargs,
        )

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids

        if not is_unfinished_response and self._should_append_eos_to_response():
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]

        if return_offsets:
            return query_token_ids, response_token_ids, offsets
        else:
            return query_token_ids, response_token_ids

    def _safety_check_tokenization(
        self,
        query_token_ids: List[str],
        response_token_ids: List[str],
        query: str,
        response: str,
        episode_text: str,
        check_query_reconstruction: bool = True,
        check_response_reconstruction: bool = True,
    ):
        decoding_kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }

        decoded_instance = self.tokenizer.decode(
            query_token_ids + response_token_ids, **decoding_kwargs
        )

    def _should_append_bos_to_query(self) -> bool:
        """
        Determine whether to append BOS to the query based on the tokenizer
        """
        #TODO: support auto 
        if self.append_bos_to_query != "auto":
            return self.append_bos_to_query

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.bos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append BOS for tokenizer {self.tokenizer.name_or_path}"
            )

    def _should_append_eos_to_response(self) -> bool:
        """
        Determine whether to append EOS to the response based on the tokenizer
        """
        if self.append_eos_to_response != "auto":
            return self.append_eos_to_response

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.eos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append EOS for tokenizer {self.tokenizer.name_or_path}"
            )


class EpisodeGeneratorWithGroupAdvantages(EpisodeGeneratorWithRewardFunction, TreeEpisodeUtils):
    def __init__(): pass

    def _generate_episodes(
        self, 
        inference_results: Dataset, 
        iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        episodes = []
        for instance in inference_results:
            tree = json.loads(instance["_treetune__reasoning_tree"])
            paths = self.extract_paths_from_tree(tree)
            for path in paths:
                assert len(path["node_chain"]) == 2, "Does not support multi-hop paths."

                query_text = path["node_chain"][0]["text"]
                full_text = path["node_chain"][-1]["full_text"]
                response_text = full_text[len(query_text) :]
        
                reward, is_unfinished_response = self.reward_function(
                    query_text, response_text, instance
                )

                try:
                    query_token_ids, response_token_ids = (
                        self._tokenize_query_and_response(
                            query_text,
                            response_text,
                            allow_append_eos=not is_unfinished_response,
                        )
                    )
                except Exception as e:
                    return []

                episode = Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    scores=reward,
                )
                episodes.append(episode)
        return episodes


    def extract_reasoning_steps(self, response_text: str) -> int:
        """ Function that takes in the response text and find all 
            Reasoning tokens which are contained within the <think> </think> tags. 
        """
        pass

        
