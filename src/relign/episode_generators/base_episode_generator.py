import json
import random
from abc import ABC
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Any
from pathlib import Path

from accelerate import PartialState
from datasets import Dataset
import wandb

from relign.common.dataset import EpisodeDataset
from relign.utils import logging
from relign.tokenization.base_tokenizer import Tokenizer

logger = logging.get_logger(__name__)

"""
Episodes can be either generated via the policy interacting with some environment or via sampling from the model itself. 
"""


@dataclass
class Episode:
    """
    A single episode.
    """

    query_token_ids: List[int]
    response_token_ids: List[int]
    scores: float  # Final scores of the episode
    process_rewards: Optional[List[float]] = (
        None  # Process rewards for each token in the response
    )
    advantages: Optional[List[float]] = (
        None  # Advantages for each token in the response
    )
    group: Optional[int] = None  # GRPO for grouped advantages/rewards

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.response_token_ids) > 0
        assert self.scores is not None

        if self.advantages is not None:
            assert len(self.advantages) == len(
                self.response_token_ids
            ), "advantages have to be the same length as the response token ids"

        if self.process_rewards is not None:
            if len(self.process_rewards) != len(self.response_token_ids):
                print(
                    f"[DEBUG] Token count mismatch: process_rewards length = {len(self.process_rewards)}; response_token_ids length = {len(self.response_token_ids)}"
                )
            assert len(self.process_rewards) == len(
                self.response_token_ids
            ), "process_rewards have to be the same length as the response token ids"


class EpisodeGeneratorStrategy(ABC):
    def __call__(self, paths):
        raise NotImplementedError


class BaseEpisodeGenerator:
    can_precompute_episodes: bool = False
    supports_distributed: bool = False

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        distributed_state: PartialState = None,
        num_episodes_per_iteration: int = None,
        project_root_dir: Optional[Path] = None,
        cloud_logger: Optional[Any] = None,
        seed: int = 69,
    ):
        self.distributed_state = distributed_state
        self.cloud_logger = cloud_logger
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.tokenizer = tokenizer
        self.project_root_dir = project_root_dir

        if project_root_dir is not None:
            self._init_episode_dir()

    def _init_episode_dir(self):
        """
        Instantiate the working directory for the episodes checkpoints
        """
        self.episodes_checkpoint_dir = self.project_root_dir / "episodes"
        print("espiode generator root dir", self.episodes_checkpoint_dir)
        self.episodes_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def get_episode_checkpoint_path(self, iteration: int) -> Path:
        """
        Sets the checkpoint directory based on the iteration
        """
        if self.episodes_checkpoint_dir is None:
            raise ValueError("episodes_checkpoint_dir is not set")

        return self.episodes_checkpoint_dir / f"episodes_{str(iteration).zfill(4)}.json"

    def checkpoint_episodes(self, episodes: EpisodeDataset, iteration: int) -> Path:
        """
        Saves the episodes to disk
        """
        assert isinstance(
            episodes, EpisodeDataset
        ), f"episodes have to be of type EpisodeDataset, not {type(episodes)}"
        checkpoint_path = self.get_episode_checkpoint_path(iteration)
        episodes.save_to_disk(checkpoint_path)
        return checkpoint_path

    def is_main_process(self) -> bool:
        return self.distributed_state.is_main_process

    def generate(
        self, iteration: Optional[int] = None
    ) -> Union[List[Episode], Dataset]:
        raise NotImplementedError

    def precompute_episodes(self):
        raise NotImplementedError

    def set_models(self, models_weakref) -> None:
        pass

    def set_trainer(self, trainer_weakref) -> None:
        pass

    def log_episodes(
        self,
        episodes: Union[List[Episode], Dataset],
        iteration_idx: int,
        num_examples: int = 100,
        num_examples_for_wandb: int = 128,
        seed: int = 42,
        log_to_cloud: bool = True,
    ):
        if not self.is_main_process():
            return

        table = wandb.Table(
            columns=[
                "idx",
                "query",
                "response",
                "query_tokens",
                "response_tokens",
                "advantages",
                "reward",
                "instance_length",
            ]
        )

        logger.info(f"Logging {num_examples} examples:")
        rng = random.Random(seed)

        num_console_logs = min(num_examples, len(episodes))
        num_wandb_logs = min(num_examples_for_wandb, len(episodes))
        indices = rng.sample(range(len(episodes)), num_wandb_logs)
        #
        for idx in indices:
            episode = episodes[idx]
            if not isinstance(episode, dict):
                episode = asdict(episode)

            query_token_ids = episode["query_token_ids"]
            response_token_ids = episode["response_token_ids"]
            reward = episode["reward"]

            query_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in query_token_ids
            ]
            query = self.tokenizer.decode(query_token_ids)

            response_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in response_token_ids
            ]
            response = self.tokenizer.decode(response_token_ids)

            advantages = episode.get("advantages")
            instance_length = len(query_token_ids) + len(response_token_ids)

            table.add_data(
                idx,
                query,
                response,
                ", ".join(query_tokens),
                ", ".join(response_tokens),
                ", ".join(
                    [str(a) for a in advantages] if advantages is not None else []
                ),
                reward,
                instance_length,
            )

            if len(table.data) >= num_console_logs:
                continue

            logger.info(f"Example {idx}")
            for k, v in episode.items():
                logger.info(f"{k}: `{v}`")
            logger.info(f"Query: `{query}`")
            logger.info(f"Response: `{response}`")
            logger.info(f"Instance Length: {instance_length}")
            logger.info(f"Reward = Scores: {reward}")

            if advantages is not None:
                # Log aligned advantages with response tokens
                logger.info("Advantages:")
                for i, (adv, tok) in enumerate(zip(advantages, response_tokens)):
                    logger.info(f"{str(i).zfill(4)}: {tok:<20} -> {adv}")

            logger.info("-" * 100)

        if log_to_cloud and self.cloud_logger is not None:
            self.cloud_logger.log({f"episodes/iteration_{iteration_idx:04}": table})


# class OnPolicyEpisodeGenerator(BaseEpisodeGenerator):
#     """
#     Allow for a policy path (model weights) or
#     pass down the current policy to infer from
#     during episode generation.
#     """

#     def __init__(
#         self,
#         policy_path: str,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.policy_path = policy_path

#     def set_policy_path(self, policy_path: str) -> None:
#         self.policy_path = policy_path


class DebugEpisodeGenerator(BaseEpisodeGenerator):
    """
    Generate episodes from a json file for debugging purposes.
    """

    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.debug_data = json.load(open(file_path, "r"))

    def generate(
        self, num_episodes_per_iteration: int, iteration: int, return_path: bool = False
    ) -> Union[EpisodeDataset, Path]:
        episodes = []
        all_queries = self.debug_data["query"]
        all_responses = self.debug_data["response"]
        all_rewards = self.debug_data["reward"]

        for i in range(num_episodes_per_iteration):
            query_token_ids = all_queries[i]
            response_token_ids = all_responses[i]
            reward = all_rewards[i]
            episodes.append(
                Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    scores=reward,
                )
            )

        episodes_dataset = EpisodeDataset.from_episode_list(episodes)
        if return_path:
            path = self.checkpoint_episodes(
                episodes=episodes_dataset, iteration=iteration
            )
            return path

        return episodes_dataset


class EpisodeGeneratorInference(BaseEpisodeGenerator):
    """
    Generate episode by querying the model and generating actions (i.e., reasoning steps)
    Until the final answer.
    [q, a, a, r, o, a, a, r]
    """

    def __init__(self, model_path: str):
        self.inference_strategy

    def generate_episodes(self, num_episodes_per_iteration: int) -> EpisodeDataset:
        pass
