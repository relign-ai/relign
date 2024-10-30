import json

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pathlib import Path

from accelerate import PartialState

from common.dataset import EpisodeDataset
from episode_generators.environment.base_environment import Env

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
    reward: float
    advantages: Optional[List[float]] = None

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.response_token_ids) > 0
        assert self.reward is not None

        if self.advantages is not None:
            assert len(self.advantages) == len(self.response_token_ids), "advantages have to be the same length as the response token ids"

class BaseEpisodeGenerator(ABC):
    def __init__(
            self,
            seed,
            project_root_dir: Path,
            distributed_state: PartialState, 
            supports_distributed: bool
        ):
        self.seed = seed
        self.project_root_dir = project_root_dir # Gets initialized by triggering set root direcotry in the runner 
        self.episodes_checkpoint_dir = None
        self.distributed_state = distributed_state 
        self.supports_distributed = supports_distributed
        self._init_episode_dir()

    @abstractmethod
    def generate_episodes(
        self, 
        num_episodes_per_iteration: int, 
        iteration: int, 
        return_path: bool
    ) -> Union[EpisodeDataset, Path]:
        pass

    def _init_episode_dir(self):
        """
        Instantiate the working directory for the episodes checkpoints
        """
        self.episodes_checkpoint_dir = (self.project_root_dir / "episodes")
        print("espiode generator root dir", self.episodes_checkpoint_dir)
        self.episodes_checkpoint_dir.mkdir(exist_ok=True, parents=True)
 
    def get_episode_checkpoint_path(self, iteration: int) -> Path:
        """
        Sets the checkpoint directory based on the iteration 
        """
        return (self.episodes_checkpoint_dir / f"episodes_{str(iteration).zfill(4)}.json")
    
    def checkpoint_episodes(self, episodes: EpisodeDataset, iteration: int) -> Path:
        """
        Saves the episodes to disk
        """
        assert isinstance(episodes, EpisodeDataset), f"episodes have to be of type EpisodeDataset, not {type(episodes)}"
        checkpoint_path = self.get_episode_checkpoint_path(iteration)
        episodes.save_to_disk(checkpoint_path) 
        return checkpoint_path

class OnPolicyEpisodeGenerator(BaseEpisodeGenerator):
    """
        Allow for a policy path (model weights) or
        pass down the current policy to infer from 
        during episode generation. 
    """
    def __init__(
            self, 
            policy_path: str,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy_path = policy_path
        
    def set_policy_path(self, policy_path: str) -> None:
        self.policy_path = policy_path


class DebugEpisodeGenerator(OnPolicyEpisodeGenerator):
    """
    Generate episodes from a json file for debugging purposes. 
    """
    def __init__(
            self, 
            file_path: str,
            **kwargs
    ):
        kwargs['supports_distributed'] = False
        super().__init__(**kwargs)
        self.debug_data = json.load(open(file_path, "r"))

    def generate_episodes(
            self, 
            num_episodes_per_iteration: int, 
            iteration: int,
            return_path: bool = False
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
                    reward=reward,
                )
            )

        episodes_dataset = EpisodeDataset.from_episode_list(episodes) 
        if return_path:
            path = self.checkpoint_episodes(episodes=episodes_dataset, iteration=iteration) 
            return path

        return episodes_dataset 
         

#TODO: What best way to differentiate between the more classical 
# RL environments and the ones where we rollout episodes from some
# Model itself. 
class EpisodeGeneratorEnvironment(BaseEpisodeGenerator):
    """
    Generate episodes by interacting with some environment, 
    in an policy step, environment step kind of way. 
    [o, a, r, o, a, r]
    """
    def __init__(self, environment: Env, policy):
        self.environment = environment
        self.policy = policy

    def generate_episodes(self) -> List[Episode]:
        pass

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

