import json

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
from deepspeed import DeepSpeedEngine

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
            project_root_dir: Path
        ):
        self.project_root_dir = project_root_dir # Gets initialized by triggering set root direcotry in the runner 
        self.episodes_checkpoint_dir = None
        self._init_episode_dir()

    @abstractmethod
    def generate_episodes(self, num_episodes_per_iteration) -> EpisodeDataset:
        pass

    def set_deepspeed(self, distributed_state: DeepSpeedEngine):
        self.distributed_state = distributed_state

    def _init_episode_dir(self):
        """
        Makes the working direcotry for the checkpoint episodes
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
        super().__init__(**kwargs)
        self.debug_data = json.load(open(file_path, "r"))

    def generate_episodes(self, num_episodes_per_iteration: int, iteration: int) -> EpisodeDataset:
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

        self.checkpoint_episodes(episodes=episodes, iteration=iteration)
        return EpisodeDataset.from_episode_list(episodes)
         

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

