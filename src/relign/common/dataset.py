from typing import List, Optional

import numpy as np
from datasets import Dataset

from relign.common.types import Episode

class EpisodeDataset:
    """
    wrapper around Dataset for handling episodes
    """
    def __init__(
            self,
            query_token_ids: np.ndarray,
            response_token_ids: np.ndarray,
            rewards: np.ndarray,
            advantages: Optional[np.ndarray] = None,
    ):
        data = {
            'query_token_ids': query_token_ids.tolist(),
            'response_token_ids': response_token_ids.tolist(),
            'rewards': rewards.tolist(),
        }
        if advantages is not None:
            assert len(advantages) is len(response_token_ids),"advantage dim must match response_token_ids dim"
            data['advantages'] = advantages.tolist()
        self.dataset = Dataset.from_dict(data)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def from_episode_list(episodes: List[Episode]) -> 'EpisodeDataset':
        """
        Create a dataset from a list of episodes.
        """
        query_token_ids = []
        response_token_ids = []
        rewards = []
        #has_advantage = episodes[0].advantages is not None 
        has_advantage = any(episode.advantages is not None for episode in episodes)
        advantages = []

        for episode in episodes:
            query_token_ids.append(episode.query_token_ids)
            response_token_ids.append(episode.response_token_ids)
            rewards.append(episode.reward)
            if episode.advantages is not None:
                advantages.append(episode.advantages)

        return EpisodeDataset(
            query_token_ids=np.array(query_token_ids, dtype=object),
            response_token_ids=np.array(response_token_ids, dtype=object),
            rewards=np.array(rewards, dtype=np.float32),
            advantages=np.array(advantages, dtype=object) if has_advantage else None
        )

    def save_to_disk(self, path):
        self.dataset.save_to_disk(path)

class PPODataset(Dataset):
    """    
        It contains the following keys:
        - "input_ids": The token ids of the entire episode (query + responses).
                Shape: (batch_size, max_seq_len)
        - "labels": The token ids of the entire episode (query + responses).
                Shape: (batch_size, max_seq_len)
        - "attention_mask": The attention mask of the entire episode (query + responses).
                Shape: (batch_size, max_seq_len)
        - "advantages": The advantages of the responses.
                (batch_size, max_seq_len)
        - "scores": The scores of the responses. It should be a 1D scalar tensor.
                Shape: (batch_size,)
        - "values": The values of the response states. (Obtained from critic)
                (batch_size, max_seq_len)
        - "ref_shifted_log_probs": The reference log probabilities of the responses.
                Shape: (batch_size, max_seq_len-1)
        - "actor_shifted_log_probs": The actor log probabilities of the responses.
                Shape: (batch_size, max_seq_len-1)
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

