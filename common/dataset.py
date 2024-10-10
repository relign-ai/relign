from typing import List, Optional

import numpy as np
from datasets import Dataset

from common.types import Episode

class EpisodeDataset(Dataset):
    """
        It should have the following columns:
        - "query_token_ids": The token ids of the query.
        - "response_token_ids": The token ids of the response.
        - "scores": The reward of the response (single scalar per response)
        - "advantages": The advantages of the response. (Optional)
    """
    
    def __init__(
            self,
            query_token_ids: np.ndarray,
            response_token_ids: np.ndarray,
            scores: np.ndarray,
            advantages: Optional[np.ndarray] = None,
    ):
        self.query_token_ids = query_token_ids
        self.response_token_ids = response_token_ids
        self.scores = scores
        self.advantages = advantages


        
    @staticmethod
    def from_episode_list(episodes: List[Episode]) -> 'EpisodeDataset':
        """
        Create a dataset from a list of episodes.
        """
        query_token_ids = []
        response_token_ids = []
        scores = []
        advantages = []

        for episode in episodes:
            query_token_ids.append(np.array(episode.query_token_ids, dtype=np.int32))
            response_token_ids.append(np.array(episode.response_token_ids, dtype=np.int32))
            scores.append(episode.scores)
            if episode.advantages is not None:
                advantages.append(np.array(episode.advantages, dtype=np.float32))
            else:
                advantages.append(None)

        # Convert lists to numpy arrays
        query_token_ids = np.array(query_token_ids, dtype=object)  # Using dtype=object for variable-length sequences
        response_token_ids = np.array(response_token_ids, dtype=object)
        scores = np.array(scores, dtype=np.float32)
        advantages = np.array(advantages, dtype=object) if any(advantages) else None

        return EpisodeDataset(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            scores=scores,
            advantages=advantages
        )


    def __len__(self):
        return len(self.scores)
    

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

