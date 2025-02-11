from typing import List, Dict, Any
from collections import defaultdict

import random
import math

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset

from relign.utils.logging import get_logger

COLUMN_REF_SHIFTED_LOGPS = "ref_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_ACTOR_SHIFTED_LOGPS = "actor_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_VALUES = "critic_values"

logger = get_logger(__name__)


class GroupedBatchSampler(Sampler):
    """
    A sampler that groups samples by a specified 'group' column and returns
    items from one or more groups in each iteration (step).

    Args:
        dataset (Dataset or HF Dataset): Your dataset object
        group_column (str): The name of the column that indicates the group
        shuffle_groups (bool): Whether to shuffle the order of groups
        groups_per_step (int): How many groups to sample in one iteration. Defaults to 1.
    """

    def __init__(
        self,
        dataset,
        group_column: str,
        shuffle_groups: bool = True,
        groups_per_step: int = 1,
    ):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.group_column = group_column
        self.shuffle_groups = shuffle_groups
        self.groups_per_step = groups_per_step

        # Build a mapping: group -> list of indices
        self.group2indices = defaultdict(list)
        for idx in range(len(dataset)):
            group_key = dataset[idx][group_column]
            self.group2indices[group_key].append(idx)

        # Flatten out just the group keys
        self.all_groups = list(self.group2indices.keys())

    def __iter__(self):
        """
        Yields indices of one or more groups per iteration, as a single batch.
        """
        # Potentially shuffle the list of group keys
        if self.shuffle_groups:
            random.shuffle(self.all_groups)

        # Collect groups_per_step groups into one batch
        for i in range(0, len(self.all_groups), self.groups_per_step):
            combined_indices = []
            group_slice = self.all_groups[i : i + self.groups_per_step]

            # Collect all sample indices from these groups
            for group in group_slice:
                combined_indices.extend(self.group2indices[group])

            # Yield them as one batch
            yield combined_indices

    def __len__(self):
        """
        The number of batches is the ceiling of (total groups / groups_per_step).
        """
        return math.ceil(len(self.all_groups) / self.groups_per_step)


class GRPODataCollator:
    """
    Collates the given data instances into a batch.
    Every data instance should have the following keys:
    - "query_token_ids": The token ids of the query.
    - "response_token_ids": The token ids of the response.
    - "score": The reward of the response (single scalar per response)
    - "advantages": The advantages of the response.
    - "group: The group of the response. It is a scalar tensor.

    Args:
        data_instances (List[Dict[str, Any]]):
            The data instances to collate.
    Returns:
        Dict[str, torch.Tensor]:
            It contains the following keys:
            - "input_ids": The token ids of the entire episode (query + responses).
                    Description: Query + response tokens + padding
                    Shape: (batch_size, max_seq_len)
            - "labels": The token ids of the entire episode (query + responses).
                    Description: -100 for padding and query tokens, response tokens for labels.
                    Shape: (batch_size, max_seq_len)
            - "attention_mask": The attention mask of the entire episode (query + responses),
                    Description:  1 for input tokens, 0 for padding tokens.
                    Shape: (batch_size, max_seq_len)
            - "advantages": The advantages of the responses.
                    Description: 0.0 for padding and query tokens, advantages at response token index
                    Shape: (batch_size, max_seq_len)
            - "scores": The scores of the responses. It should be a 1D scalar tensor.
                    Description: The scores of the responses.
                    Shape: (batch_size,)
            - "values": The values of the response states.
                    (batch_size, max_seq_len)
            - "ref_shifted_log_probs": The reference log probabilities of the responses.
                    Shape: (batch_size, max_seq_len-1)
            - "actor_shifted_log_probs": The actor log probabilities of the responses.
                    Shape: (batch_size, max_seq_len-1)
    """

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_seq_length = max(
            len(instance["query_token_ids"]) + len(instance["response_token_ids"])
            for instance in instances
        )

        # Create the batch
        batch = {
            "input_ids": [], 
            "labels": [], 
            "attention_mask": [], 
            "len_query_token_ids": [],
            "len_response_token_ids": [],
            "max_seq_length": [],
            "normalized_scores": [],
        }

        has_group = "group" in instances[0] and instances[0]["group"] is not None
        if has_group:
            batch["group"] = []

        has_advantages = (
            "advantages" in instances[0] and instances[0]["advantages"] is not None
        )

        if has_advantages:
            batch["advantages"] = []

        has_process_rewards = (
            "process_rewards" in instances[0]
            and instances[0]["process_rewards"] is not None
        )
        if has_process_rewards:
            batch["process_rewards"] = []

        has_scores = "scores" in instances[0] and instances[0]["scores"] is not None
        if has_scores:
            batch["scores"] = []

        pad_token_id = 0  # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_label = (
            -100
        )  # -100 is the default value for the padding token in the loss function
        pad_logp = -float(1e9)  # Crazy value to show up it in case of a bug

        for instance in instances:
            # attention masks: length: max_seq length, '1' for real tokens, '0' for padding tokens
            # We want to send these to the batch in collation
            query_token_ids = instance["query_token_ids"]
            response_token_ids = instance["response_token_ids"]
            # Create the input ids and attention mask
            input_ids = query_token_ids + response_token_ids
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_length - len(input_ids)

            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)

            batch["len_query_token_ids"].append(len(query_token_ids))
            batch["len_response_token_ids"].append(len(response_token_ids))
            batch["max_seq_length"].append(max_seq_length) 

            # Create the labels
            labels = (
                [pad_label] * len(query_token_ids)
                + response_token_ids
                + [pad_label] * num_pad_at_end
            )
            batch["labels"].append(labels)

            if has_group:
                batch["group"].append(instance["group"])
                # calculate the mean for each group and add it to the instance
                # calculate the std for each group and add it to the instance

            if has_advantages:
                advantages = instance["advantages"]
                # Advantages are the same length as the reponse_token_ids
                advantages = (
                    [0.0] * len(query_token_ids) + advantages + [0.0] * num_pad_at_end
                )
                assert len(labels) == len(advantages)
                batch["advantages"].append(advantages)

            if has_scores:
                assert isinstance(instance["scores"], float)
                batch["scores"].append(instance["scores"])

        group2scores = defaultdict(list)
        for example in batch:
            if "group" not in example:
                raise KeyError(
                    "Each example must have a 'group' key to enable group-based normalization."
                )
            if "scores" not in example:
                raise KeyError(
                    "Each example must have a 'scores' key to enable group-based normalization."
                )
            group2scores[example["group"]].append(example["scores"])

        # Compute mean & std for each group in the episodes
        group2meanstd = {}
        for group, scores in group2scores.items():
            arr = np.array(scores, dtype=np.float32)
            mean = arr.mean()
            std = arr.std()
            group2meanstd[group] = (mean, std)

        # 3. Replace 'scores' with normalized 'rewards'
        for example in batch:
            group = example["group"]
            mean, std = group2meanstd[group]
            if std < 1e-12:
                example["rewards"] = 0.0
            else:
                example["rewards"] = (example["scores"] - mean) / std

            # Remove the old 'scores' to ensure only normalized rewards remain
            del example["scores"]

        # Convert the lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch
