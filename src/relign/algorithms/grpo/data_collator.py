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
    Collates the given data instances into a batch and normalizes scores by group if present.
    """

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Identify the longest sequence length in the batch
        max_seq_length = max(
            len(instance["query_token_ids"]) + len(instance["response_token_ids"])
            for instance in instances
        )

        # 2. Determine which optional fields are present
        has_group = "group" in instances[0] and instances[0]["group"] is not None
        has_advantages = (
            "advantages" in instances[0] and instances[0]["advantages"] is not None
        )
        has_process_rewards = (
            "process_rewards" in instances[0]
            and instances[0]["process_rewards"] is not None
        )
        has_scores = "scores" in instances[0] and instances[0]["scores"] is not None
        has_ref_shifted_logps = COLUMN_REF_SHIFTED_LOGPS in instances[0]

        # 3. If we have group + scores, gather them first to compute group-based normalization
        if has_group and has_scores:
            group2scores = defaultdict(list)

            # Collect raw scores by group
            for inst in instances:
                group_val = inst["group"]
                score_val = inst["scores"]
                group2scores[group_val].append(score_val)

            # Compute mean & std for each group
            group2meanstd = {}
            for group_val, score_list in group2scores.items():
                arr = np.array(score_list, dtype=np.float32)
                mean = arr.mean()
                std = arr.std()
                group2meanstd[group_val] = (mean, std)

            # Replace each instance's "scores" with its normalized version
            for inst in instances:
                g = inst["group"]
                mean, std = group2meanstd[g]
                # If group is degenerate (std=0), just set the normalized score to 0
                if std < 1e-12:
                    inst["scores"] = 0.0
                else:
                    inst["scores"] = (inst["scores"] - mean) / std

        pad_logp = -float(1e9)  # Crazy value to show up it in case of a bug

        def prepare_shifted_logps(shifted_logps_with_query, query_len, response_len):
            assert len(shifted_logps_with_query) == (query_len + response_len - 1), (
                f"We assume the ref. log probs are provided for the entire sequence",
                f"(query + response) but got {len(shifted_logps_with_query)}",
                f"instead of {query_len + response_len - 1}",
            )

            shifted_logps_without_query = shifted_logps_with_query[query_len - 1 :]
            assert len(shifted_logps_without_query) == response_len

            n_pads_at_end = (max_seq_length - 1) - len(shifted_logps_with_query)
            shifted_logs = (
                [pad_logp] * (query_len - 1)
                + shifted_logps_without_query
                + [pad_logp] * n_pads_at_end
            )
            return shifted_logs

        # 4. Initialize our batch dictionary
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "len_query_token_ids": [],
            "len_response_token_ids": [],
            "max_seq_length": [],
        }
        if has_group:
            batch["group"] = []
        if has_advantages:
            batch["advantages"] = []
        if has_process_rewards:
            batch["process_rewards"] = []
        if has_scores:
            batch["scores"] = []
        if has_ref_shifted_logps:
            batch[COLUMN_REF_SHIFTED_LOGPS] = []

        # 5. Padding/label specifics
        pad_token_id = 0  # We'll mask it out anyway
        pad_label = -100  # Default ignore index in HF training

        # 6. Build out each field in the batch
        for instance in instances:
            query_token_ids = instance["query_token_ids"]
            response_token_ids = instance["response_token_ids"]

            # Make input_ids and attention_mask
            input_ids = query_token_ids + response_token_ids
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_length - len(input_ids)

            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end

            labels = (
                [pad_label] * len(query_token_ids)
                + response_token_ids
                + [pad_label] * num_pad_at_end
            )

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
            batch["len_query_token_ids"].append(len(query_token_ids))
            batch["len_response_token_ids"].append(len(response_token_ids))
            batch["max_seq_length"].append(max_seq_length)

            # Group
            if has_group:
                batch["group"].append(instance["group"])

            # Advantages
            if has_advantages:
                advantages = instance["advantages"]
                # advantage vector should match response_token_ids in length
                # but we prefix with 0.0 for the query tokens & pad with 0.0
                advantages = (
                    [0.0] * len(query_token_ids) + advantages + [0.0] * num_pad_at_end
                )
                batch["advantages"].append(advantages)

            # Process rewards
            if has_process_rewards:
                batch["process_rewards"].append(instance["process_rewards"])

            # Scores
            if has_scores:
                sc = instance["scores"]
                batch["scores"].append(sc)

            if has_ref_shifted_logps:
                shifted_ref_logps = prepare_shifted_logps(
                    instance[COLUMN_REF_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(response_token_ids),
                )
                assert len(shifted_ref_logps) == max_seq_length - 1
                batch[COLUMN_REF_SHIFTED_LOGPS].append(shifted_ref_logps)

        # 7. Convert the lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch
