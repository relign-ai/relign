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


class GRPODataCollator:
    """
    Collates the given data instances into a batch.
    
    NOTE: We no longer compute group-based means/stdev here. 
    Instead, advantages (if needed) are precomputed & stored in each entry.
    """

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Identify the longest sequence length in the batch.
        max_seq_length = max(
            len(instance["query_token_ids"]) + len(instance["response_token_ids"])
            for instance in instances
        )

        # Check which optional fields are present
        has_group = "group" in instances[0] and instances[0]["group"] is not None
        has_process_rewards = (
            "process_rewards" in instances[0]
            and instances[0]["process_rewards"] is not None
        )
        has_ref_shifted_logps = COLUMN_REF_SHIFTED_LOGPS in instances[0]
        # We now rely on `_compute_group_based_advantages` for advantage calculation:
        has_advantages = ("advantages" in instances[0]) and (instances[0]["advantages"] is not None)

        # We'll collate into these lists:
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
        if has_ref_shifted_logps:
            batch[COLUMN_REF_SHIFTED_LOGPS] = []

        pad_token_id = 0
        pad_label = -100
        pad_logp = -float(1e9)

        def prepare_shifted_logps(shifted_logps_with_query, query_len, response_len):
            expected_len = query_len + response_len - 1
            assert len(shifted_logps_with_query) == expected_len, (
                "We assume the ref. log probs are provided for the entire sequence "
                f"(query + response) but got {len(shifted_logps_with_query)} instead of {expected_len}"
            )
            # remove the query portion except the last token
            shifted_logps_without_query = shifted_logps_with_query[query_len - 1 :]
            assert len(shifted_logps_without_query) == response_len
            n_pads_at_end = (max_seq_length - 1) - len(shifted_logps_with_query)
            # Fill in alignment
            shifted_logs = (
                [pad_logp] * (query_len - 1)
                + shifted_logps_without_query
                + [pad_logp] * n_pads_at_end
            )
            return shifted_logs

        # Build each field in the batch
        for instance in instances:
            query_ids = instance["query_token_ids"]
            response_ids = instance["response_token_ids"]

            input_ids = query_ids + response_ids
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_length - len(input_ids)

            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end

            labels = (
                [pad_label] * len(query_ids)
                + response_ids
                + [pad_label] * num_pad_at_end
            )

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
            batch["len_query_token_ids"].append(len(query_ids))
            batch["len_response_token_ids"].append(len(response_ids))
            batch["max_seq_length"].append(max_seq_length)

            if has_group:
                batch["group"].append(instance["group"])
            if has_advantages:
                batch["advantages"].append(instance["advantages"])
            if has_process_rewards:
                batch["process_rewards"].append(instance["process_rewards"])
            if has_ref_shifted_logps:
                shifted_ref_logps = prepare_shifted_logps(
                    instance[COLUMN_REF_SHIFTED_LOGPS],
                    len(query_ids),
                    len(response_ids),
                )
                batch[COLUMN_REF_SHIFTED_LOGPS].append(shifted_ref_logps)

        # If we still want numeric group indices, we can just batch them as before:
        if has_group:
            group_strs = batch["group"]  # list of string hashes
            unique_groups = sorted(list(set(group_strs)))
            group2id = {g: i for i, g in enumerate(unique_groups)}
            group_ids = [group2id[g] for g in group_strs]
            batch["group_ids"] = group_ids
            del batch["group"]

        # Convert numeric fields to tensors
        for key, value in list(batch.items()):
            if (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], (int, float))
            ):
                # If any float => float tensor, else int
                has_float = any(isinstance(v, float) for v in value)
                dtype = torch.float if has_float else torch.long
                # group_ids -> long
                if key.endswith("_ids"):
                    dtype = torch.long
                batch[key] = torch.tensor(value, dtype=dtype)
            elif (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], list)
            ):
                # Nested lists => token style => long
                batch[key] = torch.tensor(value, dtype=torch.long)

        return batch