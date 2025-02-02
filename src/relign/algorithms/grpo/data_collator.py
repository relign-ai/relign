from typing import List, Dict, Any
import torch

from relign.utils.logging import get_logger

COLUMN_REF_SHIFTED_LOGPS = "ref_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_ACTOR_SHIFTED_LOGPS = "actor_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_VALUES = "critic_values"

logger = get_logger(__name__)

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
                The collated batch.
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
        batch = {"input_ids": [], "labels": [], "attention_mask": []}

        has_advantages = "advantages" in instances[0]
        if has_advantages:
            batch["advantages"] = []

        has_scores = "rewards" in instances[0]
        if has_scores:
            batch["rewards"] = []

        has_ref_shifted_logps = COLUMN_REF_SHIFTED_LOGPS in instances[0]
        if has_ref_shifted_logps:
            batch[COLUMN_REF_SHIFTED_LOGPS] = []

        has_actor_logps = COLUMN_ACTOR_SHIFTED_LOGPS in instances[0]
        if has_actor_logps:
            batch[COLUMN_ACTOR_SHIFTED_LOGPS] = []


        pad_token_id = 0  # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_label = (-100)  # -100 is the default value for the padding token in the loss function
        pad_logp = -float(1e9)  # Crazy value to show up it in case of a bug

        def prepare_shifted_logps(shifted_logps_with_query, query_len, response_len):
            assert len(shifted_logps_with_query) == (
                (query_len + response_len - 1)
            ), ( 
                f"We assume the ref. log probs are provided for the entire sequence",
                f"(query + response) but got {len(shifted_logps_with_query)}",
                f"instead of {query_len + response_len - 1}"
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

        for instance in instances:
            # attention masks: length: max_seq length, '1' for real tokens, '0' for padding tokens
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

            # Create the labels
            labels = (
                [pad_label] * len(query_token_ids)
                + response_token_ids
                + [pad_label] * num_pad_at_end
            )
            batch["labels"].append(labels)

            if has_advantages:
                advantages = instance["advantages"]
                # Advantages are the same length as the reponse_token_ids 
                advantages = [0.0] * len(query_token_ids) + advantages + [0.0] * num_pad_at_end
                assert len(labels) == len(advantages)
                batch["advantages"].append(instance["advantages"])

            if has_scores:
                assert isinstance(instance['rewards'], float)
                batch["rewards"].append(instance["rewards"])

            if has_ref_shifted_logps:
                shifted_ref_logps = prepare_shifted_logps(
                    instance[COLUMN_REF_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(response_token_ids),
                )
                assert len(shifted_ref_logps) == max_seq_length - 1
                batch[COLUMN_REF_SHIFTED_LOGPS].append(shifted_ref_logps)

            if has_actor_logps:
                shifted_actor_logps = prepare_shifted_logps(
                    instance[COLUMN_ACTOR_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(response_token_ids),
                )
                assert len(shifted_actor_logps) == (max_seq_length - 1)
                batch[COLUMN_ACTOR_SHIFTED_LOGPS].append(shifted_actor_logps)


        # Convert the lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch