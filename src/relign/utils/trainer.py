from typing import Tuple, Iterator, Dict, Optional, Any
from deepspeed import comm as dist
from torch.utils.data import Dataset, BatchSampler, RandomSampler 
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, DataLoader


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


@torch.no_grad()
def get_global_statistics(
    accelerator, xs: torch.Tensor, mask=None, device="cpu"
) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor(
        [xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device
    )
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count

    return global_mean.to(device), global_var.to(device), count.to(device)


@torch.no_grad()
def get_global_statistics_no_move(
    dist, xs: torch.Tensor, mask=None, unbiased=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """

    if mask is not None:
        sum_and_count = torch.tensor([(xs * mask).sum(), mask.sum()], device=xs.device)
    else:
        sum_and_count = [xs.sum(), xs.numel()]

    dist.all_reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    dist.all_reduce(sum_var)
    global_var = sum_var / count

    if unbiased:
        if count == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = count / (count - 1)
        global_var = global_var * bessel_correction

    return global_mean, global_var


def logprobs_from_logits(logits, labels, gather=True):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(
    values, mask, shift_mean=True, distributed=False, unbiased_variance=False
):
    """Whiten values with masked values."""
    from deepspeed import comm as dist

    if distributed and dist.is_initialized():
        mean, var = get_global_statistics_no_move(
            dist, values, mask=mask, unbiased=unbiased_variance
        )
    else:
        mean, var = masked_mean(values, mask), masked_var(
            values, mask, unbiased=unbiased_variance
        )
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_rescale_by_std(values, mask, distributed=False, unbiased_variance=False):
    """Whiten values with masked values."""
    from deepspeed import comm as dist

    if distributed and dist.is_initialized():
        mean, var = get_global_statistics_no_move(
            dist, values, mask=mask, unbiased=unbiased_variance
        )
    else:
        mean, var = masked_mean(values, mask), masked_var(
            values, mask, unbiased=unbiased_variance
        )
    whitened = values * torch.rsqrt(var + 1e-8)
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extension to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


def prepare_data_loader_for_training(
    dataset: Dataset,
    per_device_batch_size: int,
    seed: int,
    drop_last: bool = True,
    even_batches: bool = True,
    data_loader_kwargs: Optional[Dict[str, Any]] = None,
    grouped_sample = False,
) -> DataLoader:
    data_loader_kwargs = data_loader_kwargs or {}

    generator = torch.Generator()
    generator.manual_seed(seed)

    non_dist_batch_sampler = BatchSampler(
        RandomSampler(dataset, generator=generator),
        batch_size=per_device_batch_size,
        drop_last=drop_last,
    )
    dist_batch_sampler = BatchSamplerShard(
        non_dist_batch_sampler,
        num_processes=dist.get_world_size(),
        process_index=dist.get_rank(),
        split_batches=False,
        even_batches=even_batches,
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=dist_batch_sampler,
        **data_loader_kwargs,
    )
    return data_loader


def monitor_tensor_anomalies(
    tensor: torch.Tensor, mask: torch.Tensor, z_threshold: float = 3
) -> Dict[str, int]:
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    # Mask NaN and Inf values to compute mean and std of the finite values only
    finite_tensor = tensor[~nan_mask & ~inf_mask & mask]

    # Compute the mean and standard deviation of the finite values
    if finite_tensor.numel() == 0:
        mean = torch.tensor(float("nan"), device=tensor.device)
        std = torch.tensor(float("nan"), device=tensor.device)
    else:
        mean, std = masked_mean(tensor, mask), masked_var(tensor, mask, unbiased=False)

    # Handle case where std is 0
    if std == 0 or std.isnan():
        # If std is 0, all values are the same. No anomalies based on z-score.
        z_scores = torch.zeros_like(tensor)
    else:
        # Compute z-scores for finite values
        z_scores = (tensor - mean) / std

    # Check for NaN values
    num_nan = (nan_mask * mask).sum().item()

    # Check for infinity values
    num_inf = (inf_mask * mask).sum().item()

    # Check for values with high z-scores (anomalies)
    high_z_mask = z_scores.abs() > z_threshold
    num_high_z = (high_z_mask * mask).sum().item()

    # Total anomalies
    total_anomalies = num_nan + num_inf + num_high_z

    return {
        "num_nan": num_nan,
        "num_inf": num_inf,
        "num_high_z": num_high_z,
        "total_anomalies": total_anomalies,
    }