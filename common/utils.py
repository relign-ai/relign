import os

def get_rank() -> int:
    try:
        from transformers import is_torch_tpu_available

        if is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            return xm.get_ordinal()
    except ImportError:
        pass

    try:
        import torch.distributed as dist

        if dist.is_available():
            if dist.is_initialized():
                return dist.get_rank()
            else:
                return int(os.environ.get("RANK", 0))
    except ImportError:
        pass

    return -1
