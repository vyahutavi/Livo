"""Device utilities for LIVO training and inference."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def configure_runtime(device: torch.device) -> None:
    """Configure runtime optimizations for the given device."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


@contextmanager
def autocast_context(device: torch.device, precision: str = "fp16"):
    """Context manager for automatic mixed precision."""
    if precision == "fp32" or device.type == "cpu":
        yield
        return

    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        yield


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a batch dict of tensors to the specified device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def use_grad_scaler(device: torch.device, precision: str = "fp16") -> bool:
    """Determine whether to use GradScaler for mixed precision."""
    return device.type == "cuda" and precision == "fp16"
