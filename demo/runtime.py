"""Runtime helpers for device and checkpoint loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from app.config import resolve_data_root
from app.deep_learning.checkpoints import (
    download_official_imdn_checkpoint,
    download_pretrained_fsrcnn_checkpoint,
)
from app.deep_learning.fsrcnn import load_pretrained_fsrcnn
from app.deep_learning.imdn import load_pretrained_imdn


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_cuda() -> bool:
    return torch.cuda.is_available()


def device_label() -> str:
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"


def checkpoint_root() -> Path:
    return resolve_data_root() / "checkpoints"


@lru_cache(maxsize=3)
def load_fsrcnn(scale: int, device_name: str) -> Any:
    checkpoint = download_pretrained_fsrcnn_checkpoint(checkpoint_root(), scale)
    return load_pretrained_fsrcnn(checkpoint, scale, torch.device(device_name))


@lru_cache(maxsize=3)
def load_imdn(scale: int, device_name: str) -> Any:
    checkpoint = download_official_imdn_checkpoint(checkpoint_root(), scale)
    return load_pretrained_imdn(checkpoint, scale, torch.device(device_name))
