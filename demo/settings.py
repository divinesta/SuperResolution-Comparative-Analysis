"""Shared settings for the Phase 7 demo."""

from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT
from app.traditional.nedi import NEDIConfig


SCALES = (2, 3, 4)
MAX_LR_SIDE = 512

EXPERIMENT_MODE = "Experiment-style: upload HR, create LR automatically"
DIRECT_LR_MODE = "Direct LR: upload LR image"

NEDI_CONFIG = NEDIConfig(window_size=8, edge_threshold=8.0)
XAI_ROOT = PROJECT_ROOT / "results" / "figures" / "xai"
EXAMPLES_ROOT = PROJECT_ROOT / "demo" / "examples"

XAI_MAPS: dict[str, Path] = {
    "Set5: baby (x2)": XAI_ROOT / "Set5_baby_x2",
    "Set5: butterfly (x3)": XAI_ROOT / "Set5_butterfly_x3",
    "Set14: baboon (x4)": XAI_ROOT / "Set14_baboon_x4",
    "Urban100: img_004 (x3)": XAI_ROOT / "Urban100_img_004_x3",
}
