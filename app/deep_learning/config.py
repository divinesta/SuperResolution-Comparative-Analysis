"""Validated model and checkpoint conventions for Phase 3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DeepLearningModelName = Literal["fsrcnn", "imdn"]
SUPPORTED_DEEP_LEARNING_MODELS = ("fsrcnn", "imdn")
SUPPORTED_SCALES = (2, 3, 4)


@dataclass(frozen=True)
class DeepLearningModelConfig:
    """Identify one architecture, scale-specific weight file, and input policy."""

    model: DeepLearningModelName
    scale: int
    checkpoint_root: Path

    def __post_init__(self) -> None:
        normalized_model = self.model.strip().lower()
        if normalized_model not in SUPPORTED_DEEP_LEARNING_MODELS:
            supported = ", ".join(SUPPORTED_DEEP_LEARNING_MODELS)
            raise ValueError(
                f"Model must be one of {supported}; received {self.model!r}."
            )
        if self.scale not in SUPPORTED_SCALES:
            raise ValueError(
                f"Scale must be 2, 3, or 4; received {self.scale}."
            )

        object.__setattr__(self, "model", normalized_model)
        object.__setattr__(self, "checkpoint_root", Path(self.checkpoint_root))

    @property
    def checkpoint_path(self) -> Path:
        """Return the fixed location for this model's scale-specific weights."""
        return self.checkpoint_root / self.model / f"{self.model}_x{self.scale}.pth"

    @property
    def colour_policy(self) -> str:
        """Describe the colour-space policy that evaluation must record."""
        if self.model == "fsrcnn":
            return "y_model_with_bicubic_cbcr"
        return "rgb_model"
