"""Target-size alignment shared by neural super-resolution evaluators."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class AlignedReconstruction:
    """One reconstruction aligned to its uncropped HR reference dimensions."""

    image: Image.Image
    native_size: tuple[int, int]
    target_size: tuple[int, int]
    dimension_adjusted: bool


def align_reconstruction_to_target(
    reconstruction: Image.Image,
    target_size: tuple[int, int],
) -> AlignedReconstruction:
    """Resize a native model output once when floor-scaled LR loses edge pixels."""
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Target size must be positive; received {target_size}.")

    native = reconstruction.convert("RGB")
    native_size = native.size
    dimension_adjusted = native_size != target_size
    if dimension_adjusted:
        aligned = native.resize(target_size, resample=Image.Resampling.BICUBIC)
    else:
        aligned = native
    return AlignedReconstruction(
        image=aligned,
        native_size=native_size,
        target_size=target_size,
        dimension_adjusted=dimension_adjusted,
    )
