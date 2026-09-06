"""LR-consistency fusion inspired by back-projection SR ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.evaluation.metrics import calculate_quality_metrics
from app.fusion.weighted import weighted_fusion


@dataclass(frozen=True)
class BackProjectionFusionResult:
    """One fused image and the LR-consistency values used to create it."""

    image: Image.Image
    imdn_weight: float
    nedi_weight: float
    nedi_lr_mse: float
    imdn_lr_mse: float
    fused_lr_mse: float


def _downsample_to_lr(image: Image.Image, lr_size: tuple[int, int]) -> Image.Image:
    width, height = lr_size
    if width <= 0 or height <= 0:
        raise ValueError(f"LR dimensions must be positive; received {lr_size}.")
    return image.convert("RGB").resize(lr_size, Image.Resampling.BICUBIC)


def _mean_squared_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.square(first - second)))


def calculate_back_projection_weight(
    original_lr: Image.Image,
    nedi_image: Image.Image,
    imdn_image: Image.Image,
) -> tuple[float, float, float]:
    """Find the IMDN weight that best reproduces the observed LR pixels.

    The calculation solves one least-squares problem and clips the result to
    the valid range from zero to one. It uses only the LR input, never HR.
    """
    if nedi_image.size != imdn_image.size:
        raise ValueError(
            "NEDI and IMDN outputs must have the same dimensions; "
            f"received {nedi_image.size} and {imdn_image.size}."
        )
    observed = np.asarray(original_lr.convert("RGB"), dtype=np.float64)
    nedi_lr = np.asarray(
        _downsample_to_lr(nedi_image, original_lr.size), dtype=np.float64
    )
    imdn_lr = np.asarray(
        _downsample_to_lr(imdn_image, original_lr.size), dtype=np.float64
    )
    nedi_error = _mean_squared_error(observed, nedi_lr)
    imdn_error = _mean_squared_error(observed, imdn_lr)

    direction = imdn_lr - nedi_lr
    denominator = float(np.sum(np.square(direction)))
    if denominator <= np.finfo(np.float64).eps:
        # The two candidates are indistinguishable after degradation. Use the
        # independently selected best model rather than an arbitrary mixture.
        return 1.0, nedi_error, imdn_error
    numerator = float(np.sum((observed - nedi_lr) * direction))
    imdn_weight = float(np.clip(numerator / denominator, 0.0, 1.0))
    return imdn_weight, nedi_error, imdn_error


def back_projection_fusion(
    original_lr: Image.Image,
    nedi_image: Image.Image,
    imdn_image: Image.Image,
) -> BackProjectionFusionResult:
    """Fuse NEDI and IMDN using a weight derived from LR consistency."""
    imdn_weight, nedi_error, imdn_error = calculate_back_projection_weight(
        original_lr, nedi_image, imdn_image
    )
    fused = weighted_fusion(nedi_image, imdn_image, imdn_weight)
    fused_lr = np.asarray(
        _downsample_to_lr(fused, original_lr.size), dtype=np.float64
    )
    observed = np.asarray(original_lr.convert("RGB"), dtype=np.float64)
    fused_error = _mean_squared_error(observed, fused_lr)
    best_endpoint_error = min(nedi_error, imdn_error)
    if fused_error > best_endpoint_error:
        # Rounding the fused 8-bit HR image can very slightly change the
        # least-squares optimum. Never keep a mixture whose actual degraded
        # result is worse than both available endpoints.
        if imdn_error <= nedi_error:
            imdn_weight = 1.0
            fused = imdn_image.convert("RGB").copy()
            fused_error = imdn_error
        else:
            imdn_weight = 0.0
            fused = nedi_image.convert("RGB").copy()
            fused_error = nedi_error
    return BackProjectionFusionResult(
        image=fused,
        imdn_weight=imdn_weight,
        nedi_weight=1.0 - imdn_weight,
        nedi_lr_mse=nedi_error,
        imdn_lr_mse=imdn_error,
        fused_lr_mse=fused_error,
    )


def evaluate_back_projection_fusion(
    reference_hr: Image.Image,
    original_lr: Image.Image,
    nedi_image: Image.Image,
    imdn_image: Image.Image,
    scale: int,
) -> dict[str, Any]:
    """Evaluate one adaptive fusion using the fixed project metrics."""
    if scale not in {2, 3, 4}:
        raise ValueError(f"Scale must be 2, 3, or 4; received {scale}.")
    result = back_projection_fusion(original_lr, nedi_image, imdn_image)
    return {
        "method": "back_projection_nedi_imdn_fusion",
        "imdn_weight": result.imdn_weight,
        "nedi_weight": result.nedi_weight,
        "nedi_lr_mse": result.nedi_lr_mse,
        "imdn_lr_mse": result.imdn_lr_mse,
        "fused_lr_mse": result.fused_lr_mse,
        **calculate_quality_metrics(reference_hr, result.image, border=scale),
    }
