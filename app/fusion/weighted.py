"""Transparent pixel-wise weighted fusion of NEDI and IMDN outputs."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from app.evaluation.metrics import calculate_quality_metrics


DEFAULT_IMDN_WEIGHTS = tuple(round(step / 10, 1) for step in range(11))
METRIC_NAMES = ("psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb")


def _validate_weight(imdn_weight: float) -> float:
    weight = float(imdn_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(
            f"IMDN weight must be a finite number from 0 to 1; received {imdn_weight}."
        )
    return weight


def weighted_fusion(
    nedi_image: Image.Image,
    imdn_image: Image.Image,
    imdn_weight: float,
) -> Image.Image:
    """Combine matching NEDI and IMDN pixels using one fixed IMDN weight.

    An IMDN weight of 0 returns NEDI, 1 returns IMDN, and 0.9 means
    90% IMDN plus 10% NEDI.
    """
    weight = _validate_weight(imdn_weight)
    nedi_rgb = nedi_image.convert("RGB")
    imdn_rgb = imdn_image.convert("RGB")
    if nedi_rgb.size != imdn_rgb.size:
        raise ValueError(
            "NEDI and IMDN outputs must have the same dimensions; "
            f"received {nedi_rgb.size} and {imdn_rgb.size}."
        )

    nedi_pixels = np.asarray(nedi_rgb, dtype=np.float64)
    imdn_pixels = np.asarray(imdn_rgb, dtype=np.float64)
    fused_pixels = np.rint(
        (1.0 - weight) * nedi_pixels + weight * imdn_pixels
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(fused_pixels, mode="RGB")


def evaluate_weight_grid(
    reference_hr: Image.Image,
    nedi_image: Image.Image,
    imdn_image: Image.Image,
    scale: int,
    *,
    weights: Sequence[float] = DEFAULT_IMDN_WEIGHTS,
) -> list[dict[str, float]]:
    """Measure every requested fusion weight with the fixed project metrics."""
    if scale not in {2, 3, 4}:
        raise ValueError(f"Scale must be 2, 3, or 4; received {scale}.")
    if not weights:
        raise ValueError("At least one fusion weight is required.")

    validated_weights = [_validate_weight(weight) for weight in weights]
    if len(set(validated_weights)) != len(validated_weights):
        raise ValueError("Fusion weights must not contain duplicates.")

    records: list[dict[str, float]] = []
    for imdn_weight in validated_weights:
        fused = weighted_fusion(nedi_image, imdn_image, imdn_weight)
        records.append(
            {
                "imdn_weight": imdn_weight,
                "nedi_weight": 1.0 - imdn_weight,
                **calculate_quality_metrics(reference_hr, fused, border=scale),
            }
        )
    return records


def summarise_weight_results(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    """Average validation metrics for each weight over equal sample counts."""
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if "imdn_weight" not in record:
            raise ValueError("Every validation record must contain imdn_weight.")
        weight = _validate_weight(float(record["imdn_weight"]))
        for metric in METRIC_NAMES:
            if metric not in record:
                raise ValueError(f"Every validation record must contain {metric}.")
            value = float(record[metric])
            if not np.isfinite(value):
                raise ValueError(f"Validation metric {metric} must be finite.")
        grouped[weight].append(record)

    if not grouped:
        raise ValueError("Cannot summarise an empty validation result set.")
    sample_counts = {len(group) for group in grouped.values()}
    if len(sample_counts) != 1:
        raise ValueError("Every fusion weight must have the same number of samples.")

    summary: list[dict[str, float | int]] = []
    for imdn_weight in sorted(grouped):
        group = grouped[imdn_weight]
        summary.append(
            {
                "imdn_weight": imdn_weight,
                "nedi_weight": 1.0 - imdn_weight,
                "sample_count": len(group),
                **{
                    f"mean_{metric}": mean(float(record[metric]) for record in group)
                    for metric in METRIC_NAMES
                },
            }
        )
    return summary


def select_best_weight(
    summary_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select by PSNR-Y, then SSIM-Y, then the larger IMDN contribution."""
    records = list(summary_records)
    if not records:
        raise ValueError("Cannot select a fusion weight from an empty summary.")
    required = {"imdn_weight", "mean_psnr_y", "mean_ssim_y"}
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Weight summary is missing fields: {sorted(missing)}")
        _validate_weight(float(record["imdn_weight"]))

    return dict(
        max(
            records,
            key=lambda record: (
                float(record["mean_psnr_y"]),
                float(record["mean_ssim_y"]),
                float(record["imdn_weight"]),
            ),
        )
    )
