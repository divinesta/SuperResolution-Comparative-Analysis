"""Transparent fusion that allows NEDI to contribute only near strong edges."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.evaluation.images import bicubic_upsample
from app.evaluation.metrics import calculate_quality_metrics, rgb_to_y


DEFAULT_EDGE_PERCENTILES = (80.0, 90.0, 95.0)
DEFAULT_EDGE_NEDI_WEIGHTS = (0.05, 0.10, 0.20, 0.30)
METRIC_NAMES = ("psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb")


@dataclass(frozen=True)
class EdgeAwareFusionConfig:
    """Settings for one simple edge-only NEDI contribution."""

    edge_percentile: float
    edge_nedi_weight: float
    dilation_radius: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.edge_percentile) or not 0 < self.edge_percentile < 100:
            raise ValueError("Edge percentile must be a finite number between 0 and 100.")
        if (
            not np.isfinite(self.edge_nedi_weight)
            or not 0 <= self.edge_nedi_weight <= 1
        ):
            raise ValueError("Edge NEDI weight must be a finite number from 0 to 1.")
        if self.dilation_radius < 0:
            raise ValueError("Dilation radius cannot be negative.")

    @property
    def config_id(self) -> str:
        percentile = f"{self.edge_percentile:g}".replace(".", "p")
        weight = f"{self.edge_nedi_weight:.2f}".replace(".", "p")
        return f"edge_p{percentile}_nedi_{weight}_d{self.dilation_radius}"


@dataclass(frozen=True)
class EdgeAwareFusionResult:
    image: Image.Image
    edge_mask: NDArray[np.bool_]

    @property
    def edge_pixel_fraction(self) -> float:
        return float(self.edge_mask.mean())


def _sobel_magnitude(luminance: NDArray[np.float64]) -> NDArray[np.float64]:
    padded = np.pad(luminance, 1, mode="edge")
    top_left = padded[:-2, :-2]
    top = padded[:-2, 1:-1]
    top_right = padded[:-2, 2:]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    bottom_left = padded[2:, :-2]
    bottom = padded[2:, 1:-1]
    bottom_right = padded[2:, 2:]
    gradient_x = (
        -top_left + top_right - 2.0 * left + 2.0 * right - bottom_left + bottom_right
    )
    gradient_y = (
        -top_left - 2.0 * top - top_right
        + bottom_left + 2.0 * bottom + bottom_right
    )
    return np.hypot(gradient_x, gradient_y)


def _dilate_mask(
    mask: NDArray[np.bool_],
    radius: int,
) -> NDArray[np.bool_]:
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    height, width = mask.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            dilated |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return dilated


def detect_strong_edges(
    guide_lr: Image.Image,
    target_size: tuple[int, int],
    *,
    edge_percentile: float,
    dilation_radius: int = 1,
) -> NDArray[np.bool_]:
    """Detect edges from the available LR input, never from the reference HR."""
    config = EdgeAwareFusionConfig(edge_percentile, 0.0, dilation_radius)
    guide_hr = bicubic_upsample(guide_lr, target_size)
    guide_rgb = np.asarray(guide_hr.convert("RGB"), dtype=np.float64)
    magnitude = _sobel_magnitude(rgb_to_y(guide_rgb))
    # Floating-point BT.601 conversion can leave tiny non-zero values in a
    # perfectly flat image. They are numerical noise, not real edges.
    positive = magnitude[magnitude > 1e-9]
    if positive.size == 0:
        return np.zeros(magnitude.shape, dtype=bool)
    threshold = float(np.percentile(positive, config.edge_percentile))
    return _dilate_mask(
        (magnitude >= threshold) & (magnitude > 1e-9),
        config.dilation_radius,
    )


def edge_aware_fusion(
    guide_lr: Image.Image,
    nedi_image: Image.Image,
    imdn_image: Image.Image,
    config: EdgeAwareFusionConfig,
) -> EdgeAwareFusionResult:
    """Use IMDN everywhere and a fixed NEDI mixture only on detected edges."""
    nedi_rgb = nedi_image.convert("RGB")
    imdn_rgb = imdn_image.convert("RGB")
    if nedi_rgb.size != imdn_rgb.size:
        raise ValueError(
            "NEDI and IMDN outputs must have the same dimensions; "
            f"received {nedi_rgb.size} and {imdn_rgb.size}."
        )
    edge_mask = detect_strong_edges(
        guide_lr,
        imdn_rgb.size,
        edge_percentile=config.edge_percentile,
        dilation_radius=config.dilation_radius,
    )
    nedi_pixels = np.asarray(nedi_rgb, dtype=np.float64)
    imdn_pixels = np.asarray(imdn_rgb, dtype=np.float64)
    local_nedi_weight = edge_mask[..., None] * config.edge_nedi_weight
    fused_pixels = np.rint(
        local_nedi_weight * nedi_pixels + (1.0 - local_nedi_weight) * imdn_pixels
    ).clip(0, 255).astype(np.uint8)
    return EdgeAwareFusionResult(
        image=Image.fromarray(fused_pixels, mode="RGB"),
        edge_mask=edge_mask,
    )


def evaluate_edge_aware_grid(
    reference_hr: Image.Image,
    guide_lr: Image.Image,
    nedi_image: Image.Image,
    imdn_image: Image.Image,
    scale: int,
    *,
    edge_percentiles: Sequence[float] = DEFAULT_EDGE_PERCENTILES,
    edge_nedi_weights: Sequence[float] = DEFAULT_EDGE_NEDI_WEIGHTS,
    dilation_radius: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate IMDN alone plus every requested edge-aware configuration."""
    if scale not in {2, 3, 4}:
        raise ValueError(f"Scale must be 2, 3, or 4; received {scale}.")
    if not edge_percentiles or not edge_nedi_weights:
        raise ValueError("Edge percentiles and NEDI weights cannot be empty.")

    baseline_metrics = calculate_quality_metrics(reference_hr, imdn_image, border=scale)
    records: list[dict[str, Any]] = [
        {
            "config_id": "imdn_only",
            "edge_percentile": "not_applicable",
            "edge_nedi_weight": 0.0,
            "edge_imdn_weight": 1.0,
            "dilation_radius": 0,
            "edge_pixel_fraction": 0.0,
            **baseline_metrics,
        }
    ]
    seen: set[str] = set()
    for percentile in edge_percentiles:
        for nedi_weight in edge_nedi_weights:
            config = EdgeAwareFusionConfig(percentile, nedi_weight, dilation_radius)
            if config.config_id in seen:
                raise ValueError("Edge-aware candidate configurations must be unique.")
            seen.add(config.config_id)
            result = edge_aware_fusion(guide_lr, nedi_image, imdn_image, config)
            records.append(
                {
                    "config_id": config.config_id,
                    "edge_percentile": config.edge_percentile,
                    "edge_nedi_weight": config.edge_nedi_weight,
                    "edge_imdn_weight": 1.0 - config.edge_nedi_weight,
                    "dilation_radius": config.dilation_radius,
                    "edge_pixel_fraction": result.edge_pixel_fraction,
                    **calculate_quality_metrics(reference_hr, result.image, border=scale),
                }
            )
    return records


def summarise_edge_aware_results(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average every configuration over the same validation samples."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if "config_id" not in record:
            raise ValueError("Every edge-aware record must contain config_id.")
        for metric in METRIC_NAMES:
            if metric not in record or not np.isfinite(float(record[metric])):
                raise ValueError(f"Every {metric} value must be present and finite.")
        grouped[str(record["config_id"])].append(record)
    if not grouped:
        raise ValueError("Cannot summarise an empty edge-aware result set.")
    counts = {len(group) for group in grouped.values()}
    if len(counts) != 1:
        raise ValueError("Every edge-aware configuration must have the same sample count.")

    summary: list[dict[str, Any]] = []
    for config_id in sorted(grouped):
        group = grouped[config_id]
        first = group[0]
        summary.append(
            {
                "config_id": config_id,
                "edge_percentile": first["edge_percentile"],
                "edge_nedi_weight": float(first["edge_nedi_weight"]),
                "edge_imdn_weight": float(first["edge_imdn_weight"]),
                "dilation_radius": int(first["dilation_radius"]),
                "sample_count": len(group),
                "mean_edge_pixel_fraction": mean(
                    float(record["edge_pixel_fraction"]) for record in group
                ),
                **{
                    f"mean_{metric}": mean(float(record[metric]) for record in group)
                    for metric in METRIC_NAMES
                },
            }
        )
    return summary


def select_best_edge_aware_config(
    summary_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select by PSNR-Y and SSIM-Y, preferring less NEDI on exact ties."""
    records = list(summary_records)
    if not records:
        raise ValueError("Cannot select from an empty edge-aware summary.")
    return dict(
        max(
            records,
            key=lambda record: (
                float(record["mean_psnr_y"]),
                float(record["mean_ssim_y"]),
                -float(record["edge_nedi_weight"]),
            ),
        )
    )
