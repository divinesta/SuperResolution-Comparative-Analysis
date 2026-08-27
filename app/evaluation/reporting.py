"""Aggregation helpers for machine-readable experiment results."""

from collections import defaultdict
from statistics import mean
from typing import Any


SUMMARY_METRICS = (
    "psnr_y",
    "ssim_y",
    "psnr_rgb",
    "ssim_rgb",
    "latency_mean_ms",
    "latency_median_ms",
)


def summarize_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average per-image measurements for each dataset, scale, and method."""
    if not records:
        raise ValueError("Cannot summarize an empty result set.")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["scale"]),
            str(record["method"]),
        )
        grouped[key].append(record)

    summaries: list[dict[str, Any]] = []
    for (dataset, scale, method), group in grouped.items():
        summaries.append(
            {
                "dataset": dataset,
                "scale": scale,
                "method": method,
                "image_count": len(group),
                **{
                    metric: mean(float(record[metric]) for record in group)
                    for metric in SUMMARY_METRICS
                },
            }
        )
    return summaries
