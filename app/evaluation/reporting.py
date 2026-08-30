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


def summarize_deep_learning_results(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add model size, memory, and dimension counts to quality summaries."""
    summaries = summarize_results(records)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["dataset"]),
                str(record["scale"]),
                str(record["method"]),
            )
        ].append(record)

    for summary in summaries:
        group = grouped[
            (
                str(summary["dataset"]),
                str(summary["scale"]),
                str(summary["method"]),
            )
        ]
        parameter_counts = {int(record["parameter_count"]) for record in group}
        if len(parameter_counts) != 1:
            raise ValueError("One summary group contains inconsistent parameter counts.")
        summary.update(
            {
                "parameter_count": parameter_counts.pop(),
                "peak_gpu_memory_mean_mb": mean(
                    float(record["peak_gpu_memory_mb"]) for record in group
                ),
                "peak_gpu_memory_max_mb": max(
                    float(record["peak_gpu_memory_mb"]) for record in group
                ),
                "dimension_adjusted_count": sum(
                    str(record["dimension_adjusted"]).lower() == "true"
                    for record in group
                ),
            }
        )
    return summaries
