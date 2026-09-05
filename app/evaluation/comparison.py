"""Validation, tables, and figures for the controlled Phase 4 comparison."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from app.evaluation.experiment import read_results_csv, write_results_csv


DATASETS = ("Set5", "Set14", "BSD100", "Urban100")
SCALES = ("x2", "x3", "x4")
METHODS = ("bicubic", "nedi", "fsrcnn", "imdn")
EXPECTED_IMAGE_COUNTS = {"Set5": 5, "Set14": 14, "BSD100": 100, "Urban100": 100}
QUALITY_METRICS = ("psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb")
SUMMARY_METRICS = QUALITY_METRICS + ("latency_mean_ms", "latency_median_ms")
SUMMARY_PATHS = {
    "bicubic": Path("results/metrics/final/bicubic_summary_final.csv"),
    "nedi": Path("results/metrics/final/nedi_summary_final.csv"),
    "fsrcnn": Path("results/metrics/final/fsrcnn_summary_gpu.csv"),
    "imdn": Path("results/metrics/final/imdn_summary_gpu.csv"),
}
TIMING_CONTEXTS = {
    "bicubic": "CPU; Google Colab default CPU (exact model not recorded)",
    "nedi": "CPU; AWS m7i.2xlarge",
    "fsrcnn": "GPU; NVIDIA Tesla T4",
    "imdn": "GPU; NVIDIA Tesla T4",
}


def _as_finite_float(row: dict[str, str], field: str, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} has an invalid {field!r} value.") from error
    if not math.isfinite(value):
        raise ValueError(f"{source} has a non-finite {field!r} value.")
    return value


def load_phase4_summaries(repo_root: str | Path) -> list[dict[str, Any]]:
    """Load and validate the 48 required method/dataset/scale summaries."""
    root = Path(repo_root)
    combined: list[dict[str, Any]] = []

    for expected_method, relative_path in SUMMARY_PATHS.items():
        source = root / relative_path
        rows = read_results_csv(source)
        if not rows:
            raise ValueError(f"Required Phase 4 summary is missing or empty: {source}")

        expected_keys = {(dataset, scale) for dataset in DATASETS for scale in SCALES}
        actual_keys: set[tuple[str, str]] = set()
        for row in rows:
            dataset = row.get("dataset", "")
            scale = row.get("scale", "")
            method = row.get("method", "").lower()
            key = (dataset, scale)
            if method != expected_method:
                raise ValueError(f"{source} contains method {method!r}, expected {expected_method!r}.")
            if key in actual_keys:
                raise ValueError(f"{source} contains duplicate group {dataset} {scale}.")
            actual_keys.add(key)
            try:
                image_count = int(row["image_count"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{source} has an invalid image_count.") from error
            if dataset not in EXPECTED_IMAGE_COUNTS or image_count != EXPECTED_IMAGE_COUNTS[dataset]:
                raise ValueError(f"{source} has the wrong image count for {dataset} {scale}.")

            record: dict[str, Any] = {
                "dataset": dataset,
                "scale": scale,
                "method": method,
                "image_count": image_count,
                **{field: _as_finite_float(row, field, source) for field in SUMMARY_METRICS},
                "timing_device": "gpu" if method in {"fsrcnn", "imdn"} else "cpu",
                "timing_context": TIMING_CONTEXTS[method],
                "parameter_count": int(row["parameter_count"]) if row.get("parameter_count") else "",
                "peak_gpu_memory_mean_mb": (
                    float(row["peak_gpu_memory_mean_mb"])
                    if row.get("peak_gpu_memory_mean_mb")
                    else ""
                ),
                "peak_gpu_memory_max_mb": (
                    float(row["peak_gpu_memory_max_mb"])
                    if row.get("peak_gpu_memory_max_mb")
                    else ""
                ),
            }
            combined.append(record)

        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing or extra:
            raise ValueError(f"{source} groups do not match the required datasets and scales.")

    lookup = {(row["dataset"], row["scale"], row["method"]): row for row in combined}
    for dataset in DATASETS:
        for scale in SCALES:
            baseline = lookup[(dataset, scale, "bicubic")]
            group = [lookup[(dataset, scale, method)] for method in METHODS]
            for metric in QUALITY_METRICS:
                ranked = sorted(group, key=lambda row: float(row[metric]), reverse=True)
                for rank, row in enumerate(ranked, start=1):
                    row[f"{metric}_rank"] = rank
            for row in group:
                row["psnr_y_vs_bicubic_db"] = float(row["psnr_y"]) - float(baseline["psnr_y"])
                row["ssim_y_vs_bicubic"] = float(row["ssim_y"]) - float(baseline["ssim_y"])

    dataset_order = {name: index for index, name in enumerate(DATASETS)}
    scale_order = {name: index for index, name in enumerate(SCALES)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    combined.sort(
        key=lambda row: (
            dataset_order[row["dataset"]],
            scale_order[row["scale"]],
            method_order[row["method"]],
        )
    )
    return combined


def validate_detailed_results(
    detail_paths: Iterable[str | Path],
    summaries: list[dict[str, Any]],
    method: str,
    *,
    tolerance: float = 1e-9,
) -> dict[str, int]:
    """Confirm detailed rows are complete and reproduce the supplied summary."""
    records: list[dict[str, str]] = []
    for path in detail_paths:
        records.extend(read_results_csv(path))
    if not records:
        raise ValueError(f"No detailed {method} records were found.")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    unique_images: set[tuple[str, str, str]] = set()
    for record in records:
        if record.get("method", "").lower() != method:
            raise ValueError(f"Detailed {method} input contains another method.")
        key = (record.get("dataset", ""), record.get("scale", ""))
        image_key = (*key, record.get("image", ""))
        if image_key in unique_images:
            raise ValueError(f"Detailed {method} results contain duplicate image {image_key}.")
        unique_images.add(image_key)
        grouped[key].append(record)

    summary_lookup = {
        (row["dataset"], row["scale"]): row
        for row in summaries
        if row["method"] == method
    }
    expected_groups = {(dataset, scale) for dataset in DATASETS for scale in SCALES}
    if set(grouped) != expected_groups or set(summary_lookup) != expected_groups:
        raise ValueError(f"Detailed {method} results do not cover all 12 required groups.")

    for (dataset, scale), group in grouped.items():
        if len(group) != EXPECTED_IMAGE_COUNTS[dataset]:
            raise ValueError(f"Detailed {method} count is wrong for {dataset} {scale}.")
        summary = summary_lookup[(dataset, scale)]
        for field in SUMMARY_METRICS:
            calculated = mean(float(record[field]) for record in group)
            if not math.isclose(calculated, float(summary[field]), abs_tol=tolerance):
                raise ValueError(
                    f"Detailed {method} {dataset} {scale} does not reproduce summary field {field}."
                )

    return {"record_count": len(records), "group_count": len(grouped)}


def build_method_overview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create image-count-weighted quality averages and win counts per method."""
    overview: list[dict[str, Any]] = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        total = sum(int(row["image_count"]) for row in group)
        overview.append(
            {
                "method": method,
                "evaluated_image_scale_pairs": total,
                "weighted_psnr_y": sum(float(row["psnr_y"]) * int(row["image_count"]) for row in group) / total,
                "weighted_ssim_y": sum(float(row["ssim_y"]) * int(row["image_count"]) for row in group) / total,
                "psnr_y_group_wins": sum(int(row["psnr_y_rank"]) == 1 for row in group),
                "ssim_y_group_wins": sum(int(row["ssim_y_rank"]) == 1 for row in group),
                "timing_device": group[0]["timing_device"],
                "timing_context": group[0]["timing_context"],
            }
        )
    overview.sort(key=lambda row: float(row["weighted_psnr_y"]), reverse=True)
    return overview


def build_deep_learning_efficiency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize comparable Tesla T4 model size and peak-memory measurements."""
    output: list[dict[str, Any]] = []
    for method in ("fsrcnn", "imdn"):
        for scale in SCALES:
            group = [row for row in rows if row["method"] == method and row["scale"] == scale]
            parameter_counts = {int(row["parameter_count"]) for row in group}
            if len(parameter_counts) != 1:
                raise ValueError(f"{method} {scale} has inconsistent parameter counts.")
            total = sum(int(row["image_count"]) for row in group)
            output.append(
                {
                    "method": method,
                    "scale": scale,
                    "parameter_count": parameter_counts.pop(),
                    "weighted_peak_gpu_memory_mean_mb": sum(
                        float(row["peak_gpu_memory_mean_mb"]) * int(row["image_count"])
                        for row in group
                    ) / total,
                    "peak_gpu_memory_max_mb": max(float(row["peak_gpu_memory_max_mb"]) for row in group),
                    "timing_device": "gpu",
                    "device_name": "NVIDIA Tesla T4",
                }
            )
    return output


def write_phase4_tables(
    rows: list[dict[str, Any]], output_directory: str | Path, *, overwrite: bool = True
) -> dict[str, Path]:
    """Write the combined comparison and concise method-level tables."""
    output = Path(output_directory)
    return {
        "comparison": write_results_csv(
            rows, output / "comparative_summary.csv", overwrite=overwrite
        ),
        "overview": write_results_csv(
            build_method_overview(rows), output / "method_overview.csv", overwrite=overwrite
        ),
        "deep_learning_efficiency": write_results_csv(
            build_deep_learning_efficiency(rows),
            output / "deep_learning_efficiency.csv",
            overwrite=overwrite,
        ),
    }


def generate_phase4_figures(
    rows: list[dict[str, Any]], output_directory: str | Path
) -> list[Path]:
    """Generate report-ready quality and efficiency figures."""
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    colours = {"bicubic": "#7f8c8d", "nedi": "#d9822b", "fsrcnn": "#2d72b8", "imdn": "#2f855a"}
    lookup = {(row["dataset"], row["scale"], row["method"]): row for row in rows}
    paths: list[Path] = []

    for metric, ylabel, filename in (
        ("psnr_y", "PSNR-Y (dB; higher is better)", "psnr_y_comparison.png"),
        ("ssim_y", "SSIM-Y (higher is better)", "ssim_y_comparison.png"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
        positions = np.arange(len(DATASETS))
        width = 0.19
        for axis, scale in zip(axes, SCALES):
            for index, method in enumerate(METHODS):
                values = [float(lookup[(dataset, scale, method)][metric]) for dataset in DATASETS]
                axis.bar(positions + (index - 1.5) * width, values, width, label=method.upper(), color=colours[method])
            axis.set_title(scale)
            axis.set_xticks(positions, DATASETS, rotation=20)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.suptitle(f"{ylabel.split(' (')[0]} comparison by dataset and scale", y=0.98, fontsize=14)
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
            ncol=4, frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        path = output / filename
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    positions = np.arange(len(DATASETS))
    width = 0.19
    for axis, scale in zip(axes, SCALES):
        for index, method in enumerate(METHODS):
            values = [float(lookup[(dataset, scale, method)]["latency_mean_ms"]) for dataset in DATASETS]
            axis.bar(positions + (index - 1.5) * width, values, width, label=method.upper(), color=colours[method])
        axis.set_yscale("log")
        axis.set_title(scale)
        axis.set_xticks(positions, DATASETS, rotation=20)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean latency (ms, logarithmic scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "Observed latency by dataset and scale (different hardware contexts)\n"
        "Google Colab CPU for Bicubic; AWS CPU for NEDI; Tesla T4 GPU for FSRCNN/IMDN",
        y=0.99,
        fontsize=14,
    )
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.87),
        ncol=4, frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.78))
    latency_path = output / "latency_by_hardware_context.png"
    figure.savefig(latency_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(latency_path)

    efficiency = build_deep_learning_efficiency(rows)
    labels = [f"{row['method'].upper()} {row['scale']}" for row in efficiency]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = np.arange(len(labels))
    bar_colours = [colours[row["method"]] for row in efficiency]
    axes[0].bar(positions, [int(row["parameter_count"]) for row in efficiency], color=bar_colours)
    axes[0].set_ylabel("Trainable parameters")
    axes[0].ticklabel_format(axis="y", style="plain")
    axes[1].bar(positions, [float(row["peak_gpu_memory_max_mb"]) for row in efficiency], color=bar_colours)
    axes[1].set_ylabel("Maximum peak GPU memory (MB)")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Deep-learning efficiency on NVIDIA Tesla T4", fontsize=14)
    figure.tight_layout()
    efficiency_path = output / "deep_learning_efficiency.png"
    figure.savefig(efficiency_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(efficiency_path)
    return paths
