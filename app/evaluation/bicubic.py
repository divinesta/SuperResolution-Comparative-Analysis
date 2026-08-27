"""Reusable bicubic baseline evaluation and command-line entry point."""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import skimage
from PIL import Image, __version__ as pillow_version

from app.config import dataset_hr_directory, dataset_lr_directory
from app.evaluation.images import (
    align_hr_to_lr,
    bicubic_upsample,
    load_rgb_image,
    pair_image_paths,
    validate_scale,
)
from app.evaluation.metrics import calculate_quality_metrics
from app.evaluation.timing import measure_runtime


@dataclass(frozen=True)
class BicubicEvaluationConfig:
    """Settings shared by every image in a bicubic evaluation run."""

    dataset: str
    scale: int
    warmup_runs: int = 3
    timed_runs: int = 10

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("Dataset name cannot be empty.")
        validate_scale(self.scale)
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative.")
        if self.timed_runs <= 0:
            raise ValueError("timed_runs must be greater than zero.")


def _environment_fields() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
        "skimage_version": skimage.__version__,
        "operating_system": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


def _save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def evaluate_bicubic_image(
    hr_path: str | Path,
    lr_path: str | Path,
    config: BicubicEvaluationConfig,
    *,
    sr_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one prepared HR/LR image pair with the bicubic baseline."""
    hr_image_path = Path(hr_path)
    lr_image_path = Path(lr_path)
    source_hr = load_rgb_image(hr_image_path)
    lr_image = load_rgb_image(lr_image_path)
    reference_hr = align_hr_to_lr(source_hr, lr_image, config.scale)

    reconstruction, timing = measure_runtime(
        lambda: bicubic_upsample(lr_image, reference_hr.size),
        warmup_runs=config.warmup_runs,
        timed_runs=config.timed_runs,
    )
    metrics = calculate_quality_metrics(
        reference_hr,
        reconstruction,
        border=config.scale,
    )

    if sr_output_dir is not None:
        sr_name = f"{hr_image_path.stem}_bicubic_x{config.scale}.png"
        _save_image(reconstruction, Path(sr_output_dir) / sr_name)

    hr_width, hr_height = reference_hr.size
    lr_width, lr_height = lr_image.size
    return {
        "dataset": config.dataset,
        "image": hr_image_path.name,
        "scale": f"x{config.scale}",
        "method": "bicubic",
        "degradation": "prepared_bicubic_lr",
        "lr_source_file": lr_image_path.name,
        "metric_border_pixels": config.scale,
        **metrics,
        **timing.as_dict(),
        "hr_width": hr_width,
        "hr_height": hr_height,
        "lr_width": lr_width,
        "lr_height": lr_height,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **_environment_fields(),
    }


def evaluate_bicubic_dataset(
    hr_directory: str | Path,
    lr_directory: str | Path,
    config: BicubicEvaluationConfig,
    *,
    sr_output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every complete prepared HR/LR pair in a dataset."""
    return [
        evaluate_bicubic_image(
            hr_path,
            lr_path,
            config,
            sr_output_dir=sr_output_dir,
        )
        for hr_path, lr_path in pair_image_paths(hr_directory, lr_directory)
    ]


def write_results_csv(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write records to CSV without silently replacing an existing experiment."""
    if not records:
        raise ValueError("Cannot write an empty result set.")

    csv_path = Path(output_path)
    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"Result file already exists: {csv_path}. "
            "Choose a new filename or explicitly allow overwrite."
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(csv_path)
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the bicubic SR baseline.")
    parser.add_argument("--dataset", required=True, help="Dataset label, for example Set5.")
    parser.add_argument("--hr-dir", type=Path, help="Direct HR image directory.")
    parser.add_argument("--lr-dir", type=Path, help="Matching prepared LR image directory.")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Dataset root containing folders such as Set5/Set5_HR.",
    )
    parser.add_argument("--scale", required=True, type=int, choices=(2, 3, 4))
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--sr-output-dir", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_root is not None and (args.hr_dir is not None or args.lr_dir is not None):
        parser.error("Use --data-root or the --hr-dir/--lr-dir pair, not both.")
    if (args.hr_dir is None) != (args.lr_dir is None):
        parser.error("--hr-dir and --lr-dir must be provided together.")

    config = BicubicEvaluationConfig(
        dataset=args.dataset,
        scale=args.scale,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    if args.hr_dir is not None:
        hr_directory = args.hr_dir
        lr_directory = args.lr_dir
    else:
        hr_directory = dataset_hr_directory(args.dataset, args.data_root)
        lr_directory = dataset_lr_directory(args.dataset, args.scale, args.data_root)

    records = evaluate_bicubic_dataset(
        hr_directory,
        lr_directory,
        config,
        sr_output_dir=args.sr_output_dir,
    )
    output_path = write_results_csv(
        records,
        args.output_csv,
        overwrite=args.overwrite,
    )

    print(f"Evaluated {len(records)} images from {config.dataset} at x{config.scale}.")
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
