"""NEDI x2 evaluation using the project's fixed super-resolution protocol."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import dataset_hr_directory, dataset_lr_directory
from app.evaluation.experiment import environment_fields, save_image, write_results_csv
from app.evaluation.images import (
    load_rgb_image,
    pair_image_paths,
    validate_hr_lr_dimensions,
)
from app.evaluation.metrics import calculate_quality_metrics
from app.evaluation.timing import measure_runtime
from app.traditional.nedi import NEDIConfig, nedi_upsample_x2_rgb


@dataclass(frozen=True)
class NEDIEvaluationConfig:
    """Settings shared by every native NEDI x2 image evaluation."""

    dataset: str
    scale: int = 2
    window_size: int = 8
    edge_threshold: float = 8.0
    warmup_runs: int = 3
    timed_runs: int = 10

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("Dataset name cannot be empty.")
        if self.scale != 2:
            raise ValueError("The native NEDI evaluator currently supports x2 only.")
        NEDIConfig(
            window_size=self.window_size,
            edge_threshold=self.edge_threshold,
        )
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative.")
        if self.timed_runs <= 0:
            raise ValueError("timed_runs must be greater than zero.")


def evaluate_nedi_image(
    hr_path: str | Path,
    lr_path: str | Path,
    config: NEDIEvaluationConfig,
    *,
    sr_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one prepared HR/LR pair with the native NEDI x2 method."""
    hr_image_path = Path(hr_path)
    lr_image_path = Path(lr_path)
    reference_hr = load_rgb_image(hr_image_path)
    lr_image = load_rgb_image(lr_image_path)
    validate_hr_lr_dimensions(reference_hr, lr_image, config.scale)
    nedi_config = NEDIConfig(
        window_size=config.window_size,
        edge_threshold=config.edge_threshold,
    )

    reconstruction_result, timing = measure_runtime(
        lambda: nedi_upsample_x2_rgb(lr_image, reference_hr.size, nedi_config),
        warmup_runs=config.warmup_runs,
        timed_runs=config.timed_runs,
    )
    metrics = calculate_quality_metrics(
        reference_hr,
        reconstruction_result.image,
        border=config.scale,
    )

    if sr_output_dir is not None:
        sr_name = f"{hr_image_path.stem}_nedi_x2.png"
        save_image(reconstruction_result.image, Path(sr_output_dir) / sr_name)

    hr_width, hr_height = reference_hr.size
    lr_width, lr_height = lr_image.size
    return {
        "dataset": config.dataset,
        "image": hr_image_path.name,
        "scale": "x2",
        "method": "nedi",
        "degradation": "prepared_bicubic_lr",
        "lr_source_file": lr_image_path.name,
        "metric_border_pixels": config.scale,
        "dimension_policy": "native_nedi_grid_then_target_size_adjustment",
        "ssim_protocol": "gaussian_11x11_sigma_1.5_population_covariance",
        "nedi_window_size": config.window_size,
        "nedi_edge_threshold": config.edge_threshold,
        "nedi_scale_strategy": "native_x2",
        "nedi_native_passes": 1,
        "nedi_edge_pixel_count": reconstruction_result.stats.edge_pixel_count,
        "nedi_pixel_count": reconstruction_result.stats.nedi_pixel_count,
        "nedi_bilinear_fallback_count": (
            reconstruction_result.stats.bilinear_fallback_count
        ),
        "nedi_numerical_fallback_count": (
            reconstruction_result.stats.numerical_fallback_count
        ),
        "nedi_dimension_adjustment": reconstruction_result.dimension_adjusted,
        "nedi_native_width": reconstruction_result.native_size[0],
        "nedi_native_height": reconstruction_result.native_size[1],
        "nedi_sampling_grid": "even_insertion_2x_half_pixel_aligned",
        **metrics,
        **timing.as_dict(),
        "hr_width": hr_width,
        "hr_height": hr_height,
        "lr_width": lr_width,
        "lr_height": lr_height,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **environment_fields(),
    }


def evaluate_nedi_dataset(
    hr_directory: str | Path,
    lr_directory: str | Path,
    config: NEDIEvaluationConfig,
    *,
    sr_output_dir: str | Path | None = None,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate prepared x2 pairs, optionally limiting a pilot to early files."""
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be greater than zero when supplied.")

    pairs = pair_image_paths(hr_directory, lr_directory)
    if max_images is not None:
        pairs = pairs[:max_images]
    return [
        evaluate_nedi_image(
            hr_path,
            lr_path,
            config,
            sr_output_dir=sr_output_dir,
        )
        for hr_path, lr_path in pairs
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate native NEDI x2.")
    parser.add_argument("--dataset", required=True, help="Dataset label, for example Set5.")
    parser.add_argument("--hr-dir", type=Path, help="Direct HR image directory.")
    parser.add_argument("--lr-dir", type=Path, help="Matching prepared LR image directory.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--sr-output-dir", type=Path)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--edge-threshold", type=float, default=8.0)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is not None and (args.hr_dir is not None or args.lr_dir is not None):
        raise ValueError("Use --data-root or the --hr-dir/--lr-dir pair, not both.")
    if (args.hr_dir is None) != (args.lr_dir is None):
        raise ValueError("--hr-dir and --lr-dir must be provided together.")

    config = NEDIEvaluationConfig(
        dataset=args.dataset,
        window_size=args.window_size,
        edge_threshold=args.edge_threshold,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    if args.hr_dir is not None:
        hr_directory = args.hr_dir
        lr_directory = args.lr_dir
    else:
        hr_directory = dataset_hr_directory(args.dataset, args.data_root)
        lr_directory = dataset_lr_directory(args.dataset, 2, args.data_root)

    records = evaluate_nedi_dataset(
        hr_directory,
        lr_directory,
        config,
        sr_output_dir=args.sr_output_dir,
        max_images=args.max_images,
    )
    output_path = write_results_csv(records, args.output_csv, overwrite=args.overwrite)
    print(f"Evaluated {len(records)} images from {config.dataset} with NEDI x2.")
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
