"""Validation for prepared benchmark HR/LR image pairs."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import dataset_hr_directory, dataset_lr_directory
from app.evaluation.images import (
    load_rgb_image,
    pair_image_paths,
    validate_hr_lr_dimensions,
)


EXPECTED_IMAGE_COUNTS = {
    "Set5": 5,
    "Set14": 14,
    "BSD100": 100,
    "Urban100": 100,
}


@dataclass(frozen=True)
class DatasetValidation:
    dataset: str
    scale: int
    image_count: int
    hr_directory: Path
    lr_directory: Path


def validate_prepared_dataset(
    dataset: str,
    scale: int,
    data_root: str | Path | None = None,
) -> DatasetValidation:
    """Validate filenames, image count, readability, and dimensions for one dataset."""
    hr_directory = dataset_hr_directory(dataset, data_root)
    lr_directory = dataset_lr_directory(dataset, scale, data_root)
    pairs = pair_image_paths(hr_directory, lr_directory)

    expected_count = EXPECTED_IMAGE_COUNTS.get(dataset)
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(
            f"{dataset} should contain {expected_count} image pairs; found {len(pairs)}."
        )

    for hr_path, lr_path in pairs:
        hr_image = load_rgb_image(hr_path)
        lr_image = load_rgb_image(lr_path)
        validate_hr_lr_dimensions(hr_image, lr_image, scale)

    return DatasetValidation(
        dataset=dataset,
        scale=scale,
        image_count=len(pairs),
        hr_directory=hr_directory,
        lr_directory=lr_directory,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate prepared SR benchmark pairs.")
    parser.add_argument("--dataset", required=True, choices=tuple(EXPECTED_IMAGE_COUNTS))
    parser.add_argument("--scale", required=True, type=int, choices=(2, 3, 4))
    parser.add_argument("--data-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_prepared_dataset(args.dataset, args.scale, args.data_root)
    print(
        f"VALID: {result.dataset} x{result.scale} has "
        f"{result.image_count} complete HR/LR pairs."
    )
    print(f"HR: {result.hr_directory}")
    print(f"LR: {result.lr_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
