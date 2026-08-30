"""Shared experiment metadata, image saving, and CSV writing utilities."""

from __future__ import annotations

import csv
import platform
from pathlib import Path
from typing import Any

import numpy as np
import skimage
from PIL import Image, __version__ as pillow_version


def environment_fields() -> dict[str, str]:
    """Return software and host information for one experiment record."""
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
        "skimage_version": skimage.__version__,
        "operating_system": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


def save_image(image: Image.Image, output_path: str | Path) -> None:
    """Save a reconstructed image, creating its parent directory when needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_results_csv(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write records without silently replacing an existing experiment."""
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


def read_results_csv(output_path: str | Path) -> list[dict[str, str]]:
    """Read an existing result checkpoint, or return an empty list if absent."""
    csv_path = Path(output_path)
    if not csv_path.exists():
        return []
    if not csv_path.is_file():
        raise ValueError(f"Result checkpoint is not a file: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
