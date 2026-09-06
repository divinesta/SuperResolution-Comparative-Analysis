"""Create reproducible full-image and close-up figures for Phase 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.evaluation.metrics import calculate_quality_metrics


SAMPLES = (
    ("Set5", "baby", "smooth face and gradual colour regions"),
    ("Set5", "butterfly", "fine lines and high-frequency texture"),
    ("Set14", "baboon", "difficult natural texture"),
    ("Urban100", "img_004", "architectural edges and repetitive structure"),
)
SCALES = ("x2", "x3", "x4")
METHODS = ("bicubic", "nedi", "fsrcnn", "imdn")
DISPLAY_METHODS = ("reference_hr", *METHODS)
CROP_REGIONS = {
    ("Set5", "baby"): (110, 120, 405, 365),
    ("Set5", "butterfly"): (40, 35, 225, 225),
    ("Set14", "baboon"): (25, 10, 245, 205),
    ("Urban100", "img_004"): (645, 0, 1024, 345),
}


def validate_visual_samples(sample_root: str | Path) -> dict[str, int]:
    """Validate every selected LR, reference, and reconstruction image."""
    root = Path(sample_root)
    image_count = 0
    for dataset, image, _ in SAMPLES:
        crop = CROP_REGIONS[(dataset, image)]
        for scale in SCALES:
            group = root / dataset / image / scale
            paths = {name: group / f"{name}.png" for name in ("input_lr", *DISPLAY_METHODS)}
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing visual sample files: {missing}")
            with Image.open(paths["reference_hr"]) as reference:
                reference_size = reference.size
                if reference.mode != "RGB":
                    raise ValueError(f"Reference image is not RGB: {paths['reference_hr']}")
            left, top, right, bottom = crop
            if not (0 <= left < right <= reference_size[0] and 0 <= top < bottom <= reference_size[1]):
                raise ValueError(f"Crop {crop} is outside {reference_size} for {dataset}/{image}.")
            for method in METHODS:
                with Image.open(paths[method]) as reconstructed:
                    if reconstructed.mode != "RGB" or reconstructed.size != reference_size:
                        raise ValueError(f"Invalid {method} visual sample: {paths[method]}")
            image_count += len(paths)
    return {"group_count": len(SAMPLES) * len(SCALES), "image_count": image_count}


def calculate_visual_sample_metrics(sample_root: str | Path) -> list[dict[str, Any]]:
    """Calculate the fixed project metrics for the selected reconstructions."""
    validate_visual_samples(sample_root)
    root = Path(sample_root)
    records: list[dict[str, Any]] = []
    for dataset, image, reason in SAMPLES:
        for scale_label in SCALES:
            scale = int(scale_label[1:])
            group = root / dataset / image / scale_label
            with Image.open(group / "reference_hr.png") as source:
                reference = source.convert("RGB")
            for method in METHODS:
                with Image.open(group / f"{method}.png") as source:
                    reconstruction = source.convert("RGB")
                records.append(
                    {
                        "dataset": dataset,
                        "image": f"{image}.png",
                        "scale": scale_label,
                        "method": method,
                        "selection_reason": reason,
                        **calculate_quality_metrics(reference, reconstruction, border=scale),
                    }
                )
    return records


def generate_visual_comparison_figures(
    sample_root: str | Path, output_directory: str | Path
) -> list[Path]:
    """Generate figures with identical full views and close-ups for every method."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    validate_visual_samples(sample_root)
    root = Path(sample_root)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    titles = {
        "reference_hr": "Reference HR",
        "bicubic": "Bicubic",
        "nedi": "NEDI",
        "fsrcnn": "FSRCNN",
        "imdn": "IMDN",
    }

    for dataset, image, reason in SAMPLES:
        crop = CROP_REGIONS[(dataset, image)]
        left, top, right, bottom = crop
        for scale in SCALES:
            group = root / dataset / image / scale
            loaded = {}
            for method in DISPLAY_METHODS:
                with Image.open(group / f"{method}.png") as source:
                    loaded[method] = source.convert("RGB")

            figure, axes = plt.subplots(2, len(DISPLAY_METHODS), figsize=(17, 8))
            for column, method in enumerate(DISPLAY_METHODS):
                axes[0, column].imshow(loaded[method], interpolation="nearest")
                axes[0, column].add_patch(
                    patches.Rectangle(
                        (left, top), right - left, bottom - top,
                        linewidth=1.8, edgecolor="#ffcc00", facecolor="none",
                    )
                )
                axes[0, column].set_title(titles[method], fontsize=12)
                axes[0, column].axis("off")
                axes[1, column].imshow(
                    loaded[method].crop(crop), interpolation="nearest"
                )
                axes[1, column].axis("off")
            axes[0, 0].set_ylabel("Full image", fontsize=11)
            axes[1, 0].set_ylabel("Identical close-up", fontsize=11)
            figure.suptitle(
                f"{dataset} — {image}.png — {scale}\n{reason}", fontsize=15
            )
            figure.tight_layout(rect=(0, 0, 1, 0.93))
            path = output / f"{dataset}_{image}_{scale}_comparison.png"
            figure.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(figure)
            generated.append(path)
    return generated
