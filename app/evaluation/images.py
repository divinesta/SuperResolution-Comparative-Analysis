"""Image loading, alignment, and bicubic resizing operations."""

from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_scale(scale: int) -> None:
    """Raise an error when a super-resolution scale is unsupported."""
    if scale not in {2, 3, 4}:
        raise ValueError(f"Scale must be 2, 3, or 4; received {scale}.")


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load an image, apply its EXIF orientation, and return an RGB copy."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def modcrop(image: Image.Image, scale: int) -> Image.Image:
    """Crop the bottom and right edges so both dimensions divide by scale."""
    validate_scale(scale)

    width, height = image.size
    cropped_width = width - (width % scale)
    cropped_height = height - (height % scale)

    if cropped_width == 0 or cropped_height == 0:
        raise ValueError(
            f"Image size {image.size} is too small for scale x{scale}."
        )

    return image.crop((0, 0, cropped_width, cropped_height))


def bicubic_downsample(hr_image: Image.Image, scale: int) -> tuple[Image.Image, Image.Image]:
    """Align an HR image and create its controlled bicubic LR counterpart."""
    aligned_hr = modcrop(hr_image.convert("RGB"), scale)
    width, height = aligned_hr.size
    lr_image = aligned_hr.resize(
        (width // scale, height // scale),
        resample=Image.Resampling.BICUBIC,
    )
    return aligned_hr, lr_image


def bicubic_upsample(lr_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Reconstruct an RGB image at the requested size using bicubic resizing."""
    width, height = target_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Target size must be positive; received {target_size}.")

    return lr_image.convert("RGB").resize(
        target_size,
        resample=Image.Resampling.BICUBIC,
    )


def align_hr_to_lr(
    hr_image: Image.Image,
    lr_image: Image.Image,
    scale: int,
) -> Image.Image:
    """Align an HR reference to the exact dimensions represented by a paired LR image."""
    validate_scale(scale)
    target_width = lr_image.width * scale
    target_height = lr_image.height * scale
    width_difference = hr_image.width - target_width
    height_difference = hr_image.height - target_height

    if not (0 <= width_difference < scale and 0 <= height_difference < scale):
        raise ValueError(
            "HR/LR dimensions do not form a valid paired sample at "
            f"x{scale}: HR={hr_image.size}, LR={lr_image.size}."
        )

    return hr_image.convert("RGB").crop((0, 0, target_width, target_height))


def list_image_paths(directory: str | Path) -> list[Path]:
    """Return supported image files in deterministic filename order."""
    image_directory = Path(directory)
    if not image_directory.is_dir():
        raise NotADirectoryError(f"HR directory does not exist: {image_directory}")

    paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found in: {image_directory}")
    return paths


def pair_image_paths(
    hr_directory: str | Path,
    lr_directory: str | Path,
) -> list[tuple[Path, Path]]:
    """Match prepared HR and LR images by filename and reject incomplete pairs."""
    hr_paths = {path.name: path for path in list_image_paths(hr_directory)}
    lr_paths = {path.name: path for path in list_image_paths(lr_directory)}

    missing_lr = sorted(hr_paths.keys() - lr_paths.keys())
    extra_lr = sorted(lr_paths.keys() - hr_paths.keys())
    if missing_lr or extra_lr:
        details: list[str] = []
        if missing_lr:
            details.append(f"missing LR files: {', '.join(missing_lr)}")
        if extra_lr:
            details.append(f"LR files without HR matches: {', '.join(extra_lr)}")
        raise ValueError("HR/LR filename mismatch; " + "; ".join(details))

    return [(hr_paths[name], lr_paths[name]) for name in sorted(hr_paths)]
