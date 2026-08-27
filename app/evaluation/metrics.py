"""PSNR and SSIM measurements for RGB and luminance image data."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    # Some notebook environments can retain an incomplete lazy-loader interface
    # after upgrading scikit-image in a running kernel. The implementation
    # modules remain available and provide the same metric functions.
    from skimage.metrics._structural_similarity import structural_similarity
    from skimage.metrics.simple_metrics import peak_signal_noise_ratio


ImageInput: TypeAlias = Image.Image | NDArray[np.generic]
SSIM_SETTINGS = {
    "data_range": 255.0,
    "gaussian_weights": True,
    "sigma": 1.5,
    "use_sample_covariance": False,
}


def _as_rgb_array(image: ImageInput) -> NDArray[np.float64]:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float64)
    else:
        array = np.asarray(image, dtype=np.float64)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3); got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite pixel values.")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("Image pixel values must be within the range 0 to 255.")
    return array


def crop_border(image: NDArray[np.float64], border: int) -> NDArray[np.float64]:
    """Remove an equal number of pixels from all four image edges."""
    if border < 0:
        raise ValueError(f"Border cannot be negative; received {border}.")
    if border == 0:
        return image
    if image.shape[0] <= 2 * border or image.shape[1] <= 2 * border:
        raise ValueError(
            f"Image shape {image.shape} is too small for a {border}-pixel border crop."
        )
    return image[border:-border, border:-border, ...]


def rgb_to_y(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert 0-255 RGB pixels to the BT.601 limited-range Y channel."""
    return (
        16.0
        + (65.481 * image[..., 0]) / 255.0
        + (128.553 * image[..., 1]) / 255.0
        + (24.966 * image[..., 2]) / 255.0
    )


def calculate_quality_metrics(
    reference: ImageInput,
    reconstruction: ImageInput,
    border: int,
) -> dict[str, float]:
    """Calculate RGB and Y-channel PSNR/SSIM after a shared border crop."""
    reference_rgb = _as_rgb_array(reference)
    reconstruction_rgb = _as_rgb_array(reconstruction)

    if reference_rgb.shape != reconstruction_rgb.shape:
        raise ValueError(
            "Reference and reconstruction must have identical shapes; "
            f"received {reference_rgb.shape} and {reconstruction_rgb.shape}."
        )

    reference_rgb = crop_border(reference_rgb, border)
    reconstruction_rgb = crop_border(reconstruction_rgb, border)
    reference_y = rgb_to_y(reference_rgb)
    reconstruction_y = rgb_to_y(reconstruction_rgb)

    return {
        "psnr_y": float(
            peak_signal_noise_ratio(reference_y, reconstruction_y, data_range=255.0)
        ),
        "ssim_y": float(
            structural_similarity(
                reference_y,
                reconstruction_y,
                **SSIM_SETTINGS,
            )
        ),
        "psnr_rgb": float(
            peak_signal_noise_ratio(
                reference_rgb,
                reconstruction_rgb,
                data_range=255.0,
            )
        ),
        "ssim_rgb": float(
            structural_similarity(
                reference_rgb,
                reconstruction_rgb,
                channel_axis=2,
                **SSIM_SETTINGS,
            )
        ),
    }
