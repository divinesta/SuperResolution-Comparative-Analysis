"""Native x2 New Edge-Directed Interpolation for luminance images.

The implementation follows the two-stage interpolation described by Li and
Orchard. It deliberately contains no dataset loading, metric calculation, or
result writing; those responsibilities belong to the evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image


FloatImage = NDArray[np.float64]

# Keys cubic weights for a +0.5 pixel shift. Even-insertion NEDI places LR
# samples at HR sites (2i, 2j). Bicubic-prepared benchmark pairs, and this
# project's bicubic baseline, sit on the Pillow/MATLAB resize grid, which is
# half a pixel away. See Li and Orchard (2001) for the insertion lattice and
# the evaluation protocol for why the shift is applied only when assembling
# an RGB result against an HR reference.
_HALF_PIXEL_KERNEL = np.array([-0.0625, 0.5625, 0.5625, -0.0625], dtype=np.float64)


@dataclass(frozen=True)
class NEDIConfig:
    """Parameters for one native NEDI x2 luminance pass."""

    window_size: int = 8
    edge_threshold: float = 8.0

    def __post_init__(self) -> None:
        if self.window_size < 4 or self.window_size % 2 != 0:
            raise ValueError("window_size must be an even integer of at least 4.")
        if self.edge_threshold < 0:
            raise ValueError("edge_threshold cannot be negative.")


@dataclass(frozen=True)
class NEDIStats:
    """Counts describing how missing pixels were reconstructed."""

    interpolated_pixel_count: int
    edge_pixel_count: int
    nedi_pixel_count: int
    bilinear_fallback_count: int
    numerical_fallback_count: int


@dataclass(frozen=True)
class NEDIResult:
    """A native x2 luminance reconstruction and its processing counts."""

    image: FloatImage
    stats: NEDIStats


@dataclass(frozen=True)
class NEDIRGBResult:
    """An RGB reconstruction assembled from NEDI luminance and bicubic chroma."""

    image: Image.Image
    stats: NEDIStats
    native_size: tuple[int, int]
    target_size: tuple[int, int]
    dimension_adjusted: bool


@dataclass
class _MutableStats:
    interpolated_pixel_count: int = 0
    edge_pixel_count: int = 0
    nedi_pixel_count: int = 0
    bilinear_fallback_count: int = 0
    numerical_fallback_count: int = 0

    def freeze(self) -> NEDIStats:
        return NEDIStats(
            interpolated_pixel_count=self.interpolated_pixel_count,
            edge_pixel_count=self.edge_pixel_count,
            nedi_pixel_count=self.nedi_pixel_count,
            bilinear_fallback_count=self.bilinear_fallback_count,
            numerical_fallback_count=self.numerical_fallback_count,
        )


def _validate_luminance(luminance: NDArray[np.generic]) -> FloatImage:
    image = np.asarray(luminance, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D luminance image; got shape {image.shape}.")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(
            f"Luminance image must be at least 2x2; got shape {image.shape}."
        )
    if not np.isfinite(image).all():
        raise ValueError("Luminance image contains NaN or infinite values.")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("Luminance values must be within the range 0 to 255.")
    return image


def _bilinear_grid(source: FloatImage) -> FloatImage:
    """Build the 2h x 2w even-insertion grid used by Li and Orchard.

    Original samples occupy ``output[2i, 2j]``. Missing sites are initialised
    with bilinear interpolation; covariance-based NEDI later replaces edge
    pixels. The extra final row and column have no further original sample to
    their south or east, so they replicate or average from the available side.
    """
    height, width = source.shape
    output = np.empty((2 * height, 2 * width), dtype=np.float64)
    output[::2, ::2] = source
    output[::2, 1:-1:2] = (source[:, :-1] + source[:, 1:]) / 2.0
    output[::2, -1] = source[:, -1]
    output[1:-1:2, ::2] = (source[:-1, :] + source[1:, :]) / 2.0
    output[-1, ::2] = source[-1, :]
    output[1:-1:2, 1:-1:2] = (
        source[:-1, :-1]
        + source[:-1, 1:]
        + source[1:, :-1]
        + source[1:, 1:]
    ) / 4.0
    output[1:-1:2, -1] = (source[:-1, -1] + source[1:, -1]) / 2.0
    output[-1, 1:-1:2] = (source[-1, :-1] + source[-1, 1:]) / 2.0
    output[-1, -1] = source[-1, -1]
    return output


def _shift_half_pixel(image: FloatImage) -> FloatImage:
    """Shift an even-insertion image by +0.5 px onto the bicubic resize grid.

    Separable Keys cubic (a = -0.5) evaluated at offset 0.5. Output pixel
    ``(r, c)`` is sampled from input ``(r - 0.5, c - 0.5)``.
    """
    padded_rows = np.pad(image, ((2, 1), (0, 0)), mode="edge")
    shifted_rows = (
        _HALF_PIXEL_KERNEL[0] * padded_rows[:-3]
        + _HALF_PIXEL_KERNEL[1] * padded_rows[1:-2]
        + _HALF_PIXEL_KERNEL[2] * padded_rows[2:-1]
        + _HALF_PIXEL_KERNEL[3] * padded_rows[3:]
    )
    padded_columns = np.pad(shifted_rows, ((0, 0), (2, 1)), mode="edge")
    return (
        _HALF_PIXEL_KERNEL[0] * padded_columns[:, :-3]
        + _HALF_PIXEL_KERNEL[1] * padded_columns[:, 1:-2]
        + _HALF_PIXEL_KERNEL[2] * padded_columns[:, 2:-1]
        + _HALF_PIXEL_KERNEL[3] * padded_columns[:, 3:]
    )


def _is_edge(neighbours: FloatImage, threshold: float) -> bool:
    return float(np.var(neighbours)) > threshold


def _solve_weights(
    data_matrix: FloatImage,
    observations: FloatImage,
) -> FloatImage | None:
    if data_matrix.shape[0] < 4:
        return None

    try:
        weights, _, _, _ = np.linalg.lstsq(
            data_matrix,
            observations,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(weights).all():
        return None
    return np.asarray(weights, dtype=np.float64)


def _stage_one_observations(
    source: FloatImage,
    row: int,
    column: int,
    window_size: int,
) -> tuple[FloatImage, FloatImage] | None:
    """Build the local LR system for an odd-row, odd-column output pixel."""
    half_window = window_size // 2
    row_start = row - half_window + 1
    column_start = column - half_window + 1
    row_stop = row_start + window_size
    column_stop = column_start + window_size

    if (
        row_start < 1
        or column_start < 1
        or row_stop > source.shape[0] - 1
        or column_stop > source.shape[1] - 1
    ):
        return None

    window = source[row_start - 1 : row_stop + 1, column_start - 1 : column_stop + 1]
    observations = window[1:-1, 1:-1].reshape(-1)
    predictors = np.column_stack(
        (
            window[:-2, :-2].reshape(-1),
            window[:-2, 2:].reshape(-1),
            window[2:, :-2].reshape(-1),
            window[2:, 2:].reshape(-1),
        )
    )
    return predictors, observations


def _stage_two_observations(
    partial: FloatImage,
    target_row: int,
    target_column: int,
    window_size: int,
) -> tuple[FloatImage, FloatImage] | None:
    """Build the second-stage system in the 45-degree rotated lattice.

    After stage one, known output samples form a checkerboard lattice. NEDI's
    second stage is the same covariance procedure applied after a 45-degree
    rotation. In original image coordinates, diagonal neighbours in that
    rotated lattice become axial neighbours two pixels away.

    Samples that fall outside the image are skipped rather than aborting the
    whole window, so interior pixels near a boundary can still use the
    in-bounds portion of the 8x8 training set.
    """
    half_window = window_size // 2
    rotated_row = (target_row + target_column) // 2
    rotated_column = (target_column - target_row) // 2
    height, width = partial.shape

    observations: list[float] = []
    predictors: list[list[float]] = []
    for delta_u in range(-half_window + 1, half_window + 1):
        for delta_v in range(-half_window + 1, half_window + 1):
            sample_row = (rotated_row + delta_u) - (rotated_column + delta_v)
            sample_column = (rotated_row + delta_u) + (rotated_column + delta_v)
            if (
                sample_row < 2
                or sample_column < 2
                or sample_row >= height - 2
                or sample_column >= width - 2
            ):
                continue

            observations.append(float(partial[sample_row, sample_column]))
            predictors.append(
                [
                    float(partial[sample_row, sample_column - 2]),
                    float(partial[sample_row - 2, sample_column]),
                    float(partial[sample_row + 2, sample_column]),
                    float(partial[sample_row, sample_column + 2]),
                ]
            )

    if len(observations) < 4:
        return None
    return np.asarray(predictors), np.asarray(observations)


def _interpolate_candidate(
    output: FloatImage,
    target_row: int,
    target_column: int,
    neighbours: FloatImage,
    local_system: tuple[FloatImage, FloatImage] | None,
    config: NEDIConfig,
    stats: _MutableStats,
) -> None:
    stats.interpolated_pixel_count += 1
    if not _is_edge(neighbours, config.edge_threshold):
        stats.bilinear_fallback_count += 1
        return

    stats.edge_pixel_count += 1
    if local_system is None:
        stats.bilinear_fallback_count += 1
        return

    data_matrix, observations = local_system
    weights = _solve_weights(data_matrix, observations)
    if weights is None:
        stats.bilinear_fallback_count += 1
        stats.numerical_fallback_count += 1
        return

    prediction = float(np.dot(weights, neighbours))
    if not np.isfinite(prediction):
        stats.bilinear_fallback_count += 1
        stats.numerical_fallback_count += 1
        return

    output[target_row, target_column] = np.clip(prediction, 0.0, 255.0)
    stats.nedi_pixel_count += 1


def nedi_upsample_x2_luminance(
    luminance: NDArray[np.generic],
    config: NEDIConfig | None = None,
) -> NEDIResult:
    """Enlarge one 0-255 luminance image with a native two-stage NEDI pass.

    An input of shape ``(height, width)`` produces the insertion-grid shape
    ``(2 * height, 2 * width)``, with original samples at even coordinates.
    The RGB reconstruction layer is responsible for aligning that lattice with
    a bicubic-prepared HR reference.
    """
    settings = config or NEDIConfig()
    source = _validate_luminance(luminance)
    output = _bilinear_grid(source)
    stats = _MutableStats()
    height, width = source.shape

    # Stage 1: fill pixels centred between four original LR samples.
    for row in range(height - 1):
        for column in range(width - 1):
            target_row = 2 * row + 1
            target_column = 2 * column + 1
            neighbours = np.asarray(
                [
                    source[row, column],
                    source[row, column + 1],
                    source[row + 1, column],
                    source[row + 1, column + 1],
                ],
                dtype=np.float64,
            )
            _interpolate_candidate(
                output,
                target_row,
                target_column,
                neighbours,
                _stage_one_observations(
                    source,
                    row,
                    column,
                    settings.window_size,
                ),
                settings,
                stats,
            )

    # Stage 2: fill the remaining horizontal and vertical lattice positions.
    for target_row in range(output.shape[0]):
        for target_column in range(output.shape[1]):
            if (target_row + target_column) % 2 == 0:
                continue

            # In the rotated lattice, these are the four cell corners in the
            # same order as the rotated covariance predictors: left, up,
            # down, then right.
            neighbours = np.asarray(
                [
                    output[target_row, target_column - 1]
                    if target_column > 0
                    else output[target_row, target_column],
                    output[target_row - 1, target_column]
                    if target_row > 0
                    else output[target_row, target_column],
                    output[target_row + 1, target_column]
                    if target_row + 1 < output.shape[0]
                    else output[target_row, target_column],
                    output[target_row, target_column + 1]
                    if target_column + 1 < output.shape[1]
                    else output[target_row, target_column],
                ],
                dtype=np.float64,
            )
            _interpolate_candidate(
                output,
                target_row,
                target_column,
                neighbours,
                _stage_two_observations(
                    output,
                    target_row,
                    target_column,
                    settings.window_size,
                ),
                settings,
                stats,
            )

    # Reassert exact source samples after both interpolation stages.
    output[::2, ::2] = source
    return NEDIResult(image=output, stats=stats.freeze())


def nedi_upsample_x2_rgb(
    image: Image.Image,
    target_size: tuple[int, int],
    config: NEDIConfig | None = None,
) -> NEDIRGBResult:
    """Apply native x2 NEDI to luminance and bicubic interpolation to chroma.

    Native NEDI produces a ``(2h, 2w)`` even-insertion grid. Before comparison
    with a bicubic-prepared HR image, luminance is shifted by half a pixel so
    that it occupies the same sampling grid as Pillow bicubic resizing. When
    a prepared pair is one pixel larger than that native 2x size, the complete
    reconstruction is bicubically resized to the target. This keeps every
    output pixel on the same grid as the HR reference.
    """
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Target size must be positive; received {target_size}.")

    source_ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.uint8)
    luminance_result = nedi_upsample_x2_luminance(source_ycbcr[..., 0], config)
    aligned_luminance = np.clip(_shift_half_pixel(luminance_result.image), 0.0, 255.0)
    native_height, native_width = aligned_luminance.shape
    native_size = (native_width, native_height)

    cb_image = Image.fromarray(source_ycbcr[..., 1])
    cr_image = Image.fromarray(source_ycbcr[..., 2])
    cb_native = np.asarray(
        cb_image.resize(native_size, resample=Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    cr_native = np.asarray(
        cr_image.resize(native_size, resample=Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    assembled_ycbcr = np.empty((native_height, native_width, 3), dtype=np.uint8)
    assembled_ycbcr[..., 0] = np.rint(aligned_luminance).astype(np.uint8)
    assembled_ycbcr[..., 1] = cb_native
    assembled_ycbcr[..., 2] = cr_native
    reconstruction = Image.fromarray(assembled_ycbcr, mode="YCbCr").convert("RGB")

    dimension_adjusted = native_size != target_size
    if dimension_adjusted:
        reconstruction = reconstruction.resize(
            target_size,
            resample=Image.Resampling.BICUBIC,
        )

    return NEDIRGBResult(
        image=reconstruction,
        stats=luminance_result.stats,
        native_size=native_size,
        target_size=target_size,
        dimension_adjusted=dimension_adjusted,
    )
