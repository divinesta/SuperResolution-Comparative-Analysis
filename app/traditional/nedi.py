"""Native x2 New Edge-Directed Interpolation for luminance images.

The implementation follows the two-stage interpolation described by Li and
Orchard. It deliberately contains no dataset loading, metric calculation, or
result writing; those responsibilities belong to the evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatImage = NDArray[np.float64]


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
    height, width = source.shape
    output = np.empty((2 * height - 1, 2 * width - 1), dtype=np.float64)
    output[::2, ::2] = source
    output[::2, 1::2] = (source[:, :-1] + source[:, 1:]) / 2.0
    output[1::2, ::2] = (source[:-1, :] + source[1:, :]) / 2.0
    output[1::2, 1::2] = (
        source[:-1, :-1]
        + source[:-1, 1:]
        + source[1:, :-1]
        + source[1:, 1:]
    ) / 4.0
    return output


def _is_edge(neighbours: FloatImage, threshold: float) -> bool:
    return float(np.var(neighbours)) > threshold


def _solve_weights(
    data_matrix: FloatImage,
    observations: FloatImage,
) -> FloatImage | None:
    if data_matrix.shape[0] < 4:
        return None

    try:
        weights, _, rank, _ = np.linalg.lstsq(
            data_matrix,
            observations,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None

    if rank < 4 or not np.isfinite(weights).all():
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

    observations: list[float] = []
    predictors: list[list[float]] = []
    for sample_row in range(row_start, row_stop):
        for sample_column in range(column_start, column_stop):
            observations.append(float(source[sample_row, sample_column]))
            predictors.append(
                [
                    float(source[sample_row - 1, sample_column - 1]),
                    float(source[sample_row - 1, sample_column + 1]),
                    float(source[sample_row + 1, sample_column - 1]),
                    float(source[sample_row + 1, sample_column + 1]),
                ]
            )

    return np.asarray(predictors), np.asarray(observations)


def _stage_two_observations(
    partial: FloatImage,
    target_row: int,
    target_column: int,
    window_size: int,
) -> tuple[FloatImage, FloatImage] | None:
    """Build the rotated local system for the second interlacing lattice."""
    radius = window_size
    if (
        target_row - radius < 1
        or target_column - radius < 1
        or target_row + radius >= partial.shape[0] - 1
        or target_column + radius >= partial.shape[1] - 1
    ):
        return None

    candidates: list[tuple[int, int, int]] = []
    for sample_row in range(target_row - radius, target_row + radius + 1):
        for sample_column in range(
            target_column - radius,
            target_column + radius + 1,
        ):
            if (sample_row + sample_column) % 2 != 0:
                continue
            distance = (sample_row - target_row) ** 2 + (
                sample_column - target_column
            ) ** 2
            candidates.append((distance, sample_row, sample_column))

    candidates.sort()
    selected = candidates[: window_size * window_size]
    if len(selected) < window_size * window_size:
        return None

    observations: list[float] = []
    predictors: list[list[float]] = []
    for _, sample_row, sample_column in selected:
        observations.append(float(partial[sample_row, sample_column]))
        predictors.append(
            [
                float(partial[sample_row - 1, sample_column - 1]),
                float(partial[sample_row - 1, sample_column + 1]),
                float(partial[sample_row + 1, sample_column - 1]),
                float(partial[sample_row + 1, sample_column + 1]),
            ]
        )

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
    ``(2 * height - 1, 2 * width - 1)``. A later reconstruction layer is
    responsible for reconciling that native grid with an exact HR target size.
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

            neighbours = np.asarray(
                [
                    output[target_row - 1, target_column]
                    if target_row > 0
                    else output[target_row, target_column],
                    output[target_row, target_column + 1]
                    if target_column + 1 < output.shape[1]
                    else output[target_row, target_column],
                    output[target_row, target_column - 1]
                    if target_column > 0
                    else output[target_row, target_column],
                    output[target_row + 1, target_column]
                    if target_row + 1 < output.shape[0]
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
