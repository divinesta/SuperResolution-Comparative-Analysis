"""Focused tests for the native NEDI x2 luminance core."""

import unittest

import numpy as np
from PIL import Image

from app.traditional.nedi import (
    NEDIConfig,
    _stage_two_observations,
    nedi_upsample_x2_luminance,
    nedi_upsample_x2_rgb,
)


class NEDIConfigurationTests(unittest.TestCase):
    def test_window_size_must_be_even_and_at_least_four(self) -> None:
        for invalid_size in (2, 5):
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaises(ValueError):
                    NEDIConfig(window_size=invalid_size)

    def test_edge_threshold_cannot_be_negative(self) -> None:
        with self.assertRaises(ValueError):
            NEDIConfig(edge_threshold=-1)


class NEDICoreTests(unittest.TestCase):
    def test_constant_image_remains_constant(self) -> None:
        source = np.full((12, 10), 123.0)

        result = nedi_upsample_x2_luminance(source)

        self.assertEqual(result.image.shape, (23, 19))
        np.testing.assert_allclose(result.image, 123.0)
        self.assertEqual(result.stats.nedi_pixel_count, 0)
        self.assertEqual(
            result.stats.interpolated_pixel_count,
            result.stats.bilinear_fallback_count,
        )

    def test_native_pass_preserves_original_samples(self) -> None:
        source = np.arange(12 * 11, dtype=np.float64).reshape(12, 11)

        result = nedi_upsample_x2_luminance(source)

        np.testing.assert_array_equal(result.image[::2, ::2], source)

    def test_smooth_region_fallback_is_exact_bilinear_interpolation(self) -> None:
        source = np.asarray(
            [
                [0.0, 100.0, 200.0],
                [50.0, 150.0, 250.0],
            ]
        )
        expected = np.asarray(
            [
                [0.0, 50.0, 100.0, 150.0, 200.0],
                [25.0, 75.0, 125.0, 175.0, 225.0],
                [50.0, 100.0, 150.0, 200.0, 250.0],
            ]
        )

        result = nedi_upsample_x2_luminance(
            source,
            NEDIConfig(edge_threshold=1_000_000),
        )

        np.testing.assert_array_equal(result.image, expected)

    def test_textured_image_uses_adaptive_interpolation(self) -> None:
        random = np.random.default_rng(42)
        source = random.integers(0, 256, size=(24, 24)).astype(np.float64)

        result = nedi_upsample_x2_luminance(
            source,
            NEDIConfig(window_size=4, edge_threshold=0),
        )

        self.assertGreater(result.stats.edge_pixel_count, 0)
        self.assertGreater(result.stats.nedi_pixel_count, 0)
        self.assertEqual(
            result.stats.nedi_pixel_count
            + result.stats.bilinear_fallback_count,
            result.stats.interpolated_pixel_count,
        )
        self.assertTrue(np.isfinite(result.image).all())
        self.assertGreaterEqual(float(result.image.min()), 0)
        self.assertLessEqual(float(result.image.max()), 255)

    def test_synthetic_edges_produce_valid_outputs(self) -> None:
        sources = []

        horizontal = np.zeros((24, 24), dtype=np.float64)
        horizontal[12:, :] = 255
        sources.append(horizontal)

        vertical = np.zeros((24, 24), dtype=np.float64)
        vertical[:, 12:] = 255
        sources.append(vertical)

        diagonal = np.fromfunction(
            lambda row, column: np.where(row > column, 255.0, 0.0),
            (24, 24),
        )
        sources.append(diagonal)

        for source in sources:
            with self.subTest():
                result = nedi_upsample_x2_luminance(source)
                self.assertEqual(result.image.shape, (47, 47))
                self.assertTrue(np.isfinite(result.image).all())
                self.assertGreaterEqual(float(result.image.min()), 0)
                self.assertLessEqual(float(result.image.max()), 255)

    def test_invalid_luminance_inputs_are_rejected(self) -> None:
        invalid_images = (
            np.zeros((4, 4, 3)),
            np.array([[0.0, np.nan], [1.0, 2.0]]),
            np.array([[0.0, 256.0], [1.0, 2.0]]),
        )

        for image in invalid_images:
            with self.subTest(shape=image.shape):
                with self.assertRaises(ValueError):
                    nedi_upsample_x2_luminance(image)

    def test_rgb_reconstruction_uses_native_size_and_rgb_mode(self) -> None:
        random = np.random.default_rng(24)
        source = Image.fromarray(random.integers(0, 256, size=(12, 10, 3), dtype=np.uint8))

        result = nedi_upsample_x2_rgb(source, (19, 23))

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (19, 23))
        self.assertEqual(result.native_size, (19, 23))
        self.assertFalse(result.dimension_adjusted)

    def test_rgb_reconstruction_records_target_size_adjustment(self) -> None:
        source = Image.new("RGB", (10, 12), "blue")

        result = nedi_upsample_x2_rgb(source, (20, 24))

        self.assertEqual(result.image.size, (20, 24))
        self.assertEqual(result.native_size, (19, 23))
        self.assertTrue(result.dimension_adjusted)

    def test_second_stage_uses_rotated_lattice_neighbours(self) -> None:
        partial = np.arange(31 * 31, dtype=np.float64).reshape(31, 31)

        local_system = _stage_two_observations(
            partial,
            target_row=10,
            target_column=13,
            window_size=4,
        )

        self.assertIsNotNone(local_system)
        predictors, observations = local_system
        self.assertEqual(predictors.shape, (16, 4))
        self.assertEqual(observations.shape, (16,))
        self.assertEqual(observations[0], partial[10, 10])
        np.testing.assert_array_equal(
            predictors[0],
            [partial[10, 8], partial[8, 10], partial[12, 10], partial[10, 12]],
        )


if __name__ == "__main__":
    unittest.main()
