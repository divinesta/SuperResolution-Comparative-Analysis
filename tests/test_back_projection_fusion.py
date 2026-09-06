"""Tests for LR-consistency back-projection fusion."""

import unittest

import numpy as np
from PIL import Image

from app.fusion.back_projection import (
    back_projection_fusion,
    calculate_back_projection_weight,
    evaluate_back_projection_fusion,
)


def solid(value: int, size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, (value, value, value))


class BackProjectionFusionTests(unittest.TestCase):
    def test_equal_errors_select_an_exact_midpoint(self) -> None:
        original_lr = solid(100, (8, 6))
        nedi = solid(80, (16, 12))
        imdn = solid(120, (16, 12))

        result = back_projection_fusion(original_lr, nedi, imdn)

        self.assertAlmostEqual(result.imdn_weight, 0.5)
        self.assertAlmostEqual(result.nedi_weight, 0.5)
        np.testing.assert_array_equal(result.image, np.full((12, 16, 3), 100))
        self.assertEqual(result.fused_lr_mse, 0.0)

    def test_weight_moves_towards_the_closer_candidate(self) -> None:
        weight, nedi_error, imdn_error = calculate_back_projection_weight(
            solid(115, (8, 6)),
            solid(80, (16, 12)),
            solid(120, (16, 12)),
        )

        self.assertAlmostEqual(weight, 0.875)
        self.assertGreater(nedi_error, imdn_error)

    def test_weight_is_clipped_to_valid_endpoint(self) -> None:
        weight, _, _ = calculate_back_projection_weight(
            solid(240, (8, 6)),
            solid(80, (16, 12)),
            solid(120, (16, 12)),
        )
        self.assertEqual(weight, 1.0)

    def test_identical_degraded_candidates_fall_back_to_imdn(self) -> None:
        weight, _, _ = calculate_back_projection_weight(
            solid(100, (8, 6)),
            solid(90, (16, 12)),
            solid(90, (16, 12)),
        )
        self.assertEqual(weight, 1.0)

    def test_mismatched_candidate_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            back_projection_fusion(
                solid(100, (8, 6)),
                solid(80, (16, 12)),
                solid(120, (17, 12)),
            )

    def test_evaluation_returns_weights_and_quality_metrics(self) -> None:
        reference = solid(100, (24, 24))
        record = evaluate_back_projection_fusion(
            reference,
            solid(100, (12, 12)),
            solid(80, (24, 24)),
            solid(120, (24, 24)),
            2,
        )

        self.assertEqual(record["method"], "back_projection_nedi_imdn_fusion")
        self.assertAlmostEqual(record["imdn_weight"], 0.5)
        self.assertIn("psnr_y", record)
        self.assertIn("ssim_y", record)


if __name__ == "__main__":
    unittest.main()
