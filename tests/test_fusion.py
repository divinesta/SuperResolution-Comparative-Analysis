"""Tests for the transparent Phase 5 weighted fusion."""

import unittest

import numpy as np
from PIL import Image

from app.fusion.weighted import (
    evaluate_weight_grid,
    select_best_weight,
    summarise_weight_results,
    weighted_fusion,
)


def solid(value: int, size: tuple[int, int] = (20, 20)) -> Image.Image:
    return Image.new("RGB", size, (value, value, value))


class WeightedFusionTests(unittest.TestCase):
    def test_endpoints_exactly_reproduce_the_source_methods(self) -> None:
        nedi = solid(30)
        imdn = solid(210)

        np.testing.assert_array_equal(weighted_fusion(nedi, imdn, 0), np.asarray(nedi))
        np.testing.assert_array_equal(weighted_fusion(nedi, imdn, 1), np.asarray(imdn))

    def test_midpoint_uses_both_images(self) -> None:
        fused = weighted_fusion(solid(20), solid(100), 0.25)
        np.testing.assert_array_equal(np.asarray(fused), np.full((20, 20, 3), 40))

    def test_invalid_weight_and_mismatched_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "from 0 to 1"):
            weighted_fusion(solid(0), solid(1), 1.1)
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            weighted_fusion(solid(0), solid(1, (21, 20)), 0.5)

    def test_weight_grid_uses_scale_as_metric_border(self) -> None:
        reference = solid(100)
        records = evaluate_weight_grid(
            reference,
            solid(80),
            solid(100),
            2,
            weights=(0.0, 0.5, 1.0),
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(records[-1]["imdn_weight"], 1.0)
        self.assertTrue(np.isinf(records[-1]["psnr_y"]))

    def test_summary_requires_equal_samples_and_selects_best_psnr_y(self) -> None:
        records = [
            {
                "imdn_weight": weight,
                "psnr_y": psnr + offset,
                "ssim_y": 0.8,
                "psnr_rgb": psnr,
                "ssim_rgb": 0.7,
            }
            for weight, psnr in ((0.8, 30.0), (1.0, 29.0))
            for offset in (0.0, 2.0)
        ]

        summary = summarise_weight_results(records)
        selected = select_best_weight(summary)

        self.assertEqual([row["sample_count"] for row in summary], [2, 2])
        self.assertEqual(selected["imdn_weight"], 0.8)

        with self.assertRaisesRegex(ValueError, "same number"):
            summarise_weight_results(records[:-1])


if __name__ == "__main__":
    unittest.main()
