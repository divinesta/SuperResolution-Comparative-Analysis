"""Tests for the Phase 5 edge-aware fusion method."""

import unittest

import numpy as np
from PIL import Image

from app.fusion.edge_aware import (
    EdgeAwareFusionConfig,
    detect_strong_edges,
    edge_aware_fusion,
    evaluate_edge_aware_grid,
    select_best_edge_aware_config,
    summarise_edge_aware_results,
)


def rgb(array: np.ndarray) -> Image.Image:
    pixels = np.repeat(array[..., None], 3, axis=2).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


class EdgeAwareFusionTests(unittest.TestCase):
    def test_flat_guide_keeps_imdn_unchanged(self) -> None:
        guide = rgb(np.full((10, 10), 80))
        imdn = rgb(np.full((20, 20), 100))
        nedi = rgb(np.full((20, 20), 20))

        result = edge_aware_fusion(
            guide, nedi, imdn, EdgeAwareFusionConfig(90, 0.3)
        )

        self.assertEqual(result.edge_pixel_fraction, 0.0)
        np.testing.assert_array_equal(result.image, np.asarray(imdn))

    def test_edge_guide_changes_only_a_limited_region(self) -> None:
        guide_pixels = np.zeros((12, 12), dtype=np.uint8)
        guide_pixels[:, 6:] = 255
        guide = rgb(guide_pixels)
        imdn = rgb(np.full((24, 24), 100))
        nedi = rgb(np.full((24, 24), 200))

        result = edge_aware_fusion(
            guide, nedi, imdn, EdgeAwareFusionConfig(90, 0.2, 1)
        )
        output = np.asarray(result.image)[..., 0]

        self.assertGreater(result.edge_pixel_fraction, 0.0)
        self.assertLess(result.edge_pixel_fraction, 0.5)
        self.assertTrue(np.all(output[result.edge_mask] == 120))
        self.assertTrue(np.all(output[~result.edge_mask] == 100))

    def test_edge_mask_is_based_on_lr_guide_at_target_size(self) -> None:
        guide_pixels = np.zeros((8, 10), dtype=np.uint8)
        guide_pixels[4:, :] = 255
        mask = detect_strong_edges(
            rgb(guide_pixels), (30, 24), edge_percentile=90, dilation_radius=0
        )

        self.assertEqual(mask.shape, (24, 30))
        self.assertGreater(mask.sum(), 0)

    def test_grid_includes_imdn_baseline_once(self) -> None:
        reference = rgb(np.tile(np.arange(20, dtype=np.uint8), (20, 1)))
        records = evaluate_edge_aware_grid(
            reference,
            reference.resize((10, 10)),
            reference,
            reference,
            2,
            edge_percentiles=(80, 90),
            edge_nedi_weights=(0.1, 0.2),
        )

        self.assertEqual(len(records), 5)
        self.assertEqual(sum(row["config_id"] == "imdn_only" for row in records), 1)

    def test_summary_and_selection_prefer_best_psnr(self) -> None:
        records = []
        for config_id, weight, psnr in (("imdn_only", 0.0, 30.0), ("edge", 0.1, 31.0)):
            for _ in range(2):
                records.append(
                    {
                        "config_id": config_id,
                        "edge_percentile": "not_applicable" if weight == 0 else 90,
                        "edge_nedi_weight": weight,
                        "edge_imdn_weight": 1.0 - weight,
                        "dilation_radius": 0 if weight == 0 else 1,
                        "edge_pixel_fraction": 0.0 if weight == 0 else 0.1,
                        "psnr_y": psnr,
                        "ssim_y": 0.9,
                        "psnr_rgb": psnr - 1,
                        "ssim_rgb": 0.8,
                    }
                )

        summary = summarise_edge_aware_results(records)
        selected = select_best_edge_aware_config(summary)

        self.assertEqual(selected["config_id"], "edge")
        self.assertEqual(selected["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
