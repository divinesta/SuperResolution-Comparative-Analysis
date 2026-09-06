"""Focused validation tests for Phase 4 visual-comparison inputs."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.evaluation.visual_comparison import (
    CROP_REGIONS,
    DISPLAY_METHODS,
    SAMPLES,
    SCALES,
    validate_visual_samples,
)


class VisualComparisonTests(unittest.TestCase):
    def test_configured_crops_fit_the_selected_reference_sizes(self) -> None:
        reference_sizes = {
            ("Set5", "baby"): (512, 512),
            ("Set5", "butterfly"): (256, 256),
            ("Set14", "baboon"): (500, 480),
            ("Urban100", "img_004"): (1024, 680),
        }
        for key, crop in CROP_REGIONS.items():
            width, height = reference_sizes[key]
            left, top, right, bottom = crop
            self.assertTrue(0 <= left < right <= width)
            self.assertTrue(0 <= top < bottom <= height)

    def test_complete_rgb_samples_are_accepted(self) -> None:
        sizes = {
            ("Set5", "baby"): (512, 512),
            ("Set5", "butterfly"): (256, 256),
            ("Set14", "baboon"): (500, 480),
            ("Urban100", "img_004"): (1024, 680),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset, image, _ in SAMPLES:
                width, height = sizes[(dataset, image)]
                for scale in SCALES:
                    group = root / dataset / image / scale
                    group.mkdir(parents=True)
                    Image.new("RGB", (width // int(scale[1:]), height // int(scale[1:]))).save(
                        group / "input_lr.png"
                    )
                    for method in DISPLAY_METHODS:
                        Image.new("RGB", (width, height)).save(group / f"{method}.png")
            result = validate_visual_samples(root)

        self.assertEqual(result["group_count"], 12)
        self.assertEqual(result["image_count"], 72)


if __name__ == "__main__":
    unittest.main()
