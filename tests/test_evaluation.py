"""Focused tests for the shared evaluation infrastructure."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.evaluation.bicubic import (
    BicubicEvaluationConfig,
    evaluate_bicubic_dataset,
    write_results_csv,
)
from app.evaluation.images import bicubic_downsample, bicubic_upsample, modcrop
from app.evaluation.metrics import calculate_quality_metrics
from app.evaluation.timing import measure_runtime


class ImagePreparationTests(unittest.TestCase):
    def test_modcrop_makes_dimensions_divisible_by_scale(self) -> None:
        image = Image.new("RGB", (101, 98), "white")

        cropped = modcrop(image, 3)

        self.assertEqual(cropped.size, (99, 96))

    def test_downsample_and_upsample_have_expected_sizes(self) -> None:
        image = Image.new("RGB", (48, 36), "white")

        aligned_hr, lr_image = bicubic_downsample(image, 4)
        reconstruction = bicubic_upsample(lr_image, aligned_hr.size)

        self.assertEqual(aligned_hr.size, (48, 36))
        self.assertEqual(lr_image.size, (12, 9))
        self.assertEqual(reconstruction.size, aligned_hr.size)


class MetricTests(unittest.TestCase):
    def test_identical_images_have_perfect_metrics(self) -> None:
        image = np.full((24, 24, 3), 128, dtype=np.uint8)

        metrics = calculate_quality_metrics(image, image.copy(), border=2)

        self.assertTrue(math.isinf(metrics["psnr_y"]))
        self.assertTrue(math.isinf(metrics["psnr_rgb"]))
        self.assertAlmostEqual(metrics["ssim_y"], 1.0)
        self.assertAlmostEqual(metrics["ssim_rgb"], 1.0)

    def test_border_crop_excludes_edge_only_errors(self) -> None:
        reference = np.full((24, 24, 3), 100, dtype=np.uint8)
        reconstruction = reference.copy()
        reconstruction[:2, :, :] = 0
        reconstruction[-2:, :, :] = 0
        reconstruction[:, :2, :] = 0
        reconstruction[:, -2:, :] = 0

        uncropped = calculate_quality_metrics(reference, reconstruction, border=0)
        cropped = calculate_quality_metrics(reference, reconstruction, border=2)

        self.assertFalse(math.isinf(uncropped["psnr_rgb"]))
        self.assertTrue(math.isinf(cropped["psnr_rgb"]))


class TimingTests(unittest.TestCase):
    def test_runtime_uses_warmups_and_repeated_measurements(self) -> None:
        calls = 0

        def operation() -> int:
            nonlocal calls
            calls += 1
            return calls

        result, stats = measure_runtime(operation, warmup_runs=2, timed_runs=4)

        self.assertEqual(calls, 6)
        self.assertEqual(result, 6)
        self.assertEqual(stats.warmup_runs, 2)
        self.assertEqual(stats.timed_runs, 4)
        self.assertGreaterEqual(stats.latency_mean_ms, 0)


class DatasetEvaluationTests(unittest.TestCase):
    def test_dataset_evaluation_writes_a_non_destructive_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            hr_directory = root / "Set5_HR"
            hr_directory.mkdir()
            output_path = root / "metrics" / "Set5_x2_bicubic_final.csv"

            gradient = np.zeros((24, 24, 3), dtype=np.uint8)
            gradient[..., 0] = np.arange(24, dtype=np.uint8)[:, None] * 10
            gradient[..., 1] = np.arange(24, dtype=np.uint8)[None, :] * 10
            gradient[..., 2] = 100
            Image.fromarray(gradient).save(hr_directory / "sample.png")

            records = evaluate_bicubic_dataset(
                hr_directory,
                BicubicEvaluationConfig(
                    dataset="Set5",
                    scale=2,
                    warmup_runs=0,
                    timed_runs=1,
                ),
            )
            write_results_csv(records, output_path)

            with output_path.open(newline="", encoding="utf-8") as file:
                saved_records = list(csv.DictReader(file))

            self.assertEqual(len(saved_records), 1)
            self.assertEqual(saved_records[0]["dataset"], "Set5")
            self.assertEqual(saved_records[0]["scale"], "x2")
            self.assertIn("psnr_y", saved_records[0])
            self.assertIn("psnr_rgb", saved_records[0])

            with self.assertRaises(FileExistsError):
                write_results_csv(records, output_path)


if __name__ == "__main__":
    unittest.main()
