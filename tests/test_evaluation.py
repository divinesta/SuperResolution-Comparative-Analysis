"""Focused tests for the shared evaluation infrastructure."""

import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.config import dataset_hr_directory, dataset_lr_directory, resolve_data_root
from app.evaluation.bicubic import (
    BicubicEvaluationConfig,
    evaluate_bicubic_dataset,
    write_results_csv,
)
from app.evaluation.data_validation import validate_prepared_dataset
from app.evaluation.images import (
    align_hr_to_lr,
    bicubic_downsample,
    bicubic_upsample,
    modcrop,
    pair_image_paths,
)
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


class DataPathTests(unittest.TestCase):
    def test_explicit_data_root_builds_the_expected_hr_path(self) -> None:
        root = Path("/example/data")

        hr_directory = dataset_hr_directory("Set14", root)

        self.assertEqual(hr_directory, root / "Set14" / "Set14_HR")

    def test_data_root_builds_the_expected_lr_path(self) -> None:
        root = Path("/example/data")

        lr_directory = dataset_lr_directory("BSD100", 4, root)

        self.assertEqual(lr_directory, root / "BSD100" / "BSD100_LR_x4")

    def test_environment_variable_can_select_the_data_root(self) -> None:
        with patch.dict("os.environ", {"FYP_SR_DATA_ROOT": "/mounted/datasets"}):
            root = resolve_data_root()

        self.assertEqual(root, Path("/mounted/datasets"))


class MetricTests(unittest.TestCase):
    def test_identical_images_have_perfect_metrics(self) -> None:
        image = np.full((24, 24, 3), 128, dtype=np.uint8)

        metrics = calculate_quality_metrics(image, image.copy(), border=2)

        self.assertTrue(math.isinf(metrics["psnr_y"]))
        self.assertTrue(math.isinf(metrics["psnr_rgb"]))
        self.assertAlmostEqual(metrics["ssim_y"], 1.0)
        self.assertAlmostEqual(metrics["ssim_rgb"], 1.0)

    def test_ssim_uses_the_standard_gaussian_protocol(self) -> None:
        reference = np.full((24, 24, 3), 128, dtype=np.uint8)
        reconstruction = reference.copy()
        reconstruction[10:14, 10:14, :] = 100

        with patch(
            "app.evaluation.metrics.structural_similarity",
            wraps=__import__(
                "skimage.metrics",
                fromlist=["structural_similarity"],
            ).structural_similarity,
        ) as mocked_ssim:
            calculate_quality_metrics(reference, reconstruction, border=2)

        self.assertEqual(mocked_ssim.call_count, 2)
        for call in mocked_ssim.call_args_list:
            self.assertTrue(call.kwargs["gaussian_weights"])
            self.assertEqual(call.kwargs["sigma"], 1.5)
            self.assertFalse(call.kwargs["use_sample_covariance"])
            self.assertEqual(call.kwargs["data_range"], 255.0)

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


class PreparedPairTests(unittest.TestCase):
    def test_hr_reference_is_aligned_to_prepared_lr_dimensions(self) -> None:
        hr_image = Image.new("RGB", (101, 98), "white")
        lr_image = Image.new("RGB", (33, 32), "white")

        aligned_hr = align_hr_to_lr(hr_image, lr_image, scale=3)

        self.assertEqual(aligned_hr.size, (99, 96))

    def test_filename_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            hr_directory = root / "HR"
            lr_directory = root / "LR"
            hr_directory.mkdir()
            lr_directory.mkdir()
            Image.new("RGB", (24, 24)).save(hr_directory / "original.png")
            Image.new("RGB", (12, 12)).save(lr_directory / "different.png")

            with self.assertRaisesRegex(ValueError, "filename mismatch"):
                pair_image_paths(hr_directory, lr_directory)

    def test_known_dataset_requires_its_expected_image_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            hr_directory = root / "Set5" / "Set5_HR"
            lr_directory = root / "Set5" / "Set5_LR_x2"
            hr_directory.mkdir(parents=True)
            lr_directory.mkdir(parents=True)
            Image.new("RGB", (24, 24)).save(hr_directory / "only_one.png")
            Image.new("RGB", (12, 12)).save(lr_directory / "only_one.png")

            with self.assertRaisesRegex(ValueError, "should contain 5"):
                validate_prepared_dataset("Set5", 2, root)


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
            lr_directory = root / "Set5_LR_x2"
            hr_directory.mkdir()
            lr_directory.mkdir()
            output_path = root / "metrics" / "Set5_x2_bicubic_final.csv"

            gradient = np.zeros((24, 24, 3), dtype=np.uint8)
            gradient[..., 0] = np.arange(24, dtype=np.uint8)[:, None] * 10
            gradient[..., 1] = np.arange(24, dtype=np.uint8)[None, :] * 10
            gradient[..., 2] = 100
            Image.fromarray(gradient).save(hr_directory / "sample.png")
            prepared_lr = Image.fromarray(gradient).resize(
                (12, 12),
                resample=Image.Resampling.BICUBIC,
            )
            prepared_lr.save(lr_directory / "sample.png")

            records = evaluate_bicubic_dataset(
                hr_directory,
                lr_directory,
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
