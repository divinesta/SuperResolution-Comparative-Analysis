"""Tests for Phase 4 result validation and comparison tables."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.evaluation.comparison import (
    DATASETS,
    EXPECTED_IMAGE_COUNTS,
    METHODS,
    SCALES,
    build_method_overview,
    load_phase4_summaries,
)


class Phase4ComparisonTests(unittest.TestCase):
    def _write_summary(self, root: Path, method: str) -> None:
        path = root / "results" / "metrics" / "final" / f"{method}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "dataset", "scale", "method", "image_count", "psnr_y", "ssim_y",
            "psnr_rgb", "ssim_rgb", "latency_mean_ms", "latency_median_ms",
        ]
        if method in {"fsrcnn", "imdn"}:
            fields += ["parameter_count", "peak_gpu_memory_mean_mb", "peak_gpu_memory_max_mb"]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for dataset in DATASETS:
                for scale in SCALES:
                    score = {"bicubic": 30, "nedi": 29, "fsrcnn": 32, "imdn": 33}[method]
                    row = {
                        "dataset": dataset, "scale": scale, "method": method,
                        "image_count": EXPECTED_IMAGE_COUNTS[dataset], "psnr_y": score,
                        "ssim_y": score / 40, "psnr_rgb": score - 1,
                        "ssim_rgb": score / 41, "latency_mean_ms": 2,
                        "latency_median_ms": 2,
                    }
                    if method in {"fsrcnn", "imdn"}:
                        row.update(parameter_count=100, peak_gpu_memory_mean_mb=20, peak_gpu_memory_max_mb=25)
                    writer.writerow(row)

    def test_complete_summaries_are_ranked_and_annotated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for method in METHODS:
                self._write_summary(root, method)
                paths[method] = Path(f"results/metrics/final/{method}.csv")
            with patch("app.evaluation.comparison.SUMMARY_PATHS", paths):
                rows = load_phase4_summaries(root)

        self.assertEqual(len(rows), 48)
        imdn = next(row for row in rows if row["method"] == "imdn")
        nedi = next(row for row in rows if row["method"] == "nedi")
        self.assertEqual(imdn["psnr_y_rank"], 1)
        self.assertEqual(nedi["psnr_y_vs_bicubic_db"], -1)

    def test_method_overview_counts_all_image_scale_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for method in METHODS:
                self._write_summary(root, method)
                paths[method] = Path(f"results/metrics/final/{method}.csv")
            with patch("app.evaluation.comparison.SUMMARY_PATHS", paths):
                rows = load_phase4_summaries(root)
            overview = build_method_overview(rows)

        self.assertEqual(overview[0]["method"], "imdn")
        self.assertEqual(overview[0]["evaluated_image_scale_pairs"], 657)
        self.assertEqual(overview[0]["psnr_y_group_wins"], 12)


if __name__ == "__main__":
    unittest.main()
