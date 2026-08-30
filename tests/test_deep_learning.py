"""Tests for Phase 3 model and checkpoint conventions."""

import unittest
from pathlib import Path

from app.deep_learning.checkpoints import IMDN_CHECKPOINTS
from app.deep_learning.config import DeepLearningModelConfig


class DeepLearningModelConfigTests(unittest.TestCase):
    def test_official_imdn_metadata_covers_every_required_scale(self) -> None:
        self.assertEqual(set(IMDN_CHECKPOINTS), {2, 3, 4})
        for scale, checkpoint in IMDN_CHECKPOINTS.items():
            self.assertEqual(checkpoint.scale, scale)
            self.assertEqual(len(checkpoint.sha256), 64)
            self.assertGreater(checkpoint.size_bytes, 0)
            self.assertEqual(checkpoint.degradation, "bicubic")

    def test_checkpoint_is_model_and_scale_specific(self) -> None:
        config = DeepLearningModelConfig("imdn", 4, Path("/checkpoints"))

        self.assertEqual(
            config.checkpoint_path,
            Path("/checkpoints/imdn/imdn_x4.pth"),
        )

    def test_model_name_is_normalized(self) -> None:
        config = DeepLearningModelConfig("FSRCNN", 2, Path("weights"))  # type: ignore[arg-type]

        self.assertEqual(config.model, "fsrcnn")
        self.assertEqual(config.colour_policy, "y_model_with_bicubic_cbcr")

    def test_each_model_has_an_explicit_colour_policy(self) -> None:
        fsrcnn = DeepLearningModelConfig("fsrcnn", 2, Path("weights"))
        imdn = DeepLearningModelConfig("imdn", 2, Path("weights"))

        self.assertNotEqual(fsrcnn.colour_policy, imdn.colour_policy)
        self.assertEqual(imdn.colour_policy, "rgb_model")

    def test_unsupported_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fsrcnn, imdn"):
            DeepLearningModelConfig("edsr", 2, Path("weights"))  # type: ignore[arg-type]

    def test_unsupported_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2, 3, or 4"):
            DeepLearningModelConfig("imdn", 8, Path("weights"))


if __name__ == "__main__":
    unittest.main()
