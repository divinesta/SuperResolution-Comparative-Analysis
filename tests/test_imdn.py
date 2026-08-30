"""Shape and conversion tests for the IMDN architecture."""

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in the Colab environment.")
class IMDNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global Image, IMDN, imdn_upsample, torch
        import torch
        from PIL import Image

        from app.deep_learning.imdn import IMDN, imdn_upsample

    def test_each_supported_scale_has_the_expected_output_shape(self) -> None:
        input_tensor = torch.zeros((1, 3, 7, 9), dtype=torch.float32)

        for scale in (2, 3, 4):
            with self.subTest(scale=scale):
                output = IMDN(upscale=scale)(input_tensor)
                self.assertEqual(tuple(output.shape), (1, 3, 7 * scale, 9 * scale))

    def test_pillow_inference_returns_rgb_at_native_scaled_size(self) -> None:
        image = Image.new("RGB", (9, 7), (50, 100, 150))
        model = IMDN(upscale=2).eval()

        output = imdn_upsample(model, image, "cpu")

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (18, 14))

    def test_unsupported_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2, 3, or 4"):
            IMDN(upscale=8)


if __name__ == "__main__":
    unittest.main()
