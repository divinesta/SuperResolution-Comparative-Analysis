"""Shape, colour-policy, and inference tests for FSRCNN."""

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in the Colab environment.")
class FSRCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global FSRCNN, Image, fsrcnn_upsample, pil_to_luminance_tensor, torch
        import torch
        from PIL import Image

        from app.deep_learning.fsrcnn import (
            FSRCNN,
            fsrcnn_upsample,
            pil_to_luminance_tensor,
        )

    def test_each_supported_scale_has_the_expected_output_shape(self) -> None:
        input_tensor = torch.zeros((1, 1, 7, 9), dtype=torch.float32)

        for scale in (2, 3, 4):
            with self.subTest(scale=scale):
                output = FSRCNN(scale_factor=scale)(input_tensor)
                self.assertEqual(tuple(output.shape), (1, 1, 7 * scale, 9 * scale))

    def test_pillow_inference_returns_rgb_at_native_scaled_size(self) -> None:
        image = Image.new("RGB", (9, 7), (50, 100, 150))
        model = FSRCNN(scale_factor=2).eval()

        output = fsrcnn_upsample(model, image, "cpu")

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (18, 14))

    def test_model_input_is_one_luminance_channel(self) -> None:
        image = Image.new("RGB", (9, 7), (50, 100, 150))

        tensor = pil_to_luminance_tensor(image, "cpu")

        self.assertEqual(tuple(tensor.shape), (1, 1, 7, 9))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_published_configuration_has_expected_parameter_count(self) -> None:
        model = FSRCNN(scale_factor=2)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(parameter_count, 12_809)

    def test_unsupported_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2, 3, or 4"):
            FSRCNN(scale_factor=8)


if __name__ == "__main__":
    unittest.main()
