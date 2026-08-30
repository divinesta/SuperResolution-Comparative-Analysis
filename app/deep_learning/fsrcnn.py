"""FSRCNN architecture and luminance-only image inference for Phase 3.

The layer names match the pretrained FSRCNN(56, 12, 4) checkpoints published
by yjn870. The network reconstructs limited-range BT.601 luminance; chroma is
enlarged separately with bicubic interpolation before RGB assembly.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn


class FSRCNN(nn.Module):
    """Fast Super-Resolution CNN using the paper's 56-12-4 configuration."""

    def __init__(
        self,
        *,
        scale_factor: int,
        num_channels: int = 1,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ) -> None:
        super().__init__()
        if scale_factor not in {2, 3, 4}:
            raise ValueError(
                f"FSRCNN scale must be 2, 3, or 4; received {scale_factor}."
            )
        if num_channels != 1:
            raise ValueError("The fixed FSRCNN pipeline is luminance-only.")

        self.scale_factor = scale_factor
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        middle: list[nn.Module] = [
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        ]
        for _ in range(m):
            middle.extend(
                (
                    nn.Conv2d(s, s, kernel_size=3, padding=1),
                    nn.PReLU(s),
                )
            )
        middle.extend((nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)))
        self.mid_part = nn.Sequential(*middle)
        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in (*self.first_part, *self.mid_part):
            if isinstance(module, nn.Conv2d):
                standard_deviation = sqrt(
                    2 / (module.out_channels * module.weight.data[0][0].numel())
                )
                nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.last_part.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias)

    def forward(self, luminance: Tensor) -> Tensor:
        features = self.first_part(luminance)
        features = self.mid_part(features)
        return self.last_part(features)


def load_pretrained_fsrcnn(
    checkpoint_path: str | Path,
    scale: int,
    device: str | torch.device,
) -> FSRCNN:
    """Load one verified scale-specific FSRCNN state dictionary strictly."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"FSRCNN checkpoint does not exist: {path}")

    target_device = torch.device(device)
    model = FSRCNN(scale_factor=scale)
    saved_state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved_state, dict):
        raise TypeError(f"Expected an FSRCNN state dictionary in {path}.")
    keys = tuple(saved_state)
    if keys and all(str(key).startswith("module.") for key in keys):
        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in saved_state.items()
        }
    else:
        state_dict = saved_state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(target_device)


def _rgb_to_ycbcr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16.0 + (64.738 * red + 129.057 * green + 25.064 * blue) / 256.0
    cb = 128.0 + (-37.945 * red - 74.494 * green + 112.439 * blue) / 256.0
    cr = 128.0 + (112.439 * red - 94.154 * green - 18.285 * blue) / 256.0
    return np.stack((y, cb, cr), axis=-1)


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> Image.Image:
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    red = 298.082 * y / 256.0 + 408.583 * cr / 256.0 - 222.921
    green = (
        298.082 * y / 256.0
        - 100.291 * cb / 256.0
        - 208.120 * cr / 256.0
        + 135.576
    )
    blue = 298.082 * y / 256.0 + 516.412 * cb / 256.0 - 276.836
    rgb = np.stack((red, green, blue), axis=-1)
    pixels = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def pil_to_luminance_tensor(
    image: Image.Image,
    device: str | torch.device,
) -> Tensor:
    """Convert one RGB image to normalized limited-range BT.601 luminance."""
    luminance = _rgb_to_ycbcr(image)[..., 0] / 255.0
    contiguous = np.ascontiguousarray(luminance[np.newaxis, np.newaxis, ...])
    return torch.from_numpy(contiguous).to(device)


def tensor_and_lr_to_pil(output: Tensor, lr_image: Image.Image) -> Image.Image:
    """Combine predicted luminance with bicubic-upscaled LR chroma."""
    if output.ndim == 4:
        if output.shape[0] != 1:
            raise ValueError(f"Expected one output image; got shape {tuple(output.shape)}.")
        output = output[0]
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected a CHW luminance tensor; got {tuple(output.shape)}.")

    predicted_y = output.detach().clamp(0.0, 1.0).mul(255.0).cpu().numpy()[0]
    target_size = (int(predicted_y.shape[1]), int(predicted_y.shape[0]))
    lr_ycbcr = _rgb_to_ycbcr(lr_image)
    enlarged_chroma = []
    for channel in (1, 2):
        plane = Image.fromarray(lr_ycbcr[..., channel], mode="F")
        resized = plane.resize(target_size, resample=Image.Resampling.BICUBIC)
        enlarged_chroma.append(np.asarray(resized, dtype=np.float32))
    return _ycbcr_to_rgb(
        np.stack((predicted_y, enlarged_chroma[0], enlarged_chroma[1]), axis=-1)
    )


def fsrcnn_upsample(
    model: FSRCNN,
    image: Image.Image,
    device: str | torch.device,
) -> Image.Image:
    """Run deterministic Y-only FSRCNN inference and bicubic chroma assembly."""
    with torch.inference_mode():
        output = model(pil_to_luminance_tensor(image, device))
    return tensor_and_lr_to_pil(output, image)
