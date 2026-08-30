"""IMDN architecture and image inference used by the Phase 3 pipeline.

The module layout intentionally follows the authors' published checkpoint
names so their x2, x3, and x4 state dictionaries can be loaded strictly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn


def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
) -> nn.Conv2d:
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
    )


class _ContrastChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        spatial_mean = features.mean(dim=(2, 3), keepdim=True)
        spatial_variance = (features - spatial_mean).square().mean(
            dim=(2, 3),
            keepdim=True,
        )
        contrast = spatial_variance.sqrt()
        attention = self.conv_du(contrast + self.avg_pool(features))
        return features * attention


class _InformationMultiDistillationBlock(nn.Module):
    def __init__(self, in_channels: int, distillation_rate: float = 0.25) -> None:
        super().__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels
        self.c1 = _conv(in_channels, in_channels, 3)
        self.c2 = _conv(self.remaining_channels, in_channels, 3)
        self.c3 = _conv(self.remaining_channels, in_channels, 3)
        self.c4 = _conv(self.remaining_channels, self.distilled_channels, 3)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = _conv(in_channels, in_channels, 1)
        self.cca = _ContrastChannelAttention(self.distilled_channels * 4)

    def _split(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return torch.split(
            features,
            (self.distilled_channels, self.remaining_channels),
            dim=1,
        )

    def forward(self, features: Tensor) -> Tensor:
        distilled_1, remaining_1 = self._split(self.act(self.c1(features)))
        distilled_2, remaining_2 = self._split(self.act(self.c2(remaining_1)))
        distilled_3, remaining_3 = self._split(self.act(self.c3(remaining_2)))
        distilled_4 = self.c4(remaining_3)
        combined = torch.cat(
            (distilled_1, distilled_2, distilled_3, distilled_4),
            dim=1,
        )
        return self.c5(self.cca(combined)) + features


class IMDN(nn.Module):
    """Information Multi-Distillation Network for RGB super-resolution."""

    def __init__(
        self,
        *,
        in_nc: int = 3,
        nf: int = 64,
        num_modules: int = 6,
        out_nc: int = 3,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        if num_modules != 6:
            raise ValueError("The official pretrained IMDN uses exactly six modules.")
        if upscale not in {2, 3, 4}:
            raise ValueError(f"IMDN scale must be 2, 3, or 4; received {upscale}.")

        self.upscale = upscale
        self.fea_conv = _conv(in_nc, nf, 3)
        self.IMDB1 = _InformationMultiDistillationBlock(nf)
        self.IMDB2 = _InformationMultiDistillationBlock(nf)
        self.IMDB3 = _InformationMultiDistillationBlock(nf)
        self.IMDB4 = _InformationMultiDistillationBlock(nf)
        self.IMDB5 = _InformationMultiDistillationBlock(nf)
        self.IMDB6 = _InformationMultiDistillationBlock(nf)
        self.c = nn.Sequential(
            nn.Conv2d(nf * num_modules, nf, 1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )
        self.LR_conv = _conv(nf, nf, 3)
        self.upsampler = nn.Sequential(
            _conv(nf, out_nc * upscale**2, 3),
            nn.PixelShuffle(upscale),
        )

    def forward(self, image: Tensor) -> Tensor:
        shallow = self.fea_conv(image)
        block_1 = self.IMDB1(shallow)
        block_2 = self.IMDB2(block_1)
        block_3 = self.IMDB3(block_2)
        block_4 = self.IMDB4(block_3)
        block_5 = self.IMDB5(block_4)
        block_6 = self.IMDB6(block_5)
        fused = self.c(
            torch.cat(
                (block_1, block_2, block_3, block_4, block_5, block_6),
                dim=1,
            )
        )
        restored_features = self.LR_conv(fused) + shallow
        return self.upsampler(restored_features)


def load_pretrained_imdn(
    checkpoint_path: str | Path,
    scale: int,
    device: str | torch.device,
) -> IMDN:
    """Load an official scale-specific state dictionary with strict matching."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"IMDN checkpoint does not exist: {path}")

    target_device = torch.device(device)
    model = IMDN(upscale=scale)
    saved_state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved_state, dict):
        raise TypeError(f"Expected an IMDN state dictionary in {path}.")
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


def pil_to_tensor(image: Image.Image, device: str | torch.device) -> Tensor:
    """Convert one RGB Pillow image to a normalized NCHW float tensor."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    contiguous = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).unsqueeze(0).to(device)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    """Convert one normalized NCHW/CHW tensor to an 8-bit RGB image."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected one output image; got shape {tuple(tensor.shape)}.")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected a CHW RGB tensor; got shape {tuple(tensor.shape)}.")

    pixels = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def imdn_upsample(model: IMDN, image: Image.Image, device: str | torch.device) -> Image.Image:
    """Run deterministic, gradient-free RGB inference on one LR image."""
    with torch.inference_mode():
        output = model(pil_to_tensor(image, device))
    return tensor_to_pil(output)
