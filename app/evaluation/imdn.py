"""IMDN evaluation through the project's fixed super-resolution protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import torch

from app.deep_learning.alignment import align_reconstruction_to_target
from app.deep_learning.checkpoints import IMDN_CHECKPOINTS
from app.deep_learning.imdn import IMDN, pil_to_tensor, tensor_to_pil
from app.evaluation.experiment import environment_fields, save_image
from app.evaluation.images import load_rgb_image, validate_hr_lr_dimensions
from app.evaluation.metrics import calculate_quality_metrics
from app.evaluation.timing import TimingStats


@dataclass(frozen=True)
class IMDNEvaluationConfig:
    """Settings shared by one dataset-scale IMDN GPU evaluation."""

    dataset: str
    scale: int
    warmup_runs: int = 3
    timed_runs: int = 10

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("Dataset name cannot be empty.")
        if self.scale not in {2, 3, 4}:
            raise ValueError(f"Scale must be 2, 3, or 4; received {self.scale}.")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative.")
        if self.timed_runs <= 0:
            raise ValueError("timed_runs must be greater than zero.")


def _measure_cuda_forward(
    model: IMDN,
    input_tensor: torch.Tensor,
    config: IMDNEvaluationConfig,
) -> tuple[torch.Tensor, TimingStats, float]:
    if input_tensor.device.type != "cuda":
        raise ValueError("IMDN GPU timing requires a CUDA input tensor.")

    with torch.inference_mode():
        for _ in range(config.warmup_runs):
            model(input_tensor)
    torch.cuda.synchronize(input_tensor.device)
    torch.cuda.reset_peak_memory_stats(input_tensor.device)

    durations_ms: list[float] = []
    output: torch.Tensor | None = None
    with torch.inference_mode():
        for _ in range(config.timed_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(input_tensor)
            end.record()
            torch.cuda.synchronize(input_tensor.device)
            durations_ms.append(float(start.elapsed_time(end)))

    peak_memory_mb = torch.cuda.max_memory_allocated(input_tensor.device) / 1024**2
    timing = TimingStats(
        latency_mean_ms=mean(durations_ms),
        latency_median_ms=median(durations_ms),
        latency_std_ms=pstdev(durations_ms),
        latency_min_ms=min(durations_ms),
        latency_max_ms=max(durations_ms),
        warmup_runs=config.warmup_runs,
        timed_runs=config.timed_runs,
    )
    return output, timing, peak_memory_mb  # type: ignore[return-value]


def evaluate_imdn_image(
    hr_path: str | Path,
    lr_path: str | Path,
    model: IMDN,
    config: IMDNEvaluationConfig,
    device: torch.device,
    *,
    sr_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one prepared HR/LR pair with a loaded official IMDN model."""
    if device.type != "cuda":
        raise ValueError("The full IMDN GPU evaluator requires a CUDA device.")
    if model.upscale != config.scale:
        raise ValueError(
            f"Loaded IMDN is x{model.upscale}, but evaluation requested x{config.scale}."
        )

    hr_image_path = Path(hr_path)
    lr_image_path = Path(lr_path)
    reference_hr = load_rgb_image(hr_image_path)
    lr_image = load_rgb_image(lr_image_path)
    validate_hr_lr_dimensions(reference_hr, lr_image, config.scale)

    input_tensor = pil_to_tensor(lr_image, device)
    output_tensor, timing, peak_memory_mb = _measure_cuda_forward(
        model,
        input_tensor,
        config,
    )
    native_reconstruction = tensor_to_pil(output_tensor)
    expected_native_size = (
        lr_image.width * config.scale,
        lr_image.height * config.scale,
    )
    if native_reconstruction.size != expected_native_size:
        raise RuntimeError(
            f"IMDN x{config.scale} returned {native_reconstruction.size}; "
            f"expected {expected_native_size}."
        )

    aligned = align_reconstruction_to_target(native_reconstruction, reference_hr.size)
    metrics = calculate_quality_metrics(
        reference_hr,
        aligned.image,
        border=config.scale,
    )
    if sr_output_dir is not None:
        output_name = f"{hr_image_path.stem}_imdn_x{config.scale}.png"
        save_image(aligned.image, Path(sr_output_dir) / output_name)

    provenance = IMDN_CHECKPOINTS[config.scale]
    device_properties = torch.cuda.get_device_properties(device)
    return {
        "dataset": config.dataset,
        "image": hr_image_path.name,
        "scale": f"x{config.scale}",
        "method": "imdn",
        "degradation": "prepared_bicubic_lr",
        "lr_source_file": lr_image_path.name,
        "metric_border_pixels": config.scale,
        "dimension_policy": "native_model_grid_then_full_target_size_adjustment",
        "ssim_protocol": "gaussian_11x11_sigma_1.5_population_covariance",
        "colour_policy": "rgb_model",
        "timing_device": "gpu",
        "timing_scope": "model_forward_only_cuda_events",
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_mb": device_properties.total_memory / 1024**2,
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_original_filename": provenance.original_filename,
        "checkpoint_sha256": provenance.sha256,
        "checkpoint_source_url": provenance.source_url,
        "checkpoint_training_dataset": provenance.training_dataset,
        "checkpoint_degradation": provenance.degradation,
        "native_width": aligned.native_size[0],
        "native_height": aligned.native_size[1],
        "target_width": aligned.target_size[0],
        "target_height": aligned.target_size[1],
        "dimension_adjusted": aligned.dimension_adjusted,
        **metrics,
        **timing.as_dict(),
        "hr_width": reference_hr.width,
        "hr_height": reference_hr.height,
        "lr_width": lr_image.width,
        "lr_height": lr_image.height,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda or "unknown",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **environment_fields(),
    }
