"""Interactive super-resolution demonstration interface."""

from __future__ import annotations

import tempfile
import time
import warnings
import zipfile
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch
from PIL import Image

from app.config import PROJECT_ROOT, resolve_data_root
from app.deep_learning.alignment import align_reconstruction_to_target
from app.deep_learning.checkpoints import (
    download_official_imdn_checkpoint,
    download_pretrained_fsrcnn_checkpoint,
)
from app.deep_learning.fsrcnn import fsrcnn_upsample, load_pretrained_fsrcnn
from app.deep_learning.imdn import imdn_upsample, load_pretrained_imdn
from app.evaluation.images import bicubic_upsample
from app.evaluation.metrics import calculate_quality_metrics
from app.fusion.back_projection import back_projection_fusion
from app.traditional.nedi import NEDIConfig, nedi_upsample_rgb


SCALES = (2, 3, 4)
MAX_LR_SIDE = 512
NEDI_CONFIG = NEDIConfig(window_size=8, edge_threshold=8.0)
XAI_ROOT = PROJECT_ROOT / "results" / "figures" / "xai"
EXAMPLES_ROOT = PROJECT_ROOT / "demo" / "examples"


@dataclass(frozen=True)
class MethodOutput:
    image: Image.Image
    seconds: float
    notes: str


@dataclass(frozen=True)
class PreparedInput:
    image: Image.Image
    original_size: tuple[int, int]
    resized: bool


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_cuda() -> bool:
    return torch.cuda.is_available()


def _device_label() -> str:
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"


def _checkpoint_root() -> Path:
    return resolve_data_root() / "checkpoints"


@lru_cache(maxsize=3)
def _load_fsrcnn(scale: int, device_name: str) -> Any:
    checkpoint = download_pretrained_fsrcnn_checkpoint(_checkpoint_root(), scale)
    return load_pretrained_fsrcnn(checkpoint, scale, torch.device(device_name))


@lru_cache(maxsize=3)
def _load_imdn(scale: int, device_name: str) -> Any:
    checkpoint = download_official_imdn_checkpoint(_checkpoint_root(), scale)
    return load_pretrained_imdn(checkpoint, scale, torch.device(device_name))


def _clean_image(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _target_size(lr_image: Image.Image, scale: int) -> tuple[int, int]:
    return (lr_image.width * scale, lr_image.height * scale)


def _resize_for_demo(lr_image: Image.Image) -> PreparedInput:
    largest_side = max(lr_image.size)
    if largest_side <= MAX_LR_SIDE:
        return PreparedInput(
            image=lr_image,
            original_size=lr_image.size,
            resized=False,
        )

    ratio = MAX_LR_SIDE / largest_side
    resized_size = (
        max(1, round(lr_image.width * ratio)),
        max(1, round(lr_image.height * ratio)),
    )
    resized = lr_image.resize(resized_size, Image.Resampling.BICUBIC)
    return PreparedInput(
        image=resized,
        original_size=lr_image.size,
        resized=True,
    )


def _resize_note(prepared: PreparedInput) -> str | None:
    if not prepared.resized:
        return None
    old_width, old_height = prepared.original_size
    new_width, new_height = prepared.image.size
    return (
        f"Input resized from {old_width}x{old_height}px to "
        f"{new_width}x{new_height}px for demo speed."
    )


def _time_call(function: Any, *args: Any) -> MethodOutput:
    start = time.perf_counter()
    image, notes = function(*args)
    return MethodOutput(image=image, seconds=time.perf_counter() - start, notes=notes)


def _run_bicubic(lr_image: Image.Image, scale: int) -> tuple[Image.Image, str]:
    return bicubic_upsample(lr_image, _target_size(lr_image, scale)), "Bicubic"


def _run_nedi(lr_image: Image.Image, scale: int) -> tuple[Image.Image, str]:
    result = nedi_upsample_rgb(lr_image, scale, _target_size(lr_image, scale), NEDI_CONFIG)
    return result.image, f"NEDI ({result.native_passes} pass)"


def _run_fsrcnn(lr_image: Image.Image, scale: int, device: torch.device) -> tuple[Image.Image, str]:
    try:
        model = _load_fsrcnn(scale, str(device))
        native = fsrcnn_upsample(model, lr_image, device)
        result = align_reconstruction_to_target(native, _target_size(lr_image, scale))
        return result.image, "FSRCNN (12.8K)"
    except Exception as exc:
        fallback = bicubic_upsample(lr_image, _target_size(lr_image, scale))
        return fallback, f"FSRCNN (Unavailable: {exc})"


def _run_imdn(lr_image: Image.Image, scale: int, device: torch.device) -> tuple[Image.Image, str]:
    try:
        model = _load_imdn(scale, str(device))
        native = imdn_upsample(model, lr_image, device)
        result = align_reconstruction_to_target(native, _target_size(lr_image, scale))
        return result.image, "IMDN (703K)"
    except Exception as exc:
        fallback = bicubic_upsample(lr_image, _target_size(lr_image, scale))
        return fallback, f"IMDN (Unavailable: {exc})"


def _metric_row(
    method: str,
    output: MethodOutput,
    reference_hr: Image.Image | None,
    scale: int,
) -> dict[str, str | float]:
    row: dict[str, str | float] = {
        "Method": method,
        "PSNR-Y": "—",
        "SSIM-Y": "—",
        "Latency": f"{output.seconds:.3f}s",
        "Notes": output.notes,
    }
    if reference_hr is not None and reference_hr.size == output.image.size:
        metrics = calculate_quality_metrics(reference_hr, output.image, border=scale)
        row["PSNR-Y"] = f"{metrics['psnr_y']:.2f} dB"
        row["SSIM-Y"] = f"{metrics['ssim_y']:.4f}"
    return row


def _save_zip(outputs: dict[str, MethodOutput], scale: int) -> str:
    temp_file = tempfile.NamedTemporaryFile(prefix=f"sr_x{scale}_", suffix=".zip", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, out in outputs.items():
            img_path = temp_path.with_name(f"{name}_x{scale}.png")
            out.image.save(img_path)
            archive.write(img_path, arcname=f"{name}_x{scale}.png")
            img_path.unlink(missing_ok=True)
    return str(temp_path)


XAI_MAPS = {
    "Set5: baby (x2)": XAI_ROOT / "Set5_baby_x2",
    "Set5: butterfly (x3)": XAI_ROOT / "Set5_butterfly_x3",
    "Set14: baboon (x4)": XAI_ROOT / "Set14_baboon_x4",
    "Urban100: img_004 (x3)": XAI_ROOT / "Urban100_img_004_x3",
}


def _get_xai_images(case: str) -> tuple[str | None, str | None]:
    folder = XAI_MAPS.get(case)
    if not folder or not folder.is_dir():
        return None, None
    lime = folder / "lime_importance_final.png"
    shap = folder / "shap_importance_final.png"
    return (str(lime) if lime.exists() else None, str(shap) if shap.exists() else None)


def _get_presets() -> list[list[Any]]:
    presets = []
    if (EXAMPLES_ROOT / "geometric_lr.png").exists():
        presets.append([str(EXAMPLES_ROOT / "geometric_lr.png"), 2, str(EXAMPLES_ROOT / "geometric_hr.png")])
    if (EXAMPLES_ROOT / "concentric_lr.png").exists():
        presets.append([str(EXAMPLES_ROOT / "concentric_lr.png"), 2, str(EXAMPLES_ROOT / "concentric_hr.png")])
    if (EXAMPLES_ROOT / "lattice_lr.png").exists():
        presets.append([str(EXAMPLES_ROOT / "lattice_lr.png"), 2, str(EXAMPLES_ROOT / "lattice_hr.png")])
    return presets


def run_super_resolution(
    lr_image: Image.Image | None,
    scale: int,
    reference_hr: Image.Image | None,
) -> tuple[
    tuple[Image.Image, Image.Image] | None,
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    pd.DataFrame,
    str | None,
    dict[str, Image.Image],
]:
    if lr_image is None:
        raise gr.Error("Upload an image first.")

    scale = int(scale)
    if scale not in SCALES:
        raise gr.Error(f"Scale must be one of {SCALES}.")

    prepared = _resize_for_demo(_clean_image(lr_image))
    lr_image = prepared.image
    reference = (
        _clean_image(reference_hr)
        if reference_hr is not None and not prepared.resized
        else None
    )
    device = _device()

    bicubic = _time_call(_run_bicubic, lr_image, scale)
    nedi = _time_call(_run_nedi, lr_image, scale)
    fsrcnn = _time_call(_run_fsrcnn, lr_image, scale, device)
    imdn = _time_call(_run_imdn, lr_image, scale, device)

    fusion_start = time.perf_counter()
    fusion_res = back_projection_fusion(lr_image, nedi.image, imdn.image)
    fusion = MethodOutput(
        image=fusion_res.image,
        seconds=time.perf_counter() - fusion_start,
        notes=f"Fusion (IMDN {fusion_res.imdn_weight:.2f}, NEDI {fusion_res.nedi_weight:.2f})",
    )

    outputs = {
        "bicubic": bicubic,
        "nedi": nedi,
        "fsrcnn": fsrcnn,
        "imdn": imdn,
        "fusion": fusion,
    }

    records = [_metric_row(k.upper(), v, reference, scale) for k, v in outputs.items()]
    df = pd.DataFrame(records)
    zip_path = _save_zip(outputs, scale)

    target_size = _target_size(lr_image, scale)
    image_store = {
        "lr": lr_image.resize(target_size, Image.Resampling.NEAREST),
        "bicubic": bicubic.image,
        "nedi": nedi.image,
        "fsrcnn": fsrcnn.image,
        "imdn": imdn.image,
        "fusion": fusion.image,
    }
    slider = (bicubic.image, imdn.image)

    resize_note = _resize_note(prepared)
    if resize_note is not None:
        df.loc[len(df)] = {
            "Method": "DEMO",
            "PSNR-Y": "—",
            "SSIM-Y": "—",
            "Latency": "—",
            "Notes": f"{resize_note} PSNR/SSIM disabled because the LR input was resized.",
        }

    return (
        slider,
        bicubic.image,
        nedi.image,
        fsrcnn.image,
        imdn.image,
        fusion.image,
        df,
        zip_path,
        image_store,
    )


def update_slider(pair: str, store: dict[str, Image.Image] | None) -> tuple[Image.Image, Image.Image] | None:
    if not store:
        return None
    if "LR" in pair:
        return (store.get("lr", store["bicubic"]), store["imdn"])
    if "NEDI" in pair:
        return (store["nedi"], store["imdn"])
    if "FSRCNN" in pair:
        return (store["fsrcnn"], store["imdn"])
    return (store["bicubic"], store["imdn"])


EMIL_CSS = """
/* Layout Container */
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 16px !important;
}

/* Header */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color-primary, #e4e4e7);
    margin-bottom: 24px;
}

.dark .app-header {
    border-bottom-color: #27272a;
}

.app-title-group {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.app-title {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    margin: 0 !important;
    color: var(--body-text-color, #09090b);
}

.app-subtitle {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued, #71717a);
}

.app-device-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.775rem;
    font-family: var(--font-mono);
    color: var(--body-text-color-subdued, #71717a);
    background: var(--background-fill-secondary, #f4f4f5);
    padding: 4px 10px;
    border-radius: 9999px;
    border: 1px solid var(--border-color-primary, #e4e4e7);
}

.dark .app-device-badge {
    background: #18181b;
    border-color: #27272a;
}

.device-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
}

.device-dot-cpu {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #a1a1aa;
}

/* Primary Button: Emil Kowalski physical response */
button.primary {
    background: #18181b !important;
    color: #fafafa !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: -0.01em !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    transition: transform 120ms cubic-bezier(0.23, 1, 0.32, 1), background-color 120ms ease !important;
}

.dark button.primary {
    background: #f4f4f5 !important;
    color: #09090b !important;
    border-color: rgba(0, 0, 0, 0.1) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
}

button.primary:hover {
    background: #27272a !important;
}

.dark button.primary:hover {
    background: #e4e4e7 !important;
}

button.primary:active {
    transform: scale(0.975) !important;
}

/* Form & Input Polish */
.gr-form, .gr-box {
    border-radius: 8px !important;
    border: 1px solid var(--border-color-primary, #e4e4e7) !important;
}

.dark .gr-form, .dark .gr-box {
    border-color: #27272a !important;
}

/* Segmented / Radio */
.gr-radio {
    gap: 4px !important;
}

/* Clean Image Slider Frame */
.image-slider {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-color-primary, #e4e4e7) !important;
}

.dark .image-slider {
    border-color: #27272a !important;
}

/* Table Polish */
table {
    font-size: 0.825rem !important;
    border-collapse: collapse !important;
}

th {
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: var(--body-text-color-subdued, #71717a) !important;
    border-bottom: 1px solid var(--border-color-primary, #e4e4e7) !important;
    padding: 8px 12px !important;
}

.dark th {
    border-bottom-color: #27272a !important;
}

td {
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border-color-primary, #f4f4f5) !important;
    font-family: var(--font-mono) !important;
}

.dark td {
    border-bottom-color: #18181b !important;
}

td:first-child, td:last-child {
    font-family: var(--font-sans) !important;
}

/* Secondary Drawer Accordions */
.gr-accordion {
    border: 1px solid var(--border-color-primary, #e4e4e7) !important;
    border-radius: 8px !important;
    margin-top: 12px !important;
}

.dark .gr-accordion {
    border-color: #27272a !important;
}

.gr-accordion > .label-wrap {
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
"""


def build_theme() -> gr.Theme:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.zinc,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        button_primary_background_fill="#18181b",
        button_primary_background_fill_hover="#27272a",
        button_primary_text_color="#fafafa",
        block_border_width="1px",
        block_radius="8px",
    )
    return theme


def build_interface() -> gr.Blocks:
    theme = build_theme()
    dot_class = "device-dot" if _is_cuda() else "device-dot-cpu"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The 'theme' parameter in the Blocks constructor will be removed.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The 'css' parameter in the Blocks constructor will be removed.*",
            category=DeprecationWarning,
        )
        demo_context = gr.Blocks(
            theme=theme,
            css=EMIL_CSS,
            title="Super-Resolution Demo",
        )

    with demo_context as demo:
        stored_images = gr.State({})

        # Header
        gr.HTML(f"""
        <div class="app-header">
            <div class="app-title-group">
                <h1 class="app-title">Super-Resolution Studio</h1>
                <span class="app-subtitle">NEDI · FSRCNN · IMDN · Fusion</span>
            </div>
            <div class="app-device-badge">
                <span class="{dot_class}"></span>
                <span>{_device_label()}</span>
            </div>
        </div>
        """)

        with gr.Row():
            # Left: Controls
            with gr.Column(scale=4):
                lr_input = gr.Image(label="Low-Resolution Input", type="pil", height=230)
                with gr.Row():
                    scale_select = gr.Radio(choices=[2, 3, 4], value=2, label="Scale", info="Magnification")
                with gr.Accordion("HR Reference (Optional)", open=False):
                    ref_input = gr.Image(label="Ground Truth (Calculates PSNR/SSIM)", type="pil", height=160)

                run_button = gr.Button("Upscale", variant="primary", size="lg")

                presets = _get_presets()
                if presets:
                    gr.Examples(
                        examples=presets,
                        inputs=[lr_input, scale_select, ref_input],
                        label="Sample Inputs",
                    )

            # Right: Interactive Slider & Metrics
            with gr.Column(scale=6):
                with gr.Row():
                    pair_select = gr.Dropdown(
                        choices=[
                            "Bicubic vs IMDN",
                            "LR vs IMDN",
                            "NEDI vs IMDN",
                            "FSRCNN vs IMDN",
                        ],
                        value="Bicubic vs IMDN",
                        label="Inspect Pair",
                        container=False,
                        scale=3,
                    )
                slider_view = gr.ImageSlider(
                    label="Comparison View",
                    type="pil",
                    height=350,
                )

                metrics_table = gr.Dataframe(label="Results", interactive=False)
                download_file = gr.File(label="Export Reconstructions (.zip)", interactive=False)

        # Output gallery (collapsible drawer)
        with gr.Accordion("Individual Method Outputs", open=False):
            with gr.Row():
                bicubic_out = gr.Image(label="Bicubic", type="pil")
                nedi_out = gr.Image(label="NEDI", type="pil")
                fsrcnn_out = gr.Image(label="FSRCNN", type="pil")
                imdn_out = gr.Image(label="IMDN", type="pil")
                fusion_out = gr.Image(label="Fusion", type="pil")

        # XAI Inspection (collapsible drawer)
        with gr.Accordion("XAI Feature Attribution (IMDN)", open=False):
            case_dropdown = gr.Dropdown(
                choices=list(XAI_MAPS.keys()),
                value="Set5: baby (x2)",
                label="Benchmark Case",
            )
            with gr.Row():
                lime_img = gr.Image(label="LIME Attribution", type="filepath")
                shap_img = gr.Image(label="SHAP Attribution", type="filepath")

        # Wire events
        run_button.click(
            fn=run_super_resolution,
            inputs=[lr_input, scale_select, ref_input],
            outputs=[
                slider_view,
                bicubic_out,
                nedi_out,
                fsrcnn_out,
                imdn_out,
                fusion_out,
                metrics_table,
                download_file,
                stored_images,
            ],
        )

        pair_select.change(
            fn=update_slider,
            inputs=[pair_select, stored_images],
            outputs=slider_view,
        )

        def _update_xai(case: str):
            return _get_xai_images(case)

        case_dropdown.change(
            fn=_update_xai,
            inputs=case_dropdown,
            outputs=[lime_img, shap_img],
        )

        demo.load(
            fn=_update_xai,
            inputs=case_dropdown,
            outputs=[lime_img, shap_img],
        )

    return demo


def main() -> None:
    parser = ArgumentParser(description="Launch super-resolution demo.")
    parser.add_argument("--share", action="store_true", help="Public Colab link.")
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()
    build_interface().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
