"""Gradio interface composition for the Phase 7 demo."""

from __future__ import annotations

import warnings

import gradio as gr

from demo.inference import get_presets, run_super_resolution, update_slider
from demo.runtime import device_label, is_cuda
from demo.settings import DIRECT_LR_MODE, EXPERIMENT_MODE, XAI_MAPS
from demo.styles import EMIL_CSS, build_theme
from demo.xai import get_xai_images


def _make_blocks() -> gr.Blocks:
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
        return gr.Blocks(
            theme=build_theme(),
            css=EMIL_CSS,
            title="Super-Resolution Demo",
        )


def build_interface() -> gr.Blocks:
    dot_class = "device-dot" if is_cuda() else "device-dot-cpu"

    with _make_blocks() as demo:
        stored_images = gr.State({})

        gr.HTML(f"""
        <div class="app-header">
            <div class="app-title-group">
                <h1 class="app-title">Super-Resolution Studio</h1>
                <span class="app-subtitle">NEDI · FSRCNN · IMDN · Fusion</span>
            </div>
            <div class="app-device-badge">
                <span class="{dot_class}"></span>
                <span>{device_label()}</span>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(label="Input Image", type="pil", height=230)
                input_mode = gr.Radio(
                    choices=[EXPERIMENT_MODE, DIRECT_LR_MODE],
                    value=EXPERIMENT_MODE,
                    label="Input Mode",
                    info="Experiment-style is best for fair project demonstrations.",
                )
                scale_select = gr.Radio(
                    choices=[2, 3, 4],
                    value=2,
                    label="Scale",
                    info="Magnification",
                )

                with gr.Accordion("HR Reference (Optional)", open=False):
                    ref_input = gr.Image(
                        label="Ground Truth for Direct LR mode only",
                        type="pil",
                        height=160,
                    )
                    gr.Markdown(
                        "Use this only if you already have a matching LR/HR pair. "
                        "In experiment-style mode, the uploaded input image is already used as the HR reference."
                    )

                run_button = gr.Button("Upscale", variant="primary", size="lg")

                presets = get_presets()
                if presets:
                    gr.Examples(
                        examples=presets,
                        inputs=[input_image, scale_select, input_mode, ref_input],
                        label="Sample Inputs",
                    )

            with gr.Column(scale=6):
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

        with gr.Accordion("Individual Method Outputs", open=False):
            with gr.Row():
                bicubic_out = gr.Image(label="Bicubic", type="pil")
                nedi_out = gr.Image(label="NEDI", type="pil")
                fsrcnn_out = gr.Image(label="FSRCNN", type="pil")
                imdn_out = gr.Image(label="IMDN", type="pil")
                fusion_out = gr.Image(label="Fusion", type="pil")

        with gr.Accordion("XAI Feature Attribution (IMDN)", open=False):
            case_dropdown = gr.Dropdown(
                choices=list(XAI_MAPS.keys()),
                value="Set5: baby (x2)",
                label="Benchmark Case",
            )
            with gr.Row():
                lime_img = gr.Image(label="LIME Attribution", type="filepath")
                shap_img = gr.Image(label="SHAP Attribution", type="filepath")

        run_button.click(
            fn=run_super_resolution,
            inputs=[input_image, scale_select, input_mode, ref_input],
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

        case_dropdown.change(
            fn=get_xai_images,
            inputs=case_dropdown,
            outputs=[lime_img, shap_img],
        )

        demo.load(
            fn=get_xai_images,
            inputs=case_dropdown,
            outputs=[lime_img, shap_img],
        )

    return demo
