# Phase 7 Gradio demo notes

**Updated:** 7 September 2026  
**App entry point:** `app/demo.py`  
**Demo package:** `demo/`  
**Colab launcher:** `notebooks/19_phase7_gradio_demo_colab.ipynb`

## Purpose & Architecture

Phase 7 provides a publication-grade interactive demonstration and research showcase of the final project. The interface is organized into four specialized tabs:

The runnable command remains `python -m app.demo`, but the implementation is
split into the `demo/` package:

- `demo/interface.py`: Gradio layout and event wiring.
- `demo/inference.py`: Bicubic, NEDI, FSRCNN, IMDN, fusion, metrics, and ZIP export callbacks.
- `demo/runtime.py`: GPU/CPU detection and pretrained checkpoint loading.
- `demo/settings.py`: shared constants, paths, scale options, and XAI case mapping.
- `demo/styles.py`: Gradio theme and custom CSS.
- `demo/xai.py`: XAI figure lookup helpers.

1. **Interactive Studio (`studio_tab`):**
   - Upload either one HR image for experiment-style simulation or a prepared LR image for direct upscaling.
   - Select scaling factors: x2, x3, or x4.
   - In experiment-style mode, the app creates the LR image internally through bicubic downsampling, then evaluates reconstructions against the aligned HR reference.
   - In direct LR mode, an optional matching HR reference can be uploaded for live Y-PSNR (dB) and Y-SSIM evaluation with scale-dependent border cropping.
   - **Interactive Before/After Split Slider (`gr.ImageSlider`):** Live draggable split slider comparing Bicubic vs IMDN, LR vs IMDN, NEDI vs IMDN, or FSRCNN vs IMDN.
   - **Multi-Model Gallery:** 5 side-by-side output cards (Bicubic, NEDI, FSRCNN, IMDN, and Back-Projection Fusion).
   - Evaluation metrics table, status notes, and ZIP archive download for all reconstructed PNGs.

2. **Research Benchmark (`benchmark_tab`):**
   - Executive takeaway cards highlighting key empirical conclusions:
     - Quality Winner (IMDN: 29.38 dB PSNR-Y, 0.8357 SSIM-Y, 12/12 group wins).
     - Pareto Efficiency Champion (FSRCNN: 12.8K parameters, 6.41 GFLOPs at x3).
     - Scientific Negative Result Transparency on Fusion (rejection on validation data).
     - NEDI Grid Phase Alignment Audit (~3 dB PSNR restoration).
   - Full global performance overview table (`method_overview.csv`).
   - Deep learning peak VRAM and efficiency table (`deep_learning_efficiency.csv`).
   - Model computational complexity table (`model_complexity_fixed_256.csv`).
   - Interactive publication figure selector (PSNR-Y, SSIM-Y, VRAM, latency, and GFLOPs).

3. **Saved Phase 6 XAI Results (`xai_tab`):**
   - This section is a viewer for precomputed Phase 6 XAI figures, not a live XAI generator for uploaded demo images.
   - The interface states this clearly so users do not confuse LIME/SHAP explanations with another upscaling model.
   - Case-by-case browser covering all 4 benchmark evaluations:
     - `Set5 — baby.png (x2)`: Smooth facial skin and gradual contours.
     - `Set5 — butterfly.png (x3)`: High-frequency wing veins and edges.
     - `Set14 — baboon.png (x4)`: Chaotic natural fur texture at 4x scale.
     - `Urban100 — img_004.png (x3)`: Repetitive architectural grids.
   - Side-by-side display of LIME vs SHAP feature attribution heatmaps.
   - Empirical degradation metrics: original target PSNR-Y, average perturbation drop, and attribution runtime.
   - Detailed academic interpretation explaining IMDN's local receptive field dependency.

4. **Methodology & Protocol (`protocol_tab`):**
   - Comprehensive thesis summary, problem formulation, luminance-channel evaluation rules, border-cropping protocol, and citations.

## Visual Design & Styling

- Modern typography pairing: **Plus Jakarta Sans** for UI headers and body; **JetBrains Mono** for numerical metrics and parameters.
- Soft modern color palette: Indigo/Violet primary accents (`#4f46e5`), Slate neutrals, Emerald status pills.
- Gradient hero header with academic eyebrow badge (`🎓 FINAL YEAR RESEARCH PROJECT • 2026`).
- Hardware environment badge dynamically reporting GPU (CUDA) or CPU mode.
- Double-bezel card framing with subtle ambient elevation and responsive collapse.

## Run Command

Local:
```bash
python -m app.demo
```

Google Colab:
```bash
python -m app.demo --share
```

## Large Upload Handling

The demo keeps the whole uploaded image's aspect ratio. It does not take a
small patch/crop from the middle.

In direct LR mode, if the LR input is larger than 512 pixels on its longest side, the app
automatically resizes it to fit within 512 pixels before running the methods.
For example, a 1920x1440 LR image becomes 512x384.

In experiment-style mode, the uploaded image is treated as the HR reference.
If that HR image would create an LR image larger than 512 pixels on its longest
side, the HR display reference is resized first. The app then creates the LR
image from that resized HR reference using bicubic downsampling. This preserves
the same controlled degradation setup used in the project experiments.

This keeps NEDI practical because NEDI runs on CPU and becomes slow on large
images. When automatic resizing happens, live PSNR/SSIM are disabled because
the uploaded HR reference no longer corresponds exactly to the resized LR
pipeline in direct LR mode. In experiment-style mode, metrics remain valid
because the resized HR reference and generated LR image are kept aligned.
