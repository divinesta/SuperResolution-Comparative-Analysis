# Phase 7 Gradio demo notes

**Updated:** 7 September 2026  
**App entry point:** `app/demo.py`  
**Colab launcher:** `notebooks/19_phase7_gradio_demo_colab.ipynb`

## Purpose & Architecture

Phase 7 provides a publication-grade interactive demonstration and research showcase of the final project. The interface is organized into four specialized tabs:

1. **Interactive Studio (`studio_tab`):**
   - Upload any custom LR image or choose from built-in sample presets (`demo/examples/`).
   - Select scaling factors: x2, x3, or x4.
   - Optional Ground Truth (HR) reference upload for live Y-PSNR (dB) and Y-SSIM evaluation with scale-dependent border cropping.
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

3. **Explainability / XAI Explorer (`xai_tab`):**
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

The demo keeps the whole uploaded image. It does not crop large uploads.

If the LR input is larger than 512 pixels on its longest side, the app
automatically resizes it to fit within 512 pixels before running the methods.
For example, a 1920x1440 LR image becomes 512x384.

This keeps NEDI practical because NEDI runs on CPU and becomes slow on large
images. When automatic resizing happens, live PSNR/SSIM are disabled because
the uploaded HR reference no longer corresponds exactly to the resized LR
pipeline.
