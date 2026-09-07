# Phase 6 XAI protocol

## Purpose

Phase 6 explains the best deep-learning model from the final comparison. IMDN
was selected because it achieved the best PSNR-Y and SSIM-Y results in Phase 4,
and Phase 5 fusion did not improve it.

This phase does not train a new model and does not try to improve the result.
It asks a different question:

```text
Which parts of the LR input matter most to IMDN's reconstructed HR output?
```

## Why this needs a clear target

LIME and SHAP usually explain a single prediction, such as a class label in an
image-classification model. Super-resolution is different. IMDN does not output
a label; it outputs a new image.

Therefore, the explanation must be tied to a measurable image-output target.
For this project, the target is the quality of a selected HR region after IMDN
reconstruction.

In simple terms:

```text
Change parts of the LR input.
Run IMDN again.
Check how much the selected HR output region changes or loses quality.
The input areas that cause the biggest change are treated as important.
```

## Model and data

- Model explained: IMDN only.
- Scales: x2, x3, and x4.
- Image source: selected representative images already used in Phase 4 visual
  comparison.
- Image types covered:
  - smooth regions: Set5/baby.png
  - fine texture: Set5/butterfly.png
  - difficult natural texture: Set14/baboon.png
  - architectural edges and repetitive structure: Urban100/img_004.png

Using the Phase 4 visual samples keeps Phase 6 connected to results that have
already been discussed in the report.

## Explanation target

Each explanation run selects one visible HR region and measures IMDN quality in
that region using PSNR-Y.

PSNR-Y is used because it is the main metric in this project. Y means luminance,
which is the brightness/detail channel commonly used for super-resolution
evaluation.

The HR reference is used only to score the reconstructed region. It is not used
by IMDN to generate the output.

## LIME approach

LIME will divide the LR input image into simple regions. Then it will hide or
soften some of those regions, rerun IMDN, and measure how the selected output
region's PSNR-Y changes.

If hiding a region causes a large quality drop, that LR region is important for
the selected output region.

Expected output:

- original LR input
- IMDN reconstruction
- selected HR target region
- LIME importance map
- short written interpretation

## SHAP approach

SHAP will use the same idea of changing LR input regions and measuring the
effect on the selected output region. It estimates how much each region
contributes to the final measured quality.

Because SHAP can be slow for images, this project will use small image regions,
a small number of samples, and will record the runtime/settings.

Expected output:

- original LR input
- IMDN reconstruction
- selected HR target region
- SHAP importance map
- short written interpretation

## Required records

Every explanation result should record:

- dataset and image name
- scale
- selected region coordinates
- XAI method used
- perturbation settings
- number of samples
- runtime
- output paths for generated maps
- short interpretation

## Interpretation rule

The report should connect each heatmap to visible super-resolution behaviour.
For example:

- strong attention around edges may support edge reconstruction;
- strong attention around repeated structures may support pattern rebuilding;
- weak or scattered attention in texture regions may explain difficult texture
  recovery;
- attention on irrelevant smooth/background areas should be reported as a
  limitation, not hidden.

## Completion test

Phase 6 is complete when the generated LIME and SHAP maps are reproducible,
readable, saved with their settings, and explained in relation to IMDN's visible
super-resolution output.
