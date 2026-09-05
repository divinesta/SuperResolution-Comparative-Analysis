# Phase 4 visual comparison selection

**Written:** 5 September 2026  
**Notebook:** `notebooks/13_phase4_flops_and_visuals_colab.ipynb`

## Purpose

The visual comparison uses a small, predefined selection instead of showing
only images that make one method look favourable. Every method receives the
same prepared LR input, and every output is compared with the same HR reference.

## Selected images

| Dataset | Image | Main characteristic |
|---|---|---|
| Set5 | `baby.png` | Smooth face and gradual colour regions |
| Set5 | `butterfly.png` | Fine lines and high-frequency texture |
| Set14 | `baboon.png` | Difficult natural texture |
| Urban100 | `img_004.png` | Architectural edges and repetitive structure |

Each image is reconstructed at x2, x3, and x4 using Bicubic, NEDI, FSRCNN, and
IMDN. The notebook also saves the LR input and HR reference. This produces 12
comparison groups.

## Interpretation rule

The final inspection will use identical full images and identical crop
coordinates for every method. It will describe:

- edge sharpness or blurring;
- texture preservation or smoothing;
- cleanliness of smooth regions;
- ringing, jagged lines, false detail, or other visible artefacts.

PSNR and SSIM results remain the quantitative evidence. Visual examples are
supporting evidence and will not replace the complete numerical evaluation.

