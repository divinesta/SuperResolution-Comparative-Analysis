# Phase 5 weighted-fusion protocol

## Method

Phase 5 combines the final RGB output from NEDI with the final RGB output from
IMDN. IMDN was selected because it was the best deep-learning method in Phase
4. For every matching pixel and colour channel:

```text
fusion = (NEDI weight × NEDI pixel) + (IMDN weight × IMDN pixel)
NEDI weight = 1 - IMDN weight
```

The result is rounded to the nearest 8-bit pixel value. This is a fixed,
transparent calculation; no new model is trained.

## Weight selection fixed before the final test

- Candidate IMDN weights: 0.0, 0.1, ..., 1.0.
- Candidate NEDI weight: one minus the IMDN weight.
- Validation data: DIV2K validation images 0801-0805 only.
- Each image contributes one fixed 360 × 360 centre HR crop.
- Scales: x2, x3, and x4, giving 15 validation image-scale samples per weight.
- LR creation: the project's controlled bicubic downsampling.
- Main selection value: mean PSNR-Y across all 15 samples.
- Tie-breaks: mean SSIM-Y, then the larger IMDN weight.
- One global weight is selected and used unchanged for every final dataset and
  scale. No final Set5, Set14, BSD100, or Urban100 result is used to choose it.

The 360 × 360 crop is divisible by x2, x3, and x4. It keeps validation runtime
manageable while applying every scale to the same image area.

## Required records

The validation run saves every per-image result, the average for every weight,
the selected weight, the source image IDs, crop rule, checkpoint hashes, and a
plot. The endpoints are included: IMDN weight 0 is NEDI alone, and IMDN weight
1 is IMDN alone. This endpoint comparison is the Phase 5 ablation.

If no mixture beats IMDN alone, the selected weight may be 1.0. That result
must be reported honestly rather than claiming that fusion improved quality.

## Edge-aware follow-up

The global weighted validation selected 100% IMDN: all 15 validation cases had
their best PSNR-Y at IMDN weight 1.0. Therefore, one final simple edge-aware
experiment is permitted before closing Phase 5.

- Edge guide: Sobel gradient strength calculated from bicubic-upsampled LR
  luminance. The HR reference is never used to locate edges.
- Edge cut-offs: 80th, 90th, and 95th percentiles.
- NEDI contribution at detected edges: 5%, 10%, 20%, and 30%.
- NEDI contribution outside detected edges: 0%.
- Detected edge mask expansion: one pixel.
- Validation images, scales, crop rule, metrics, and selection rule remain the
  same as the global weighted experiment.
- IMDN alone is included as a candidate and must be beaten on mean PSNR-Y.

This is the final fusion variation. If it does not beat IMDN on validation, no
further fusion variants will be tried; the negative result will be reported.
