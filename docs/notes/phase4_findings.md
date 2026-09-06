# Phase 4 controlled comparison findings

**Written:** 5 September 2026  
**Notebook:** `notebooks/12_phase4_comparative_analysis.ipynb`

## Data validation

- All four methods have summaries for the four datasets at x2, x3, and x4.
- NEDI, FSRCNN, and IMDN each have 657 detailed image-scale records.
- Each detailed result set contains all 12 required dataset-scale groups and
  reproduces its final summary.
- The repository does not currently contain the final per-image Bicubic records,
  so Bicubic is included through its validated final summary.

## Main quality result

The image-count-weighted Y-channel results are:

| Method | PSNR-Y | SSIM-Y | PSNR-Y group wins | SSIM-Y group wins |
|---|---:|---:|---:|---:|
| IMDN | 29.3772 | 0.8357 | 12 | 12 |
| FSRCNN | 28.0485 | 0.8026 | 0 | 0 |
| Bicubic | 26.4251 | 0.7524 | 0 | 0 |
| NEDI | 26.1573 | 0.7382 | 0 | 0 |

IMDN achieved the highest PSNR-Y and SSIM-Y in every dataset-scale group.
FSRCNN ranked second throughout the quality comparison. Bicubic generally
outperformed NEDI on the numerical quality metrics, although visual edge
behaviour still needs to be inspected before the qualitative conclusion is
written.

## Timing limitation

Bicubic was timed on Google Colab's default CPU; the exact CPU model was not
recorded. NEDI was timed on an AWS m7i.2xlarge CPU instance. FSRCNN and IMDN
were timed on an NVIDIA Tesla T4 GPU. These values describe the completed
experiments but must not be presented as a direct same-hardware speed ranking.

## Completion

FLOPs and the fixed visual comparison are now complete. The detailed values are
stored in `results/metrics/final/model_complexity_fixed_256.csv` and
`results/metrics/final/visual_sample_metrics.csv`. The final full-image and
identical-crop figures are stored under `results/figures/visual_comparisons/`,
and their interpretation is documented in
`docs/notes/phase4_visual_findings.md`.

The final per-image Bicubic CSV is not available in the repository. Its complete
12-group final summary remains the source used for Phase 4. This limitation is
recorded rather than silently replacing it with preliminary results generated
under an older schema.
