# Phase 4 visual and complexity findings

**Completed:** 6 September 2026  
**Generation notebook:** `notebooks/13_phase4_flops_and_visuals_colab.ipynb`  
**Comparison figures:** `results/figures/visual_comparisons/`

## What was inspected

Four images were fixed before the outputs were inspected: `baby.png` for smooth
facial regions, `butterfly.png` for fine lines, `baboon.png` for natural
texture, and Urban100 `img_004.png` for repetitive architectural structure.
Each image was compared at x2, x3, and x4. Every figure uses the same crop for
the HR reference, Bicubic, NEDI, FSRCNN, and IMDN.

## Visual findings

- On `baby.png`, IMDN and FSRCNN preserve the eye boundaries, eyelashes, and
  hat texture more clearly. Bicubic and NEDI are smoother and lose more detail,
  especially at x3 and x4. The smooth skin regions remain visually stable for
  all methods.
- On `butterfly.png`, IMDN provides the clearest wing veins and white spots,
  with FSRCNN second. Bicubic and NEDI increasingly soften and widen the dark
  lines as the scale increases.
- On `baboon.png`, IMDN retains the most visible fur texture and the clearest
  boundary around the eye. FSRCNN retains more texture than the classical
  methods, while Bicubic and NEDI lose much of the fine fur at x4.
- On Urban100 `img_004.png`, IMDN and FSRCNN preserve the repeated circular
  holes and their spacing more clearly. Bicubic and NEDI blur or partially
  merge this pattern at the higher scales.
- The difference between methods is smaller at x2 and becomes easier to see at
  x3 and x4. No severe ringing or obviously invented structure was found in
  these selected crops, although the neural outputs have stronger edge contrast.

These visual observations support the complete PSNR/SSIM results: IMDN gives
the strongest reconstruction quality, FSRCNN is second, and NEDI does not show
a consistent visual advantage over Bicubic under this bicubic-degradation
protocol.

## FLOPs and model-complexity findings

FLOPs estimate how many arithmetic operations one model inference requires;
they are not a timing measurement. Both networks were profiled with a batch of
one and a fixed 256x256 LR input. THOP reported multiply-accumulates (MACs), and
the project reports two FLOPs per MAC.

| Model | Scale | GFLOPs | Parameters |
|---|---:|---:|---:|
| FSRCNN | x2 | 3.44 | 12,809 |
| FSRCNN | x3 | 6.41 | 12,809 |
| FSRCNN | x4 | 10.57 | 12,809 |
| IMDN | x2 | 90.37 | 694,404 |
| IMDN | x3 | 91.50 | 703,059 |
| IMDN | x4 | 93.09 | 715,176 |

IMDN requires about 26.3 times the FSRCNN FLOPs at x2, 14.3 times at x3, and
8.8 times at x4. IMDN also has about 54-56 times as many parameters. Therefore,
IMDN provides the best measured reconstruction quality, while FSRCNN is the
lighter neural model.

## Phase 4 conclusion

IMDN is selected as the best-performing deep-learning model for Phase 5's
fusion experiment because it ranked first in PSNR-Y and SSIM-Y in all 12
dataset-scale comparisons and also looked closest to the HR references in the
fixed visual sample. FSRCNN remains the preferred model when minimal neural
computation and model size are more important than maximum reconstruction
quality.

Latency is reported separately with its hardware context: Bicubic used Google
Colab's default CPU, NEDI used an AWS m7i.2xlarge CPU instance, and the neural
models used a Tesla T4 GPU. Those latency values are observed measurements, not
a same-hardware speed contest.

