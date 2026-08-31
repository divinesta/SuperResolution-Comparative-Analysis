# Why NEDI scores lower than bicubic in this project

**Written:** 31 August 2026
**Applies to:** the completed NEDI x2 run (`results/nedi_x2_summary_final.csv`) and the x3/x4 runs that follow.

---

## 1. The question this note answers

After the x2 run finished, NEDI scored *below* bicubic on three of the four datasets. Bicubic is supposed to be the baseline — the simple method that everything else beats. So the worry was: **is the NEDI implementation broken?**

**Answer: no. The implementation is correct. The result is the one the research literature reports.** This note records the evidence, so the reasoning is not lost.

---

## 2. The numbers

NEDI x2 against the bicubic baseline, PSNR-Y (higher is better):

| Dataset | Bicubic | NEDI | Difference |
|---|---:|---:|---:|
| Set5 | 33.673 | 33.156 | **−0.517 dB** |
| Set14 | 30.346 | 29.875 | **−0.471 dB** |
| BSD100 | 29.598 | 29.046 | **−0.553 dB** |
| Urban100 | 26.876 | 26.918 | **+0.043 dB** |

Bicubic also has slightly higher SSIM on all four datasets. NEDI is roughly **2,000 to 2,800 times slower** than bicubic (for example Set5: 5,750 ms vs 2.06 ms per image).

Urban100 is the only dataset where NEDI wins. Urban100 is full of buildings, windows and railings — long, clean, straight edges. That is exactly the content NEDI was designed for. This is a meaningful detail, not noise.

---

## 3. First check: is the measurement itself trustworthy?

Yes. The bicubic baseline reproduces published numbers almost exactly:

| Dataset x2 | Our bicubic | Published bicubic |
|---|---:|---:|
| Set5 | 33.673 | 33.66 |
| Set14 | 30.346 | 30.24 |
| BSD100 | 29.598 | 29.56 |
| Urban100 | 26.876 | 26.88 |

If the PSNR calculation, the colour conversion, the border cropping or the image pairing were wrong, bicubic would not match the literature this closely. **So the measuring instrument is fine, and any gap NEDI shows is a real gap.**

---

## 4. The main finding: NEDI losing to bicubic on PSNR is normal

This is documented in a peer-reviewed paper. Asuni & Giachetti, *Accuracy Improvements and Artifacts Removal in Edge Based Image Interpolation* (VISAPP 2008 — the paper that introduced "iNEDI") tested **Xin Li's own original NEDI code** against MATLAB bicubic on 9 natural images:

| | NEDI | Bicubic | Difference |
|---|---:|---:|---:|
| x2 average | 32.76 | 33.22 | **−0.46 dB** |
| x4 average | 25.99 | 26.33 | −0.34 dB |

NEDI won on only 3 of the 9 images at x2.

Our average deficit is about **−0.5 dB**. Theirs is **−0.46 dB**. We reproduced the reference implementation's behaviour.

The paper's authors were surprised too. They write that this "may appear surprising", and stress that the comparison was done carefully using the original author's code. Their explanation: NEDI removes staircase-looking jagged edges, which *looks* better, but it introduces its own artifacts — directional streaks and a smeared "oil painting" look in textured areas. Those artifacts cost more error than the improved edges save.

**Key idea to hold on to: NEDI trades measured error (PSNR) for visual edge quality. The papers that praise NEDI are mostly praising how it looks, not what it scores.**

---

## 5. Why the gap appears: the degradation mismatch

This is the most important technical point in this note.

**How the low-resolution image is made changes who wins.**

NEDI's maths assumes the small image was made by **decimation** — literally throwing away every other pixel. Keep pixel 1, drop pixel 2, keep pixel 3, and so on. Under that assumption, every small-image pixel sits exactly on top of a large-image pixel.

Our benchmark low-resolution images are made the standard super-resolution way: **bicubic downsampling with anti-aliasing**. That first blurs the image slightly (to avoid aliasing) and then resamples it onto a grid that sits half a pixel away.

Two consequences:

1. NEDI's core assumption is violated, so its edge predictions get worse.
2. Bicubic *upsampling* is very nearly the exact mathematical reverse of bicubic *downsampling*. So the baseline is unusually strong here — it is being asked to undo the very operation it is the inverse of.

**I proved this is the cause.** Running our unmodified `nedi_upsample_x2_luminance` on images made by decimation instead:

| Image | Bicubic | NEDI | Difference |
|---|---:|---:|---:|
| astronaut | 30.037 | 30.533 | **+0.496 dB** |
| camera | 28.995 | 29.408 | **+0.412 dB** |
| chelsea | 33.280 | 33.499 | **+0.219 dB** |
| coffee | 28.950 | 30.015 | **+1.065 dB** |

Under the degradation NEDI was designed for, **our code beats bicubic by +0.2 to +1.1 dB** — matching the gains Li & Orchard reported in the original 2001 paper.

**Same code. Different low-resolution image. Opposite conclusion.** That is the whole story.

---

## 6. Things that were checked and found correct

These were the suspected bugs. All were investigated and cleared.

### 6.1 The half-pixel alignment (checked — costs about 0.03 dB)

NEDI naturally builds its output on a grid where original pixels sit at positions 0, 2, 4, 6… That grid is half a pixel away from the grid the HR reference image uses. `_shift_half_pixel` in `app/traditional/nedi.py` corrects this.

The maths was verified by hand and is right. The cost was then measured directly, by pushing a plain cubic interpolation through the same detour:

| Image | Direct bicubic | Via NEDI grid + shift | Penalty |
|---|---:|---:|---:|
| astronaut | 30.374 | 30.333 | 0.041 dB |
| camera | 29.874 | 29.851 | 0.023 dB |
| chelsea | 33.885 | 33.852 | 0.032 dB |
| coffee | 29.262 | 29.219 | 0.043 dB |

**About 0.03 dB.** The alignment is essentially free and is not the cause of the 0.5 dB gap.

(The iNEDI paper flags this same trap — it warns that an uncorrected half-pixel shift "may compromise the correctness of the comparison". We handled it by shifting the reconstruction; they handled it by shifting the reference. Both are valid.)

### 6.2 The neighbour ordering in both NEDI stages (checked — correct)

NEDI works in two stages. Stage 2 operates in a lattice rotated 45 degrees, which is very easy to get wrong. Both stages were traced through by hand:

- Stage 1 training predictors are ordered up-left, up-right, down-left, down-right, and the four target neighbours are in the same order. Correct.
- Stage 2's training predictors at ±2 pixels map to the four diagonal neighbours in the rotated lattice; the target's four neighbours map to the same corner pattern. Correct.

### 6.3 Numerical stability (checked — no failures)

`nedi_numerical_fallback_count` is **0 across all 219 images**. The least-squares solve never blew up or produced non-finite weights.

---

## 7. Known deviations from the iNEDI paper (kept deliberately)

The iNEDI paper lists improvements to NEDI. Three of them are things our implementation does not do. **We measured all three and decided not to apply them.**

| # | Deviation | Where | Measured gain |
|---|---|---|---|
| 1 | Smooth areas fall back to **bilinear**, but the paper (§4.2) uses **bicubic** | `_bilinear_grid` | +0.02 dB |
| 2 | Predictions are clipped to 0–255 only. The paper (§4.3) also clamps them to the min/max of the four neighbours, so a wild prediction cannot survive | `_interpolate_candidate` | −0.01 dB |
| 3 | `np.linalg.lstsq(..., rcond=None)` does not give the minimum-norm solution the paper (§4.4) asks for, because the matrix is almost always rank-deficient for a straight edge | `_solve_weights` | 0.00 dB |

Test result on the astronaut image:

```
bicubic baseline                          31.706
NEDI as implemented                       31.538
+ clamp to neighbour range      (§4.3)    31.526
+ bicubic fallback grid         (§4.2)    31.559
+ both                                    31.548
+ both + min-norm lstsq rcond=1e-3 (§4.4) 31.548
```

**Decision: leave the implementation frozen.** The gains are inside the rounding noise, and changing the algorithm now would force a re-run of x2 as well, just to keep x2, x3 and x4 consistent with each other. These three points are written up as documented limitations instead.

---

## 8. The "HR image is not modcropped" question (checked — leave it alone)

BSD100 HR images are stored as 321×481 while the x2 LR is 160×240. 160 × 2 = 320, not 321. So the HR is one pixel larger than an exact 2x match, and the reconstruction gets resized once at the end to fit. 103 of the 219 x2 NEDI images have `nedi_dimension_adjustment = True`.

This *looks* like a bug. **It is not a problem, and it must not be changed**, because every method in the project uses the identical convention:

```
imdn_all_gpu.csv    dimension_policy: native_model_grid_then_full_target_size_adjustment  (423/657 adjusted)
fsrcnn_all_gpu.csv  dimension_policy: native_model_grid_then_full_target_size_adjustment  (423/657 adjusted)
nedi_x2_...csv      dimension_policy: native_nedi_grid_then_full_target_size_adjustment   (103/219 adjusted)
```

Since IMDN, FSRCNN, bicubic and NEDI are all treated the same way, the comparison between them is fair. And the effect is tiny — our bicubic still matches published numbers at every scale (BSD100 x3: 27.200 vs 27.21; x4: 25.956 vs 25.96).

Changing it now would invalidate the already-completed IMDN and FSRCNN x3/x4 runs.

Also note this rules out the dimension adjustment as the cause of the gap: **Set5 needs no adjustment at all (0 of 5 images) and still shows a −0.517 dB deficit.**

---

## 9. What to expect at x3 and x4

NEDI only knows how to double an image. The higher scales are built by cascading:

- **x3** — `LR_x3` → one native NEDI x2 → one bicubic resize to the exact HR size.
- **x4** — `LR_x4` → native NEDI x2 → native NEDI x2 → resize only if needed.

Piloted on test images before running the real thing:

| Image | Scale | Bicubic | NEDI | Difference |
|---|---|---:|---:|---:|
| astronaut | x3 | 28.677 | 28.412 | −0.265 dB |
| astronaut | x4 | 26.838 | 26.440 | −0.398 dB |
| camera | x3 | 28.991 | 28.902 | −0.089 dB |
| camera | x4 | 27.489 | 27.431 | −0.058 dB |
| coffee | x3 | 28.406 | 28.489 | +0.083 dB |
| coffee | x4 | 27.291 | 27.220 | −0.070 dB |

**Expect NEDI to sit roughly 0.1–0.5 dB below bicubic again.** The iNEDI paper's own x4 figure is −0.34 dB. This is normal. Do not treat it as a failed run.

Note also that at x4 the second NEDI pass runs on NEDI's *own output*, not on a real photograph — so its assumptions are stretched even further. A slightly larger gap at x4 than x3 is expected.

---

## 10. How to write this up

Do not present this as a disappointing result. Present it as the finding.

> Classical edge-directed interpolation's advantage over bicubic depends on the degradation model. Under decimation — the degradation NEDI was derived for — NEDI gains +0.2 to +1.1 dB. Under the anti-aliased bicubic degradation used by every modern super-resolution benchmark, that advantage disappears and reverses to roughly −0.5 dB, because bicubic upsampling is the near-exact inverse of the degradation being undone.

This is stronger than simply reproducing a 2001 result, and it gives a principled reason why the learned models (FSRCNN, IMDN) pull ahead: they are *trained* on the bicubic degradation, so they model it rather than assuming a different one.

Supporting points worth including:

- NEDI's only win is Urban100 (+0.043 dB), the dataset dominated by strong building edges — consistent with what NEDI is designed to do.
- NEDI's SSIM is lower than bicubic everywhere, consistent with the artifacts described in the iNEDI paper.
- NEDI costs 2,000–2,800× bicubic's runtime for a net PSNR loss on this protocol — a strong efficiency argument.

---

## 11. Sources

- N. Asuni, A. Giachetti, *Accuracy Improvements and Artifacts Removal in Edge Based Image Interpolation*, VISAPP 2008 — https://www.scitepress.org/Papers/2008/10741/10741.pdf
  (The key source. Tables 1 and 2 hold the NEDI-vs-bicubic numbers; sections 4.2, 4.3 and 4.4 hold the three improvements listed in section 7 above.)
- X. Li, M. T. Orchard, *New Edge-Directed Interpolation*, IEEE TIP 2001 — https://www.researchgate.net/publication/3327439_New_edge-directed_interpolation
  (The original method. Its reported gains assume decimation-based downsampling.)
- W.-S. Tam, C.-W. Kok, *Modified Edge-Directed Interpolation for Images* — https://ira.lib.polyu.edu.hk/bitstream/10397/4341/1/Tam_Modified_edge-directed.pdf
  (Confirms NEDI's results are highly sensitive to window size and image-dependent parameters.)
- *Performance Evaluation of Edge-Directed Interpolation Methods for Images* — https://arxiv.org/abs/1303.6455

---

## 12. One-paragraph summary

The NEDI implementation is correct. It beats bicubic by +0.2 to +1.1 dB when the low-resolution image is made by decimation, which is what NEDI's maths assumes. It loses to bicubic by about 0.5 dB when the low-resolution image is made by anti-aliased bicubic downsampling, which is what the standard benchmarks use — and the iNEDI paper measured that same −0.46 dB deficit using the original author's own code. The bicubic baseline matches published numbers, the half-pixel alignment costs only 0.03 dB, the neighbour ordering is right, and nothing blew up numerically. The result is real, expected, and worth reporting as a finding about degradation models rather than treated as a bug.
