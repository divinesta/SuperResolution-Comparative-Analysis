# Phase 6 XAI findings

**Completed:** 7 September 2026  
**Final notebook:** `notebooks/18_phase6_xai_imdn_final_colab.ipynb`  
**Metrics:** `results/metrics/final/xai/phase6_xai_final_summary.csv`  
**Settings:** `results/metrics/final/xai/phase6_xai_final_settings.json`  
**Figures:** `results/figures/xai/`

## What was explained

Phase 6 explains IMDN, the best-performing deep-learning method from the final
comparison. The fusion experiments in Phase 5 did not improve IMDN, so the XAI
stage focuses on IMDN alone.

IMDN outputs a reconstructed image rather than a class label. Therefore, the
explanation target was defined as the PSNR-Y of a selected 96x96 HR region.
Input regions in the LR image were softened, IMDN was run again, and the change
in the selected region's PSNR-Y was measured.

In simple terms, this phase asks:

```text
Which parts of the LR input matter most for reconstructing this selected HR
region?
```

## Final settings

Four representative cases were selected from the Phase 4 visual examples:

| Dataset | Image | Scale | Region type |
|---|---|---:|---|
| Set5 | baby.png | x2 | smooth face and gradual colour regions |
| Set5 | butterfly.png | x3 | fine lines and high-frequency texture |
| Set14 | baboon.png | x4 | difficult natural texture |
| Urban100 | img_004.png | x3 | architectural edges and repetitive structure |

Each LR image was divided into an 8x8 grid, giving 64 input segments. Both LIME
and SHAP used 96 samples per case. Hidden LR segments were replaced with a
Gaussian-blurred version of the LR image, using radius 3.

## Quantitative result

Across the four final cases, hiding LR input segments reduced the selected
region quality by an average of **9.7068 dB PSNR-Y**. This confirms that the
perturbation-based explanations were affecting IMDN's output rather than
producing empty heatmaps.

| Case | Original target PSNR-Y | Mean PSNR-Y drop |
|---|---:|---:|
| Set5 baby.png x2 | 46.2768 | 13.5840 |
| Set5 butterfly.png x3 | 31.7918 | 14.4204 |
| Set14 baboon.png x4 | 29.6276 | 4.4257 |
| Urban100 img_004.png x3 | 24.5294 | 6.3972 |

The total recorded runtime for the final XAI run was about **35.0 seconds** on
the Colab GPU environment. LIME took about 18.1 seconds in total, while SHAP
took about 15.5 seconds in total.

## Visual interpretation

On `baby.png` at x2, both LIME and SHAP concentrated importance around the
selected nose region. This indicates that IMDN relied most strongly on nearby
facial structure when reconstructing that smooth local region.

On `butterfly.png` at x3, the importance maps were less clean than `baby.png`,
but still highlighted wing lines and nearby high-frequency texture. This
supports the Phase 4 observation that butterfly wing detail is highly sensitive
to the available LR structure.

On `baboon.png` at x4, the heatmap remained close to the target area but was
less sharply concentrated. This is treated as a limitation case: difficult
natural texture at high scale is harder for both IMDN and perturbation-based XAI
to explain cleanly.

On Urban100 `img_004.png` at x3, LIME and SHAP both highlighted repeated
architectural structures near the selected target region. This supports the
interpretation that IMDN uses nearby repeated patterns and edge structure when
reconstructing regular man-made textures.

## Main finding

The final explanation maps suggest that IMDN relies mostly on local and nearby
LR image structure, especially edges, repeated patterns, and visible texture
around the target output region.

This should not be described as the model understanding objects. A safer and
more accurate statement is that IMDN's reconstruction quality is strongly
affected by nearby structural information in the LR input.

## Limitations

The explanations are local to the selected regions and selected images. They do
not prove how IMDN behaves on every possible input. The heatmaps also depend on
the perturbation rule, grid size, number of samples, and selected target region.

The XAI results should therefore be reported as supporting visual evidence for
IMDN's reconstruction behaviour, not as a complete proof of the model's internal
reasoning.

## Conclusion

Phase 6 produced reproducible LIME and SHAP explanation maps for selected IMDN
super-resolution examples. The clearest evidence came from smooth facial detail
and repeated architectural structure, where both explanation methods focused on
meaningful nearby regions. The natural-texture example was less clean and is
reported as a limitation.
