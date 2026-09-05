# NEDI scaling strategy for x2, x3, and x4

**Written:** 5 September 2026  
**Final results:** `results/metrics/final/nedi_summary_final.csv`

## Why the scales are handled differently

The original NEDI algorithm performs one operation: it doubles an image's
width and height. In other words, its native scale factor is x2. The project
also evaluates x3 and x4, so those scales must be constructed from the native
x2 operation.

## Strategy used at each scale

### x2: native NEDI

```text
LR_x2 -> one NEDI x2 pass -> HR-sized reconstruction
```

This is a direct use of NEDI. No additional scaling method is needed unless a
small final dimension adjustment is required to match the stored HR image.

### x3: hybrid NEDI and bicubic

```text
LR_x3 -> one NEDI x2 pass -> bicubic resize to the exact HR size
```

NEDI has no native x3 operation. One NEDI pass first enlarges the LR image by
x2, and bicubic performs the remaining resize to the required x3 target. The
x3 result must therefore be described as **hybrid NEDI-bicubic**, not as native
x3 NEDI.

### x4: two native NEDI passes

```text
LR_x4 -> NEDI x2 -> NEDI x2 -> HR-sized reconstruction
```

Applying x2 twice gives x4 because `2 x 2 = 4`. This is a cascaded NEDI result:
the second pass processes the output produced by the first pass. A final size
adjustment is used only when the stored HR dimensions are not an exact x4
match.

## Reporting rule

The three scales can be presented together in the final comparison table, but
their implementation must be stated accurately:

| Evaluation scale | Description to report |
|---|---|
| x2 | One native NEDI x2 pass |
| x3 | One native NEDI x2 pass followed by bicubic resizing |
| x4 | Two consecutive native NEDI x2 passes |

These strategies were fixed before the full runs and are the strategies used
to produce the final NEDI results. They do not indicate an error in the x3 or
x4 evaluation; they are the documented way this project extends a
doubling-only algorithm to the required scales.
