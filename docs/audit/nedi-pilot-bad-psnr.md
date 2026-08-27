# Audit: NEDI pilot looked much worse than bicubic

**Date:** 2026-08-27  
**Image:** Set5 `baby.png`, scale x2  
**Code:** `app/traditional/nedi.py` (the Colab notebook only calls this code)

## What we saw

The NEDI Colab pilot kept scoring far below bicubic. Three runs, three commits, same picture:

| Run | What we changed | NEDI PSNR | vs bicubic |
|---|---|---|---|
| 1 | First full NEDI pipeline | 34.30 | −2.77 dB |
| 2 | Rewrote NEDI stage 2 | 34.48 | −2.60 dB |
| 3 | Stopped stretching 511→512; pasted instead | 33.20 | −3.88 dB |

Bicubic stayed at **37.08 dB**. So bicubic was fine. NEDI was the problem.

The notebook was not the bug. All three scores came from the same evaluator calling `nedi_upsample_x2_rgb`.

## Simple picture of the mistake

Think of the HR image as a 512×512 sheet of graph paper.

Bicubic draws the big image on that same sheet, on the same squares.

Old NEDI did two wrong things:

1. It drew on a **511×511** sheet, then forced that onto 512×512. One row and one column were missing.
2. Even after making the size 512×512, NEDI’s pixels were sitting **half a square away** from bicubic’s pixels.

If two pictures are almost the same, but one is slid by half a pixel, PSNR thinks they are very different. That is not “NEDI failed at edges”. That is “we compared the wrong grid”.

A local check proved this:

- Put LR pixels on NEDI’s old grid, fill the rest with simple bilinear: **33.25 dB**
- Slide that same image by half a pixel: **36.17 dB**

Same pixels. Only the position changed. About **3 dB** came back.

## What the earlier commits did *not* fix

- Run 2 changed how stage 2 estimates neighbours. That was a small internal tweak. It did not move the image onto bicubic’s grid.
- Run 3 stopped stretching 511→512 and pasted the 511 image into the top-left of 512. That made the size mismatch more honest, and the score got **worse**.

We never built the grid the original NEDI paper uses, and we never lined it up with our bicubic test images.

## What the paper actually does

Li and Orchard (2001) double an image like this:

- LR size `256` → NEDI size `512` (not 511)
- Copy each LR pixel onto every other HR pixel
- Fill the gaps with NEDI (edges) or bilinear (smooth areas)

Our bicubic LR images were made with bicubic resize, not by “keep every other pixel”. So NEDI’s copied pixels do not sit on the same spots as the bicubic baseline. After NEDI finishes, we now slide the result by **half a pixel** so both methods are compared on the same grid.

## The fix

In `app/traditional/nedi.py`:

1. Output size is now **exactly 2× LR** (256→512).
2. After NEDI, shift luminance by **+0.5 pixel** before scoring.
3. Near the image border, skip missing training samples instead of giving up on the whole window.
4. If the local maths is “rank-deficient” (common on a real edge), still use the NEDI weights. That is how NEDI follows an edge.

The native NEDI pass still keeps the original LR values on even pixels. The half-pixel slide happens only when we build the RGB image used for scoring.

## Numbers after the fix (local, same baby x2)

| Method | PSNR (Y) | SSIM (Y) |
|---|---|---|
| Bicubic | 37.08 | 0.952 |
| NEDI (fixed) | 36.12 | 0.942 |
| Difference | **−0.96 dB** | −0.010 |

Output size was **512×512**. No extra resize.

That ~1 dB gap is normal for original NEDI on bicubic-downsampled photos. Papers often show NEDI a bit below bicubic on PSNR, even when edges look sharper. A 3 dB hole was the grid bug. A 1 dB hole is the method.

## What to do on the next Colab pilot

Pull the latest code and re-run `notebooks/04_nedi_colab_pilot.ipynb`.

The notebook now also prints how many pixels used NEDI vs bilinear, and whether the size was adjusted. Check those lines. For baby x2 you want native size **512×512** and `adjusted=False`.

Do not treat “NEDI a bit below bicubic on PSNR” as another code failure.
