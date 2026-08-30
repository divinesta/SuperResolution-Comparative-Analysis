# NEDI Implementation Design

This document fixes the design of the New Edge-Directed Interpolation (NEDI)
phase before implementation. The implementation will follow Xin Li and Michael
Orchard's original NEDI method and will reuse the project's existing evaluation
infrastructure.

## Purpose

NEDI receives the same prepared bicubic low-resolution (LR) images used by the
bicubic baseline and reconstructs them at the matching high-resolution (HR)
size. The original HR image is used only as the reference for evaluation; it is
never available to the reconstruction algorithm.

The NEDI results will be compared with bicubic using the fixed rules in
`docs/EVALUATION_PROTOCOL.md`.

## Reference Algorithm

The primary source is:

- Xin Li and Michael T. Orchard, "New Edge-Directed Interpolation," *IEEE
  Transactions on Image Processing*, vol. 10, no. 10, pp. 1521-1527, 2001.
  DOI: <https://doi.org/10.1109/83.951537>

The original algorithm estimates local covariance information from the LR
image. For each missing HR pixel near an edge, it uses that covariance to find
four interpolation weights. The missing value is predicted from its four
nearest known neighbours in the relevant diagonal or axial arrangement.

For a local observation vector `y` and a local data matrix `C`, the four weights
`alpha` solve the least-squares problem:

```text
alpha = argmin ||C alpha - y||²
```

The implementation will solve this least-squares system directly rather than
forming a matrix inverse. This is mathematically equivalent for a full-rank
system and is more numerically stable.

## NEDI x2 Core

One native x2 pass will work in two interpolation stages:

1. Place the original LR samples at their corresponding locations in the x2
   output grid and estimate the missing diagonal/interlacing pixels.
2. Rotate the resulting checkerboard lattice by 45 degrees in coordinate space,
   then repeat the same covariance procedure to estimate the remaining
   horizontal and vertical pixels.

The default local covariance window will be 8x8 and the default edge activity
threshold will be 8, matching the settings reported in the original paper.

The hybrid switching rule from the paper will be retained:

- Use covariance-based NEDI interpolation for detected edge pixels.
- Use bilinear interpolation for smooth pixels.

## Colour Handling

NEDI will operate on image luminance rather than independently processing the
red, green, and blue channels:

1. Convert the LR RGB image to YCbCr.
2. Apply NEDI to the Y (luminance/brightness) channel.
3. Upsample the Cb and Cr colour channels using bicubic interpolation.
4. Combine the reconstructed Y, Cb, and Cr channels and convert back to RGB.

This gives NEDI control over the edges and visible structure while avoiding
independent colour-channel interpolation that can create colour misalignment.
The final RGB image will still be evaluated with both Y-channel and RGB metrics.

## Scaling Policy

The original NEDI method natively supports magnification factors that are
powers of two. The project requires x2, x3, and x4, so the following policy will
be recorded in every result:

- **x2:** one native NEDI x2 pass.
- **x4:** two consecutive native NEDI x2 passes.
- **x3:** one native NEDI x2 pass followed by bicubic resizing to the exact x3
  target size. This is a documented hybrid scale procedure, not native x3 NEDI.

The native x2 insertion grid is an exact `2 x LR` lattice, matching Li and
Orchard and Xin Li's original MATLAB: original samples occupy even
coordinates, so `256x256` becomes `512x512`. Missing pixels in the last odd
row and column have no further original neighbour and are filled from the
available side.

That even-insertion lattice is half a pixel away from the sampling grid used
by bicubic-prepared HR/LR pairs and by this project's bicubic baseline
(Pillow resize). After the native NEDI pass, luminance is therefore shifted
by +0.5 pixels with a Keys cubic kernel before it is compared with the HR
reference. This is an evaluation-alignment step, not a change to the
covariance interpolator. Without it, even a correct NEDI reconstruction
measures several dB below bicubic solely from the grid offset.

If a prepared pair is larger than the native `2 x LR` size, the complete NEDI
result is bicubically resized to the exact HR size. This keeps all NEDI pixels
on the same sampling grid as the HR reference; pasting the native result into
only the top-left of a larger image would misalign the image. The result record
states whether a size adjustment occurred and records `nedi_sampling_grid`.

## Boundary and Numerical Fallbacks

Bilinear interpolation will be used when:

- a pixel is in a smooth region;
- fewer than four usable covariance samples are available near an image
  boundary;
- the least-squares system produces a non-finite result.

Rank-deficient local systems are still solved. On a true step edge the
4-neighbour data matrix is often rank 1 or 2; that degeneracy is how NEDI
learns to interpolate along the edge. Discarding those systems sends the
algorithm to bilinear exactly where it is supposed to be adaptive.

Original LR sample values must remain unchanged at their corresponding
even coordinates during each native x2 pass. The half-pixel alignment shift
is applied afterwards, only when an RGB result is assembled for evaluation.
Reconstructed luminance and colour values will be clipped to the valid 0-255
image range before conversion to an 8-bit RGB image.

These fallbacks must be counted and written to the result records so that the
amount of adaptive NEDI processing remains visible.

## Code Structure

The NEDI phase will add:

```text
app/traditional/nedi.py       # NEDI reconstruction algorithm only
app/evaluation/nedi.py        # Dataset evaluation and command-line interface
tests/test_nedi.py            # Focused algorithm and integration tests
notebooks/04_nedi_colab.ipynb # Colab pilot and full-run workflow
```

Shared behaviour will continue to come from the existing modules:

- image loading and HR/LR validation from `app/evaluation/images.py`;
- Y and RGB PSNR/SSIM from `app/evaluation/metrics.py`;
- warm-ups and repeated timing from `app/evaluation/timing.py`;
- CSV protection and reporting from the existing evaluation framework.

## NEDI Configuration and Result Fields

The evaluator will record at least:

```text
nedi_window_size
nedi_edge_threshold
nedi_scale_strategy
nedi_native_passes
nedi_edge_pixel_count
nedi_pixel_count
nedi_bilinear_fallback_count
nedi_numerical_fallback_count
nedi_dimension_adjustment
nedi_sampling_grid
```

All existing quality, timing, dimension, environment, and dataset fields from
the bicubic evaluation will also be retained.

## Validation Before a Full Run

The implementation must pass these checks:

1. A constant image remains constant after interpolation.
2. Output dimensions and RGB mode match the requested target.
3. Output values are finite and remain within 0-255.
4. Original samples are preserved during a native x2 pass.
5. Horizontal, vertical, and diagonal synthetic edges produce valid outputs.
6. Singular or insufficient neighbourhoods use the documented fallback instead
   of crashing.
7. A Set5 x2 pilot completes successfully in Colab before any full dataset run.

The pilot will also estimate NEDI's runtime. The full run will retain the fixed
three warm-ups and ten timed runs unless the pilot demonstrates that this is
impractical; any change must be documented before the full evaluation and must
not be made silently.

On bicubic-prepared natural images, original NEDI is typically a little below
bicubic on PSNR (often around 0.5-1 dB) while looking sharper on edges. A gap
of several dB is an implementation or alignment error, not the expected
behaviour of the paper.

## Output Protection

Bicubic results will not be changed or overwritten. NEDI outputs will use
separate run directories and filenames, including:

```text
results/metrics/final/nedi_summary_final.csv
```

Detailed per-image CSV files and reconstructed images will remain in the
timestamped Google Drive run directory. The final summary and selected visual
comparisons can then be copied into the repository.
