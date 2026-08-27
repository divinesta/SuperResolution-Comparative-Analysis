# Super-Resolution Evaluation Protocol

This document records the fixed rules that must be applied to bicubic, NEDI,
FSRCNN, IMDN, and the later fusion method. Using the same rules makes the
comparison fair and reproducible.

## Experiment flow

For every dataset image and scale:

1. Load the original high-resolution (HR) image as RGB.
2. Load its matching prepared bicubic low-resolution (LR) image for the selected
   scale.
3. Verify that the HR and LR filenames match and that their dimensions have the
   correct scale relationship.
4. Reconstruct the prepared LR image to the original HR image's exact width and
   height. Do not crop the HR image to `LR size x scale`; the downloaded LR files
   were created from the complete original image, including dimensions that are
   not evenly divisible by the scale.
5. Give that same prepared LR input to every reconstruction method.
6. Compare each reconstructed image with the original HR image.
7. Save per-image measurements to CSV before creating summaries or report tables.

The required scales are x2, x3, and x4. The required test datasets are Set5,
Set14, BSD100, and Urban100.

## Quality measurements

Each reconstruction records both:

- Y-channel PSNR and SSIM as the primary thesis measurements.
- RGB PSNR and SSIM as additional measurements.

Before calculating either set of metrics, crop the same border from the HR and
reconstructed images:

- x2: 2 pixels from every edge.
- x3: 3 pixels from every edge.
- x4: 4 pixels from every edge.

The Y channel uses the BT.601 conversion implemented in
`app/evaluation/metrics.py`.

SSIM uses the standard super-resolution settings: an 11x11 Gaussian window,
sigma 1.5, and population covariance. These settings match the original SSIM
method and the evaluation implementation used by BasicSR.

## Runtime measurements

For the bicubic baseline, run three untimed warm-ups followed by ten timed runs
per image. Record the mean, median, standard deviation, minimum, and maximum
latency in milliseconds.

The main experiments run in Google Colab:

- Bicubic and NEDI run on the Colab CPU.
- FSRCNN and IMDN run on the Colab GPU.
- FSRCNN and IMDN are also timed on the Colab CPU for a direct CPU comparison
  with NEDI.
- GPU timings are reported separately from CPU timings.

Software versions and available hardware information must be stored with the
result records.

## Dataset locations

Datasets remain outside Git. The expected structure is:

```text
FYP_SR_Data/
├── Set5/
│   ├── Set5_HR/
│   ├── Set5_LR_x2/
│   ├── Set5_LR_x3/
│   └── Set5_LR_x4/
├── Set14/            # Same HR and LR folder pattern
├── BSD100/           # Same HR and LR folder pattern
└── Urban100/         # Same HR and LR folder pattern
```

The code chooses the data root in this order:

1. A path explicitly passed using `--data-root`.
2. The `FYP_SR_DATA_ROOT` environment variable.
3. `/content/drive/MyDrive/FYP_SR_Data` when that mounted Colab folder exists.
4. The repository's local `data/` folder.

Direct paired directories can instead be supplied together with `--hr-dir` and
`--lr-dir`.

## Result protection

The existing CSV files under `results/metrics/` are preliminary RGB baseline
results and must not be deleted. Final reruns should use clearly different
filenames. The evaluator refuses to replace an existing CSV unless overwrite is
explicitly requested.

The prepared LR images remain unchanged in Google Drive. Reconstructed images
and full experiment outputs also remain in Google Drive. Final CSV files and
selected report images can then be copied into the repository.
