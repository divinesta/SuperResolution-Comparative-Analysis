# Phase 5 fusion findings

**Completed:** 6 September 2026

Three simple methods were tested for combining NEDI and IMDN. Evaluation
used five DIV2K validation images at x2, x3, and x4, giving 15 validation cases.
The final test datasets were not used to choose any settings.

## Method 1: whole-image weighted fusion

Every pixel in the NEDI and IMDN outputs was combined using fixed percentages.
Eleven settings were tested, from 100% NEDI to 100% IMDN.

The best setting was **100% IMDN and 0% NEDI**, with mean PSNR-Y of 36.7041 dB
and SSIM-Y of 0.917439. Image quality improved steadily as the NEDI percentage
was reduced. Therefore, whole-image weighted fusion did not improve IMDN.

## Method 2: edge-aware fusion

IMDN was kept unchanged outside strong edges. At detected edges, NEDI
contributions of 5%, 10%, 20%, and 30% were tested using three edge-strength
cut-offs. Edges were detected from the LR input rather than the reference HR
image.

IMDN alone remained the best overall setting. The strongest actual mixture
used 5% NEDI only around the strongest 5% of edges. It achieved mean PSNR-Y of
36.6910 dB, which was 0.0131 dB below IMDN. It improved 2 of the 15 validation
cases but reduced PSNR-Y in the other 13.

## Method 3: back-projection fusion

The NEDI and IMDN reconstructions were shrunk back to the original LR size.
For each image, the fusion percentage was calculated automatically to make the
combined result reproduce the observed LR image as closely as possible. The HR
reference was not used to calculate the percentage.

The method assigned an average weight of 99.9618% to IMDN and 0.0382% to NEDI.
At x3 and x4 it selected 100% IMDN for every image. After pixel rounding, all
15 fused images were identical to the IMDN outputs, producing no PSNR-Y or
SSIM-Y improvement.

## Conclusion

None of the three fusion methods improved IMDN overall. Whole-image weighting
reduced quality, edge-aware fusion reduced average quality, and back-projection
effectively rejected NEDI. Under this project's bicubic degradation and
evaluation protocol, NEDI did not provide complementary information that
improved the stronger IMDN reconstruction. This is retained as a valid negative
result, and no additional fusion settings will be selected using the final test
datasets.
