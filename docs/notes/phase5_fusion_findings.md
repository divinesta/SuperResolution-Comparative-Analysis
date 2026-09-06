# Phase 5 fusion findings

**Status:** Weighted and edge-aware experiments completed 6 September 2026

Two simple methods were tested for combining NEDI and IMDN. Weight selection
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

## Conclusion

Neither of these two fusion methods improved IMDN overall. Under this project's bicubic
degradation and evaluation protocol, NEDI did not provide complementary detail
that improved the stronger IMDN reconstruction. A final back-projection fusion
will be evaluated separately using the same validation cases. No fusion
settings will be selected using the final test datasets.
