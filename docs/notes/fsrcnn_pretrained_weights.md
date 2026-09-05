# FSRCNN pretrained weights

**Written:** 5 September 2026  
**Evaluation notebook:** `notebooks/10_fsrcnn_colab_evaluation.ipynb`

## Decision

The project uses existing pretrained FSRCNN weights for x2, x3, and x4. The
weights were trained on the 91-image training dataset; this provenance is
recorded in the detailed FSRCNN evaluation results.

The completed FSRCNN experiment is an inference evaluation of those pretrained
weights on Set5, Set14, BSD100, and Urban100. The project will not train FSRCNN
again or repeat the completed evaluation solely because the weights were
pretrained.

## Reporting rule

The methodology and final report must state clearly that:

- FSRCNN used pretrained weights rather than weights trained within this project.
- The pretrained weights came from the 91-image training dataset.
- The reported FSRCNN results were produced using those same verified weights.

