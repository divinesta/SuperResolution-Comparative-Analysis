# Phase 3: FSRCNN and IMDN

This phase may run alongside the remaining Phase 2 NEDI experiment. Phase 3
reuses the prepared LR images and shared metrics, but it does not consume NEDI
outputs. NEDI results are only required later, when the controlled comparison
tables are assembled.

## First milestone: pretrained inference

Training is intentionally not the first task. The first milestone is to prove
that each model can load one scale-specific checkpoint, reconstruct one prepared
LR image, and produce a record through the shared evaluation protocol. This
separates pipeline errors from the much slower training process.

The order is:

1. Validate the Colab GPU and PyTorch environment.
2. Fix model, colour, scale, and checkpoint conventions.
3. Implement and test FSRCNN and IMDN forward inference.
4. Load provenance-recorded pretrained weights for x2, x3, and x4.
5. Run one Set5 image as a smoke test before any dataset-wide run.
6. Add GPU and CPU timing, parameter count, memory, and FLOPs records.
7. Decide whether fine-tuning on DIV2K is necessary from the pilot evidence.

## Fixed model conventions

- Every model has a separate checkpoint for x2, x3, and x4.
- Checkpoints live outside Git under
  `FYP_SR_Data/checkpoints/<model>/<model>_x<scale>.pth`.
- FSRCNN follows its original luminance-only policy: the network reconstructs Y,
  while Cb and Cr are enlarged with bicubic interpolation before RGB assembly.
- IMDN reconstructs RGB directly, matching its official implementation.
- The initial FSRCNN checkpoints are the published yjn870 FSRCNN(56,12,4)
  x2/x3/x4 weights trained on the 91-image dataset. Results must retain this
  training-dataset label; they are pilot evidence for deciding whether the
  project needs a controlled DIV2K fine-tune.
- Both models receive the same prepared LR files used by bicubic and NEDI.
- Both are evaluated against the same uncropped HR file, with the scale-sized
  metric border defined in `docs/EVALUATION_PROTOCOL.md`.
- A native neural output that is smaller than the uncropped HR dimensions is
  bicubically adjusted once to the exact target size, with the native size and
  adjustment flag recorded in the result.

## Checkpoint provenance

No downloaded weight is accepted merely because its filename matches. For each
checkpoint, record its source URL, original filename, SHA-256 checksum, model
variant, scale, training dataset, degradation method, and license. A checkpoint
whose architecture or degradation cannot be established must not be used in the
final comparison.

## Parallel-work boundary

Safe while NEDI Step 6 runs:

- Colab/GPU setup
- architecture and checkpoint-loader implementation
- unit and shape tests
- single-image pretrained inference
- parameter/FLOPs tooling

Wait for the final controlled-comparison phase:

- ranking all methods
- selecting the best deep model
- drawing final thesis conclusions
- choosing the model used by the later fusion phase
