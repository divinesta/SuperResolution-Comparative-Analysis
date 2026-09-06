# Project Execution Plan

## Project

**Title:** Comparative Analysis of Edge-Directed Interpolation and Lightweight Deep Learning Models for Image Super-Resolution

This plan converts the approved Chapters 1 and 2 scope into implementation, experimentation, evaluation, and reporting tasks. The Word document containing Chapters 1 and 2 is the academic source of truth. The roadmap file provides supporting project-management guidance.

## Fixed Scope

The project will compare:

- New Edge-Directed Interpolation (NEDI)
- Fast Super-Resolution Convolutional Neural Network (FSRCNN)
- Information Multi-Distillation Network (IMDN)
- A simple fusion of NEDI and the best-performing deep learning model

The deep-learning evaluation uses verified pretrained weights at x2, x3, and
x4. IMDN's weights were trained on DIV2K, while FSRCNN's weights were trained
on the 91-image dataset. Both models are tested on Set5, Set14, BSD100, and
Urban100, and their training provenance is reported with the results.

The evaluation will cover:

- Reconstruction quality: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM)
- Efficiency: inference latency, memory usage, parameter count, and floating-point operations (FLOPs)
- Qualitative behaviour: edge preservation, texture recovery, smooth-region quality, and visible artefacts
- Interpretability: Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) for the best-performing deep learning model
- Usability: a Gradio demonstration interface

## Working Arrangement

### Local machine

Use the local machine for:

- Repository and project-structure management
- Python environment and dependency work
- Bicubic baseline and NEDI implementation
- Small experiments and visual inspection
- Evaluation utilities and result processing
- Gradio development and testing
- Figures, tables, documentation, and report writing

### Google Colab

Use Google Colab for:

- PyTorch and BasicSR setup
- GPU inference with the verified pretrained checkpoints
- DIV2K preparation and larger experiments
- Batch evaluation of all test datasets
- Model statistics and repeated timing experiments
- LIME and SHAP runs when they are slow locally

The repository remains the source of truth for code and notebooks. Datasets, model weights, generated images, and large experiment outputs remain outside GitHub in Google Drive or another agreed storage location.

## Repository Structure

```text
FYP-SuperResolution-Comparative-Analysis/
├── data/                 # Metadata or instructions only; datasets stay outside GitHub
├── app/
│   ├── traditional/      # NEDI and classical baselines
│   ├── dl_models/        # FSRCNN and IMDN definitions/inference
│   ├── evaluation/       # Metrics, timing, model statistics, and reports
│   ├── xai/              # LIME and SHAP experiments
│   └── utils/            # Shared image, configuration, and logging utilities
├── notebooks/            # Exploratory and Colab notebooks
├── results/
│   ├── images/           # Selected visual comparisons
│   ├── metrics/          # CSV/JSON experiment results
│   ├── figures/          # Report-ready plots
│   └── xai/              # Explanation maps and summaries
├── demo/                 # Gradio application
├── docs/                 # Experiment notes and technical documentation
├── requirements.txt
├── .gitignore
└── PLAN.md
```

## Experimental Rules

These rules must be fixed before full evaluation begins:

1. Generate low-resolution inputs from high-resolution ground truth using the same bicubic degradation and anti-aliasing process for every method.
2. Evaluate every method on the same images, datasets, scaling factors, and image channels.
3. Keep test images separate from all training and fine-tuning data.
4. Record the exact model weights, configuration, random seed, software versions, hardware, and date for every experiment.
5. Measure inference time after warm-up and report the measurement procedure clearly.
6. Report both average results and dataset-level results; do not select only favourable images.
7. Keep bicubic as the initial baseline throughout the project.
8. Do not claim that one method is universally best. Conclusions must describe the quality-efficiency trade-off supported by the measurements.

Before full evaluation, we will explicitly record the remaining protocol choices: colour-space/channel used for metrics, border-cropping rule, batch size, timing device, and the exact degradation implementation.

## Phase Plan

### Phase 0: Project Setup and Baseline

**Tasks:**

- Create the local Python environment and repository structure.
- Add dependency and data-management documentation.
- Create the Google Drive folders for DIV2K and the four test datasets.
- Implement the bicubic baseline on one image, then a small sample.
- Save reconstructed images and PSNR, SSIM, and runtime results.
- Create the experiment log format.

**Deliverable:** A reproducible bicubic baseline running locally and in Colab.

**Completion test:** The same input and protocol produce consistent results in both environments.

### Phase 1: Evaluation Infrastructure

**Tasks:**

- Implement shared image loading, resizing, colour handling, and output-saving utilities.
- Implement PSNR and SSIM calculation.
- Implement runtime measurement with warm-up and repeated trials.
- Add parameter, FLOPs, and memory measurement utilities for neural networks.
- Define CSV/JSON schemas for experiment results.
- Create a small visual comparison notebook.

**Deliverable:** Evaluation tools that can accept any super-resolution method through the same interface.

**Completion test:** Bicubic results can be generated for all three scaling factors on a small test sample.

### Phase 2: NEDI

**Tasks:**

- Study the NEDI covariance and geometric-duality procedure.
- Implement NEDI in small, testable functions.
- Validate the implementation on synthetic edges and simple images.
- Compare NEDI with bicubic at x2, x3, and x4.
- Record quality, runtime, memory behaviour, edge preservation, and artefacts.

**Deliverable:** Tested NEDI implementation and initial comparison results.

**Completion test:** NEDI produces valid outputs for all required scaling factors and its behaviour is documented.

### Phase 3: FSRCNN and IMDN

**Current progress (5 September 2026):** The FSRCNN and IMDN GPU evaluations
are complete for all required datasets and scales. Their detailed files each
contain 657 image-scale records. Checkpoint sources, checksums, training-data
provenance, parameter counts, GPU memory, and Tesla T4 timing are recorded.
Controlled fixed-input FLOPs were subsequently completed during Phase 4.

**Tasks:**

- Set up PyTorch/BasicSR in Colab.
- Run pretrained inference using verified model and scale-specific checkpoints.
- Record the source, checksum, and original training dataset of every checkpoint.
- Ensure separate model configurations or weights for x2, x3, and x4.
- Evaluate FSRCNN and IMDN using the shared evaluation infrastructure.
- Measure PSNR, SSIM, latency, memory, parameters, and FLOPs.

**Deliverable:** Reproducible FSRCNN and IMDN inference pipelines with logged results.

**Completion test:** Both models run successfully at x2, x3, and x4 and produce logged results on the same test protocol as NEDI.

### Phase 4: Controlled Comparative Evaluation

**Completed (6 September 2026):** All 48 method/dataset/scale summary groups
were compared. The detailed NEDI, FSRCNN, and IMDN files each contain 657
validated image-scale records and reproduce their final summaries. Controlled
FLOPs were recorded for both neural models, and 12 predefined full-image and
identical-crop visual comparisons were inspected. IMDN ranked first in PSNR-Y
and SSIM-Y in all 12 groups and is selected as the best-performing neural model
for Phase 5. The unavailable final per-image Bicubic CSV and the different
timing hardware are explicitly recorded as limitations. Analysis is generated
by `notebooks/12_phase4_comparative_analysis.ipynb`; FLOPs and visual samples
are generated by `notebooks/13_phase4_flops_and_visuals_colab.ipynb`.

**Tasks:**

- Run the complete evaluation on Set5, Set14, BSD100, and Urban100.
- Repeat timing measurements under clearly documented hardware conditions.
- Generate dataset-level and overall metric tables.
- Produce visual comparisons covering smooth areas, edges, textures, and repetitive structures.
- Analyse the quality-efficiency trade-offs rather than only ranking PSNR.

**Deliverable:** Complete comparison tables, plots, and selected visual evidence.

**Completion test:** Every required method, dataset, and scaling factor has valid results with no missing protocol information.

### Phase 5: Fusion Method

**Completed (6 September 2026):** IMDN is selected as the neural component.
Three transparent fusion methods were implemented and tested: whole-image
weighted fusion, edge-aware fusion, and back-projection fusion. Whole-image
validation selected 100% IMDN, edge-aware fusion did not beat IMDN, and
back-projection also effectively selected IMDN. Therefore, fusion is recorded
as a valid negative result: it is reproducible, but it did not improve the
best single method. The protocol and conclusions are documented in
`docs/notes/phase5_fusion_protocol.md` and
`docs/notes/phase5_fusion_findings.md`; the validation outputs are stored in
`results/metrics/final/fusion/`. No full fusion benchmark is required because
the validation stage rejected the fusion variants before final test-set use.

**Tasks:**

- Select the best deep learning model using the predefined evaluation criteria.
- Implement a transparent weighted or edge-aware fusion of NEDI and that model.
- Test multiple simple weights using a validation subset, not the final test results.
- Compare the fusion against NEDI, FSRCNN, IMDN, and bicubic.
- Perform an ablation showing the effect of the fusion component.

**Deliverable:** Fusion method, weight-selection record, and ablation results.

**Completion test:** The fusion is reproducible and its benefit or limitation is reported honestly.

### Phase 6: Explainable Artificial Intelligence

**Tasks:**

- Select representative images from different test-set characteristics.
- Apply LIME and SHAP to the best-performing deep learning model.
- Generate explanation maps for edges, textures, smooth regions, and artefact-prone areas.
- Record the explanation settings and computational cost.
- Interpret whether the model relies on meaningful structural information or suspicious patterns.

**Deliverable:** LIME and SHAP visualisations with written interpretation.

**Completion test:** Explanations are reproducible, readable, and connected to specific super-resolution behaviours.

### Phase 7: Gradio Demo

**Tasks:**

- Build an image-upload workflow in Gradio.
- Display bicubic, NEDI, FSRCNN, IMDN, and fusion outputs side by side.
- Display available PSNR, SSIM, runtime, and model information.
- Add selected XAI maps for the best deep learning model.
- Add output download support.
- Test the interface on CPU and document its limitations.

**Deliverable:** Demonstration-ready Gradio application.

**Completion test:** A user can upload an image, select a scaling factor, view the method outputs, and download results.

### Phase 8: Final Analysis and Report

**Tasks:**

- Consolidate all final metrics, figures, and visual examples.
- Answer the research objectives using measured evidence.
- Explain which methods are preferable under different resource constraints.
- Document limitations, threats to validity, and future work.
- Complete implementation, methodology, results, discussion, and conclusion chapters.
- Prepare supervisor-review materials, presentation slides, and reproducibility instructions.

**Deliverable:** Complete report, documented repository, final figures, and presentation-ready demo.

**Completion test:** The report, code, results, and demo describe the same final experiment and can be followed by another researcher.

## Results and Experiment Logging

Every experiment should record at least:

```text
experiment_id, date, method, model_checkpoint, dataset, image_count,
scale, degradation, channel, psnr, ssim, latency_ms, memory_mb,
parameters, flops, hardware, software_version, notes
```

Results should be saved as machine-readable CSV or JSON first. Report tables and figures should be generated from those records rather than typed manually.

## Risks and Responses

- **NEDI is slow or difficult to reproduce:** begin with small images, validate against synthetic edges, and document any implementation assumptions.
- **Pretrained checkpoints differ in training data:** record each checkpoint's
  training dataset and avoid claiming that checkpoint training was controlled
  within this project.
- **LIME or SHAP is computationally expensive:** explain a small, representative image sample and report the explanation settings and cost.
- **Fusion does not improve every metric:** treat that as a valid research result and analyse when fusion helps or hurts.
- **Results differ across hardware:** separate quality evaluation from timing evaluation and report hardware for every latency claim.
- **Scope expands:** protect the fixed comparison and postpone optional experiments until all required deliverables are complete.

## First Milestone

The first working milestone is the local bicubic baseline. We will complete it in this order:

1. Confirm the local Python environment.
2. Create the repository folders and dependency files.
3. Run one HR-to-LR-to-SR bicubic experiment.
4. Calculate PSNR, SSIM, and runtime.
5. Save the output image and one logged result row.
6. Re-run the same baseline in Google Colab.

After this milestone is verified, we will begin the NEDI implementation while preparing the Colab deep-learning environment in parallel.

## Working Agreement

- We will make one small, verifiable change at a time.
- We will not move to the next phase while the current phase lacks a reproducible deliverable.
- We will keep implementation decisions and experiment results documented as they are made.
- We will treat negative or mixed results as useful findings when they are produced by a fair experiment.
