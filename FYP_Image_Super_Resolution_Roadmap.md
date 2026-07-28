# Final Year Project Roadmap: Comparative Analysis of Edge-Directed Interpolation and Lightweight Deep Learning Models for Image Super-Resolution

**Student:** Emilola Divine  
**Topic Approved:** Option 1 with supervisor feedback (comparative analysis, focus on best DL algorithm, XAI with SHAP/LIME, possible ensemble, thorough evaluation)  
**Goal:** Rigorous comparison, practical insights for resource-constrained environments (e.g., Nigeria/mobile), XAI for interpretability, working demo, potential publication.

## Approved Project Scope

The Word document containing Chapters 1 and 2 is the formal academic source of truth for this roadmap. The implementation and evaluation must follow these fixed conclusions:

- **Traditional method:** New Edge-Directed Interpolation (NEDI)
- **Deep learning models:** Fast Super-Resolution Convolutional Neural Network (FSRCNN) and Information Multi-Distillation Network (IMDN)
- **Fusion:** A simple fusion of NEDI and the best-performing deep learning model
- **Explainability:** Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) applied to the best-performing deep learning model
- **Training dataset:** DIVerse 2K (DIV2K)
- **Testing datasets:** Set5, Set14, BSD100, and Urban100
- **Scaling factors:** x2, x3, and x4
- **Evaluation:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), floating-point operations, parameter count, memory usage, and inference latency
- **Demo interface:** Gradio

## Project Objectives (Refined)

1. Implement and evaluate the New Edge-Directed Interpolation algorithm (NEDI).
2. Implement/train FSRCNN and IMDN as the selected lightweight deep learning models.
3. Conduct a fair comparative evaluation using PSNR, SSIM, runtime, visual quality, and efficiency.
4. Apply LIME and SHAP to the best-performing deep learning model.
5. Develop a simple NEDI/deep learning fusion method.
6. Build a user-friendly Gradio demo.
7. Analyze trade-offs and provide recommendations.
8. Document everything for report + possible paper.

## Tech Stack Recommendations

- **Language:** Python 3
- **DL Framework:** PyTorch (via BasicSR toolbox for fairness and ease)
- **Traditional/Image Processing:** OpenCV, scikit-image, Pillow, NumPy
- **XAI:** `lime`, `shap`
- **GUI:** Gradio
- **Evaluation:** PSNR/SSIM (scikit-image), timing, memory usage, and `torchinfo`/`thop` for model statistics
- **Compute:** Google Colab (GPU), Kaggle, local if GPU available
- **Version Control:** GitHub repo
- **Datasets:** DIV2K (train), Set5/Set14/BSD100/Urban100 (test)

## Detailed Phase-by-Phase Roadmap

### Phase 0: Preparation & Setup

- Create GitHub repo: `FYP-SR-Comparative-Analysis`
- Set up Colab environment + requirements.txt
- Download datasets (DIV2K subset for quick starts)
- Implement simple baseline:
  - Load HR image → bicubic downsample to LR → bicubic upscale
  - Compute PSNR/SSIM, display images
- Document environment and baseline results
- Update proposal document with this roadmap

**Milestone:** Working baseline notebook + GitHub repo initialized

### Phase 1: Literature Review & Planning

- Summarize key papers (SRCNN, FSRCNN, NEDI, IMDN, and lightweight SR surveys)
- Note strengths/weaknesses, implementation details
- Confirm the fixed comparison: NEDI + FSRCNN + IMDN + fusion
- Define evaluation protocol (upscaling factors x2/x3/x4, same test sets, CPU/GPU timing, memory measurement, and seeds for reproducibility)
- Plan XAI experiments and GUI features

**Deliverable:** Literature notes document + detailed experiment plan

### Phase 2: Traditional Methods

- Implement or adapt NEDI (use existing GitHub impls as base, understand covariance/edge logic)
- Compare with basic bicubic as baseline
- Test on standard datasets, record metrics and visuals (especially edge preservation)
- Record edge-preservation behaviour, runtime, and memory use without changing the approved method

**Milestone:** NEDI working, quantitative comparison vs bicubic

### Phase 3: Lightweight Deep Learning Models

- Use **BasicSR** framework for reproducibility
- Implement FSRCNN and IMDN
- Train on DIV2K (or use pre-trained weights + fine-tune)
- Evaluate at x2, x3, and x4
- Measure inference time, memory usage, PSNR/SSIM, parameter count, and FLOPs
- Run fair comparisons against traditional methods

**Milestone:** Multiple DL models trained/tested, initial comparison tables

### Phase 4: XAI Integration

- Apply both LIME and SHAP to the best-performing deep learning model
- Generate explanations for sample images (e.g., which regions/pixels drive the super-resolution)
- Analyze what the model "learns" (edges, textures, etc.)
- Compare interpretability with traditional methods

**Milestone:** XAI visualizations and insights documented

### Phase 5: Ensemble & Advanced Experiments

- Implement simple ensemble (e.g., weighted average, edge-aware fusion of NEDI + DL)
- Ablation studies (what contributes most to performance?)
- Test on additional images (real-world low-res, domain-specific if time)
- Iterate based on results ("try until best" per supervisor)

**Milestone:** Proposed improvement (ensemble) that shows value

### Phase 6: Gradio Demo Application

- Build with Gradio:
  - Upload LR image
  - Show side-by-side results from all methods
  - Display metrics + XAI maps
  - Option to download results
- Deploy (e.g., Hugging Face Spaces or local)

**Milestone:** Interactive demo ready for presentation

### Phase 7: Evaluation, Analysis & Reporting

- Comprehensive tables/graphs (PSNR, SSIM, inference time, memory use, parameter count, and FLOPs)
- Qualitative analysis (visual examples of strengths/weaknesses)
- Trade-off discussion + recommendations for low-resource settings
- Limitations, future work
- Write full report/thesis chapter + draft paper
- Reproducibility: seeds, code, data links

**Milestone:** Complete report, code documentation, presentation slides

## Key Resources

- **BasicSR:** `https://github.com/XPixelGroup/BasicSR (core framework)`
- NEDI GitHubs: Search "Edge-Directed_Interpolation"
- Datasets: DIV2K official site
- XAI tutorials: LIME/SHAP image examples
- Papers: SRCNN, FSRCNN, NEDI, lightweight SR surveys

## Best Practices & Tips

- **Reproducibility:** Fix random seeds, document hyperparameters, use same test images
- **Fair Comparison:** Same hardware conditions for timing, same preprocessing
- **Scope Control:** Start with bicubic + NEDI + FSRCNN, then add IMDN, fusion, XAI, and the Gradio demo
- **Compute Management:** Use Colab GPU wisely, start with small subsets
- **Documentation:** Keep everything in the repo (notebooks, results, figures)
- **Supervisor Alignment:** Share progress and roadmap regularly
- **Publication Angle:** Highlight XAI insights, efficiency for developing regions, practical demo

## Risks & Mitigation

- Training too slow → Use pre-trained weights + fine-tune, smaller models
- XAI hard → Start with LIME on small images
- Scope creep → Prioritize core comparison + XAI + demo
- Results not "best" → Focus on analysis and your ensemble contribution

Update this roadmap as you progress and get supervisor feedback.

**Next Immediate Steps (This Week):**

1. Create GitHub repo + initial Colab baseline notebook
2. Adapt/implement NEDI
3. Set up BasicSR and run one DL inference example

You got this! Track progress against this roadmap.

---

*Created: July 2026*  
*Version: 1.0*
