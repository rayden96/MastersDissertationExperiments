# Thesis Experiment Plan — BOGrad & COSGD

**Dissertation:** Batch Orthogonalised Gradient Descent for Single-Task Classification
**Author:** Rayden Logan Burger

This document lists every experiment planned for the thesis, the plots each will produce, and where those plots live in the narrative. Based on the supervisor's decision to share datasets across COSGD and BOGrad, the same dataset suite is used throughout Chapters 3–5 so that every comparison is apples-to-apples.

---

## Shared Dataset Suite

All three method chapters (3, 4, 5) draw from this pool. Chapters 3 and 4 may use subsets for focused mechanism studies; Chapter 5 runs the full suite.

| Modality | Dataset | # Classes | Model | Scale |
|---|---|---|---|---|
| Tabular | Forest Covertype | 7 | MLP (54→128→64→7) | 581k rows |
| Image | MNIST | 10 | small CNN | small |
| Image | FashionMNIST | 10 | small CNN | small |
| Image | EMNIST-Balanced | 47 | small grayscale CNN | small |
| Image | CIFAR-10 | 10 | 3-layer CNN | small |
| Image | **CIFAR-100** | **100** | **ResNet-18** | **large** |
| Audio | ESC-50 | 50 | 2D log-mel CNN | small |
| Text | DBpedia-14 | 14 | EmbeddingBag + Linear | medium |

**Optimizer set evaluated on every dataset:** SGD, RMSprop, SignSGD, Adam, plus their COSGD and BOGrad counterparts. BOGrad is treated as base-optimizer-agnostic — the algorithm work to make it integrate cleanly with Adam-style preconditioned methods is a parallel effort, and the experiments below assume that work succeeds. No separate Adam ablation is planned.

**Headline addition (CIFAR-100 + ResNet-18)** addresses prior reviewer feedback that the evaluation was confined to small models. **Covertype** replaces Iris and Titanic with a real-world tabular benchmark (581k examples, used in the TabNet and tree-baseline literature) without losing the tabular modality. **BloodMNIST** is dropped as redundant with the other small image benchmarks.

---

# Chapter 3 Experiments — COSGD

These experiments characterise COSGD's mechanism, its high-dimensional behaviour, and — critically — its scalability limits, which sets up the motivation for BOGrad in Chapter 4.

## Experiment 3.1 — Cosine Similarity vs. Vector Dimension (Synthetic)

**Purpose.** Establish that the "random vectors are almost orthogonal in high dimension" intuition only holds when vectors are normalised, motivating why magnitude matters in COSGD's orthogonalisation.

**Method.** Generate pairs of random vectors at dimensions 2 to 10,000 (step 500), 50 trials per dimension. Measure average absolute cosine similarity in (a) unit-normalised and (b) raw-magnitude conditions.

**Plot 3.1 — Two-panel line plot.**
- Panel A: x-axis = vector dimension (linear, 0–10,000); y-axis = average |cosine similarity| (linear, 0–1). One line: normalised vectors.
- Panel B: same x-axis; y-axis = average |cosine similarity| (linear, likely 0–100+). One line: unnormalised vectors.
- Shows: normalised case collapses near 0; unnormalised case grows with dimension, confirming that magnitude variance prevents natural orthogonality.

**Location.** Section 3.4.1, directly supporting the argument that normalisation prior to Gram-Schmidt is consequential.

## Experiment 3.2 — L2 Change After Gram-Schmidt Orthogonalisation (Synthetic)

**Purpose.** Quantify how much orthogonalisation actually perturbs vectors across dimensions, and whether the perturbation is stable or blows up with magnitude.

**Method.** Same dimensional sweep as 3.1. For each trial, apply Gram-Schmidt and measure the average L2 distance between pre- and post-orthogonalisation vectors, for normalised and unnormalised cases.

**Plot 3.2 — Two-panel line plot.**
- Panel A: x = dimension; y = avg L2 change after Gram-Schmidt (normalised).
- Panel B: x = dimension; y = avg L2 change after Gram-Schmidt (unnormalised).
- Shows: stable, small perturbation when normalised; large and growing perturbation when unnormalised. Justifies the design choice to normalise gradients before orthogonalising in COSGD.

**Location.** Section 3.4.2.

## Experiment 3.3 — Magnitude-Ordering Ablation

**Purpose.** Test whether sorting per-class gradients by descending magnitude before Gram-Schmidt actually matters, or if any ordering works equally well.

**Method.** On MNIST, CIFAR-10, and EMNIST-Balanced, run COSGD with three orderings: descending magnitude (default), ascending magnitude, random permutation. Five trials each, paired data order.

**Plot 3.3 — Grouped bar chart.**
- x-axis = dataset; y-axis = test accuracy at final epoch; groups of three bars per dataset (descending / ascending / random). Error bars = std across trials.
- Shows: whether descending-magnitude priority is empirically justified, marginal, or irrelevant.

**Location.** Section 3.3.2 (Magnitude Ordering Rationale), as empirical validation of the theoretical choice.

## Experiment 3.4 — COSGD Class-Count Scalability

**Purpose.** Demonstrate the O(mn²) bottleneck that motivates BOGrad. Show that COSGD's per-step cost grows quadratically in the number of classes present in a mini-batch.

**Method.** Fix architecture and batch size. Run COSGD on CIFAR-10 (10 classes), EMNIST-Balanced (47), ESC-50 (50), and CIFAR-100 (100). Also run a synthetic control: a fixed CNN on CIFAR-10 artificially split into {2, 5, 10, 20, 50, 100} synthetic sub-classes (via label sub-hashing) to isolate class count from dataset effects. Measure wall-clock time per step.

**Plot 3.4a — Line plot (synthetic control).**
- x-axis = number of classes in batch (log scale, 2–100); y-axis = wall-clock seconds per step. Overlay a fitted quadratic reference curve.
- Shows: empirical confirmation of the O(n²) orthogonalisation cost.

**Plot 3.4b — Grouped bar chart (real datasets).**
- x-axis = dataset (CIFAR-10, EMNIST-Bal, ESC-50, CIFAR-100); y-axis = wall-clock seconds per step; two bars per dataset (SGD baseline vs COSGD).
- Shows: practical overhead penalty on high-class-count problems, with CIFAR-100 as the headline 100-class point.

**Location.** Section 3.7 (Scalability Limitations and Transition to BOGrad). This plot is the narrative hinge between COSGD and BOGrad.

## Experiment 3.5 — COSGD Memory Footprint vs Class Count

**Purpose.** Complement 3.4 with a memory-cost view of the same bottleneck.

**Method.** Same setup as 3.4. Measure peak GPU memory per step.

**Plot 3.5 — Line plot.**
- x-axis = number of classes; y-axis = peak GPU memory (MB). One line for COSGD, one flat reference line for SGD baseline.
- Shows: per-class gradient storage is the dominant memory driver.

**Location.** Section 3.7, paired with 3.4.

---

# Chapter 4 Experiments — BOGrad Mechanism

These experiments validate that BOGrad's projection does what the theory claims. They are lightweight, diagnostic, and use a small number of datasets (typically CIFAR-10 and EMNIST-Balanced) — the full bakeoff is in Chapter 5.

## Experiment 4.1 — Gradient Norm Preservation Under Projection

**Purpose.** Empirically verify Eq. (10): ‖g̃_t‖ ≤ ‖g_t‖. Also show how much norm is removed on average, which is a proxy for how much "recent-direction overlap" existed.

**Method.** During training on CIFAR-10 (SGD base, K = 8), log ‖g̃_t‖ / ‖g_t‖ every step.

**Plot 4.1 — Line plot with shaded band.**
- x-axis = training step; y-axis = ratio ‖g̃_t‖ / ‖g_t‖ ∈ (0, 1]. Line = mean across parameter tensors, shaded band = ±1 std across tensors.
- Reference horizontal line at y = 1.
- Shows: ratio is strictly ≤ 1, and reveals the training phase in which interference is strongest (typically early epochs).

**Location.** Section 4.3 (Theoretical Analysis of Between-Batch Orthogonalisation).

## Experiment 4.2 — Cosine Alignment Between Successive Updates

**Purpose.** Verify that BOGrad reduces the destructive cross-term in Eq. (8) by empirically measuring ⟨g_t, g_{t-1}⟩ / (‖g_t‖ ‖g_{t-1}‖) with and without the projection.

**Method.** On CIFAR-10 + SGD and CIFAR-10 + RMSprop, log cosine(step_t, step_{t-1}) during training for baseline vs BOGrad-wrapped. Also log cosine of the raw gradients in both cases (to separate "raw gradient alignment" from "effective update alignment").

**Plot 4.2 — Two-panel line plot.**
- Panel A: x = step; y = cosine(g_t, g_{t-1}); two lines (baseline, BOGrad).
- Panel B: x = step; y = cosine(applied_update_t, applied_update_{t-1}); two lines.
- Shows: BOGrad pushes successive update directions toward orthogonality, confirming the mechanism empirically.

**Location.** Section 4.3.

## Experiment 4.3 — Orthogonalisation Method Comparison

**Purpose.** Justify the choice of Modified Gram-Schmidt over alternatives.

**Method.** Implement and compare on CIFAR-10 + SGD (K = 8):
- Modified Gram-Schmidt (default)
- Classical Gram-Schmidt
- Householder reflections
- QR-based projection (via torch.linalg.qr on the stacked buffer)

For each, measure: (a) final test accuracy, (b) wall-clock time per step, (c) numerical orthogonality residual ‖G^T g̃‖ after projection.

**Plot 4.3a — Grouped bar chart.**
- x-axis = method (4 bars); y-axis split across two panels:
  - Panel A: final test accuracy
  - Panel B: wall-clock time per step (ms)

**Plot 4.3b — Bar chart.**
- x-axis = method; y-axis = log₁₀ orthogonality residual after projection.
- Shows: numerical quality of each method's output.

**Location.** Section 4.2.1 (Orthogonalisation Methods).

## Experiment 4.4 — Effect of Buffer Size K on Projection

**Purpose.** Characterise how much of the gradient BOGrad removes as K grows, independent of final task accuracy. This is the "theoretical side" of buffer size; the "empirical side" (which K is best for accuracy) is in Chapter 5.

**Method.** Train CIFAR-10 + SGD with K ∈ {1, 2, 4, 8, 16, 32, 64}. Log mean ‖g̃_t‖ / ‖g_t‖ across training.

**Plot 4.4 — Line plot.**
- x-axis = training step; y-axis = mean ‖g̃_t‖ / ‖g_t‖; one line per K value (colour gradient from light to dark).
- Shows: larger K strips more of the gradient, approaching the saturation implied by linear dependence among recent gradients.

**Location.** Section 4.2.2 (Buffer Theory and Scalability).

## Experiment 4.5 — Batch Size × K Interaction

**Purpose.** Test the hypothesis that small batches (noisier gradients, more temporal variance) benefit more from larger K than large batches do.

**Method.** CIFAR-10 + SGD. Cross batch size ∈ {32, 64, 128, 256, 512} with K ∈ {0 (baseline), 2, 8, 32}. Five trials each, record final test accuracy.

**Plot 4.5 — Heatmap.**
- x-axis = K; y-axis = batch size; cell colour = final test accuracy; annotate each cell with mean ± std.
- Shows: regions where BOGrad helps most (expected: small batch, moderate K).

**Location.** Section 4.2.3 (Batch Size).

---

# Chapter 5 Experiments — Unified Empirical Evaluation

This is the bakeoff: COSGD and BOGrad against baselines and each other on the full shared dataset suite. Chapter 5 is structured around the subsection headings in the thesis template (5.2 Main Results, 5.3 Sensitivity, 5.4 Model Scale, 5.5 Gradient Stats).

## Experiment 5.1 — Main Test-Accuracy Trajectories (per dataset)

**Purpose.** Primary empirical evidence. For each dataset, show test accuracy over training for every optimizer variant, enabling direct visual comparison of convergence speed and final performance.

**Method.** For each of the 8 datasets, train with: SGD, SGD+COSGD, SGD+BOGrad, RMSprop, RMSprop+COSGD, RMSprop+BOGrad, SignSGD, SignSGD+COSGD, SignSGD+BOGrad, Adam, Adam+COSGD, Adam+BOGrad. Five trials, paired data order across all methods within a trial. BOGrad is applied uniformly across all base optimizers — no Adam-specific variant is reported.

**Plot 5.1 — One line plot per dataset (8 plots total).**
- x-axis = training steps (linear); y-axis = test accuracy (linear, dataset-appropriate range).
- Lines: 12 methods (distinguish baseline / COSGD / BOGrad by linestyle: solid / dashed / dotted; distinguish optimizers by colour).
- Shaded band = ±1 std across 5 trials.
- Shows: which method family wins on which modality and scale, including whether BOGrad and COSGD have complementary strengths and whether benefits hold up on the larger ResNet-18 / CIFAR-100 setting.

**Location.** Section 5.2 (Main Results). Each plot appears next to the dataset-specific discussion.

## Experiment 5.2 — Fixed-Budget Accuracy Tables

**Purpose.** Quantitative snapshot at controlled training budgets, supporting the "faster convergence" claim numerically.

**Method.** From runs in 5.1, extract test accuracy at 10%, 50%, and 100% of the epoch budget.

**Output.** Three tables (one per budget fraction), rows = (dataset × baseline optimizer), columns = (Baseline, COSGD, BOGrad). Report mean ± std over 5 trials. Bold the best per row.

**Location.** Section 5.2, immediately after the trajectory plots. Consolidated as Tables 5.1–5.3.

## Experiment 5.3 — Final Accuracy Summary Plot

**Purpose.** A single at-a-glance visual summarising which method wins across the whole dataset suite.

**Method.** Aggregate final accuracies from 5.1.

**Plot 5.3 — Grouped bar chart.**
- x-axis = dataset (8 groups); y-axis = final test accuracy; bars within each group = methods (colour-coded as in 5.1).
- Error bars = ±1 std.
- Shows: big-picture winner/loser pattern across modalities and model scales.

**Location.** Section 5.2, at the end of the main-results discussion.

## Experiment 5.4 — Learning-Rate Sensitivity

**Purpose.** Test whether BOGrad/COSGD shift the useful learning-rate range, and whether they are more or less sensitive than baselines.

**Method.** On CIFAR-10 and CIFAR-100 (ResNet-18), sweep LR ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1} for SGD, SGD+COSGD, SGD+BOGrad, RMSprop, RMSprop+BOGrad.

**Plot 5.4 — Two line plots (one per dataset).**
- x-axis = learning rate (log scale); y-axis = final test accuracy; one line per method.
- Error bars = ±1 std.
- Shows: peak location and width of the useful LR basin per method, for both small CNN and ResNet-18 settings.

**Location.** Section 5.3 (Sensitivity Analysis).

## Experiment 5.5 — Buffer Size K Sensitivity (BOGrad)

**Purpose.** Determine the practical best-K across datasets, complementing the mechanism-level view in Experiment 4.4.

**Method.** For each of CIFAR-10, CIFAR-100, MNIST, EMNIST-Balanced, ESC-50, DBpedia-14, train SGD+BOGrad with K ∈ {1, 2, 4, 8, 16, 32}.

**Plot 5.5 — Line plot.**
- x-axis = K (log₂ scale); y-axis = final test accuracy; one line per dataset.
- Horizontal dashed reference line per dataset at SGD baseline accuracy.
- Shows: where the accuracy-vs-K curve peaks per dataset, informing the default recommendation.

**Location.** Section 5.3.

## Experiment 5.6 — Batch Size Sensitivity

**Purpose.** Confirm at the task level what Experiment 4.5 showed at the mechanism level: that BOGrad's benefit shifts with batch size.

**Method.** On CIFAR-10 and EMNIST-Balanced, vary batch size ∈ {32, 64, 128, 256, 512} with fixed best-K from 5.5. Compare SGD vs SGD+BOGrad vs SGD+COSGD.

**Plot 5.6 — Two line plots (one per dataset).**
- x-axis = batch size (log scale); y-axis = final test accuracy; three lines per plot (SGD, SGD+BOGrad, SGD+COSGD).
- Shows: which method tolerates small or large batches best.

**Location.** Section 5.3.

## Experiment 5.7 — Model-Scale Study

**Purpose.** Characterise how BOGrad's benefits change with model capacity and where it provides the best trade-off. Two complementary views:

**5.7a — Width-scaled small CNN (controlled).** CIFAR-10 CNN with channel widths scaled by s ∈ {0.125, 0.25, 0.5, 1.0, 2.0}. At each scale, retune (LR, K) on validation for SGD, SGD+COSGD, SGD+BOGrad, RMSprop, RMSprop+BOGrad. Five trials each.

**5.7b — Real architectures (reviewer-credible).** CIFAR-100 with ResNet-18, ResNet-34, and ResNet-50. Same method set as 5.7a. Five trials each.

**Plot 5.7a — Grouped bar chart (synthetic width sweep).**
- x-axis = model scale (5 groups); y-axis = test accuracy at fixed-fraction-of-epochs; bars = methods.
- Error bars = ±1 std. Shows: where BOGrad/COSGD help most relative to baseline on a controlled axis.

**Plot 5.7b — Grouped bar chart (real architectures).**
- x-axis = architecture (ResNet-18, -34, -50); y-axis = final test accuracy; bars = methods.
- Shows: whether the method ranking is preserved on production-style architectures, and whether BOGrad's relative benefit grows or shrinks with capacity.

**Plot 5.7c — Line plot.**
- x-axis = model scale (combining both 5.7a and 5.7b on a parameter-count x-axis, log scale); y-axis = wall-clock seconds per epoch; one line per method.
- Shows: how the overhead of orthogonalisation amortises (or doesn't) with model size, spanning small CNN to ResNet-50.

**Plot 5.7d — Line plot.**
- x-axis = model scale (same parameter-count axis); y-axis = (BOGrad accuracy − baseline accuracy) / (BOGrad time − baseline time). The "accuracy gain per unit overhead" ratio.
- Shows: the regime where BOGrad delivers the best bang-for-buck across scales.

**Location.** Section 5.4 (Model-Scale Study).

## Experiment 5.8 — Gradient Statistics During Training

**Purpose.** Move beyond "test accuracy improved" to show *why*: the gradient geometry during training genuinely differs under BOGrad/COSGD vs baseline.

**Method.** On CIFAR-10 + SGD and CIFAR-10 + RMSprop, log during training:
- cosine(g_t, g_{t-1})
- ‖g_t‖
- fraction of gradient removed by projection (BOGrad / COSGD)
- effective step size (‖applied update‖)

**Plot 5.8 — Four-panel plot (2×2 grid) per optimizer (so two figures total).**
- Panel A (top-left): x = step, y = cosine(g_t, g_{t-1}); lines for baseline, BOGrad, and COSGD.
- Panel B (top-right): x = step, y = ‖g_t‖ (log scale); same three lines.
- Panel C (bottom-left): x = step, y = fraction of ‖g‖ removed by projection (BOGrad and COSGD only).
- Panel D (bottom-right): x = step, y = ‖applied update‖; same three lines.
- Shows: empirical interference reduction + its effect on effective step magnitude. This is the evidence that connects mechanism to outcome.

**Location.** Section 5.5 (Gradient Statistics Analysis). Implementation details to be refined — instrumentation may add overhead, so this is run on a dedicated short training schedule rather than the full 5.1 runs.

## Experiment 5.9 — Per-Dataset Pareto Frontiers (Accuracy vs. Wall-Clock)

**Purpose.** Final trade-off summary: accuracy gain versus time cost, broken out per dataset so each modality/scale tells its own story.

**Method.** From runs in 5.1, extract mean seconds per step and final test accuracy.

**Plot 5.9 — One scatter plot per dataset (8 plots, presentable as a 2×4 grid).**
- x-axis = wall-clock seconds per step (log scale); y-axis = final test accuracy.
- Points = (optimizer × method) combinations; colour by base optimizer, marker shape by method (baseline / COSGD / BOGrad).
- Pareto frontier overlaid as a dashed line per panel.
- Shows: which methods sit on the accuracy–time Pareto frontier for each dataset, and how that frontier changes between small CNN datasets and the larger ResNet-18 / CIFAR-100 setting.

**Location.** Section 5.6 (Summary of Findings) — closing figure of the empirical chapter.

---

# Cross-Reference Table

| Thesis Section | Experiments |
|---|---|
| 3.3.2 Magnitude Ordering Rationale | 3.3 |
| 3.4.1 Cosine Similarity | 3.1 |
| 3.4.2 L2 Distance | 3.2 |
| 3.7 Scalability Limitations | 3.4, 3.5 |
| 4.2.1 Orthogonalisation Methods | 4.3 |
| 4.2.2 Buffer Theory and Scalability | 4.4 |
| 4.2.3 Batch Size | 4.5 |
| 4.3 Theoretical Analysis (Between-Batch) | 4.1, 4.2 |
| 5.2 Main Results | 5.1, 5.2, 5.3 |
| 5.3 Sensitivity Analysis | 5.4, 5.5, 5.6 |
| 5.4 Model-Scale Study | 5.7 |
| 5.5 Gradient Statistics Analysis | 5.8 |
| 5.6 Summary of Findings | 5.9 |

---

# Notes on Shared-Dataset Integration

Key consequence of shared datasets: **COSGD now appears on EMNIST-Balanced (47), ESC-50 (50), and CIFAR-100 (100)**, well beyond what its original paper faced. Two implications:

1. The high-class-count datasets stress COSGD's O(n²) scaling in the mini-batch, providing natural evidence for Experiment 3.4. CIFAR-100 in particular gives the headline 100-class data point. Whether to mitigate via hierarchical clustering or simply let runs go long is a decision to make once initial timings are in.
2. **DBpedia-14** uses an EmbeddingBag model whose parameter count differs by several orders of magnitude between the embedding table and the classifier head. This is a good stress test for per-tensor projection strategies, relevant to the model-scale discussion.

Keep the paired data-order control from the BOGrad paper: within each trial, baseline / COSGD / BOGrad see the same shuffled index order and mini-batch partitioning across all epochs. This removes mini-batch stochasticity from the cross-method comparison and is essential for credible results.

# Notes on the Adam Generalisation Effort

The experiment plan assumes BOGrad works as a generic wrapper across SGD, RMSprop, SignSGD, and Adam. The algorithmic work to make BOGrad's projection geometry compatible with Adam-style preconditioning is being done in parallel and is not itself an experiment in this plan — its outcome is a single algorithm that ships into Experiment 5.1 alongside everything else. If that work doesn't fully land in time, the fallback is to drop the Adam columns from 5.1 rather than reintroduce a separate Adam-variant ablation.
