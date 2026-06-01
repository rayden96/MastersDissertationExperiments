# Handoff document — interference research project

A complete information dump for a fresh session to come up to speed.
No instructions, no recommendations — just facts about what exists,
what we've found, and where things sit.

---

## 1. Project context

**Repo:** `C:\Masters Work\Dissertation&Experiments\MastersDissertationExperiments`

**Dissertation topic** (originally): "Batch Orthogonalised Gradient Descent
for Single-Task Classification" — proposing BoGrad and COSGD as
optimisation contributions.

**Current framing** (after research evolution): A *study* on gradient
interference in single-task supervised training, with COSGD and BoGrad
as proposed solutions to two distinct interference types, and a
comparison against implicit alternatives (dropout, momentum, etc.).

The user is the author. Two original optimisers exist (BoGrad and COSGD)
that were already developed before this research phase began. The
research phase has been about characterising what they do, building a
framework to measure it, and exploring extensions.

---

## 2. The two-type interference framework

### Definitions

Single-task supervised training: model parameters $\theta_t \in \mathbb{R}^p$,
mini-batches $B_t$ drawn iid from dataset, per-batch gradient
$g_t = \nabla_\theta \ell(\theta_t; B_t)$, optimiser produces update $u_t$
with $\theta_{t+1} = \theta_t + u_t$.

**Two distinct phenomena are called "interference":**

| Type | Definition | Method that targets it |
|---|---|---|
| **Inter-batch (intra-batch)** | Class gradients within a single batch conflict — averaging them yields a degraded direction | **COSGD** (per-class orthogonalisation within batch) |
| **Between-batch** | Successive batch gradients/updates conflict over time — trajectory zigzags or undoes recent progress | **BoGrad** (orthogonalise current step against recent buffer of past steps) |

### Headline metrics

- **Inter-batch:** `IB_%neg` = fraction of class-pair gradients within a
  batch with $\cos < 0$. Computed via `stage1/inter_batch_metric.py` —
  per-class forward+backward on a held-out probe set, then pairwise
  cosines.
- **Between-batch:**
  - `WW_K` (wasted-work ratio) = $\lVert \theta_t - \theta_{t-K}\rVert / \sum_i \lVert u_i\rVert$ over a $K$-step window. Range (0, 1].
  - Out-of-batch (OOB) forgetting magnitude — only meaningful in non-iid
    batching; structurally 0 in standard CIFAR-10.
  - Pairwise alignment distribution over $K$-step buffer
    (`PairwiseAlignmentTracker`).

### The "wasted-work hypothesis" (W) and refined (W')

(W): Net training-speed is inversely related to total wasted work
($\Sigma_t \Sigma_c \Delta_{t,c}$ — cumulative per-class forgetting).

(W'): Orthogonalisation helps only when the buffered direction is
*meaningfully separable from the descent direction* — i.e. the gradient
buffer's dominant aligned direction is not where descent points.

(W) is partially supported (under non-iid batching, F7/F12 confirm).
(W') is strongly supported by F10 (drift bias, BoGrad fails) vs F12
(perpendicular bias, BoGrad recovers ~95%).

---

## 3. Findings catalogue (F1–F18)

All formally documented in
`research/01_interference_framework/framework.md` §8 (with numbers and
context). Indexed in `TESTS_OVERVIEW.md` at root.

**F1.** Momentum induces structural lag-1 gradient correlation. Vanilla
SGD: cos(g_t, g_{t-1}) ≈ 0. SGD+momentum: +0.26. Effect arises because
momentum drives parameters along a coherent trajectory; gradients
sampled along that trajectory share signal beyond mini-batch noise.
Visible at log_every=1 only — disappears at log_every=10.

**F2.** Wasted-work ratio (WW_K=32) ranks methods by trajectory
efficiency: vanilla SGD 0.30 < SGD+mom 0.41 < BoGrad+vanilla 0.53 <
BoGrad+mom 0.61. Ranking matches the mechanism but is not a perfect
predictor of accuracy (BoGrad+vanilla has higher WW than mom-baseline
but lower accuracy).

**F3.** Distinct trajectory-lag decay profiles. Lag-1/4/16 cosines for
each method differ characteristically. Vanilla SGD: ~0 across all lags
(memoryless). Momentum: +0.77 lag-1, decays sharply to 0.15 at lag-4.
BoGrad: persistent moderate alignment across all lags. BoGrad+momentum:
combines both effects.

**F4.** Per-class forgetting magnitude is step-magnitude-confounded.
Methods with larger per-step parameter movement produce larger per-class
loss fluctuations in both directions. Direct comparison of $\sum \Delta$
across methods with different $\lVert u\rVert$ is unfair to large-step
methods. Motivated step-magnitude-normalised variants in §4.2.

**F5.** Detection ≠ slowdown. In a synthetic quadratic with zero-mean
alternating interference, framework metrics correctly detect the
injected pattern (cos collapses from +0.41 to −0.87, WW drops from 0.60
to 0.11), but the injected interference *does not slow vanilla SGD*
(SGD's averaging absorbs it). Worse: BoGrad applied to it actively hurts
(dist 0.06 → 0.62) by stripping useful averageable signal.

**F6.** Per-class total forgetting magnitude is dominated by
within-class noise, not cross-class interference. In class-disjoint
regime, total Δ is *smaller* than interleaved (0.68 vs 9.50) — opposite
of what naive interpretation predicts — because per-class loss
fluctuations from arbitrary parameter updates dominate the metric. WW_K
correctly tracked the slowdown where forgetting magnitude failed.

**F7.** Out-of-batch forgetting metric cleanly isolates cross-class
interference. For each step, partition forgetting events by whether
the class was in the current batch (in-batch, ambiguous source) or not
(out-of-batch, must be parameter drift from other classes). Disjoint
regime: Δ_OOB = 0.55. Interleaved: 0.00 by construction (no class is
ever out-of-batch). The signal F6 said was hidden is now visible.

**F8.** Pairwise alignment distribution is a regime fingerprint.
Interleaved CIFAR-10: ~42% positive / ~58% negative pairs with small
magnitudes (random-noise-dominated). Class-disjoint: 100% positive /
0% negative with strong cosines (0.80) (structure-dominated). The
sign distribution diagnoses the regime — a new diagnostic dimension.

**F9.** BoGrad-mode comparison: in highly-aligned-buffer regimes,
`negative` is the only safe mode. Disjoint CIFAR-10: BoGrad-neg +2.8pts;
BoGrad-full and BoGrad-positive both crash to chance accuracy (collapse
training because they remove the entire buffered direction, which is
also descent). Default `projection_mode="negative"` is empirically
correct.

**F10.** Drift-biased interference is BoGrad-incompatible. Synthetic
quadratic with constant bias on a coordinate subset: bias slows training
(P2 PASS, dist_opt 0.05 → 0.60), but no BoGrad mode recovers (full and
positive both worsen further; negative is no-op since 100% pos pairs).
Reason: drift bias is parallel to descent — removing it strips descent.
Motivated hypothesis (W').

**F11.** Multi-trial pairwise alignment fingerprints. SGD+momentum and
Adam under standard CIFAR-10 produce **clean 50/50 random-walk in
gradient buffer** over K=32 (within 0.5% of perfect balance, std ≤ 0.5%).
Vanilla SGD is 63/37. Update buffers tighten monotonically toward 100%
positive: vanilla 63% → momentum 66% → Adam 80% → BoGrad+vanilla 88%
→ BoGrad+mom 99% → BoGrad+Adam 100%. WW_K scales with this progression.

**F12.** Hypothesis (W') VALIDATED. Synthetic quadratic with bias on a
near-zero-eigenvalue subspace (perpendicular to descent): BoGrad full
and positive recover ~95% of the bias-induced damage (dist_inactive
9.48 → 3.32 vs clean baseline 3.06). BoGrad-neg does nothing
(100% pos pairs → skip). **Combined with F10:** orthogonalisation works
iff buffered direction is separable from descent. Pairwise alignment
fingerprint alone (100% pos in both F10 and F12) cannot distinguish —
need cos(g, g_full) to predict.

**F13.** Forgetting reduction without accuracy recovery. Permuted-classes
regime (sequential class blocks): all variants collapse to chance
accuracy (10%) because the schedule is too extreme. But BoGrad-neg
*does* reduce total OOB forgetting magnitude by 58% (1378 → 579) along
the trajectory. The framework detects what BoGrad does (reduce per-class
forgetting) even when the macro outcome (accuracy) doesn't improve.

**F14.** Two-types framework empirically validated on real CIFAR-10
(Stage 1, 2 epochs full data). IB_%neg in standard SGD+momentum
training: 70%. COSGD reduces it to 54% (mean cosine flipped −0.08 → +0.01).
BoGrad doesn't touch IB (~71% — confirming it targets a different
phenomenon) but improves WW_K (0.36 → 0.58, +62%) and accuracy (+5.7pts).
OOB is structurally 0 in iid batching. **Two methods, two metrics,
nearly orthogonal effects** — each addresses one interference type.

**F15.** MoGrad (orthogonalise-against-momentum-EMA-reference, not in
update) is a falsified hypothesis. Negative mode is no-op (~7%
projection rate, ties vanilla SGD). Full mode is catastrophic across
all start_steps, with monotonic damage scaling: start=50 → acc 0.099,
start=200 → 0.156, start=500 → 0.190 (vs SGD+mom 0.546). Cleanly
confirms F10/W': momentum reference becomes the descent direction;
removing it strips descent. The constructive corollary is
Complement-Aware Momentum (V2 from Phase 2): *amplify* the perpendicular
component instead of projecting out the parallel.

**F16.** Gradient clipping is inert on both interference axes
(`research/04_implicit_comparisons` 10-method run). clip(1.0) and
clip(5.0) reproduce baseline IB_%neg (0.69 vs 0.70), IB_⟨cos⟩ (−0.08),
and WW_K (0.36) to ≤±0.01. clip(1.0) loses 3.6 pts accuracy with no
compensating reduction in any interference metric. Confirms in
framework terms what `testing/01_attribution` showed in attribution
terms: **magnitude reduction alone does not constitute interference
reduction**.

**F17.** Dropout reduces between-batch interference monotonically with
dropout probability (`research/04_implicit_comparisons`). WW_K rises
0.361 → 0.429 → 0.507 as dropout 0 → 0.1 → 0.3, while IB_%neg stays
within ±0.03 of baseline. Dropout(0.3) reaches 87% of BoGrad's WW_K
but loses 13 pts of accuracy — **WW_K alone is not an
accuracy-predictor**; the manner of trajectory-efficiency improvement
matters. Mechanism: dropout decorrelates per-step gradients via random
unit masking. Targets the between-batch axis only.

**F18.** Adam moves the two interference axes in *opposite* directions
(`research/04_implicit_comparisons`). Adam (lr=1e-3) raises IB_%neg by
+0.08 (more inter-batch class conflict, *worse*) AND raises WW_K by
+0.09 (better trajectory efficiency, *better*). Plausible mechanism:
per-parameter adaptive scaling breaks per-class gradient alignment
symmetry. **Direct empirical confirmation that the two axes are
independently moveable** — the framework's separation hypothesis. In
the same run, BoGrad reproduces Stage 1 numbers exactly (acc 0.586,
IB_%neg 0.701, WW_K 0.582 vs Stage 1's 0.601, 0.707, 0.579), and COSGD
reproduces Stage 1 exactly too (acc 0.524, IB_%neg 0.550, WW_K 0.092
vs 0.487, 0.541, 0.092).

---

## 4. Methods

### BoGrad (`common/optimizers/BoGrad.py`)

**Production version.** Wraps any base PyTorch optimiser. Maintains a
buffer of past directions (gradients or applied updates), projects the
current direction against the buffer using sequential Gram-Schmidt
(the original BOSGD method, NOT QR — empirically QR was worse, see
phase-1 results).

Key parameters:
- `buffer_size` (K): number of past directions buffered.
- `project_stage`: `"gradient"` (project g_t before optimiser step) or
  `"update"` (project the applied delta after the step).
- `projection_mode`: `"full"` / `"negative"` (default — only subtract when
  cos < 0) / `"positive"` (only when cos > 0). F9 confirmed `negative`
  as the right default.
- `projection_scope`: per-tensor (default) or global.
- `preserve_magnitude`: optional rescale projected back to original norm.
- `random_projection`: control flag — replace buffer with random vectors.
- `buffer_dtype`, `projection_dtype`: dtype controls.

Optimal K per optimiser family (from sweep_update_K):
- SGD+momentum: 32 (peak +4.2pts same-LR, +0.9 peak-vs-peak)
- RMSprop+momentum: 16
- Adam: ≥128 (still rising at K=128, +6.0pts at default LR)
- SignSGD+momentum: 64

Legacy/reference implementation: `common/optimizers/BoGradScrutinized.py`.

### COSGD (`common/optimizers/COSGD.py`)

**Already-existing** per-class-within-batch orthogonalisation optimiser.
Brought into this repo from `C:\Masters Work\COSGD\COSGDAlgorithm\cosgd.py`.

Mechanism: for each batch, compute per-class gradients $g_c$ separately,
apply Gram-Schmidt orthogonalisation across them, sum the orthogonalised
gradients to form the update.

Non-standard step API: `optimizer.step(data, labels, unique_labels)` —
does its own forward+backward per class internally.

Key parameters:
- `orthogonalization_method`: `"gram_schmidt_normal"`,
  `"gram_schmidt_negative"`, `"modified_gs_normal"`,
  `"modified_gs_negative"`.
- `step_method`: `"single_forward"` (most efficient),
  `"multi_forward"`, `"multi_forward_with_BN"` (BN-stable).

Includes a built-in `COSGDTimer` for bottleneck analysis.

### Complement-Aware Momentum / ComplementMomentumSGD (`discoveryPhase2/enhanced_variants.py`)

The most promising future-work direction based on framework analysis.
**Doesn't fight momentum** — keeps the standard momentum step, *adds*
a γ-weighted boost in the direction perpendicular to previous velocity.

Mechanism:
```
v_prev = v_{t-1}
v_t = μ · v_{t-1} + g_t                   # standard momentum update
v̂ = v_prev / ‖v_prev‖
α = ⟨g_t, v̂⟩
g_perp = g_t − α · v̂
θ ← θ − η · (v_t + γ · g_perp)            # standard step + γ·perp boost
```

With γ=0 reduces to vanilla SGD+momentum. With γ>0, amplifies
information in the current gradient that is orthogonal to the running
momentum direction — the "new info momentum was about to dilute".

Variants for Adam, RMSprop, SignSGD also in `enhanced_variants.py`
(`ComplementAdam`, `ComplementRMSprop`, `ComplementSignSGD`). The Adam
version diverges with default settings; "Fix A" (decompose against m̂
in raw space rather than u in step space) helps but doesn't fully
resolve the geometry.

Empirical results from `discoveryPhase2/sweep_complement`:
- SGD+momentum: monotonic improvement with γ. **+5.7pts at γ=2** over
  baseline. Std shrinks from 0.029 to 0.012.
- RMSprop+momentum: inverted-U, peak at γ=0.25 (+4.3pts).
- Adam: divergence in original; Fix A doesn't resolve.
- SignSGD+momentum: small monotonic gain (+1.4pts at γ=2).

### MoGrad (`MoGrad/mograd.py`)

Tested idea: maintain a momentum-style EMA of past gradients, but
**don't use it in the update equation** — only as a reference vector
for orthogonalisation. After warmup, project the current gradient
against this reference before taking a vanilla SGD step.

Mechanism:
```
m_t = β · m_{t-1} + (1-β) · g_t          # reference EMA, NOT in update
if step > start_step:
    m̂ = m_t / ‖m_t‖
    α = ⟨g_t, m̂⟩
    if mode allows (sign-gated):
        g̃_t = g_t − α · m̂                # remove component along m_t
    else:
        g̃_t = g_t
θ ← θ − η · g̃_t                          # vanilla SGD on projected g
```

Falsified by F15. Documented as a clean negative result that supports
W'.

### Other variants in `discoveryPhase2/enhanced_variants.py`

- `AdaptiveTriggerBoGrad`: wraps a base optimiser, applies BoGrad-style
  projection only when cos(g_t, EMA(g)) < threshold. Phase-2 result:
  trigger fires too rarely to matter.
- `GradientDifferenceBoGrad`: buffers $g_t - g_{t-1}$ instead of $g_t$.
  Phase-2: doesn't help.
- `MultiScaleBoGrad`: dual buffer (short K=4 + long K=32). Phase-2:
  catastrophic on SGD+momentum, fine on SGD-vanilla.
- `SignSGD`: plain SignSGD baseline with optional momentum.
- `InPipelineRMSprop` / `InPipelineSignSGD`: project at the in-pipeline
  stage between preconditioning and momentum / before sign.
  Phase-2 in-pipeline sweep: Adam works at high K (≥128), RMSprop and
  SignSGD struggle.

### Discovery-phase Adam variant (`discovery/bograd_variants.py`)

`AdamWithBoGrad`: Adam with BoGrad projection at configurable point
(`grad`, `momentum`, `natural`, `update`). Phase-1 ablation showed
update-stage works best for Adam.

---

## 5. The framework module (`common/diagnostics/`)

### `interference.py` — main library

Three sub-trackers + a top-level wrapper:

**`InterferenceTracker`** — top-level. Lifecycle:
```
tracker = InterferenceTracker(model, criterion, probe_set, log_every=1, ...)
for x, y in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y); loss.backward()
    tracker.before_step()                      # snapshot params + grads
    optimizer.step()
    batch_classes = set(y.unique().cpu().tolist())
    tracker.after_step(loss=loss.item(), batch_classes=batch_classes)

history = tracker.get_history()
summary = tracker.summary()
```

Captures per logged step:
- `g_norm`, `u_norm` (gradient and applied update magnitudes)
- `cos_g_prev`, `cos_u_prev` — lag-1 alignment
- `cos_u_neg_g` — descent quality
- Trajectory lags `cos_u_lag1`, `cos_u_lag4`, `cos_u_lag16`
- Pairwise stats (gradient and update buffer) via `PairwiseAlignmentTracker`
- Per-class probe loss/accuracy via `ClassProbeSet`
- Per-class forgetting (and OOB partition when `batch_classes` provided)
  via `ForgettingTracker`
- Wasted-work ratio over `K`-window via `WastedWorkTracker`
- (Optional) full-batch reference gradient via `cos(g_t, g_t^{full})` if
  `full_batch_loader` provided

**`ForgettingTracker`**:
- Tracks per-class forgetting events: $\Delta_{t,c} = \max(0, L_c(\theta_t) - L_c(\theta_{t-1}))$
- When `batch_classes` is supplied, partitions into in-batch
  (ambiguous source) and out-of-batch (clean cross-class interference signal)
- Run-level summary exposes both totals.

**`WastedWorkTracker`**:
- $\mathrm{WW}_K(t) = \lVert \theta_t - \theta_{t-K}\rVert / \sum_{i=t-K+1}^t \lVert u_i\rVert$
- Updated EVERY step (cheap), reported on logged steps. (Was buggy
  pre-fix when only updated on logged steps.)

**`PairwiseAlignmentTracker`**:
- For each logged step, computes $\cos(x_t, x_{t-k})$ for all
  $k = 1..K$ against a buffer
- Aggregates: `frac_positive`, `frac_negative`, `mean_positive_cos`,
  `mean_negative_cos`, `min_cos`, `max_cos`
- Tracks both gradient and update buffers separately

### `per_class_probe.py`

`ClassProbeSet` — held-out balanced probe, GPU-resident, single-fwd-pass
eval. Default 64 examples per class. Per-example loss is clamped at 50
(prevents astronomical losses in CL-style regimes from corrupting the
forgetting-magnitude metric).

### Run-level normalised metrics in summary

- `forgetting_per_unit_step_run` = $\sum \Delta / \sum \lVert u\rVert$
- `forgetting_oob_per_unit_step` = OOB version of above
- `forgetting_per_progress` = $\sum \Delta / \lvert L_{\text{init}} - L_{\text{final}}\rvert$
- `forgetting_oob_per_progress` = OOB version
- `net_displacement` = $\lVert \theta_{\text{end}} - \theta_{\text{start}}\rVert$
- `cum_step_norm`, `cum_forgetting_magnitude`,
  `cum_forgetting_oob_magnitude`, `loss_progress`

---

## 6. Stage 1 — meeting headline numbers

CIFAR-10 / SmallCNN / lr=0.05 / momentum=0.9 / batch=128 / 2 epochs / 1 trial.

| Method | final_acc | IB_%neg | IB_⟨cos⟩ | WW_K |
|---|---|---|---|---|
| SGD+momentum baseline | 0.544 | 70% | −0.08 | 0.36 |
| + BoGrad (upd-K=32 neg) | **0.601** *(+5.7)* | 71% | −0.09 | **0.58** *(+62%)* |
| COSGD (mod-GS, single-fwd) | 0.487 | **54%** *(−16)* | **+0.01** *(flipped)* | 0.09 |

OOB structurally 0 across all (every batch contains all classes in
standard interleaved batching).

**Reading:** Two methods, two metrics, near-orthogonal effects. BoGrad
addresses between-batch trajectory (WW_K up); COSGD addresses
inter-batch class conflict (IB_%neg down, mean cosine flips sign).
Each method moves only its target metric.

Files: `stage1/MEETING_SUMMARY.md`,
`stage1/results/run_20260507_105741/`.

---

## 7. MoGrad — confirmed negative result

CIFAR-10 / 2 epochs full data / 2 trials.

| Config | acc | proj_rate | Δ vs SGD+mom |
|---|---|---|---|
| sgd_vanilla | 0.346 ± 0.002 | — | −0.200 |
| sgd_momentum | 0.546 ± 0.010 | — | 0.000 |
| mograd_negative_start50 | 0.347 ± 0.003 | 6.7% | −0.200 |
| mograd_negative_start200 | 0.347 ± 0.003 | 7.8% | −0.200 |
| mograd_negative_start500 | 0.346 ± 0.003 | 7.1% | −0.200 |
| mograd_full_start50 | 0.099 ± 0.001 | 100% | −0.447 |
| mograd_full_start200 | 0.156 ± 0.030 | 100% | −0.391 |
| mograd_full_start500 | 0.190 ± 0.004 | 100% | −0.357 |

**Negative mode**: rarely fires (~7%), no effect on accuracy across all
start_steps. Equivalent to vanilla SGD.

**Full mode**: catastrophic. Damage scales monotonically with start_step
(later start = less damage). The pattern itself is a clean framework
signature — confirming F10/W'.

Files: `MoGrad/mograd.py`, `MoGrad/run_study.py`,
`MoGrad/results/run_20260507_112138/` (longer run),
`MoGrad/results/run_20260507_105751/` (smoke test).

---

## 8. Folder map

### Root-level docs
- `README.md` — repo entry point
- `CLAUDE.md` — AI-assistant guardrails / where-to-read-first
- `TESTS_OVERVIEW.md` — running findings index (F1–F15)
- `HANDOFF.md` — this file
- `thesis_experiment_plan.md` — original thesis chapter plan (now somewhat
  superseded by current research direction)

### `common/`
- `optimizers/BoGrad.py` — production BoGrad
- `optimizers/BoGradScrutinized.py` — legacy reference
- `optimizers/COSGD.py` — COSGD (copied from external repo)
- `optimizers/__init__.py` — exports `BoGrad`, `COSGD`
- `diagnostics/interference.py` — `InterferenceTracker`,
  `ForgettingTracker`, `WastedWorkTracker`, `PairwiseAlignmentTracker`
- `diagnostics/per_class_probe.py` — `ClassProbeSet` (with loss-cap)

### `testing/` — tactical tests for specific BoGrad questions
- `_common.py` — shared model/data/loop helpers (SmallCNN, ResNet8,
  ResNet18CIFAR, train_run, evaluate, aggregate)
- `01_attribution/` ✅ done. F: direction matters, magnitude doesn't.
  Random-projection control failed; rescaling doesn't change accuracy.
- `02_rescaling/` — skipped (Test 01 settled it)
- `03_optimizations/` — queued (speed/memory ablation)
- `04_inpipeline_diagnosis/` — queued (why in-pipeline projection works
  for Adam but not RMSprop/SignSGD)
- `05_2x2_grid/` — queued (final headline ablation)

### `research/01_interference_framework/` — Stream 1 research artefacts
- `framework.md` — formal definitions, motivation, all findings
  documented in §8 (F1–F13 with full tables and reasoning)
- `baseline_reference.py` — 4-way SGD/mom/BoGrad with full diagnostics
- `synthetic_quadratic.py` — controlled cross-step interference
  (zero-mean alternating); produced F5
- `synthetic_quadratic_biased.py` — drift/rotating/decaying biased
  interference; produced F10
- `synthetic_quadratic_perpendicular.py` — perpendicular bias
  ((W') validation); produced F12
- `synthetic_class_disjoint.py` — class-disjoint vs interleaved
  mini-batches; produced F6, F7, F8, F9
- `synthetic_permuted_classes.py` — sequential class blocks; produced F13
- `pairwise_alignment_study.py` — multi-trial standard CIFAR-10
  fingerprinting; produced F11
- `results/` — run output JSONs

### `research/02_bograd_scrutiny/`, `research/03_cosgd_scrutiny/` — empty (queued)

### `research/04_implicit_comparisons/` ✅ first 10-method run complete
- `methods.md` — experiment description with hypotheses
- `run_comparison.py` — runner: SGD-vanilla, SGD+mom, +Dropout(0.1), +Dropout(0.3), +clip(1.0), +clip(5.0), Adam, Adam@lr=0.05, BoGrad, COSGD
- `analyze.py` — plots and summary table generator
- `findings.md` — full write-up of F16, F17, F18 with the two-axis story
- `results/run_20260507_133221/` — single-trial 2-epoch run, all 10 methods, with axes_scatter / acc_summary / ib_axis / ww_axis / ib_over_time / trajectory_cosines / pairwise_positive plots

### `research/05_batch_size_sensitivity/` ⏳ scaffolded (not run)
- `methods.md` — design and predictions
- `run_sensitivity.py` — sweeps batch ∈ {16, 32, 64, 128, 256, 512} with linear LR scaling
- `analyze.py` — plots and summary
- `results/` — empty

### `discovery/` — Phase 1 historical
- `bograd_variants.py` — `AdamWithBoGrad` (Phase 1 Adam projection-point
  variant)
- `ablation_cifar10.py` — Phase 1 runner
- `diagnostics.py` — early diagnostic harness (superseded by
  `common/diagnostics/`)
- `results/` — Phase 1 result data

### `discoveryPhase2/` — Phase 2 historical (sweep work)
- `enhanced_variants.py` — 10 custom optimisers (Complement variants,
  AdaptiveTrigger, GradientDifference, MultiScale, SignSGD,
  InPipeline RMSprop/SignSGD)
- `ablation_phase2.py` — Phase 2 runner
- `sweep_complement.py` — γ sweep producing the +5.7pt Complement-Aware
  Momentum result on SGD+mom
- `sweep_update_K.py` — K sweep producing per-optimiser optimal K curves
- `sweep_in_pipeline_K.py` — in-pipeline K sweep (Adam works at K=128;
  RMSprop/SignSGD struggle)
- `results/` — Phase 2 result data (6 sweep folders)

### `stage1/` — meeting prep
- `inter_batch_metric.py` — `measure_inter_batch_interference()`
- `run_comparison.py` — three-method comparison runner
- `MEETING_SUMMARY.md` — meeting handout
- `README.md`
- `results/run_20260507_105741/` — Stage 1 numbers (F14)

### `MoGrad/` — momentum-orthogonalisation experiment
- `mograd.py` — `MoGrad` optimiser
- `run_study.py` — multi-trial sweep runner
- `README.md` — algorithm explanation
- `results/run_20260507_112138/` — F15 confirmation run (2 epochs)
- `results/run_20260507_105751/` — initial smoke test

### `docs/`
- `experiment_design.md` — folder/logging/checkpointing conventions
- `results_schema.md` — JSON contract (provisional, not always followed
  by research scripts)

### `data/`
- CIFAR-10 raw data (gitignored; downloaded by torchvision)

---

## 9. Key empirical numbers for quick reference

### BoGrad on standard CIFAR-10 (SGD+momentum, lr=0.05, mu=0.9)

- Same-LR: +5.7pts (`testing/01_attribution`, Stage 1)
- Peak-vs-peak (LR-tuned): ~+1pt (`testing/01_attribution`)
- Per-optimiser optimal K for update-stage projection:
  SGD+mom: 32 (peak +4.2pts same-LR)
  RMSprop: 16
  Adam: ≥128 (+6pts at K=128)
  SignSGD+mom: 64

### COSGD on standard CIFAR-10

- Reduces IB_%neg from 70% → 54%
- Mean IB cosine flips −0.08 → +0.01
- WW_K collapses 0.36 → 0.09 (trade: intra-batch alignment ↑, inter-batch
  trajectory ↓)
- Final accuracy: slightly below baseline at lr=0.05 (0.487 vs 0.544)

### Complement-Aware Momentum on SGD+mom

- γ=0 (baseline): 0.604
- γ=2: 0.661 (+5.7pts, monotonic with γ)

### MoGrad on SGD baseline

- Negative mode: ties vanilla SGD (~0.347)
- Full mode: 0.099-0.190 depending on start_step (catastrophic)
- Never approaches SGD+mom (0.546)

### Interference fingerprints under standard CIFAR-10

- IB_%neg under SGD+momentum: 70% (intra-batch class conflict severe)
- Pairwise-K=32 gradient buffer:
  - vanilla SGD: 63% positive (some structure)
  - SGD+momentum: **50/50** (random walk, std ≤ 0.5%)
  - Adam: **50/50** (random walk, std ≤ 0.3%)
- Pairwise-K=32 update buffer:
  - vanilla: 63% pos
  - SGD+mom: 66% pos
  - Adam: 80% pos
  - BoGrad+SGD-mom: 99% pos
  - BoGrad+Adam: 100% pos
- WW_K=32:
  - vanilla SGD: 0.29
  - SGD+mom: 0.40
  - Adam: 0.50
  - BoGrad+vanilla: 0.52
  - BoGrad+mom: 0.61
  - BoGrad+Adam: 0.70

---

## 10. Open / pending work

(Listed for orientation, not as a to-do.)

- **Implicit method comparison** (Section 4 of the eventual paper):
  10-method run completed (`research/04_implicit_comparisons/`).
  Findings F16–F18 extracted. Still pending: gradDrop variant, multi-trial
  confirmation, dropout-LR sweep, Complement-Aware Momentum line in
  the same harness.
- **`testing/05_2x2_grid`**: full momentum × BoGrad ablation per
  optimiser. Code exists, not run.
- **`testing/04_inpipeline_diagnosis`** (under framework lens): why
  in-pipeline projection works for Adam but not RMSprop/SignSGD.
  Phase-2 sweep produced the question; framework not applied to it yet.
- **(W'') predictive criterion**: does cos(g_t, g_full) predict per-step
  BoGrad effect (high cos = parallel = hurts; low = perpendicular =
  helps)? Theoretical result that follows from F10/F12; not tested.
- **Statistical baseline** (§7.3 of framework.md): noise-floor under
  Gaussian-noise null model.
- **Per-progress OOB metric** (added but not used in any experiment yet).
- **Multi-trial Stage 1 run** (current is single-trial).
- **Batch-size sensitivity** scaffold built (`research/05_batch_size_sensitivity/`),
  not yet run. Will reveal whether IB_%neg / WW_K respond as predicted to
  batch-size scan over {16, 32, 64, 128, 256, 512}.
- **Complement-Aware Momentum unified treatment** across all four
  optimiser families (Adam version doesn't work with current
  formulations).

---

## 11. Plan files

The active plan describing the research direction:
`C:\Users\rayde\.claude\plans\on-the-questions-1-tidy-goblet.md`

This was the structural plan that proposed the four research streams
(interference framework, BoGrad scrutiny, COSGD scrutiny, implicit
comparison). The interference framework stream (Stream 1) is largely
complete (F1–F15 documented); other streams haven't started.

---

## 12. Conventions in use

- All experiments produce JSON outputs under `<test>/results/run_<timestamp>/`
- Per-trial files under `per_trial/<config>__t<trial>.json`
- Console summary tables include mean ± std across trials
- `--quick` flag uses 1/4 of CIFAR-10 train set
- Paired data ordering: same `generator_seed` per trial across all
  variants within a trial (so all configs see the same shuffled batches)
- Default seeds: `base_seed=2026`, trial seeds `base_seed + trial * 1000`
- Logging cadences: `log_every=1` for full step-by-step diagnostics,
  `log_every=10` for cheaper runs; `probe_every` controls the more
  expensive per-class evaluation
- Loss values clamped at 50 in `ClassProbeSet` (prevents CE overflow
  in continual-learning regimes)

---

## 13. Findings register, abbreviated

| ID | One-line statement |
|---|---|
| F1 | Momentum induces structural lag-1 gradient correlation. |
| F2 | WW_K ranks methods by trajectory efficiency. |
| F3 | Optimisers have distinct trajectory-lag decay profiles. |
| F4 | Per-class forgetting magnitude is step-magnitude-confounded. |
| F5 | Detection ≠ slowdown (zero-mean noise is detected but doesn't slow training). |
| F6 | Per-class total forgetting magnitude is dominated by within-class noise. |
| F7 | Out-of-batch forgetting cleanly isolates cross-class interference. |
| F8 | Pairwise alignment distribution is a regime fingerprint. |
| F9 | BoGrad-negative is the empirically correct default mode. |
| F10 | Drift bias (parallel to descent) is BoGrad-incompatible. |
| F11 | Multi-trial: SGD+mom and Adam show 50/50 random-walk gradient buffer. |
| F12 | Perpendicular bias is BoGrad-recoverable; (W') validated. |
| F13 | BoGrad reduces forgetting along trajectory even when accuracy can't recover. |
| F14 | Two-types interference framework empirically validated on CIFAR-10. |
| F15 | MoGrad is falsified; confirms F10/W'; Complement-Aware Momentum is the constructive corollary. |
| F16 | Gradient clipping is inert on both interference axes (only changes magnitude). |
| F17 | Dropout reduces between-batch interference monotonically with dropout probability; doesn't move inter-batch axis. |
| F18 | Adam moves IB and WW axes in opposite directions — empirical confirmation of axis-independence. |
