# Tests Overview — what's been done, what's queued

A single-page navigator across all experiments and findings. Read this when
you want to know "what tests exist, what did each one show, where is the
data". Linked from every README in the repo.

The work is organised into three layers, with different purposes:

| Layer | Purpose | When it runs | Lifetime |
|---|---|---|---|
| **`research/`** | Build understanding (what is interference? does it slow training? do our tools detect it?) | Now | Ongoing — informs the thesis chapters |
| **`testing/`** | Tactical tests answering specific implementation questions about BoGrad | Now / soon | Once-off — produces results, then chapter-ready |
| **`experiments/`** | The thesis chapter experiments (Ch3, Ch4, Ch5 from `thesis_experiment_plan.md`) | After research+testing complete | Final |
| **`discoveryPhase2/`** | Historical scratch-pad sweeps from earlier iterations | Already done | Reference only |

The current focus is `research/01_interference_framework/`. Everything in
`testing/` is queued behind it.

---

## Quick map of what's been done

### Settled findings

| Finding | Source | Statement |
|---|---|---|
| **D1** | `testing/01_attribution` | BoGrad's accuracy gain is direction-attributable, not magnitude. Random projection control fails to match; magnitude rescaling doesn't change the result. |
| **D2** | `testing/01_attribution` | BoGrad's optimal LR is lower than baseline's. Same-LR comparisons inflate the apparent gap. |
| **D3** | `discoveryPhase2/sweep_update_K` | For each base optimiser, optimal $K$ for update-stage projection differs: SGD+mom: 32, RMSprop: 16, Adam: $\geq128$, SignSGD+mom: 64. |
| **D4** | `discoveryPhase2/sweep_complement` | Complement-aware momentum (V2) gives $+5.7$pts on SGD+momentum at $\gamma=2$. Diverges on Adam. |
| **F1** | `research/01/baseline_reference` | Momentum induces structural gradient correlation: $\cos(g_t, g_{t-1})$ is $-0.04$ for vanilla SGD but $+0.26$ for SGD+momentum. |
| **F2** | `research/01/baseline_reference` | Wasted-work ratio $\mathrm{WW}_K$ ranks methods by trajectory efficiency — vanilla < momentum < BoGrad+vanilla < BoGrad+momentum. |
| **F3** | `research/01/baseline_reference` | Optimisers have distinct trajectory-lag-decay profiles. Momentum: short-horizon. BoGrad: persistent across horizons. They compose multiplicatively. |
| **F4** | `research/01/baseline_reference` | Per-class forgetting magnitude is confounded with step magnitude. Methods with bigger steps produce more forgetting in absolute terms regardless of whether they help or hurt. |
| **F5** | `research/01/synthetic_quadratic` | Detection ≠ slowdown. Zero-mean alternating noise is detected by the metrics but doesn't actually slow SGD (averaging absorbs it). BoGrad applied here actively *hurts* by stripping useful averageable signal. |
| **F6** | `research/01/synthetic_class_disjoint` | Per-class forgetting magnitude as written cannot detect biased class-level interference because it is dominated by within-class loss noise. Trajectory metric ($\mathrm{WW}_K$) tracks accuracy correctly where forgetting magnitude doesn't. |
| **F7** | `research/01/synthetic_class_disjoint` (with OOB) | Out-of-batch forgetting metric (loss regression on classes not in the batch) cleanly isolates cross-class interference. Disjoint regime: 0.552. Interleaved: 0.000 (by construction). The clean signal F6 said was hidden is now visible. |
| **F8** | `research/01/synthetic_class_disjoint` (with pairwise tracker) | Pairwise alignment statistics over the K-buffer reveal regime structure that lag-1-only metrics miss. Interleaved: ~42% positive / ~58% negative pairs with small magnitudes. Disjoint: 100% positive / 0% negative with strong cosines (0.80). The sign distribution is itself a regime fingerprint. |
| **F9** | `research/01/synthetic_class_disjoint` (mode comparison) | BoGrad-mode comparison on disjoint regime: `negative` mode is the only one that helps (+2.8pts). `Full` and `positive` collapse training to chance accuracy because they remove the buffered direction entirely (which is also the descent direction in this 100%-positive-aligned regime). Default `projection_mode="negative"` is empirically correct; `full`/`positive` are too aggressive when buffered directions are highly aligned. |
| **F10** | `research/01/synthetic_quadratic_biased` (drift mode) | Drift-biased interference (constant pull) slows training (P2 PASS) but is BoGrad-incompatible: no projection mode recovers convergence. The bias is aligned with descent, so `frac_positive` → 100% and removing the buffered direction strips the descent signal. **Refined hypothesis (W'): orthogonalisation helps only when interference is separable from descent — when `frac_negative > 0` or buffered directions are mixed-sign.** |
| **F11** | `research/01/pairwise_alignment_study` (3 trials × full CIFAR-10) | Multi-trial pairwise alignment fingerprints across optimisers under standard batching. **SGD+momentum and Adam produce a clean 50/50 random-walk fingerprint in the gradient buffer over K=32** (within 0.5% of perfect balance), while vanilla SGD is 63/37. Update buffers monotonically tighten toward 100% positive alignment as we go SGD-vanilla → momentum → Adam → BoGrad. WW_K tracks the alignment progression. **The pairwise alignment distribution is a stable regime-specific fingerprint with very tight cross-trial std.** |
| **F12** | `research/01/synthetic_quadratic_perpendicular` (W' validation) | **Hypothesis (W') VALIDATED.** When bias is perpendicular to descent (placed on a low-eigenvalue inactive subspace), BoGrad full and positive modes recover ~95% of the bias-induced damage (dist_inactive: 9.48 → 3.32, vs clean baseline 3.06). BoGrad neg does nothing (bias makes all pairs positive-aligned, neg-mode skips). Combined with F10 (BoGrad fails on parallel drift bias), the picture is: orthogonalisation works iff buffered direction is separable from descent. Pairwise alignment fingerprint alone (100% pos in both cases) cannot distinguish — need cos(g, g_full) to predict. |
| **F14** | `stage1/run_comparison` (real CIFAR-10, 2 epochs, 3-method comparison) | **Two-types-of-interference framework empirically validated.** IB_%neg metric (intra-batch class-pair gradient conflict) shows 70% of class-pair gradients within a CIFAR-10 batch are anti-aligned. **COSGD reduces this to 54%** with mean cosine flipped from −0.08 to +0.01. **BoGrad does not touch it** (~71% — confirming it targets a different phenomenon) but improves WW_K (0.36 → 0.58, +62%) and accuracy (+5.7pts). OOB forgetting is structurally 0 in standard interleaved batching. Two methods, two metrics, near-orthogonal effects — confirming each addresses a different interference type. |
| **F15** | `MoGrad/run_study` (CIFAR-10, 2 epochs full data, multi-trial) | **MoGrad (orthogonalise-against-momentum-reference) is a falsified hypothesis.** Negative mode is a no-op (~7% projection rate, accuracy identical to vanilla SGD across all start_steps). Full mode is catastrophic (0.099–0.190 vs SGD-momentum 0.546), with **monotonic improvement as start_step is delayed** (start=50→0.099, start=500→0.190). Neither approaches SGD+momentum. **Cleanly confirms F10/W' prediction**: the momentum reference becomes the descent direction; removing it (full mode) strips descent; gating on disagreement (negative mode) rarely fires. The constructive alternative — **Complement-Aware Momentum** (V2 from phase 2: *amplify* the perpendicular component instead of projecting out the parallel) — is the framework-predicted right move; +5.7pts on SGD+momentum at γ=2. |
| **F13** | `research/01/synthetic_permuted_classes` (sequential class blocks) | In an extreme continual-learning-style schedule (sequential class blocks), all four block-trained variants collapse to chance accuracy (10%). But BoGrad-negative still measurably reduces total OOB forgetting magnitude by 58% (1378 → 579) along the trajectory. **The framework detects what BoGrad does (reduce per-class forgetting) even when the macro outcome (accuracy) doesn't improve.** Disambiguates the micro-effect from the macro-question; useful for thesis claims. |
| **F16** | `research/04_implicit_comparisons/run_comparison` (10-method real-CIFAR-10 single-trial) | **Gradient clipping is inert on both interference axes.** clip(1.0) and clip(5.0) reproduce baseline IB_%neg (0.69 vs 0.70), IB_⟨cos⟩ (-0.08), and WW_K (0.36) to ≤±0.01. Clip(1.0) loses 3.6 pts accuracy with no compensating reduction in any interference metric. Confirms in framework terms what `testing/01_attribution` showed in attribution terms: **magnitude reduction alone does not constitute interference reduction.** |
| **F17** | `research/04_implicit_comparisons/run_comparison` | **Dropout reduces between-batch interference monotonically with dropout probability.** WW_K rises 0.361 → 0.429 → 0.507 as dropout 0 → 0.1 → 0.3, while IB_%neg stays within ±0.03 of baseline. Dropout(0.3) reaches 87% of BoGrad's WW_K, but loses 13 pts of accuracy — **WW_K alone is not an accuracy-predictor**; the manner of trajectory-efficiency improvement matters. Mechanism: dropout decorrelates per-step gradients via random unit masking, reducing sequential cancellation. Targets the between-batch axis only. |
| **F18** | `research/04_implicit_comparisons/run_comparison` | **Adam is the only tested method that moves the two interference axes in opposite directions.** Adam (lr=1e-3) raises IB_%neg by +0.08 (more inter-batch class conflict, worse) AND raises WW_K by +0.09 (better trajectory efficiency). Plausible mechanism: per-parameter adaptive scaling breaks per-class gradient alignment symmetry. **Direct empirical confirmation that the two axes are independently moveable** — the framework's separation hypothesis. BoGrad reproduces Stage 1 numbers exactly under the same paired-data harness (acc 0.586, IB_%neg 0.701, WW_K 0.582 vs Stage 1's 0.601, 0.707, 0.579). |

### Open questions tied to specific tests

| Question | Test that would answer it |
|---|---|
| Does $\mathrm{WW}_K$ predict end-of-training accuracy? | Multi-trial reference run (queued) |
| Does Hypothesis (W) hold under biased / non-cancelling interference? | `synthetic_quadratic_biased.py` (queued) |
| Does forgetting accumulate during sequential class blocks? | `synthetic_permuted_classes.py` (queued) |
| Why does in-pipeline projection work for Adam but not RMSprop / SignSGD? | `testing/04_inpipeline_diagnosis` (queued) |
| What's the 2×2 picture of momentum × BoGrad per optimiser? | `testing/05_2x2_grid` (queued) |

---

## The framework — what it is, what it measures

The framework is `common/diagnostics/interference.py`. It instruments any
training loop with metrics targeting three independent axes of "interference":

| Axis | What it measures | Headline metric |
|---|---|---|
| **Geometric** | Are successive updates pulling against each other? | $\cos(u_t, u_{t-1})$ for lag-1; **PairwiseAlignmentTracker** for full $K$-buffer distribution (introduced in F8) |
| **Per-class** | Is any class's accuracy being damaged by an update? | $\Delta^{\text{OOB}}_t$ — out-of-batch forgetting (introduced in F7) |
| **Trajectory** | Are we making efficient progress through parameter space? | $\mathrm{WW}_K$ — wasted-work ratio over $K$-step window |

**The PairwiseAlignmentTracker** is the new richer geometric metric (added F8).
For each step it computes $\cos(x_t, x_{t-k})$ for all $k = 1..K$ and reports:

- `frac_positive` / `frac_negative`: fraction of K pairs that are aligned vs anti-aligned
- `mean_positive_cos` / `mean_negative_cos`: average cosine of each subset
- `max_cos` / `min_cos`: distributional extremes

Across both gradient and update buffers. This generalises the lag-1-only
$\rho^g, \rho^u$ to the full $K$-window — exactly the window BoGrad
orthogonalises against. The sign distribution turns out to be a regime
fingerprint (F8) and explains *why* `projection_mode="negative"` is the
right default in practice (F9): it conditions the projection on what the
buffer geometry actually looks like.

Each is captured at every step (when `log_every=1`) and aggregated into a
run-level summary. Definitions, motivation, and limitations are in
[`research/01_interference_framework/framework.md`](research/01_interference_framework/framework.md).

The framework's verdict on which axis matters most evolves with each finding:

- After F2: trajectory looks promising.
- After F4: per-class is broken (step-magnitude confound).
- After F6: per-class total forgetting is *worse than broken* — it actively
  misranks methods.
- After F7: out-of-batch per-class forgetting works as intended. Trajectory +
  out-of-batch per-class are both reliable; geometric is the noisiest.

---

## Detailed inventory

### `research/01_interference_framework/`

The current working directory. Everything here is investigative.

#### Validation reference run — `baseline_reference.py`

**What it does.** Runs SmallCNN/CIFAR-10 for 1 epoch with full diagnostics
on four configurations: vanilla SGD, SGD+momentum, BoGrad+vanilla
(gradient-stage K=8), BoGrad+SGD+momentum (update-stage K=32). The most
informative single training run we have.

**Why it matters.** The diagnostics for these four runs are the canonical
reference profile — every subsequent change to the framework is judged by
whether the reference run continues to produce sane numbers and whether
the new change adds value beyond what's already there.

**Findings produced.** F1, F2, F3, F4 (described above).

**Run.**
```bash
python research/01_interference_framework/baseline_reference.py --include-bograd --log-every 1 --probe-every 1
```

**Most recent results.** `results/baseline_reference/run_20260504_154438`

#### Synthetic quadratic — `synthetic_quadratic.py`

**What it does.** Replaces the neural-network setup with a simple quadratic
loss $L(\theta) = \tfrac{1}{2}\theta^\top H \theta - b^\top \theta$ where
we inject controlled cross-step interference. Three variants run: clean,
with interference, with interference + BoGrad. Tests whether the framework
metrics detect injected structure AND whether the detection corresponds to
training slowdown.

**Why it matters.** The quadratic setting lets us inject interference of
known structure and verify the framework can detect it. It's the *causal*
test missing from real-data runs (where everything is correlational).

**Findings produced.** F5: detection ≠ slowdown. The metrics correctly
detect zero-mean alternating noise, but the noise doesn't actually slow
training, and BoGrad applied to it hurts. **A useful negative result** that
sharpened the framework's interpretation.

**Run.**
```bash
python research/01_interference_framework/synthetic_quadratic.py
```

**Most recent results.** `results/synthetic_quadratic/run_20260504_165513`

#### Synthetic class-disjoint — `synthetic_class_disjoint.py`

**What it does.** Trains SmallCNN/CIFAR-10 with two batch regimes:
*interleaved* (normal random-shuffle) and *disjoint* (each batch contains
exactly one class, classes cycled $0 \to 1 \to \ldots \to 9$). With and
without BoGrad. Tests whether biased per-class interference produces
detectable forgetting and whether orthogonalisation recovers some of the
slowdown.

**Why it matters.** The most direct empirical test of Hypothesis (W) we
have so far. Disjoint training is a continual-learning-style stress case
within a single dataset.

**Findings produced.** F6 (per-class forgetting magnitude is dominated by
within-class noise), F7 (out-of-batch forgetting cleanly isolates
cross-class interference), F8 (pairwise alignment reveals regime
structure), F9 (BoGrad-negative mode is empirically correct; full/positive
collapse training when buffered directions are highly aligned).

**Run.**
```bash
python research/01_interference_framework/synthetic_class_disjoint.py --epochs 2
python research/01_interference_framework/synthetic_class_disjoint.py --quick --epochs 1 --n-batches-per-epoch 100
```

**Most recent results.** `results/synthetic_class_disjoint/run_20260505_172015`

#### Synthetic biased quadratic — `synthetic_quadratic_biased.py`

**What it does.** Like `synthetic_quadratic.py` but with *biased* (non-cancelling)
interference: drift (constant pull), rotating (slow rotation), or decaying
(initial bias that fades). Tests Hypothesis (W) properly — the previous
quadratic had zero-mean interference that SGD averaged out (F5), masking
whether (W) holds.

**Why it matters.** This is the controlled test of "does orthogonalisation
fix biased interference" — a question the F5 setup couldn't answer because
its interference was averageable.

**Findings produced.** F10: drift-biased interference is BoGrad-incompatible.
Drift produces 100% positive pairwise alignment with strong cosine; removing
the buffered direction also removes the descent signal. **Refined hypothesis
(W')**: orthogonalisation helps only when interference is separable from
descent.

**Run.**
```bash
python research/01_interference_framework/synthetic_quadratic_biased.py
python research/01_interference_framework/synthetic_quadratic_biased.py --bias-mode rotating --bias-period 50
python research/01_interference_framework/synthetic_quadratic_biased.py --bias-mode decaying
```

**Most recent results.** `results/synthetic_quadratic_biased/run_20260505_171731`

### `research/04_implicit_comparisons/` ✅ FIRST RUN COMPLETE

**Question.** When measured on the framework's two-axis decomposition, what
do *implicit* interference reducers (dropout, gradient clipping, momentum,
Adam) actually do? Do any of them substitute for BoGrad or COSGD?

**Method.** Same protocol as Stage 1, broader method set. 10 variants:
sgd_vanilla, sgd_momentum, sgd_mom_dropout(0.1), sgd_mom_dropout(0.3),
sgd_mom_clip(1.0), sgd_mom_clip(5.0), Adam(1e-3), Adam(0.05), BoGrad
(reference), COSGD (reference). Single trial, paired data ordering, 2
epochs CIFAR-10.

**Findings produced.** F16 (clip is inert), F17 (dropout reduces
between-batch interference but trades accuracy), F18 (Adam moves the two
axes in opposite directions; reproduces Stage 1 BoGrad/COSGD).

**Run.**
```bash
python research/04_implicit_comparisons/run_comparison.py --epochs 2
python research/04_implicit_comparisons/analyze.py
```

**Most recent results.** `research/04_implicit_comparisons/results/run_20260507_133221/`
Findings written to `research/04_implicit_comparisons/findings.md`.

### `research/05_batch_size_sensitivity/` ⏳ SCAFFOLDED

**Question.** Does inter-batch interference (IB_%neg) shrink or grow with
batch size? Does between-batch trajectory efficiency (WW_K) follow the
expected noise-vs-progress curve?

**Method.** Single optimiser (SGD+momentum), batch sizes
{16, 32, 64, 128, 256, 512}, linear LR scaling.

**Status.** Code complete, not yet run.

### `testing/` — tactical tests

These are scoped tests answering specific BoGrad-implementation questions.
They produce results that go directly into thesis claims rather than
informing methodology.

#### `01_attribution` ✅ DONE

**Question.** Is BoGrad's accuracy gain from better step direction or just
implicit step-size reduction (a hidden LR cut)?

**Method.** 4-way comparison: baseline, BoGrad, BoGrad-with-rescaling
(direction only), BoGrad-with-random-projection (magnitude only). LR sweep
within each.

**Result.** Direction matters. Magnitude reduction in BoGrad is inert
(D1, D2 above). Settled.

#### `02_rescaling` ⏭ SKIPPED

Per Test 01 result, magnitude doesn't matter — rescaling ablation is
unnecessary. Marked skipped.

#### `03_optimizations` ⏳ QUEUED

**Question.** Which speed/memory optimisations to BoGrad's projection
should become defaults? How does the overhead scale with model size?

**Method.** Microbenchmark on SmallCNN; iso-accuracy comparison of
sync-removed, fp16-buffer, global-scope variants. Then test the best on
SmallCNN, ResNet-8, ResNet-18.

**Status.** Queued. Needs to run before any large-architecture experiment.

#### `04_inpipeline_diagnosis` ⏳ QUEUED — to be reframed

**Original question.** Why does in-pipeline projection work for Adam at
$K=128$ but not for RMSprop ($K=16$ peak then collapses) or SignSGD
($K=64$ non-monotonic)?

**Reframing under interference framework.** What does the framework's OOB
forgetting + WW_K say about each optimiser's interference profile? Maybe
the failing families' optimisers don't have *interference* to remove —
their wasted-work is curvature-driven instead.

**Status.** Queued. Will use the framework instrumentation.

#### `05_2x2_grid` ⏳ QUEUED — the final headline

**Question.** For each optimiser family (SGD, RMSprop, Adam, SignSGD),
decompose the contribution of momentum × BoGrad in a 2×2 grid.

**Method.** 4 cells per family (no-mom × no-BG, no-mom × BG, mom × no-BG,
mom × BG), with within-cell LR sweep, 3 trials each. Best-LR results
reported per cell.

**Status.** Queued. Should be the last test we run, after the framework
and per-method scrutiny are stable.

### `discoveryPhase2/` — historical sweeps

Already-completed sweep work that produced findings D3, D4 above and
informed Phase-1 / Phase-2 decisions. See `discoveryPhase2/README.md` for
context. **Do not extend.** Use `testing/` for new tactical tests.

Contents to be aware of:
- `enhanced_variants.py` — 10 custom optimiser classes, useful as drop-ins
  for new experiments (Complement variants, AdaptiveTrigger, MultiScale,
  InPipeline RMSprop/SignSGD).
- `sweep_update_K.py` — produced the per-optimiser optimal-K curves
  (D3 finding).
- `sweep_complement.py` — produced the Complement-aware momentum result
  (D4 finding).
- `sweep_in_pipeline_K.py` — produced the in-pipeline data that motivates
  testing/04.

---

## How to navigate from here

If you want to **understand the framework**, read in this order:
1. [`research/01_interference_framework/framework.md`](research/01_interference_framework/framework.md) §3 — how we got to the formal definition.
2. [`research/01_interference_framework/framework.md`](research/01_interference_framework/framework.md) §4 — formal metric definitions.
3. [`research/01_interference_framework/framework.md`](research/01_interference_framework/framework.md) §8 — empirical findings F1–F7 with numbers.

If you want to **understand a specific result**, follow the table at the
top of this document. Each finding has a one-line statement and a source
test name.

If you want to **run a specific test**, the "Run" line under each test
above gives the command. Outputs are JSONs under
`<test>/results/run_<timestamp>/`.

If you want to **decide what to do next**, the immediate options are:

1. **Build the (W') validation experiment** — design a setup where bias is
   PERPENDICULAR to descent (not aligned, like F10's drift). E.g. quadratic
   loss with sparse θ★ and bias on the zero-θ★ coordinates. This tests
   whether BoGrad recovers when interference is genuinely separable from
   descent. The most direct remaining test of Hypothesis (W').
2. **Move to `testing/04_inpipeline_diagnosis`** under the new framework.
   The framework's OOB, WW_K, and pairwise-alignment metrics now give us
   a way to answer "what kind of interference does each optimiser face"
   rather than just "what K works" — F11 already gives baseline regime
   fingerprints; we'd extend to RMSprop and SignSGD.
3. **Statistical baseline (Gaussian null)** — define the noise floor
   under a Gaussian-noise null model so we can identify "real" structural
   interference vs random fluctuation in the framework metrics.
4. **Move to Stream 2 (BoGrad scrutiny)** — use the framework as our lens
   for the BoGrad theoretical / regime characterisation work. Defer
   `testing/05_2x2_grid` until BoGrad is fully understood under the framework.

My recommendation: option 1 (closes the (W') story), then option 4
(BoGrad scrutiny under the framework), then `testing/05_2x2_grid` as the
final headline.

---

*Document last updated 2026-05-05 with Findings F1–F11, F13 + OOB metric +
PairwiseAlignmentTracker + per-progress OOB. Updated whenever a major test
completes.*
