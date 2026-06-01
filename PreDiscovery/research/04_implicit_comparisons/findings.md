# Implicit-method comparison — findings

> Run: `research/04_implicit_comparisons/results/run_20260507_133221/`
> CIFAR-10, SmallCNN, batch=128, lr=0.05, mu=0.9, 2 epochs, single trial,
> paired data ordering (same `seed=2026`, same shuffled batch sequence
> across all ten methods).

## Headline table

| method | acc | IB_%neg | IB_⟨cos⟩ | WW_K | wall_s |
|---|---:|---:|---:|---:|---:|
| sgd_vanilla (no momentum) | 0.344 | 0.731 | −0.096 | **0.186** | 60 |
| **sgd_momentum (baseline)** | **0.546** | 0.700 | −0.079 | 0.361 | 62 |
| sgd_mom + Dropout(0.1) | 0.510 | 0.733 | −0.090 | 0.429 | 75 |
| sgd_mom + Dropout(0.3) | 0.415 | 0.717 | −0.091 | 0.507 | 132 |
| sgd_mom + clip(1.0) | 0.510 | 0.693 | −0.087 | 0.362 | 126 |
| sgd_mom + clip(5.0) | 0.550 | 0.695 | −0.079 | 0.361 | 127 |
| Adam (lr=1e-3) | 0.517 | **0.776** | −0.097 | 0.448 | 119 |
| Adam (lr=0.05) | 0.280 | 0.627 | −0.081 | 0.495 | 120 |
| **+ BoGrad upd-K=32 neg** | **0.586** | 0.701 | −0.088 | **0.582** | 143 |
| **COSGD (modified-GS)** | 0.524 | **0.550** | **+0.005** | **0.092** | 143 |

## Four findings (F-impl-1 to F-impl-4)

### F-impl-1: gradient clipping is inert on both interference axes

clip(1.0) and clip(5.0) reproduce the baseline IB_%neg, IB_⟨cos⟩,
and WW_K to within ±0.01:

| | IB_%neg | IB_⟨cos⟩ | WW_K |
|---|---:|---:|---:|
| baseline | 0.700 | −0.079 | 0.361 |
| clip(1.0) | 0.693 | −0.087 | 0.362 |
| clip(5.0) | 0.695 | −0.079 | 0.361 |

Confirms the prediction: **clipping changes magnitude, not direction**,
and therefore moves neither interference axis. clip(1.0) loses 3.6 pts
of accuracy with no compensating reduction in any interference metric.
A clean confirmation of `testing/01_attribution`'s finding (in a
different formalism) that magnitude reduction alone does not
constitute interference reduction.

### F-impl-2: dropout reduces between-batch interference monotonically

Dropout's effect on WW_K scales with dropout probability:

| | WW_K | Δ vs baseline | acc | Δ acc |
|---|---:|---:|---:|---:|
| baseline | 0.361 | — | 0.546 | — |
| Dropout(0.1) | 0.429 | +0.07 | 0.510 | −3.6 |
| Dropout(0.3) | 0.507 | +0.15 | 0.415 | −13.1 |

Without changing IB_%neg or IB_⟨cos⟩ measurably (both within ±0.03 of
baseline). **Dropout acts on the between-batch axis, not the
inter-batch axis.** Mechanism: dropout decorrelates per-step gradients
because different units are masked each step, reducing sequential
cancellation in the trajectory.

Notably, Dropout(0.3) reaches **WW_K=0.507**, ≈ 87% of BoGrad's WW_K
(0.582). But Dropout(0.3) loses 13 pts of accuracy in the process,
while BoGrad gains 4. So **WW_K alone is not an accuracy-predictor**;
the *manner* in which trajectory efficiency is improved matters.

### F-impl-3: Adam moves the two axes in *opposite* directions

Adam (lr=1e-3) is unique in our set: it improves trajectory
efficiency *and* worsens inter-batch class conflict simultaneously.

| | IB_%neg | WW_K | acc |
|---|---:|---:|---:|
| baseline | 0.700 | 0.361 | 0.546 |
| **Adam (1e-3)** | **0.776 (+0.076)** | **0.448 (+0.087)** | 0.517 (−2.9) |

This is the framework's separation hypothesis materialising in real
data: the two axes are independent enough that a method can move one
positively and the other negatively at the same time.

Plausible mechanism: Adam's per-parameter adaptive scaling breaks
the rough symmetry between class gradients. After a few steps, some
parameters that one class needs are scaled differently from those
another class needs, increasing the angle between per-class gradients.

### F-impl-4: BoGrad and COSGD reproduce Stage 1 within seed-noise

Both reference methods, run inside the same paired-data harness as
the implicit comparison, recover the Stage 1 numbers:

| | acc | IB_%neg | WW_K |
|---|---:|---:|---:|
| Stage 1 BoGrad | 0.601 | 0.707 | 0.579 |
| **This run BoGrad** | **0.586** | **0.701** | **0.582** |
| Stage 1 COSGD | 0.487 | 0.541 | 0.092 |
| **This run COSGD** | **0.524** | **0.550** | **0.092** |

WW_K and IB_%neg are reproduced to ≤0.01. Accuracy varies by ±0.04
(the seed-noise scale at this trial count). Confirms the headlines:
**BoGrad raises WW_K cleanly without touching IB_%neg; COSGD drops
IB_%neg and flips IB_⟨cos⟩ positive while collapsing WW_K**.

## The two-axis decomposition (consolidated)

Sorted by WW_K (between-batch axis), then commenting on IB_%neg
(inter-batch axis):

```
WW_K
 0.6 |  bograd       (IB=0.701, acc=0.586, +4)
     |
 0.5 |  drop03 / adam_lr05  (drop03 IB=0.717 acc=0.415, -13;
     |                       adam_lr05 IB=0.627 acc=0.280, -27)
     |  adam (IB=0.776 [worse], acc=0.517, -3)
 0.4 |  drop01 (IB=0.733, acc=0.510, -4)
     |
 0.3 |  baseline / clip1 / clip5  (IB=0.69-0.70)
     |
 0.2 |  vanilla (IB=0.731 — no smoothing)
 0.1 |  cosgd (IB=0.550 [much better], acc=0.524, -2)
```

Read across:
- **No method moves IB_%neg into COSGD's region** (~55%) except COSGD.
- **Multiple methods raise WW_K**: BoGrad (best), drop03, adam_lr05,
  adam, drop01. None matches BoGrad's +0.22.
- **Among WW-raisers, only BoGrad gains accuracy.** All others lose
  accuracy by 3–27 pts.

## Implications

1. **Inter-batch interference is robust to standard regularisation.**
   Dropout, clipping, momentum, and Adam variants leave IB_%neg
   essentially unchanged (modulo Adam *worsening* it). COSGD remains
   the only known method to address this axis; the orthogonalisation is
   doing something the implicit alternatives cannot.
2. **Between-batch interference has multiple known mitigations**, with
   BoGrad currently at the Pareto frontier (highest WW_K with positive
   accuracy effect). Dropout and Adam approach it on the metric but
   not on the accuracy outcome — they trade trajectory efficiency for
   capacity (dropout) or conditioning (Adam).
3. **The two axes are independently moveable**, as the framework
   predicted. Adam moves them in opposite directions; BoGrad and COSGD
   each move only their target. This is direct empirical support for
   treating "interference" as not one phenomenon but two.

## Trajectory cosines (auxiliary table)

Update self-correlation at lag k (k=10 actual steps because
log_every=10):

| method | lag-1 (~10 steps) | lag-4 (~40 steps) | lag-16 (~160 steps) | pw_upd+ (32-buffer) |
|---|---:|---:|---:|---:|
| sgd_vanilla | +0.10 | +0.16 | −0.01 | 0.580 |
| baseline | +0.06 | +0.03 | +0.01 | 0.588 |
| Dropout(0.1) | +0.09 | +0.04 | +0.01 | 0.589 |
| Dropout(0.3) | +0.14 | +0.01 | +0.02 | 0.639 |
| clip(1.0) | +0.01 | +0.09 | +0.06 | 0.600 |
| clip(5.0) | +0.05 | +0.03 | +0.02 | 0.587 |
| Adam (1e-3) | +0.17 | +0.11 | +0.03 | **0.701** |
| Adam (0.05) | +0.09 | +0.02 | −0.02 | 0.581 |
| **+ BoGrad** | **+0.27** | −0.03 | +0.04 | **0.716** |
| COSGD | (na — params-snapshot interface) | | | (na) |

`pw_upd+` is what to read first: the fraction of pairwise update
cosines that are positive over the K=32 buffer. Adam (0.701) and
BoGrad (0.716) are the two methods that make consecutive update
directions strongly persistent — but only BoGrad does so by
*construction* (orthogonalising against past directions); Adam does
so via momentum + adaptive scaling.

## Open questions

- **Combined methods.** Does Dropout + BoGrad add cleanly on the WW_K
  axis (independent mechanisms targeting the same axis)? Does
  Dropout + COSGD address both axes? Does COSGD + BoGrad?
- **Adam's IB_%neg increase.** Is this fixed under different LR /
  model size? Or is the +0.08 shift a property of CIFAR-10 +
  small-CNN at this LR?
- **Multi-trial runs.** Single-trial differences ≤0.04 acc are within
  seed noise. Multi-trial confirmation of the F-impl-2 dropout effect
  would tighten the claim.
- **Combination with Complement-Aware Momentum.** ComplementMomentumSGD
  (γ=2) gives +5.7 pts on SGD+mom (Phase 2). It belongs in this table
  alongside BoGrad, but it's a different optimisation philosophy
  (amplify perpendicular instead of project parallel) and was not
  included in this batch.

## Files

- `methods.md` — experiment description and predictions.
- `run_comparison.py` — runner.
- `analyze.py` — analysis + plotting.
- `results/run_20260507_133221/` — this run's data.
  - `config.json`
  - per-method subfolders with `result.json`, `history.json`,
    `inter_batch_history.json`, `summary.json`.
  - `acc_summary.png`, `axes_scatter.png`, `ib_axis.png`, `ww_axis.png`,
    `ib_over_time.png`, `trajectory_cosines.png`,
    `pairwise_positive.png`, `summary.md`.
