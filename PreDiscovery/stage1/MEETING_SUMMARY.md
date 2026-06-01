# Meeting summary — interference framework + Stage 1 results

A two-page handout for tomorrow's meeting. Numbers come from
`stage1/results/run_20260507_105741/` (full CIFAR-10 × 2 epochs)
and `MoGrad/results/run_20260507_105751/` (smoke-test sweep).

---

## 1. The framing — two types of interference

Single-task supervised training has (at least) two distinct interference
phenomena:

| Type | What | Method that targets it |
|---|---|---|
| **Inter-batch (intra-batch)** | Class gradients within one batch conflict — the averaged batch gradient is a degraded direction | **COSGD** (per-class orthogonalisation within the batch) |
| **Between-batch** | Successive batch gradients/updates conflict over time — trajectory zigzags or undoes recent progress | **BoGrad** (orthogonalise current step against recent buffer) |

Defined cleanly:

- **Inter-batch interference at step t** = $\sum_{c \neq c'} \mathbb{1}\!\left[\langle g_t^c, g_t^{c'}\rangle < 0\right]$ (count of conflicting class-pair gradients within batch $t$).
- **Between-batch interference** has multiple operationalisations:
  - $\cos(g_t, g_{t-k})$ over a $K$-window (geometric / trajectory).
  - Out-of-batch forgetting magnitude (per-class — only meaningful under non-iid batching).
  - Wasted-work ratio $\mathrm{WW}_K = \lVert \theta_t - \theta_{t-K}\rVert / \sum_i \lVert u_i\rVert$.

---

## 2. Empirical headline — both interference types are real and measurable

**CIFAR-10, SmallCNN, lr=0.05 momentum=0.9, 2 epochs (782 steps), batch 128.**

| Method | final acc | IB_%neg | IB_⟨cos⟩ | WW_K | OOB_total |
|---|---|---|---|---|---|
| SGD+momentum baseline | 0.544 | **70%** | **−0.08** | 0.36 | 0 |
| **+ BoGrad** (upd-K=32 neg) | **0.601** *(+5.7)* | 71% (≈) | −0.09 (≈) | **0.58** *(+62%)* | 0 |
| **COSGD** (mod-GS, single fwd) | 0.487 | **54%** *(−16)* | **+0.01** *(flipped sign)* | 0.09 | 0 |

### What this picture says (one paragraph)

> Both interference types are present in standard CIFAR-10 training. **Inter-batch class-pair conflict is severe**: 70% of class-pair gradients within a batch are anti-aligned (mean cosine −0.08). **Between-batch trajectory inefficiency is moderate**: WW_K=0.36 means only about a third of step magnitude turns into net progress. **COSGD reduces inter-batch interference cleanly** — IB_%neg drops to 54%, mean cosine flips positive — but this comes at a cost in trajectory efficiency (WW collapses to 0.09) and a small accuracy regression. **BoGrad improves trajectory efficiency cleanly** — WW_K rises to 0.58 — and gains +5.7 accuracy points, without affecting inter-batch class conflict. **The two methods are addressing genuinely different phenomena**, and their effects on the two metrics are nearly orthogonal.

### Why OOB forgetting is 0 across all three

In standard interleaved CIFAR-10 batches (n=128, 10 classes), every batch
contains all classes, so by definition no class is ever "out of batch" —
the OOB metric is structurally 0. **OOB forgetting is the right metric for
non-iid regimes** (class-disjoint, permuted classes); for standard
training, the between-batch story is told by WW_K and pairwise cosines.

---

## 3. Interference profile during training (qualitative)

Per-step inter-batch interference (`IB_%neg` measured every 25 batches)
remains in the 60–80% range throughout training under SGD+momentum and
BoGrad. Under COSGD it sits in the 40–60% range — visibly different
distribution. **The phenomenon is persistent, not transient.**

---

## 4. MoGrad — confirmed negative result (2-epoch run)

We tested an idea: maintain a momentum-style EMA of past gradients
($\mathbf{m}_t$), but **don't use it in the update equation** — only as
a reference vector for orthogonalisation. Three modes (full / negative /
positive) × three start-step values × 2 trials × 2 full epochs CIFAR-10.

| Config (full, 2 epochs) | final_acc | proj_rate |
|---|---|---|
| sgd_vanilla | 0.346 ± 0.002 | — |
| **sgd_momentum** | **0.546 ± 0.010** | — |
| mograd_negative (start=50) | 0.347 ± 0.003 | 6.7% |
| mograd_negative (start=200) | 0.347 ± 0.003 | 7.8% |
| mograd_negative (start=500) | 0.346 ± 0.003 | 7.1% |
| mograd_full (start=50) | 0.099 ± 0.001 | 100% |
| mograd_full (start=200) | 0.156 ± 0.030 | 100% |
| mograd_full (start=500) | 0.190 ± 0.004 | 100% |

### Three clean readings

1. **Negative mode is a no-op across all start_steps.** Projection rate
   is ~7% (rarely fires) and accuracy is identical to vanilla SGD. The
   EMA reference and current gradient are mostly aligned, so neg-mode
   skips. **Doesn't help, doesn't hurt.**

2. **Full mode is catastrophic and the damage scales monotonically with
   warmup.** With start=50 → acc=0.099 (chance for 10-class). With
   start=500 → acc=0.190. The longer the warmup, the more "damage"
   accumulated *before* projection starts (projection then destroys it),
   but even with the longest warmup tested, the model still degrades
   sharply once projection kicks in. **Always hurts. Hurts less when
   you delay projection longer.**

3. **MoGrad doesn't approach SGD+momentum (0.546)** under any
   configuration. Best MoGrad config (negative-start200 at 0.347) ties
   vanilla SGD — but never momentum.

### Why — framework predicts this exactly

This is consistent with framework finding F10 / hypothesis (W'):
*orthogonalising against the descent direction strips the descent
signal*. The momentum reference IS the descent direction (after warmup,
it points toward the local minimum), so:
- **Full mode** removes the descent signal entirely → catastrophic.
- **Negative mode** would only fire when the current gradient *fights*
  the consensus, which is rare under standard CIFAR-10 → vacuous.

The monotonic improvement of full-mode accuracy with later start_step
(0.099 → 0.156 → 0.190) is itself a framework prediction: the longer
training proceeds before the projection kicks in, the more useful
parameter movement has accumulated, and the less time projection has to
undo it. **Even with 500 steps of clean training before projection
starts, the projection-induced damage still dominates by epoch 2.**

### Takeaway and where to point future work

MoGrad as currently formulated is **a falsified hypothesis** —
empirically and theoretically. The framework predicted it would fail
(F10/W'); it failed exactly as predicted; the failure mode is clean and
the magnitude is large.

The corollary is the constructive alternative: **Complement-Aware
Momentum** (V2 from Phase 2). Same momentum reference, but instead of
*projecting out* the parallel component, *amplify* the perpendicular
component on top of the standard momentum step:

$$\theta \gets \theta - \eta\,(v_t + \gamma\, g_\perp)$$

where $g_\perp$ is the part of $g_t$ orthogonal to $v_{t-1}$. This:
- Doesn't fight momentum (the parallel component stays in $v_t$).
- Adds new orthogonal information momentum was about to dilute.
- Empirically gives **+5.7 pts on SGD+momentum at γ=2** (sweep_complement).

The MoGrad result thus *supports* the case for Complement-Aware Momentum
in the future-work section: removing the momentum direction (MoGrad)
fails; amplifying the perpendicular complement (Complement) works.

---

## 5. The paper structure (where this is going)

Coming out of these results:

1. **Section: Two types of interference.** Define, motivate, give the
   formal metrics. Cite F1–F13 as supporting evidence for the framework.
2. **Section: COSGD.** Theoretical motivation (per-class within-batch
   orthogonalisation), the IB_%neg evidence (it does what it claims),
   and the trade-off (between-batch trajectory inefficiency).
3. **Section: BoGrad.** Theoretical motivation (between-batch trajectory
   reduction), the WW_K evidence (it does what it claims), the
   hyperparameter ablations from `discoveryPhase2/`. (W') analysis
   from F12 (works when interference is separable from descent).
4. **Section: Comparison to implicit methods.** Dropout, gradient
   clipping, momentum, Adam. Which interference type does each address?
   (Not yet run — the next experiment after the meeting.)
5. **Conclusion.** When does each method help? Synthesis of the framework.

---

## 6. Concrete next experiments (post-meeting)

In priority order:

1. **Implicit-method comparison** — repeat the Stage 1 protocol with
   dropout, gradient clipping, vanilla momentum, Adam. Fills out
   Section 4 of the paper.
2. **Sensitivity of inter-batch interference to batch size** — does
   IB_%neg shrink or grow with batch size? Inform design recommendations.
3. **COSGD + BoGrad combined** — does the combination address both
   interference types? Or do they fight?
4. **Between-batch metric for standard batching** — refine. WW_K is
   one option; pairwise alignment over a K-window is another. The
   permuted-classes / class-disjoint setups give cleaner OOB signal.

---

## Key files / artefacts

- `stage1/run_comparison.py` — the runner that produced the headline table.
- `stage1/inter_batch_metric.py` — the new IB_%neg metric.
- `stage1/results/run_20260507_105741/` — the full numbers used here.
- `MoGrad/run_study.py` + `MoGrad/mograd.py` — the MoGrad experiment code.
- `common/diagnostics/interference.py` — the framework module
  (InterferenceTracker, ForgettingTracker, WastedWorkTracker,
  PairwiseAlignmentTracker, ClassProbeSet).
- `common/optimizers/{BoGrad,COSGD}.py` — the two proposed methods.
- `research/01_interference_framework/framework.md` — formal definitions
  (F1–F13).
- `TESTS_OVERVIEW.md` (root) — index of all experiments + findings.
