# Implicit Method Comparison — methods catalogue

Stage 4 of the interference-framework investigation. Question: how do
**implicit interference reducers** (dropout, gradient clipping, momentum,
Adam) compare to the **explicit orthogonalisation methods** (BoGrad,
COSGD) on the two interference-axis metrics established in Stage 1?

This experiment extends `stage1/run_comparison.py` by adding methods
that are not orthogonalisation-based but are widely believed to reduce
"some kind of" interference. The point is not to claim BoGrad/COSGD beat
them, but to characterise *which axis of interference each method
addresses*.

## Methods compared

| Code | Method | Mechanism (claimed) | Interference type targeted (claim) |
|---|---|---|---|
| `sgd_vanilla` | Plain SGD (no momentum) | None — control | None |
| `sgd_momentum` | SGD + momentum | Smooths trajectory; averages successive updates | Between-batch (implicit) |
| `sgd_mom_dropout01` | SGD + momentum + Dropout(0.1) | Random unit masking | Within-batch noise — possibly inter-batch |
| `sgd_mom_dropout03` | SGD + momentum + Dropout(0.3) | Random unit masking | Within-batch noise — possibly inter-batch |
| `sgd_mom_clip1` | SGD + momentum + grad clip 1.0 | Step-magnitude bounding | Trajectory stability (not interference per se) |
| `sgd_mom_clip5` | SGD + momentum + grad clip 5.0 | Loose clip (most steps unbounded) | — |
| `adam` | Adam (lr=1e-3) | Per-parameter adaptive LR + momentum | Between-batch (implicit) |
| `adam_lr05` | Adam (lr=0.05) | Same as above, LR matched to SGD | — |
| `bograd` | SGD+mom + BoGrad upd-K=32 neg | Explicit between-batch orthogonalisation | Between-batch (explicit) |
| `cosgd` | COSGD modified-GS single-fwd | Explicit per-class within-batch orthogonalisation | Inter-batch (explicit) |

(BoGrad/COSGD are included only as reference points — they were the
headlines in Stage 1.)

## Metrics captured (same as Stage 1)

For every method:
- `final_acc` — test accuracy at end of last epoch.
- `IB_%neg` — fraction of class-pair gradient cosines within a
  batch that are < 0 (lower = less inter-batch conflict).
- `IB_⟨cos⟩` — mean cosine across class pairs.
- `OOB_total` — out-of-batch forgetting magnitude (only meaningful for
  non-iid batching; structurally 0 here).
- `WW_K` — wasted-work ratio over a 32-step window
  (1 = perfectly efficient, lower = more cancellation).
- `OOB_per_step` — OOB forgetting per unit parameter movement
  (step-magnitude-normalised — useful when methods have different step
  norms).

## Hypotheses about expected outcomes

These are predictions to test, not assertions:

1. **Dropout** should affect inter-batch metrics (it changes the gradient
   distribution within a batch by stochastic masking). Prediction:
   `IB_%neg` falls slightly relative to baseline.
2. **Gradient clipping** changes step *magnitude*, not direction.
   Prediction: nearly identical interference profile to baseline; only
   `WW_K` may shift.
3. **Momentum** is in the baseline. Comparing `sgd_vanilla` (no momentum)
   to `sgd_momentum` reveals momentum's between-batch contribution.
4. **Adam** combines momentum-style averaging with per-parameter
   adaptive LR. Prediction: substantial trajectory smoothing
   (higher `WW_K` than vanilla SGD); inter-batch interference unchanged.
5. **BoGrad** raises `WW_K` (explicit between-batch reduction); does not
   change `IB_%neg` (Stage 1 confirmed). Reference.
6. **COSGD** drops `IB_%neg` substantially; collapses `WW_K` (Stage 1
   confirmed). Reference.

The ideal result for the framework: implicit methods cluster around
their predicted axes; the two orthogonalisation methods each move *one
axis specifically* in a way the implicit alternatives cannot.

## Files

- `run_comparison.py` — runner. Identical metric-capture protocol to
  `stage1/run_comparison.py`; broader method set.
- `results/run_<id>/` — per-run output: `config.json`, per-method
  subfolders with `history.json`, `inter_batch_history.json`,
  `summary.json`, `result.json`.

## Running

```
python research/04_implicit_comparisons/run_comparison.py --quick
python research/04_implicit_comparisons/run_comparison.py --epochs 2
python research/04_implicit_comparisons/run_comparison.py --only sgd_vanilla,adam
```
