# Batch-size sensitivity of interference

Stage 5 of the interference investigation. Question: does the
**inter-batch interference** signal `IB_%neg` shrink or grow with batch
size? And what about between-batch trajectory efficiency `WW_K`?

## Motivation

The Stage 1 result was at batch=128 (CIFAR-10): IB_%neg ≈ 70% under
SGD+momentum. Two competing intuitions:

- **Smaller batches** → fewer per-class examples per batch → noisier
  per-class gradients → more conflict → IB_%neg ↑.
- **Larger batches** → averaging more examples per class → less
  conflict per pair → IB_%neg ↓ (until it saturates near full-dataset).

The trajectory side has the standard noise-vs-progress story already
characterised in the optimisation literature, but it's worth confirming
the WW_K axis behaves the way that literature would predict (large
batches ↑ WW_K, small batches ↓ WW_K).

## Design

Same protocol as `stage1/run_comparison.py` and
`research/04_implicit_comparisons/run_comparison.py`. Single optimiser
held constant — **SGD+momentum** — to keep the comparison clean. Vary:

- batch size: 16, 32, 64, 128 (= Stage 1 default), 256, 512
- LR scaling rule: linear scaling (lr = 0.05 × batch / 128) so per-step
  effective progress is comparable

Captured metrics per batch-size run (same as Stage 1):
- IB_%neg, IB_<cos>
- WW_K
- final test acc

## Hypotheses

H1. IB_%neg is monotonically decreasing in batch size (within the
    range tested).
H2. WW_K is monotonically increasing in batch size.
H3. Final accuracy peaks at moderate batch (around 128–256) — small
    batches lose to noise, large batches lose to update count under
    the linear scaling rule.

## Files

- `run_sensitivity.py` — runner.
- `analyze.py` — plotting + summary, mirrors 04_'s analyzer.
- `results/run_<id>/` — per-batch-size subdirs.

## Status

**Not yet run.** Scaffold only. Wait for `04_implicit_comparisons` to
finish before launching to avoid GPU contention.
