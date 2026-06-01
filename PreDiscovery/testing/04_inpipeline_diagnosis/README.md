# Test 04 — In-pipeline diagnosis

## Aim

Phase-2 in-pipeline projection helps Adam dramatically (+6pts at K=128) but underwhelms on RMSprop (peak +1.6pts, then collapses) and SignSGD (peak +1.6pts, non-monotonic). Why?

We want:
1. A diagnostic story — what's geometrically different about each family's buffer?
2. A recommendation — for RMSprop and SignSGD, is there a better buffer content or projection point?
3. (Optional) A working variant for at least one of the failing families.

## Hypotheses to test

For each, we capture per-step diagnostics and compare patterns across the four optimisers (Adam succeeds; RMSprop, SignSGD struggle; SGD+momentum is reference).

| Hypothesis | Diagnostic |
|---|---|
| **H1: Buffer geometry differs.** RMSprop/SignSGD buffers span almost full parameter space quickly (low effective rank vs nominal K), so projection over-strips. | Singular-value spread of the stacked buffer at each step. |
| **H2: g_ratio distribution differs.** For successful families, projection rarely fires (negative-mode); for failing families, it fires too often or removes too much. | Histogram of `g_ratio` across steps. |
| **H3: Trigger rate differs.** In `negative` mode, what fraction of buffer iterations result in subtraction? | `trigger_rate = (#dot < 0) / (#dot total)`. |
| **H4: SignSGD's pre-sign vector has weird geometry.** The `m_t` for SignSGD has different magnitude characteristics than for SGD+momentum. | `‖m_t‖` distribution; coordinate-wise extreme value frequency. |
| **H5: RMSprop's preconditioned grad has uneven coordinates.** The `1/√v` rescaling makes some coordinates dominate the buffer entries. | Per-coordinate variance of buffered vectors. |

## Method

### Phase 4a — Diagnostic capture

For each family at its best K from the in-pipeline sweep:

| Family | Best K (in-pipeline) | Reference |
|---|---|---|
| Adam | 128 (still rising) | succeeds |
| RMSprop | 16 | declines after |
| SignSGD | 64 | non-monotonic |

Run 1 epoch of training with high-resolution diagnostics (`log_every=5`):

- buffer condition number (top vs bottom singular value)
- buffer effective rank (sum-eigenvalue / max-eigenvalue style estimate)
- per-step `g_ratio` (mean, std)
- per-step trigger rate (mode=negative)
- buffer-content magnitude distribution

Save the per-step JSON; produce summary plots if `--plot` is set.

### Phase 4b — Alternative buffer content (targeted fixes)

Based on what 4a reveals, test variants on the failing families:

For **RMSprop**:
- `pre_precond` — buffer raw `g_t` (not preconditioned), keep in-pipeline projection point.
  Hypothesis: preconditioned buffer accumulates ill-conditioned vectors; raw g is better.
- `update_stage_K_finer` — finer K sweep on update-stage to verify the K=16 peak.

For **SignSGD**:
- `tanh_smoothed` — buffer `tanh(m_t / scale)` instead of raw `m_t`.
  Hypothesis: large `m_t` magnitudes after long momentum runs make the buffer spiky; smoothing helps.
- `gradient_buffer` — buffer raw `g_t` instead of `m_t`.
- `update_stage_only` — skip in-pipeline; rely solely on update-stage K sweep.

Each variant: 3 trials, 5 epochs, single LR. ~5 runs.

## Decision criteria

For each failing family, we want either:
- A diagnostic-supported explanation (e.g., "RMSprop's buffer has condition number 100×, projection strips signal") → recommendation: don't use in-pipeline for this family, use update-stage instead.
- A better-performing alternative variant → recommendation: use this variant in Test 5's 2×2.

Either is a publishable finding for Chapter 4.

## Setup

- 4a: 3 families × 1 run with diagnostics ≈ 30 min.
- 4b: ~5 variants × 3 trials × 5 epochs ≈ 60 min.
- Total: ~90 min.

## Run

```bash
# Phase 4a — diagnostic capture only
python testing/04_inpipeline_diagnosis/run.py --phase a

# Phase 4b — alternative variants on failing families
python testing/04_inpipeline_diagnosis/run.py --phase b

# Both
python testing/04_inpipeline_diagnosis/run.py --phase both
```

## Output

- `results/run_<timestamp>/diagnostics/<family>__per_step.json` — per-step diagnostics for 4a.
- `results/run_<timestamp>/results.json` — variant comparison results for 4b.
- Summary table prints recommendations per family.
