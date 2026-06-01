# Test 02 — Rescaling ablation

## Aim

If Test 01 shows that the magnitude-preserving variant (`preserve_magnitude=True`) wins or matters, characterise *which form* of magnitude rescaling works best. This test treats `preserve_magnitude` as the design choice and ablates the variations:

- `no_rescale` — current default (magnitude reduction left in place).
- `full_rescale` — `g̃ ← g̃ · ‖g‖/‖g̃‖`, no clipping. Risks blow-up when ‖g̃‖ → 0.
- `clipped_2x` — clip rescale factor at 2.0.
- `clipped_5x` — clip rescale factor at 5.0.

(A "mean rescale" variant — rescaling to a running average ‖g‖ rather than current — is omitted from the default sweep but easy to add via a flag.)

## Why clipping matters

When the buffer fully spans the gradient (possible at large K and high redundancy), the projected vector approaches zero. The ratio `‖g‖/‖g̃‖` then explodes. Without clipping, a single near-degenerate step can take a giant jump in an essentially arbitrary direction.

With `max_rescale=5`: even worst-case, the step is at most 5× the original. A dampened safety net.

## Method

Same architecture and dataset as Test 01 (SmallCNN, CIFAR-10), same SGD+momentum=0.9 base optimizer with `update-stage K=32 negative` BoGrad config. The only difference is the rescaling configuration.

Two K values are tested to see if rescaling matters more at large K:
- K=32 (the prior peak for SGD+momentum)
- K=128 (where the buffer is more likely to span the gradient and the degenerate case is more likely)

## Configurations

| variant | preserve_magnitude | max_rescale |
|---|---|---|
| `no_rescale` | False | n/a |
| `full_rescale` | True | None (no clipping) |
| `clipped_2x` | True | 2.0 |
| `clipped_5x` | True | 5.0 |

Single LR (lr=0.05) — the LR sweep was the point of Test 01; here we focus on the rescaling axis.

## Decision criteria

- Pick the variant with **highest mean accuracy and lowest variance**.
- Confirm clipping doesn't hurt at small K — if `clipped_5x` ≈ `full_rescale` at K=32, clipping is safe.
- At K=128: if `full_rescale` shows occasional outlier runs (high std), the clipping is doing useful work.

The winner becomes the default for `preserve_magnitude=True`. If `no_rescale` ties or wins at K=32, the rescaling story is less compelling and we report it as a per-K knob.

## Setup

- Total: 4 variants × 2 K values × 3 trials = **24 runs ≈ 60 min on T4.**

## Run

```bash
python testing/02_rescaling/run.py
python testing/02_rescaling/run.py --Ks 32                   # only K=32
python testing/02_rescaling/run.py --quick                    # 1/4 train set
```

## Skip condition

If Test 01 conclusively shows magnitude doesn't matter, this test is unnecessary — document the no-rescale default and move to Test 04.
