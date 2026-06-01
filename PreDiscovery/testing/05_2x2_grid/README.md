# Test 05 — 2×2 grid: momentum × BoGrad per optimiser

## Aim

The headline ablation. For each optimiser family, decompose the contribution of momentum and BoGrad and quantify their interaction. This is the result that goes into Chapter 4.

## The 2×2 design

For each optimiser:

| | no momentum | momentum |
|---|---|---|
| **no BoGrad** | A | B |
| **BoGrad** | C | D |

Effects of interest:
- **B − A** — pure momentum effect (well-established, positive control).
- **C − A** — pure BoGrad effect (validates that BoGrad helps without momentum).
- **D − B** — BoGrad on top of momentum (the thesis claim).
- **D − C** — momentum on top of BoGrad (sanity check; symmetric to D − B).

If `D > B` and `D > C`, momentum and BoGrad are *both* contributing and either complement each other or both improve baseline. If `D ≈ max(B, C)`, only one mechanism is doing useful work. If `D < B`, BoGrad fights momentum even at the right configuration.

## LR sweep within each cell

Per Test 01's recommendation, each cell runs a 3-point LR sweep so we compare best-tuned to best-tuned. The cell's reported result is the best mean across LRs.

This means each cell is 3 LR × 3 trials = 9 runs.

## Optimiser mapping for "no momentum"

| Family | momentum=on | momentum=off | Notes |
|---|---|---|---|
| SGD | `momentum=0.9` | `momentum=0.0` | clean |
| RMSprop | `momentum=0.9` | `momentum=0.0` | clean — uses torch.optim.RMSprop without momentum |
| Adam | `betas=(0.9, 0.999)` | `betas=(0.0, 0.999)` | β₁=0 disables first-moment EMA |
| SignSGD | `momentum=0.9` | `momentum=0.0` | clean |

## BoGrad config per cell

Use the best (stage, K, mode) combination determined by the prior sweeps. Updated as Tests 01–04 land:

| Family | momentum off — BoGrad config | momentum on — BoGrad config |
|---|---|---|
| SGD | `gradient`-stage K=8 negative (Phase 1 winner) | `update`-stage K=32 negative (sweep_update_K winner) |
| RMSprop | `gradient`-stage K=8 negative or `update`-stage K=8 (TBD from Test 04) | `update`-stage K=16 negative |
| Adam | `update`-stage K=8 negative (no momentum proxy) | `update`-stage K=128 negative |
| SignSGD | `gradient`-stage K=8 negative | `update`-stage K=64 negative or in-pipeline K=64 (TBD) |

Note: "BoGrad without momentum" for momentum-based families is the more theoretically clean cell — gradient-stage projection works without fighting an EMA.

If Test 02 confirms `preserve_magnitude=True` is the right default, all BoGrad cells use it.

## Setup

- 4 families × 4 cells × 3 LRs × 3 trials = **144 runs ≈ 7 hr on T4** at 5 epochs.
- Optionally drop the LR sweep to 1 (best-known LR per cell) → 48 runs ≈ 2.5 hr.
- The full sweep is the rigorous version we want for the thesis. If time-constrained, we can run a "quick" version first to confirm the pattern.

## Output

Per family, a 2×2 table:

```
SGD
              no momentum    momentum    Δ(momentum)
no BoGrad     A: 0.4xxx      B: 0.6xxx   +0.xx
BoGrad        C: 0.6xxx      D: 0.7xxx   +0.xx
Δ(BoGrad)     +0.xx          +0.xx       (interaction)
```

Plus the underlying per-cell mean ± std and best LR.

## Run

```bash
# Full sweep (~7 hr)
python testing/05_2x2_grid/run.py --epochs 5 --trials 3

# Single LR per cell (~2.5 hr)
python testing/05_2x2_grid/run.py --no-lr-sweep

# One family
python testing/05_2x2_grid/run.py --families sgd --epochs 5

# Quick (1/4 train set)
python testing/05_2x2_grid/run.py --quick
```
