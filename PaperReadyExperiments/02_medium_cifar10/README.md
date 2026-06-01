# 02 — Medium: CIFAR-10 with a small CNN

## Claim being tested

The interference metrics behave on **real image data** the way the 2D toy in `01_small_2d/` says they should. Same metrics, real classifier, real per-class subgradient structure.

The model is small (~93k params, no BatchNorm) so the per-class diagnostic forwards are cheap and the gradient direction at any θ is roughly stable across the reference horizon.

## Design

- **Dataset.** CIFAR-10, 10 classes, balanced.
- **Model.** Small 3-conv CNN (`PreDiscovery/FocusedWork/run_experiments.py:SmallCIFARCNN`). No BN, no dropout, so per-class subgradient forwards don't perturb model state.
- **Optimizer.** Vanilla SGD with momentum 0.9, lr 0.05.
- **Batch.** 128, shuffled per epoch.
- **Reference set.** 200 balanced samples per class (2000 total). Reference gradient refreshed every 100 logged steps; held constant between refreshes.
- **Logged steps.** Every 50 steps (per-class subgrad computation is the dominant overhead).
- **K-window values.** $K \in \{4, 32, 128\}$.
- **Multi-seed.** 3 seeds by default.

## What's measured

Everything in `interference.InterferenceMeter`:

- **§03 inter-batch**: cancellation index $I_{\text{inter}}$, pairwise cosine and magnitude stats, useful/wasted decomposition vs reference, useful descent fraction of the batch gradient.
- **§04 between-batch**: cancellation index $I_{\text{between},K}$ for each $K$, pairwise stats within window, useful path fraction.
- **Deficit**: per-step first-order $D_t$, cumulative.
- **Calibration**: measured reference loss at each refresh.

## Predicted behaviour

- $I_{\text{inter}}$ should be **low** under standard SGD+momentum on CIFAR-10 (per-class gradients conflict — known empirical result from PreDiscovery, ~0.10 in early Stage-1 runs).
- Mean pairwise cosine between per-class subgradients should be **negative or near zero**.
- $I_{\text{between},K}$ should rise as $K$ grows from 4 to 128 — short windows oscillate more than long ones in a smooth descent.
- $D_t$ should correlate negatively with $I_{\text{inter}}$ (high cancellation → high training-hurt).

## Outputs

- `results/run_<id>/logs_seed<S>.json` — per-step logs.
- `results/run_<id>/summary_seed<S>.json` — per-run summary.
- `results/run_<id>/headline.json` — mean ± std across seeds.
- `results/run_<id>/metrics_timeseries.png`, `correlations.png` — figures.

## How to run

```bash
python PaperReadyExperiments/02_medium_cifar10/run.py
python PaperReadyExperiments/02_medium_cifar10/plot.py
```

Knobs in `run.py`: `--epochs`, `--lr`, `--batch_size`, `--seeds`, `--log_every`, `--ref_refresh_every`.
