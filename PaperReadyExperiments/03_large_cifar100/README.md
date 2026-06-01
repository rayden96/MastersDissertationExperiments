# 03 — Large: CIFAR-100 with ResNet18

## Claim being tested

The interference metrics scale to a **bigger model on a harder problem** without breaking down, and their qualitative behaviour (sign / direction / correlations) matches the smaller-scale runs.

This is the dissertation's "yes, the metric works at realistic scale" experiment. ResNet18 (~11.2M params, BatchNorm) on CIFAR-100 (100 classes, ~10× more class structure than CIFAR-10) is a meaningful stress test of the per-class subgradient pipeline and the cancellation indices.

## Design

- **Dataset.** CIFAR-100, 100 classes, balanced.
- **Model.** ResNet18 adapted for CIFAR (small stem, no maxpool). ~11.2M params. Has BatchNorm layers — the `TorchClassificationProblem` snapshots and restores BN running stats around every diagnostic forward so the metric machinery doesn't pollute training-time BN state.
- **Optimizer.** Vanilla SGD with momentum 0.9, lr 0.1 (standard CIFAR-100 ResNet starting LR).
- **Batch.** 128.
- **Reference set.** 100 balanced samples per class (10,000 total — same scale as CIFAR-10's 200/class × 10 = 2,000, scaled to the larger class count).
- **Logged steps.** Every 100 steps (per-class subgrad is 100× the cost of a single forward, so we space the logs out).
- **K-window values.** $K \in \{4, 32, 128\}$.
- **Multi-seed.** 3 seeds.

## Predicted behaviour

- $I_{\text{inter}}$ should be **even lower** than CIFAR-10 (more classes → more pairwise conflict — pairwise count grows as $C(C-1)/2$, so any per-pair anti-alignment dominates the mean more visibly).
- Mean pairwise cosine should be negative.
- The correlation between geometric metrics and $D_t$ should hold qualitatively (we expect $|r| \gtrsim 0.4$ for the inter-batch metrics on a clean SGD trajectory).
- Useful descent fraction of the batch gradient should still be positive (the batch gradient is on the right side of descent, just noisy).
- ResNet18's BatchNorm doesn't break the metric — the `_bn_state_preserved` context restores running stats correctly.

## Cost note

Per-class subgradient computation on a 100-class batch is expensive: each logged step does ~100 extra forward-backwards on small per-class slices. With `log_every=100`, this caps overhead at ~1 extra epoch per 100 logged training steps. Wall-clock estimate (RTX 3050 Laptop, 5 epochs, 3 seeds): ~45–60 min.

## Outputs

- `results/run_<id>/logs_seed<S>.json`, `summary_seed<S>.json`, `headline.json`.
- `results/run_<id>/metrics_timeseries.png`, `correlations.png`.

## How to run

```bash
python PaperReadyExperiments/03_large_cifar100/run.py
python PaperReadyExperiments/03_large_cifar100/plot.py
```
