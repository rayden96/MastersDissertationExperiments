# PaperReadyExperiments

Experiments that will be in the dissertation.

## What's here

- `interference/` — clean, reusable implementation of the interference metrics defined in `PreDiscovery/FocusedWork/01..04_*.md`. Single source of truth for measurement code.
- `01_small_2d/` — toy experiment on a 2D mixture-of-quadratics. Lets us visualize the trajectory in parameter space alongside the interference metrics. Tests whether the metrics behave as expected in a controlled setting.
- `02_medium_cifar10/` — CIFAR-10 with a small CNN. Measures the same metrics on real image data.
- `03_large_cifar100/` — CIFAR-100 with ResNet18. Tests whether the metric behavior carries up in scale to a harder problem with a deeper model.

Each experiment folder is self-contained: `README.md` (claim + design), `run.py` (produces JSON), `plot.py` (reads JSON, produces figures), `results/run_<id>/` for outputs.

## What we're testing

The four `FocusedWork/` documents define interference and the procedure to measure it:

1. **Definition** (§01): interference is structured cancellation in a sum of gradient-like vectors, beyond iid sampling noise.
2. **Two types** (§02): inter-batch (within-batch per-class cancellation) and between-batch (across-batch trajectory cancellation).
3. **§03 inter-batch metrics**: cancellation index $I_{\text{inter}}$, pairwise cosine and magnitude stats over per-class subgradients, useful/wasted decomposition against the full-batch reference, first-order loss-decrease deficit.
4. **§04 between-batch metrics**: cancellation index $I_{\text{between},K}$, pairwise stats within a $K$-window, useful path fraction, per-step deficit.

These experiments empirically test that:

- The metrics are *well-defined* (give sensible values across model sizes and problem difficulties).
- The metrics are *meaningful* (correlate with training behaviour as the theory predicts).
- The two types are *empirically separable* (across scales — already shown at one scale in PreDiscovery).
- A larger model / harder problem either preserves or alters these patterns in interpretable ways.

## Conventions

- **Multi-trial** (≥3 seeds) for headline numbers. Single-trial only for exploration.
- **Reproducibility**: seeds are explicit; same seed → same result.
- **Training and plotting are separate.** `run.py` produces data; `plot.py` reads it.
- **Shared code** lives in `interference/` (or `common/` at repo root for cross-cutting concerns). Never paste utilities between experiments.
- **Outputs** under `<experiment>/results/run_<id>/`. Gitignored except for hand-curated final figures and summary docs.

## Status

| Experiment | Status |
|---|---|
| `interference/` | scaffolded |
| `01_small_2d/` | scaffolded |
| `02_medium_cifar10/` | TODO |
| `03_large_cifar100/` | TODO |
