# 01 — Small 2D toy

## Claim being tested

The interference metrics defined in `PreDiscovery/FocusedWork/03_*.md` and `04_*.md` are **well-defined and behave as the theory predicts in a controlled setting**. A 2D mixture-of-quadratics with known properties lets us:

1. visualise the optimization trajectory in parameter space alongside the metrics;
2. verify that the cancellation index responds to inter-batch class conflict (per-component disagreement) and between-batch trajectory zigzag the way §03 / §04 say it should;
3. confirm the useful-vs-wasted decomposition makes geometric sense against a known full-batch reference (the average of the mixture centres).

The 2D setting is small enough that there is no ambiguity about what "interference" looks like — it's literally visible.

## Design

**Problem.** $K$ quadratic terms in 2D, each pulling parameters $\theta \in \mathbb{R}^2$ toward its own centre $\mu_k$:

$$L_k(\theta) = \tfrac{1}{2}\lVert \theta - \mu_k\rVert^2, \quad L(\theta) = \frac{1}{K}\sum_{k=1}^K L_k(\theta).$$

Mixture centres $\mu_k$ are placed on a circle of radius $R$ around the origin. The full-data optimum is at the centroid (= origin by symmetry). The per-component gradient $\nabla L_k(\theta) = \theta - \mu_k$ pulls toward $\mu_k$. Different components disagree → strong inter-batch interference signal.

**Subgroups** in the §03 sense are the mixture components: per-subgroup gradient = per-component gradient on the batch.

**Training.** Vanilla SGD on minibatches drawn iid from $\{1, \ldots, K\}$. The "data" is just the component index.

**Reference gradient.** The mean of the per-component gradients (= $\theta - \bar\mu$). Exact, no estimation noise.

**What's measured.** Everything in `interference.InterferenceMeter`: cancellation indices $I_{\text{inter}}$ and $I_{\text{between},K}$, pairwise cosine + magnitude stats, useful/wasted decomposition, per-step first-order deficit $D_t$, cumulative deficit, loss calibration.

## Predicted behaviour

- $I_{\text{inter}}$ should be **low** early in training (mixture components disagree strongly because $\theta$ is far from the centroid → strong cancellation when averaging) and **rise** as $\theta$ approaches the centroid (per-component gradients align with the descent direction = all pointing at the centroid from $\theta$'s side).
- The mean pairwise cosine between per-component gradients should start near $0$ or negative and rise.
- $I_{\text{between},K}$ behaviour depends on $K$ and LR: large LR → trajectory zigzags → low $I_{\text{between}}$. Small LR → straight descent → high $I_{\text{between}}$.
- Useful descent fraction of the batch gradient should be high near the centroid (where batch grad ≈ full-batch grad) and lower far away.
- The per-step deficit $D_t$ should track the cancellation: when interference is high, the batch direction is far from the full-batch direction, so $D_t$ is larger (the deficit is non-negative; larger = more first-order descent lost to interference).

## Outputs

- `results/run_<id>/logs_seed<S>.json` — per-step logs per seed.
- `results/run_<id>/summary_seed<S>.json` — run summary per seed.
- `results/run_<id>/trajectory_seed<S>.npy` — full $\theta$ trajectory (every step).
- `results/run_<id>/figs/` — trajectory plot, metric time-series, multi-seed mean/std bands.

## How to run

```bash
python PaperReadyExperiments/01_small_2d/run.py        # default config, 5 seeds
python PaperReadyExperiments/01_small_2d/plot.py       # reads latest run, makes figures
```

Configuration knobs in `run.py`: `n_groups`, `R` (radius of centre circle), `batch_size`, `lr`, `n_steps`, `K_values`, `seeds`.
