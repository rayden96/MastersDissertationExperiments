# MoGrad — Momentum-orthogonalised gradient

## The idea

Maintain a momentum-style EMA of past *gradients* (or equivalently
*update directions*) — call it $\mathbf{m}_t$. **Do NOT use $\mathbf{m}_t$
in the update equation** (this is the key distinction from standard
momentum). Instead, use $\mathbf{m}_t$ purely as a *reference vector* for
orthogonalisation: project the current gradient against $\mathbf{m}_t$
before applying a vanilla SGD step.

$$
\begin{aligned}
\mathbf{m}_t &= \beta\, \mathbf{m}_{t-1} + (1-\beta)\, g_t \quad \text{(reference, not used in update)} \\
\tilde{g}_t &= g_t - \alpha\, \tfrac{\langle g_t, \mathbf{m}_t\rangle}{\lVert \mathbf{m}_t\rVert^2}\, \mathbf{m}_t \\
\theta_{t+1} &= \theta_t - \eta\, \tilde{g}_t
\end{aligned}
$$

with optional sign-gating ($\alpha = 1$ if `cos(g, m) < 0` for negative
mode, etc.) and a warmup period before projection begins.

## Reasoning

If $\mathbf{m}_t$ is well-stabilised, it points along the local consensus
descent direction. There are two reasonable readings of "orthogonalise
against $\mathbf{m}_t$":

**Reading A — anti-redundancy.** Remove the part of $g_t$ that is already
captured by $\mathbf{m}_t$. Successive steps explore complementary
directions instead of repeating the consensus direction. Compare:
conjugate-gradient methods, where each step is H-orthogonal to previous
steps. Here we approximate that with vanilla-orthogonal-to-momentum.

**Reading B — anti-conflict.** Remove the part of $g_t$ that *fights*
$\mathbf{m}_t$ (negative-mode projection). The current update doesn't undo
recent progress.

These two readings correspond to different `projection_mode`s:
- `"positive"` mode = Reading A (subtract redundant overlap).
- `"negative"` mode = Reading B (subtract destructive overlap).
- `"full"` mode = both.

## Why not just use momentum directly?

Standard momentum applies $\mathbf{m}_t$ in the update: $\theta \to \theta
- \eta\, \mathbf{m}_t$. The momentum vector becomes the dominant step
direction over time.

MoGrad doesn't do that. It updates with $-\eta\, \tilde{g}_t$ where
$\tilde{g}_t$ is the *current gradient with the momentum-direction
component removed/modified*. This means:

- The actual step direction is the residual of the current gradient
  after subtracting the momentum overlap.
- Step magnitudes are bounded by $\lVert g_t\rVert$ rather than amplified
  by momentum accumulation.
- It's structurally "vanilla SGD with a smarter direction", not "vanilla
  SGD with smoothed direction".

Whether this works depends entirely on what's in $\mathbf{m}_t$. If
$\mathbf{m}_t$ is the descent direction, removing it (mode="positive" or
"full") strips signal. If $\mathbf{m}_t$ has stabilised on a stale or
biased direction, removing it helps escape.

## When does projection start?

Projection requires $\mathbf{m}_t$ to have converged. With $\beta = 0.9$
the EMA effective horizon is ~10 steps; with $\beta = 0.99$ it's ~100.
Before $\mathbf{m}_t$ stabilises, projecting against it is noise.

We test multiple `start_step` values (50, 100, 200, 500, 1000) to find
empirically when projection becomes useful.

## What we expect

This is genuinely an open question. Plausible outcomes:

- **MoGrad-positive helps after a long warmup.** Once $\mathbf{m}_t$ is
  stable as the consensus, removing the redundant component leaves
  exploration in complementary directions, possibly finding better
  minima or escaping plateaus.
- **MoGrad-negative helps in noisy regimes.** Removing destructive overlap
  with the consensus direction prevents single bad steps from undoing
  recent progress — same intuition as BoGrad-negative.
- **MoGrad-full hurts (consistent with F10).** Removing the entire
  momentum direction strips the descent signal; the residual gradient is
  low-magnitude noise.
- **Without warmup, projection hurts** because $\mathbf{m}_t$ is just
  noise early in training.

If results match: this confirms the framework's W' criterion holds for
MoGrad too. If `mode="positive"` or `"negative"` helps cleanly at the
right start_step, MoGrad is a viable variant for the future-work section.

## How to run

```bash
# Default: full sweep of start_step values × 3 modes × 3 trials
python MoGrad/run_study.py

# Single mode, single start_step, multiple trials
python MoGrad/run_study.py --modes negative --start-steps 200 --trials 5

# Quick (1/4 train set)
python MoGrad/run_study.py --quick
```

## Output

`results/run_<timestamp>/`:

- `config.json` — the run configuration.
- `<variant>__<start_step>__t<trial>.json` — per-run summary.
- `summary.json` — aggregated mean ± std per (mode, start_step).
- console table at the end.
