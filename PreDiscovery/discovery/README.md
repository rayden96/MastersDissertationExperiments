# Discovery — BoGrad / Adam ablation

This folder is for exploratory experiments that aren't part of the formal thesis
plan but inform what goes into it. Right now it contains a focused ablation
investigating whether BoGrad helps any base optimiser (vanilla SGD, SGD +
momentum, Adam) when projection is applied at different points and in different
ways.

## Why this exists

The orthogonality story in the IJCNN paper assumed Euclidean orthogonality
between raw gradients was the right invariant to enforce. Two suspicions push
back on that:

1. **Adam has its own geometry.** Adam's update is `m̂_t / (√v̂_t + ε)`. Making
   raw gradients Euclidean-orthogonal does not make Adam's *updates*
   orthogonal. It also doesn't make gradients orthogonal in the natural
   `1/√v_t`-weighted inner product Adam effectively uses.
2. **Momentum re-injects what BoGrad just removed.** With β₁ = 0.9, Adam's
   momentum buffer has effective horizon ~10 steps — the same window BoGrad's
   K=8 buffer covers. The projection at step t is partially undone by the EMA
   from steps t-1, t-2, …
3. **High-dimensional gradients are nearly orthogonal already.** Persistent
   *negative* alignment is unusual; positive alignment is the normal case and
   often a *good* signal (coherent descent direction). Symmetric projection
   may be removing useful structure rather than destructive interference.

The ablation tests projection variants that target each of these issues, on
each of the three optimiser regimes.

## Variants tested

For **SGD vanilla** (no momentum, no weight decay):
- `baseline` — pure SGD
- `bograd_grad_full` — original BoGrad, full projection
- `bograd_grad_neg` — original BoGrad, asymmetric (only remove negatively-aligned components)

For **SGD + momentum** (β = 0.9):
- `baseline` — SGD + momentum
- `bograd_grad_full` — project g_t, then momentum picks up projected g
- `bograd_grad_neg` — same but asymmetric
- `bograd_update_full` — project the applied parameter delta (= projecting velocity, equivalent up to scale)
- `bograd_update_neg` — same but asymmetric

For **Adam**:
- `baseline` — pure Adam
- `baseline_lr_low` / `baseline_lr_high` — control runs at flanking LRs to bracket the implicit LR-boost effect
- `bograd_grad_full` — project g_t before Adam updates m, v (the geometrically-incoherent variant, kept for reference)
- `bograd_grad_neg` — asymmetric grad-stage
- `bograd_momentum_neg` — project m_t (the EMA, before bias correction and v scaling)
- `bograd_natural_neg` — project g_t in Adam's natural metric `⟨a, b⟩_v = Σ a_j b_j / (√v_j + ε)`
- `bograd_update_neg` — project the final Adam update u_t = m̂/√v̂

`_full` = remove full projected component; `_neg` = remove only negatively-aligned components (PCGrad-style asymmetric projection).

## Diagnostics captured per run

Per logging step, aggregated across all parameter tensors of the model:
- `g_norm` — raw gradient norm
- `u_norm` — applied update norm `‖θ_after − θ_before‖`
- `cos_g_prev` — cos(g_t, g_{t-1}) — raw gradient alignment between successive steps
- `cos_u_prev` — cos(u_t, u_{t-1}) — *update* alignment, which is the one that actually maps to interference in parameter space
- `cos_u_g` — cos(u_t, g_t) — descent-direction quality (negative ≈ valid descent)

These let us check whether each variant actually does what it claims (e.g. is `bograd_natural_neg` decorrelating updates, or just gradients?), and whether the headline accuracy effect is mediated by projection geometry or by the implicit learning-rate boost from norm reduction.

## How to run

```bash
python discovery/ablation_cifar10.py --epochs 5
python discovery/ablation_cifar10.py --epochs 5 --only adam_bograd_natural_neg,adam_baseline
python discovery/ablation_cifar10.py --epochs 5 --quick   # 1/4 of CIFAR-10 train set, fastest sanity check
```

Results are written to `discovery/results/<run_id>/`:
- `results.json` — aggregated final metrics for every variant
- `histories/<variant>.json` — per-step diagnostic series for each variant

A standalone plot script will render trajectory and diagnostic panels from those files.

## Reading the output

The runner ends with a summary table. Things to look for:

1. Does *any* `_neg` variant beat its corresponding `_full` variant? (Tests the asymmetric-projection hypothesis.)
2. Does any Adam variant beat Adam baseline at *both* the standard and re-tuned LRs? (If only at standard LR, the win is implicit-LR-boost, not projection.)
3. Compare `cos_u_prev` traces between baseline and BoGrad. If BoGrad isn't reducing `cos_u_prev` meaningfully (vs `cos_g_prev`), the projection isn't accomplishing what it set out to do for that optimiser.
