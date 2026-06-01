# Discovery Phase 2 — BoGrad enhancements

Phase 1 told us the original BoGrad mechanism (project current gradient
orthogonal to recent gradients) helps vanilla SGD substantially but provides
no value on top of momentum or Adam — and in some configurations actively
fights them.

This phase tests five candidate enhancements, each chosen to target a kind
of interference that momentum and Adam genuinely don't address.

## The five variants

### 1. Trajectory projection (long buffer of applied parameter deltas)

**Idea.** Buffer the actually-applied parameter deltas `Δθ_t` over a longer
window (K=32) than the original BoGrad's K=8. Project the next intended
update against this trajectory buffer.

**What it targets.** Anti-cycling. Momentum has effective horizon ~10 steps;
beyond that it forgets. If the optimiser is stuck in a long basin and revisits
the same parameter region every 20–30 steps, momentum can't see that — but a
long-K trajectory buffer can.

**Implementation.** Existing `BoGrad` with `project_stage="update"`,
`buffer_size=32`, `projection_mode="negative"`.

### 2. Complement-aware momentum

**Idea.** Don't fight momentum — explicitly extract the *new* information in
`g_t` that momentum hasn't yet incorporated, and add it as a perpendicular
boost.

For SGD: decompose `g_t = α·v_{t-1} + g_perp` (component along previous
velocity, plus the perpendicular complement). Step:

    θ ← θ - lr · (v_t + γ · g_perp)

For Adam: compute the standard Adam update `u_t = m̂_t / √v̂_t`, decompose
`g_t` into "along u_t" and "perpendicular to u_t", and add a preconditioned
perpendicular boost.

**What it targets.** Momentum's structural blindness to information
orthogonal to its running direction. The new info gets diluted into the EMA
instead of acted on.

**Implementation.** Custom `ComplementMomentumSGD` and `ComplementAdam`
optimisers.

### 3. Adaptive triggering

**Idea.** Default to the base optimiser. Only apply BoGrad's projection when
the current gradient *genuinely fights* the recent direction —
`cos(g_t, ema(g)) < threshold`.

**What it targets.** The phase-1 finding that BoGrad can hurt in regimes
where successive gradients are coherent (which is most of training).
Asymmetric projection improved on full projection by gating on individual
buffer dot products; this gates on the global trend.

**Implementation.** `AdaptiveTriggerBoGrad` wrapper that maintains its own
EMA proxy of recent gradient direction and conditionally enables projection.

### 4. Gradient-difference buffer

**Idea.** Buffer not gradients themselves but `(g_t - g_{t-1})` — the
*change* in gradient. Project current `g_t` against directions of recent
change.

**What it targets.** High-curvature directions where the loss is "ringing"
across mini-batches. The change in gradient is a Hessian-vector product
proxy; projecting against it damps the ringing without touching coherent
descent. A poor man's second-order signal.

**Implementation.** `GradientDifferenceBoGrad` — same projection algorithm,
different buffer contents.

### 5. Multi-scale buffer

**Idea.** Two buffers: short K_s=4 for immediate mini-batch noise, long
K_l=32 for landscape-level oscillation. Apply asymmetric projection against
both in sequence.

**What it targets.** Different timescales of interference. Momentum covers
the short horizon; the long-K signal is something it doesn't see. By
separating the buffers we let each do its own job at the right scale.

**Implementation.** `MultiScaleBoGrad` — two buffers, two projection passes.

## What's tested

For each of `sgd_vanilla`, `sgd_momentum`, `adam`:
- baseline (no BoGrad)
- the applicable subset of variants 1–5

Variant 2 (Complement-aware momentum) is skipped for SGD-vanilla (nothing to
complement). Everything else runs across all three regimes.

Same diagnostic harness as Phase 1: per-step `g_norm`, `u_norm`, `cos_g_prev`,
`cos_u_prev`, `cos_u_neg_g`. Late-training averages summarised in a final
table alongside test accuracy.

## Run

```bash
python discoveryPhase2/ablation_phase2.py --epochs 5
python discoveryPhase2/ablation_phase2.py --epochs 5 --quick
python discoveryPhase2/ablation_phase2.py --regimes adam --epochs 5
python discoveryPhase2/ablation_phase2.py --only adam_complement,adam_baseline
```

## What to look for

For each variant, the bar to clear is **the corresponding momentum/Adam
baseline**, not the vanilla SGD baseline. Phase 1 showed that's where BoGrad
needs to add value if it's going to be a contribution worth the chapter.

Specifically:
- **Variant 2 (Complement)** — should beat both SGD+momentum and Adam if the
  "explicit perpendicular amplification" thesis is right. This is the most
  theoretically clean variant.
- **Variant 3 (Adaptive)** — should at least *not hurt*. The hypothesis is
  that defer-by-default protects against the regime mismatch that broke phase
  1. If it ties the baseline cleanly, that's a win — it means the projection
  is harmless when not needed and we can safely include it as insurance.
- **Variant 4 (Gradient-difference)** — should reduce `cos_u_prev` noticeably
  (it's targeting curvature-driven oscillation which manifests as
  step-to-step update direction changes).
- **Variant 5 (Multi-scale)** — should help if there's a real long-horizon
  signal that single-K BoGrad misses. If the long buffer alone (variant 1)
  already captures it, this adds nothing.
- **Variant 1 (Trajectory)** — basically variant 5's long-buffer alone. If
  it beats baseline by itself, that's the simplest enhancement worth keeping.

If none of these beat their baselines, the honest conclusion is that BoGrad's
contribution is bounded to the vanilla-SGD regime and the thesis needs to
reframe accordingly.
