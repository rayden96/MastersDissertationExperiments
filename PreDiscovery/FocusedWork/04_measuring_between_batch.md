# Measuring between-batch interference

> Type B: across-step cancellation in the trajectory.

## What we want from the measurement

For each window of $K$ consecutive steps we instrument, we want a separate answer to three questions:

1. **How much cancellation is there over this window?** A scalar in $(0, 1]$ that says how much path length was wasted relative to net displacement.
2. **Of the trajectory we walked, how much was useful?** A decomposition of each update against the full-batch reference at the window start.
3. **Did this window hurt training?** The aggregate per-step loss-decrease deficit over the window — same quantity as in §03, accumulated over the window.

These are different quantities and are tracked separately. (3) is the headline candidate for *"how much did this region of training get hurt by between-batch interference"*.

## Quantities computed over a window of $K$ steps ending at $t$

Let $\{u_i\}_{i = t-K+1}^{t}$ be the applied updates over the window, with $\theta_t - \theta_{t-K} = \sum_i u_i$. Let $\tilde g_{t-K}$ be the full-batch gradient at the window start, with unit form $\hat{\tilde g}_{t-K} = \tilde g_{t-K}/\lVert\tilde g_{t-K}\rVert$. The "descent direction" in step space is $-\hat{\tilde g}_{t-K}$ (since updates point downhill while the gradient points uphill).

**(1) Cancellation index.**

$$I_{\text{between},K}(t) \;=\; \frac{\bigl\lVert \theta_t - \theta_{t-K}\bigr\rVert}{\sum_{i=t-K+1}^{t} \lVert u_i\rVert} \;\in\; (0,\, 1].$$

$I_{\text{between},K} = 1$ ⇒ all updates aligned in the same direction (straight-line trajectory). Lower ⇒ more cancellation/zigzag. This is the WW_K metric in earlier work, re-derived here as an instance of the cancellation principle.

**(2a) Pairwise angle statistics within the window.**

For all unordered pairs $(i, j)$ with $i < j$: $\cos(u_i, u_j)$. Aggregate to:

- fraction with cosine $< 0$;
- mean cosine;
- mean cosine as a function of lag $|i - j|$ (the angular *decay profile* over the window).

The angular signatures targeted by BoGrad's projection.

**(2b) Magnitude statistics over $\{u_i\}$.**

Per-step magnitude $\lVert u_i\rVert$. Aggregate to:

- mean and standard deviation across the window;
- max-to-min ratio.

Tracked separately from angle statistics.

**(3) Useful-vs-wasted decomposition against the window-start reference.**

For each update $u_i$ in the window:

- useful descent component (signed scalar; positive means this step was descent-aligned with the population gradient at window start): $\tilde u_i = -\langle u_i,\, \hat{\tilde g}_{t-K}\rangle$;
- wasted component magnitude (perpendicular to descent): $w_i = \bigl\lVert u_i + \tilde u_i\, \hat{\tilde g}_{t-K}\bigr\rVert$ — equivalently, $\sqrt{\lVert u_i\rVert^2 - \tilde u_i^2}$.

Window aggregates:

- *useful path*: $\sum_i \tilde u_i$ (signed; positive = net descent over the window);
- *wasted path*: $\sum_i w_i$ (always non-negative);
- *useful path fraction*: $\bigl\lvert \sum_i \tilde u_i\bigr\rvert \,/\, \sum_i \lVert u_i\rVert \;\in\; [0,\, 1]$.

The useful path fraction can stay high under a curving (but cooperative) trajectory; only true cancellation drives it down. This is what disambiguates cancellation from curvature — see *Curvature confound* below.

**(4) First-order loss-decrease accounting (per step within the window).**

Per step $i$, the first-order deficit (defined exactly as in §03):

$$D_i \;=\; \eta\bigl(\langle\tilde g_i,\, g_i\rangle - \lVert\tilde g_i\rVert^2\bigr).$$

This requires $\tilde g_i$ at every step. In practice, the reference is sampled every $N$ steps (shared with §03's protocol); $\tilde g_i$ at intermediate steps is approximated by the most recent reference.

Window total: $D_K(t) = \sum_{i=t-K+1}^{t} D_i$.

For SGD with momentum, replace $g_i$ with $v_i$ as in §03.

**(5) Measured loss change over the window (calibration).**

$\Delta L_K^{\text{meas}}(t) = L(\theta_t; \mathcal{D}) - L(\theta_{t-K}; \mathcal{D})$, computed on the reference set. Tracked sparsely.

## Run-level summaries

- $\sum_t D_t$ — cumulative per-step first-order deficit. *Same number as in §03* — the per-step deficit is a per-step quantity regardless of which type's lens we view it through. Inter-batch and between-batch are two *causes* contributing to this single training-hurt total.
- mean and distribution of $I_{\text{between},K}(t)$, for each $K$ tracked.
- mean useful path fraction.
- mean cosine within window, and the mean-as-function-of-lag decay profile.
- correlation (within a run, across logged windows): $I_{\text{between},K}$ vs $D_K$, mean pairwise cosine vs $D_K$, useful path fraction vs $D_K$. Tells us whether the geometric summaries predict training-hurt at the window level.

## On the curvature confound

A trajectory can have $I_{\text{between},K} < 1$ for two distinct reasons:

- **Cancellation.** Successive updates oppose along a shared direction. True interference.
- **Curvature.** The trajectory naturally bends on a smooth loss surface. Successive updates point in *different* but not *opposing* directions; in fact they may all be descent-aligned at each point.

The cancellation index alone cannot separate these. The useful path fraction (3) does separate them: a curving-but-cooperative trajectory has high useful path fraction (each step is descent-aligned even if the descent direction is rotating); a zigzagging trajectory has low useful path fraction (steps alternate against descent).

Track both. If $I_{\text{between},K}$ is low *and* useful path fraction is low ⇒ cancellation. If $I_{\text{between},K}$ is low *but* useful path fraction is high ⇒ curvature, not interference.

## Choice of $K$

$K$ sets the temporal scale of the cancellation question.

- *Small $K$* ($K \approx 4$): captures short-horizon zigzag — the regime momentum smooths.
- *Medium $K$* ($K \approx 32$): multi-batch trajectory structure — the typical BoGrad operating regime.
- *Large $K$* ($K \approx 128$): epoch-fraction trajectory — closer to net training direction.

Default: track all three. The headline $K$ is whichever is most predictive of $D_K$ in calibration — an empirical output, not a pre-commitment.

## Procedure

1. **Reference set and refresh.** Same as §03 — share the protocol. Reference set $R$, refreshed every $N$ steps.
2. **Maintain a rolling buffer** of the most recent $K_{\max}$ applied updates and parameter snapshots. Default $K_{\max} = 128$.
3. **Maintain a rolling buffer of past references** $\tilde g_{t-K}$ at the window-start time, sized to cover all $K$ being tracked.
4. **At each logged window end** (every step, or every $L$ steps for efficiency):
   - For each $K$ in the tracking set, compute (1), (2a), (2b), (3) using the current buffer and the corresponding window-start reference.
   - Compute $D_i$ for the just-completed step using the most recent $\tilde g$ (this contributes to the running $\sum_t D_t$).
5. **Sparse calibration.** At reference refreshes, compute $L(\theta_{t-K})$ and $L(\theta_t)$ on the reference set for the active $K$ values.

## Implementation notes

- The buffer of past updates is parameter-shaped × $K_{\max}$; potentially large for big models. If the training optimiser already maintains an update buffer (e.g. BoGrad does), share storage.
- The reference-gradient buffer (one $\tilde g$ per refresh time held in memory for window analysis) is small in count but parameter-shaped each. For deep models, store at half-precision or as a low-rank sketch.
- Computing pairwise cosines within $K$-buffer is $O(K^2)$ in number of dot products and $O(K \cdot p)$ in time; for $K = 128$ this is fine on modest GPU.

## A note on the angle-only working assumption

BoGrad's Gram-Schmidt against the buffer modifies the cosine of the current update with respect to past updates. It does not modify magnitudes and does not condition on curvature. As in §03, the framework tracks both angle and magnitude statistics so we can ask whether $D_K$ correlates with the cosine summary, the magnitude summary, or their interaction.

If with cosines, the angle-only assumption is supported and BoGrad's mechanism is well-targeted. If not, the framework remains valid but BoGrad's design needs a magnitude- or curvature-aware extension.
