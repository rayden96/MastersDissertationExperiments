# Measuring inter-batch interference

> Type A: within-batch cancellation across per-class subgradients.

## What we want from the measurement

For each batch we instrument, we want a separate answer to three questions:

1. **How much cancellation is there?** A scalar in $(0, 1]$ that says how much magnitude was lost when averaging the per-class subgradients.
2. **Of the gradient that was applied, how much was useful?** A decomposition against the full-batch reference $\tilde g_t$, separating the descent-aligned and non-descent components.
3. **Did this step hurt training?** The actual per-step loss change versus the loss change a full-batch step would have produced from the same $\theta_t$.

These are different quantities and are tracked separately. The third is the headline candidate for *"how much did interference hurt training"*.

## Quantities computed at step $t$

Let $\{g_{t,c}\}_{c \in C_t}$ be the per-class subgradients in batch $B_t$ and $C_t$ the set of classes present. Let $\tilde g_t$ be the full-batch gradient at $\theta_t$, with unit form $\hat{\tilde g}_t = \tilde g_t / \lVert\tilde g_t\rVert$.

**(1) Cancellation index.**

$$I_{\text{inter}}(t) \;=\; \frac{\bigl\lVert \sum_{c \in C_t} g_{t,c}\bigr\rVert}{\sum_{c \in C_t} \lVert g_{t,c}\rVert} \;\in\; (0,\, 1].$$

$I_{\text{inter}} = 1$ ⇒ all per-class subgradients aligned, no cancellation. Lower ⇒ more cancellation. Note this uses unweighted sums; the batch gradient itself is class-frequency-weighted, which is a separate thing — the cancellation index is a property of the *class structure*, not of the batch composition.

**(2a) Pairwise angle statistics over $\{g_{t,c}\}$.**

For all unordered pairs $(c, c')$ with $c < c'$, compute $\cos(g_{t,c}, g_{t,c'})$. Aggregate to:

- fraction with cosine $< 0$ (the historical IB_%neg);
- mean cosine;
- the full pairwise distribution (retained for later analysis).

These are the angular signatures most directly targeted by COSGD's projection.

**(2b) Magnitude statistics over $\{g_{t,c}\}$.**

For each class $c$ present, the magnitude $\lVert g_{t,c}\rVert$. Aggregate to:

- mean and standard deviation across classes;
- max-to-min ratio (PCGrad's "large magnitude difference" criterion).

Tracked separately from the angle statistics so we can later test whether angle alone explains training-hurt or whether magnitude differences contribute.

**(3) Useful-vs-wasted decomposition.**

For each per-class subgradient, project onto the reference:

- useful component (signed scalar): $u_c = \langle g_{t,c},\, \hat{\tilde g}_t\rangle$;
- wasted component magnitude: $w_c = \bigl\lVert g_{t,c} - u_c\, \hat{\tilde g}_t\bigr\rVert$.

For the batch gradient $g_t$ as a whole (post-averaging):

- useful descent fraction: $\langle g_t,\, \hat{\tilde g}_t\rangle / \lVert g_t\rVert \;\in\; [-1,\, 1]$.

Because $g_t$ and $\tilde g_t$ both point uphill, the useful descent fraction is positive when $g_t$ is descent-aligned with the population gradient. Aggregate per-class useful and wasted components into the batch-level *useful mass* $\sum_c u_c$ and *wasted mass* $\sum_c w_c$.

**(4) First-order loss-decrease accounting.**

By a first-order Taylor expansion of $L$ at $\theta_t$:

$$\Delta L_t \;\approx\; \langle\nabla L(\theta_t),\, u_t\rangle \;\approx\; \langle\tilde g_t,\, u_t\rangle.$$

For SGD with $u_t = -\eta g_t$:

- *Ideal* (full-batch step from this $\theta_t$): $\Delta L_t^{\text{ideal}} \approx -\eta\, \lVert\tilde g_t\rVert^2$.
- *Actual first-order*: $\Delta L_t^{\text{first}} \approx -\eta\, \langle\tilde g_t,\, g_t\rangle$.
- *Per-step first-order deficit*:

$$D_t \;=\; \Delta L_t^{\text{ideal}} - \Delta L_t^{\text{first}} \;=\; -\eta\,\lVert\tilde g_t\rVert^2 + \eta\,\langle\tilde g_t,\, g_t\rangle \;=\; \eta\bigl(\langle\tilde g_t,\, g_t\rangle - \lVert\tilde g_t\rVert^2\bigr).$$

A positive $D_t$ means the SGD step's first-order loss-decrease is less than the full-batch step's would have been — the per-step training-hurt attribution. If $g_t = \tilde g_t$ exactly, $D_t = 0$. If $g_t$ is anti-parallel to $\tilde g_t$, $D_t = 2\eta\lVert\tilde g_t\rVert^2$.

For SGD with momentum, replace $g_t$ with the velocity $v_t$ in the actual term: $\Delta L_t^{\text{first}} \approx -\eta\, \langle\tilde g_t,\, v_t\rangle$. The ideal can be defined either as one full-batch SGD step ($-\eta\lVert\tilde g_t\rVert^2$) or as one full-batch *momentum* step under a counterfactual full-batch velocity buffer; the SGD-step ideal is simpler and is what we use as the default.

**(5) Measured loss change (calibration).**

The full-batch loss $L(\theta_t; \mathcal{D})$ and $L(\theta_{t+1}; \mathcal{D})$, computed on the reference set used to compute $\tilde g_t$. The measured per-step change $\Delta L_t^{\text{meas}}$ includes curvature and higher-order effects that the first-order deficit (4) does not. Tracked sparsely (every $N$ steps) as a sanity check on the first-order story.

## Run-level summaries

- $\sum_t D_t$ — cumulative first-order training-hurt over the run. **Headline candidate for "how much did interference hurt training".**
- mean and distribution of $I_{\text{inter}}(t)$.
- mean useful descent fraction across logged steps.
- correlation coefficients (across logged steps within a run): $I_{\text{inter}}$ vs $D_t$, and mean pairwise cosine vs $D_t$. Tells us whether the geometric/angle summaries actually predict training-hurt at the per-step level — i.e. whether the angle-only working assumption holds.
- $\sum_t D_t^{\text{meas}} := \sum_t (\Delta L_t^{\text{ideal}} - \Delta L_t^{\text{meas}})$ on the sparse calibration grid. If this tracks $\sum_t D_t$ closely, the first-order accounting is sufficient; if it diverges, second-order effects matter and the framework needs to expand.

## Procedure

1. **Construct a reference set.** Sample a balanced subset of size $R$ from training data, large enough that the gradient on it is a low-variance estimate of the population gradient. Default: $R = 10\%$ of the training set, balanced across classes, fixed for the run. (For very small datasets, $R = $ full training set.)
2. **Refresh the reference gradient.** Every $N$ steps (default $N = 50$), compute $\tilde g_t$ on the reference set and store it. Hold $\tilde g_t$ fixed between refreshes — treated as a slowly-varying anchor.
3. **At each logged step**, on the current training batch:
   - Compute the per-class subgradients $\{g_{t,c}\}$ via masked backward passes (one backward per class present in the batch).
   - Compute (1), (2a), (2b), (3), (4) from the stored $\tilde g_t$.
   - Log everything.
4. **Sparse calibration.** At every reference refresh, additionally compute $L(\theta_t)$ before the step and $L(\theta_{t+1})$ after — gives $\Delta L_t^{\text{meas}}$.

## Implementation notes

- Per-class subgradients require one extra backward pass per class present. With CIFAR-10 and a typical batch containing all 10 classes, this is a 10× cost overhead at each logged step. Mitigation: log every $L$ steps rather than every step (the per-class subgradient computation is the dominant cost).
- The reference-set forward+backward at $N = 50$ is amortised cheap.
- Store $\tilde g_t$ in a parameter-shaped buffer; refresh in place.
- The full-batch loss calibration adds one extra forward on the reference set per refresh — also amortised cheap.

## A note on the angle-only working assumption

COSGD modifies the cosines among $\{g_{t,c}\}$ via Gram-Schmidt. It does not modify the per-class magnitudes and does not condition on Hessian information. The framework tracks both angle statistics (2a) and magnitude statistics (2b) so that, after a run, we can ask: does $\sum_t D_t$ correlate better with the cosine summary or with the magnitude summary?

If with cosines, the angle-only assumption is empirically supported and COSGD's mechanism is well-targeted. If with magnitudes (or with their interaction), then the framework's measurements remain valid but the methods need a magnitude- or curvature-aware extension — itself a separate, falsifiable contribution.
