# Types of interference

> Two scales, same algebraic kernel.

## The two scales

The cancellation signature defined in §01 applies whenever the optimiser combines multiple gradient-like vectors. Two such combinations matter for SGD-family single-task training, at two different scales.

### Type A — Inter-batch (within one batch)

The batch gradient is a weighted sum of per-class subgradients:

$$g_t \;=\; \frac{1}{|B_t|}\sum_{(x,y)\in B_t} \nabla_\theta\,\ell(x,y) \;=\; \sum_{c \in C_t} \frac{n_{t,c}}{|B_t|}\, g_{t,c},$$

where $C_t$ is the set of classes present in $B_t$, $n_{t,c}$ is the count of class-$c$ samples in $B_t$, and $g_{t,c}$ is the gradient of the average loss on those samples. The objects being summed are **per-class subgradients within one batch**.

- *Mechanism.* Different classes ask for different parameter changes. Averaging them mixes those requests; conflicting requests dilute each other in the average, reducing magnitude and degrading direction.
- *Targeted by.* COSGD — Gram-Schmidt-orthogonalises the per-class subgradients before summing.
- *Scale.* One batch.

### Type B — Between-batch (across $K$ steps)

The net displacement of the optimiser over a window of $K$ consecutive steps is the sum of applied updates:

$$\theta_t - \theta_{t-K} \;=\; \sum_{i=t-K+1}^{t} u_i.$$

The objects being summed are **applied updates over time**.

- *Mechanism.* Successive updates point in conflicting directions, so the path length $\sum_i \lVert u_i\rVert$ exceeds the net displacement $\bigl\lVert \sum_i u_i\bigr\rVert$. Work is wasted as the trajectory zigzags.
- *Targeted by.* BoGrad — orthogonalises the current step (or current gradient) against a buffer of recent updates.
- *Scale.* $K$ steps. Different choices of $K$ probe different temporal horizons.

## Why these are distinct, not redundant

| | Type A — inter-batch | Type B — between-batch |
|---|---|---|
| Summation index | classes within a single batch | steps over time |
| Object summed | per-class subgradient $g_{t,c}$ | applied update $u_i$ |
| Sum equals | batch gradient $g_t$ | net displacement $\theta_t - \theta_{t-K}$ |
| Cancellation reduces | $\lVert g_t\rVert$ vs. $\sum_c \lVert g_{t,c}\rVert$ | $\bigl\lVert\sum_i u_i\bigr\rVert$ vs. $\sum_i \lVert u_i\rVert$ |
| Reference for "useful" | full-batch gradient $\tilde g_t$ | full-batch gradient $\tilde g_{t-K}$ at window start |
| Method that targets it | COSGD | BoGrad |

The two types operate on different objects at different scales. A method that affects one need not affect the other — they are mechanistically separable. That separability is itself an empirical claim to be tested, not assumed.

## What is not in scope

- **Cancellation between simultaneous task gradients** (multi-task; PCGrad). Different problem.
- **Cancellation between task-specific parameter updates over sequential tasks** (continual learning). Different setting.
- **Cancellation within preconditioned updates** (Adam, RMSprop). The same cancellation principle applies, but the accounting changes under preconditioning. Deferred.

## A working assumption to be tested

Both COSGD and BoGrad act on the *angle* component of cancellation:

- COSGD's Gram-Schmidt step zeroes pairwise inner products among the per-class subgradients (the cosine cross-term in the inter-batch cancellation signature).
- BoGrad's projection zeroes the inner product between the current step and a buffered direction (the cosine cross-term in the between-batch cancellation signature).

Neither method modifies magnitudes or conditions on curvature.

The framework therefore takes "angle is what matters" as a **working hypothesis**: that the cosine cross-term in the cancellation signature is the part of interference that translates into training-hurt. Whether this holds — versus a stricter PCGrad-style criterion that additionally requires high curvature and large magnitude difference — is an empirical question. §03 and §04 measure both angle statistics and magnitude/curvature statistics so the assumption can be checked, not asserted.
