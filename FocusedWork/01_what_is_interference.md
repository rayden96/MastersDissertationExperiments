# What is interference (single-task supervised setting)

> Working definition. Subject to revision as measurement work in §03 and §04 lands.

## Setting

Single-task supervised classification. At step $t$:

- Parameters $\theta_t \in \mathbb{R}^p$.
- Mini-batch $B_t$ of $m$ samples drawn iid from the dataset $\mathcal{D}$.
- Per-batch loss $\ell(\theta_t; B_t)$ and gradient $g_t = \nabla_\theta\, \ell(\theta_t; B_t)$.
- The optimiser produces an applied update $u_t$, and $\theta_{t+1} = \theta_t + u_t$.

For SGD (the scope of this document), $u_t = -\eta\, g_t$. With momentum, $u_t = -\eta\, v_t$ where $v_t = \mu\, v_{t-1} + g_t$. Adam and other adaptive optimisers are deferred — see *Scope* at the end.

The full-batch gradient at $\theta_t$ is $\tilde g_t = \nabla_\theta\, \ell(\theta_t; \mathcal{D})$. Under iid mini-batch sampling, $\mathbb{E}[g_t] = \tilde g_t$, so

$$g_t = \tilde g_t + \xi_t,$$

with $\xi_t$ a mean-zero mini-batch sampling-noise term, independent across $t$. **Anything observed in the sequence $\{g_t\}$ that is *not* iid noise must come from somewhere structural** — the trajectory $\{\theta_t\}$ on a non-flat loss surface, or the batch composition itself.

## The phenomenon

In SGD-family training, the optimiser repeatedly forms sums or averages of gradient-like vectors. Two such combinations matter:

1. **Within one batch.** The batch gradient is the average of per-class subgradients. If the per-class subgradients pull in conflicting directions, the average has reduced magnitude relative to the sum of per-class magnitudes.
2. **Across many batches.** Net trajectory displacement is the sum of applied updates over a window. If successive updates pull in conflicting directions, the net displacement is shorter than the path length — the trajectory zigzags.

Both cases are a sum of vectors $\{x_i\}$. Algebraically:

$$\Bigl\lVert \sum_i x_i \Bigr\rVert^2 \;=\; \sum_i \lVert x_i\rVert^2 \;+\; 2 \sum_{i<j} \langle x_i,\, x_j\rangle.$$

The cross term $\sum_{i<j} \langle x_i, x_j\rangle$ is the **cancellation signature** of the sum. Positive ⇒ vectors cooperate. Negative ⇒ vectors oppose; the sum's magnitude collapses below what the individual magnitudes alone would predict. This identity is the algebraic kernel of every interference quantity in this framework.

## Working definition

> **Interference** is structured cancellation in a sum or average of gradient-like vectors, beyond what iid sampling noise alone would produce, that *reduces the loss-decrease the optimiser obtains* relative to a no-cancellation reference.

Three load-bearing pieces:

- *Structured.* Distinct from iid mini-batch noise, whose contribution to the cancellation signature averages to zero in expectation and shrinks with batch (or window) size.
- *Reduces loss-decrease.* The test of "did it hurt training?" is whether the actual per-step (or per-window) loss change is worse than a no-cancellation reference would have produced.
- *Reference-relative.* "No-cancellation" requires an anchor. The natural anchor is the full-batch gradient $\tilde g_t$ — it carries no mini-batch noise and represents the true population descent direction at the current parameters.

Concretely, we measure interference along three quantities, defined per type in §03 and §04:

- the cancellation index (1 = no cancellation, lower = more);
- the useful-vs-wasted decomposition of each constituent vector against the reference;
- the per-step loss-decrease deficit relative to the reference.

The first is purely geometric. The second projects geometry onto descent. The third connects the geometry to actual training outcomes.

## Distinction from related concepts

- **Mini-batch sampling noise.** $\xi_t$ above. iid, mean-zero, decays as the batch grows. Always present in stochastic optimisation. Not interference: its cancellation contribution averages away. Interference is what remains *after* sampling-noise expectation is taken into account.
- **Multi-task gradient interference** (PCGrad, GradVac). Conflict between *task-specific* gradients within one update for a multi-objective loss. Different problem.
- **Catastrophic forgetting** (continual learning). Performance regression on task A while training task B. Different setting.

This framework adopts the term "interference" for the single-task case as a working term. It is not standard — adjacent literature uses "gradient diversity", "gradient noise scale", "oscillation", and "effective step size" for partly-overlapping concepts. The choice of term is to be revisited if a clearer label emerges from the measurement work.

## Scope of this study

- **Optimiser.** SGD, optionally with momentum. The decomposition $u_t = -\eta g_t$ (or $u_t = -\eta v_t$ with $v_t$ a known function of past gradients) is what makes the loss-decrease accounting in §03 and §04 tractable.
- **Methods of interest.** COSGD targets within-batch (per-class) cancellation. BoGrad targets across-batch (trajectory) cancellation. Both currently act on the *angle* component of cancellation only — they do not modify magnitudes or condition on curvature. Whether angle alone is sufficient for training-hurt is an empirical question we test in §03 and §04, not a definitional choice.
- **Out of scope (deferred).** Adam and adaptive optimisers — preconditioning breaks the simple $u \propto g$ relationship; the cancellation principle still applies but the loss-decrease accounting needs to be re-derived under preconditioning. A focused study at that point would identify exactly which formulas change and which carry over.
