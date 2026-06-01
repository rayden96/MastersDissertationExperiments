# An Interference Framework for Single-Task Training

> **Status:** Draft v0.2. Foundational document for Stream 1 of the
> orthogonalisation research plan ([plan link](../../../../Users/rayde/.claude/plans/on-the-questions-1-tidy-goblet.md)).
> Sections 3 and 4 develop the formal framework. Section 8 reports findings
> from the first validation run that informed the formalism. Subsequent
> revisions will fold in synthetic-experiment findings as they land.

## 1. The question

> **Does short-horizon gradient interference meaningfully slow single-task
> training, and how do orthogonalisation techniques (BoGrad, COSGD) — both
> as proposed methods and as instruments of study — illuminate the answer?**

"Single-task" is the operative qualifier. Most prior work on gradient
interference lives in continual learning (forgetting between tasks) or
multi-task learning (conflict between simultaneous task gradients). The
single-task version of the question — *does training on one mini-batch
interfere with what we just learned from the previous mini-batch?* — is far
less studied. It is what BoGrad and COSGD implicitly target, but the question
has rarely been phrased directly. This framework operationalises it.

The framework defines:

1. What "interference" means in a single-task context (§3, §4).
2. How to measure it in a way that distinguishes it from random mini-batch
   noise on one side and from coherent descent signal on the other (§6, §7).
3. What it would take for interference to *cause* training slowdown
   rather than merely correlate with it (§5).

## 2. The setting

We work in standard single-task supervised training. Let $\theta_t \in \mathbb{R}^p$
be the parameter vector at step $t$. Mini-batches $B_t \subset \mathcal{D}$
are drawn iid from the dataset $\mathcal{D}$, $|B_t| = m$. The per-batch loss
is $\ell(\theta; B_t) = \frac{1}{m} \sum_{(x,y) \in B_t} \ell\bigl(\text{model}(\theta, x), y\bigr)$,
and the per-batch gradient is $g_t = \nabla_\theta \ell(\theta_t; B_t)$.

The optimiser produces an update $u_t$ such that $\theta_{t+1} = \theta_t + u_t$.
For the optimisers we study:
- **SGD**: $u_t = -\eta\, g_t$.
- **SGD+momentum**: $u_t = -\eta\, v_t$ with $v_t = \mu\, v_{t-1} + g_t$.
- **Adam**: $u_t = -\eta\, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ with
  $\hat{m}_t$ and $\hat{v}_t$ the bias-corrected first and second moments
  of $g$.

The full-batch reference gradient is $\tilde{g}_t = \nabla_\theta \ell(\theta_t; \mathcal{D})$.
For an iid batch sampler, $\mathbb{E}[g_t] = \tilde{g}_t$, so the mini-batch
gradient decomposes as
$$
g_t = \tilde{g}_t + \xi_t,
$$
where $\xi_t$ is mean-zero mini-batch sampling noise. Crucially, under iid
sampling $\xi_t$ is *independent* across $t$: the noise carries no inter-step
structure of its own. **Any inter-step structure observed in $\{g_t\}$
therefore arises from the trajectory $\{\theta_t\}$**: the loss surface is
not flat, so successive gradients sampled along it carry shared signal beyond
pure batch noise.

This observation will matter repeatedly: it lets us interpret cross-step
correlation in the gradient sequence as a *trajectory effect* rather than as
a sampling artefact.

## 3. From observation to formalism — building the framework

### 3.1 The empirical puzzle motivating the framework

The framework was constructed in response to an inconsistency in our prior
empirical work. Across [phases 1 and 2](../../discoveryPhase2/) and the
[K-sweeps](../../discoveryPhase2/results/), BoGrad's accuracy contribution
depended sharply on the base optimiser. On vanilla SGD with a small CNN on
CIFAR-10, BoGrad provides $+25$ percentage points (a transformative gain).
On SGD+momentum, the well-tuned-vs-well-tuned gap shrinks to $\sim +1$ pt.
On Adam, only $+0.9$ pt at peak after controlling for the implicit
learning-rate confound (see [testing/01_attribution](../../testing/01_attribution/)).
This pattern is inconsistent with a uniform-mechanism story: if BoGrad's
projection were the same kind of correction across optimisers, the accuracy
contribution should also be similar.

The natural interpretation is that orthogonalisation addresses some
*specific* underlying phenomenon — something already partially handled by
some optimisers (momentum, Adam's preconditioning) and not by others.
Identifying that phenomenon and pinning down how to measure it is the problem
this framework solves.

### 3.2 The terminology problem

The natural label for the phenomenon is *gradient interference*: between-step
conflict among the mini-batch gradients used by stochastic optimisation. But
this term is overloaded in the deep-learning literature.

- In **continual learning** (Kirkpatrick et al., 2017; Lopez-Paz & Ranzato,
  2017), "interference" refers to forgetting between sequentially-presented
  *tasks*: training on task $B$ causes performance regression on task $A$.
- In **multi-task learning** (Yu et al., 2020 — PCGrad; Wang et al., 2021 —
  GradVac), "interference" refers to conflict between *simultaneous*
  task-specific gradients within a single update.
- In **single-task supervised training**, the term is used informally but has
  not been given a precise operational definition. This is the gap the
  framework addresses.

Our setting is single-task supervised training: one objective, one mini-batch
stream, all gradients targeting the same loss surface. The relevant question
is: do successive mini-batch updates *interfere* with each other in a way
that meaningfully slows training?

### 3.3 First formulation: geometric interference

A natural first instinct defines interference by the angle between successive
updates:

> **Geometric interference (provisional).** Define
> $$\rho^u_t = \cos\bigl(u_t, u_{t-1}\bigr) = \frac{\langle u_t,\, u_{t-1}\rangle}{\lVert u_t\rVert \, \lVert u_{t-1}\rVert}.$$
> Negative $\rho^u_t$ indicates partial cancellation between successive
> updates and is candidate "interference".

The intuition is the two-step displacement argument:
$$
\lVert u_t + u_{t-1}\rVert^2
= \lVert u_t\rVert^2 + \lVert u_{t-1}\rVert^2 + 2\langle u_t, u_{t-1}\rangle.
$$
When $\langle u_t, u_{t-1}\rangle < 0$, the cross term is negative; the
squared net displacement after two steps is *strictly less than* the sum of
squared step magnitudes. Some "work" is wasted as the second step partially
undoes the first. Symmetrically, $\langle u_t, u_{t-1}\rangle > 0$ produces
super-additive displacement: steps cooperate.

This formulation has three structural limitations.

**Limitation P1 (sign ambiguity).** Positive alignment is not automatically
benign. Three distinct regimes produce $\langle u_t, u_{t-1}\rangle > 0$:

- *Coherent descent* — both updates point along $\tilde{g}_t$ (the
  loss-decreasing direction). This is signal.
- *Stale direction* — the second update largely re-applies the first
  (e.g. momentum-driven repetition along an already-explored direction).
  The optimiser is "double-paying" for one direction of progress.
- *Sampling correlation* — random fluctuations happen to align. Benign at
  expectation but contributes variance.

The cosine alone cannot distinguish these. Symmetric to the positive case,
small negative cosines may also be benign sampling fluctuations rather than
structural interference.

**Limitation P2 (magnitude blindness).** $\cos(u_t, u_{t-1}) = -0.1$ with
$\lVert u\rVert \approx 10^{-3}$ has very different consequences than the
same cosine with $\lVert u\rVert \approx 10$. The geometric metric collapses
both into a single number.

**Limitation P3 (disconnection from outcome).** Even a precisely-measured
cosine does not directly tell us how many steps of training were wasted, how
much accuracy was lost, or whether the interference accumulated into a
meaningful slowdown. The metric is not coupled to a quantity we ultimately
care about.

These limitations motivate two further formulations: one that addresses
(P1) by grounding interference in observable model behaviour, and one that
addresses (P3) by aggregating to the trajectory level.

### 3.4 Second formulation: per-class interference

The per-class formulation grounds interference in observable model
behaviour rather than gradient geometry. We define interference by what it
*does* to the model's predictions.

Define the class-restricted loss:
$$
L_c(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D},\, y = c}\bigl[\ell(\text{model}(\theta, x), y)\bigr],
$$
i.e. the average loss on examples of class $c$ at parameters $\theta$.

> **Forgetting event.** A forgetting event for class $c$ at step $t$ is the
> event $L_c(\theta_t) > L_c(\theta_{t-1})$. The forgetting magnitude is
> $$\Delta_{t,c} = \max\bigl(0, \; L_c(\theta_t) - L_c(\theta_{t-1})\bigr).$$

The total per-step forgetting magnitude is $\sum_c \Delta_{t,c}$. The
cumulative forgetting over a training run is $\sum_t \sum_c \Delta_{t,c}$.

This formulation directly addresses (P1): a regression in $L_c$ is
unambiguously bad. There is no positive-vs-negative-alignment debate; if the
model got worse at class $c$, that step interfered with class $c$'s
training. It also opens a connection to outcome (per §5): the cumulative
forgetting is plausibly correlated with wasted training work.

The formulation has two limitations.

**Limitation P4 (probe overhead).** Computing $L_c$ at every step requires a
forward pass on a held-out probe set. We mitigate this with a small balanced
probe (e.g. 64 examples per class), evaluable in one batch.

**Limitation P5 (step-magnitude confound).** Methods that take larger
per-step parameter movements naturally produce larger per-step changes in
$L_c$, in both directions. Naïve comparison of forgetting magnitudes between
methods with different $\lVert u\rVert$ is unfair to large-step methods. We
introduce step-magnitude-normalised variants in §5.2.

### 3.5 Third formulation: trajectory efficiency

The trajectory formulation operates at the path level, treating the
optimisation trajectory through parameter space as the object of interest.

Over a window of $K$ consecutive steps, define the **wasted-work ratio**:
$$
\mathrm{WW}_K(t) = \frac{\bigl\lVert \theta_t - \theta_{t-K}\bigr\rVert}{\sum_{i=t-K+1}^{t} \lVert u_i\rVert}.
$$

By the triangle inequality, $\mathrm{WW}_K(t) \in (0, 1]$. The ratio
approaches 1 when all steps stack productively (no cancellation); approaches
0 when steps cancel each other (high inefficiency).

This formulation directly addresses (P3): the ratio is itself a quantity of
interest. It also implicitly addresses (P2), since both numerator and
denominator scale with step magnitude — the ratio is dimensionless and
self-normalising.

The trajectory ratio admits a useful expansion. Writing the displacement as
the sum of steps,
$$
\theta_t - \theta_{t-K} = \sum_{i=t-K+1}^{t} u_i,
$$
we have
$$
\bigl\lVert \theta_t - \theta_{t-K}\bigr\rVert^2 = \sum_i \lVert u_i\rVert^2 + 2 \sum_{i < j} \langle u_i,\, u_j\rangle.
$$
The cross-term $\sum_{i<j} \langle u_i, u_j\rangle$ is the **inter-step
alignment integral** over the window. It is positive when steps cooperate
(efficient trajectory), negative when they oppose (wasted work). The
trajectory metric therefore implicitly aggregates *all pairwise* geometric
interference within the window — it generalises the two-step displacement
argument of §3.3 to $K$-step horizons.

**Limitation P6 (curvature confound).** A curved trajectory through a
non-flat loss surface naturally yields ratio $< 1$ even without inter-step
interference: each step bends slightly relative to the previous. Pure
interference (cancellation) and pure curvature (cooperative-but-curved) are
not distinguishable from this metric alone. The geometric and per-class
metrics serve as disambiguators.

### 3.6 Synthesis: three axes, one phenomenon

The three formulations are complementary, not redundant. They answer
different sub-questions:

| Formulation | Question | Resolution | Strength | Weakness |
|---|---|---|---|---|
| Geometric (§3.3) | Are successive updates pulling against each other? | Per-step, fine angular detail | Cheap to compute | Sign-ambiguous (P1), magnitude-blind (P2), outcome-disconnected (P3) |
| Per-class (§3.4) | Is any class's accuracy being damaged? | Per-step, sign-unambiguous | Tied to observable outcome | Probe overhead (P4), step-magnitude-confounded (P5) |
| Trajectory (§3.5) | Are we making efficient progress through parameter space? | $K$-step window, magnitude-normalised | Dimensionless, single-number summary | Curvature-confounded (P6) |

A complete instrumented run captures all three. A method that reduces
interference along one axis but not others is making a specific claim that
the framework will detect. A method that uniformly reduces interference along
all three is a stronger candidate for general-purpose adoption.

The thesis claim — that BoGrad-style orthogonalisation reduces single-task
gradient interference — is meaningful only when made specific to one of these
formulations. The remainder of the document develops each in turn (§4),
states the testable hypothesis connecting them to training speed (§5),
operationalises the measurement (§6), and reports first empirical evidence
(§8).

## 4. The three formal interference axes

Restating the formal definitions assembled in §3:

### 4.1 Geometric axis

The fine-grained per-step alignment metrics:
$$
\rho^g_t = \cos(g_t,\, g_{t-1}),\quad
\rho^u_t = \cos(u_t,\, u_{t-1}),\quad
\rho^{u\bar g}_t = \cos(u_t,\, -g_t).
$$
The first two measure self-consistency of the gradient and update streams;
the third measures whether the actually-applied step is a descent direction
in raw-gradient terms (positive ⇒ descent-aligned).

Trajectory-lag generalisation:
$$
\rho^u_{t,k} = \cos(u_t,\, u_{t-k}) \quad \text{for } k \in \{1, 4, 16, 64, \ldots\}.
$$
The decay of $\rho^u_{t,k}$ as a function of $k$ characterises the
*temporal memory* of the optimiser's trajectory.

### 4.2 Per-class axis

Per-step:
$$
\Delta_{t,c} = \max\bigl(0,\; L_c(\theta_t) - L_c(\theta_{t-1})\bigr),
\qquad N_t = \bigl|\{c : \Delta_{t,c} > 0\}\bigr|.
$$
$\Delta_{t,c}$ is the forgetting magnitude for class $c$ at step $t$;
$N_t$ is the number of classes that regressed at step $t$.

Run-cumulative:
$$
\Delta^{\text{tot}} = \sum_t \sum_c \Delta_{t,c}, \qquad
N^{\text{tot}} = \sum_t N_t.
$$

Step-magnitude-normalised variants (introduced post-§3.4 to address P5):
$$
\Delta^{\text{per-step}}_t = \frac{\sum_c \Delta_{t,c}}{\lVert u_t\rVert},
\qquad
\Delta^{\text{per-progress}} = \frac{\Delta^{\text{tot}}}{\lvert L^{\text{train}}_{\text{init}} - L^{\text{train}}_{\text{final}}\rvert}.
$$
$\Delta^{\text{per-step}}_t$ measures wasted work per unit parameter movement
(dimensionless, comparable across step scales).
$\Delta^{\text{per-progress}}$ measures total wasted work per unit useful
progress (the most direct test of the wasted-work hypothesis).

### 4.3 Trajectory axis

Wasted-work ratio at window $K$:
$$
\mathrm{WW}_K(t) = \frac{\bigl\lVert \theta_t - \theta_{t-K}\bigr\rVert}{\sum_{i=t-K+1}^{t} \lVert u_i\rVert} \in (0, 1].
$$
Run-summary statistics: mean, standard deviation, minimum, maximum across
all windowed evaluations.

Net-displacement-per-epoch (auxiliary, not normalised):
$$
\mathcal{D}_{\text{epoch}} = \bigl\lVert \theta_{\text{end}} - \theta_{\text{start}}\bigr\rVert.
$$
Reported alongside $\mathrm{WW}_K$ since the ratio alone is incomplete: a
high ratio with low net displacement is "moving slowly but accurately".

## 5. The wasted-work hypothesis

The framework formalises the intuition that interference causes training
slowness:

> **Hypothesis (W).** Net training-speed (loss reduction per step or per
> wall-clock unit) is inversely related to total wasted work, where wasted
> work is measured as the cumulative per-class forgetting magnitude
> $\Delta^{\text{tot}}$ over the training run, normalised by useful progress
> ($\Delta^{\text{per-progress}}$ in §4.2) or by total step magnitude
> ($\sum_t \Delta^{\text{per-step}}_t$).

A method that reduces $\Delta^{\text{per-progress}}$ without reducing useful
progress should therefore increase training speed. Orthogonalisation methods
make a specific testable prediction under this hypothesis: if BoGrad reduces
$\Delta^{\text{per-progress}}$ relative to baseline at fixed compute, AND
useful progress is preserved, then BoGrad should improve training speed.

The framework will validate (W) through:

1. **Observational comparison.** Measure $\Delta^{\text{tot}}$ and
   $\Delta^{\text{per-progress}}$ for baseline runs and runs with
   orthogonalisation. Verify that orthogonalisation reduces them.
2. **Per-epoch correlation.** Compute the ratio (loss decrease) /
   (per-class forgetting magnitude) per epoch. A higher ratio under
   orthogonalisation supports (W).
3. **Synthetic causality.** In settings where forgetting is *injected*
   (deliberately adversarial mini-batches), observe that injected forgetting
   slows training and that suppressing it (orthogonalisation) recovers
   speed.

(W) might be wrong: it is plausible that single-task training is robust to
forgetting and the per-class interference is a red herring. Validating or
refuting (W) is itself a thesis-worthy outcome.

## 6. Measurement layers

Every training run can be instrumented to capture interference at multiple
resolutions. The framework module
[`common/diagnostics/interference.py`](../../common/diagnostics/interference.py)
implements them all behind one `InterferenceTracker` class.

| Layer | Metric | Window | Cost |
|---|---|---|---|
| Step-step | $\rho^g_t$, $\rho^u_t$, $\rho^{u\bar g}_t$ | 1 step | Cheap (one dot product) |
| Per-class | $L_c(\theta_t)$, then $\Delta_{t,c}$ | 1 step | One forward pass on probe set |
| Trajectory | $\rho^u_{t,k}$ for $k \in \{1, 4, 16\}$ | spans | Cheap |
| Full-batch ref | $\cos(g_t, \tilde{g}_t)$, parallel/perp decomposition | calibration | Expensive (full epoch fwd/bwd) |
| Forgetting | $\Delta^{\text{per-step}}_t$, $\sum_c \Delta_{t,c}$ | accumulator | Cheap given probe |
| Wasted-work | $\mathrm{WW}_K$ for $K \in \{8, 32, 128\}$ | rolling | Cheap |

Default diagnostic-capture configuration (validation run §8 used these):

- `log_every = 1` (cosine + trajectory metrics every step — required to detect
  short-horizon structural correlations; see Finding F1 in §8.)
- `probe_every = 1` (per-class probe every step — gives the highest-resolution
  forgetting signal at modest cost).
- `full_grad_every = None` (skip the expensive metric by default).
- `wasted_work_K = 32`.

Cheaper configurations (`log_every = 10`, `probe_every = 10`) are appropriate
for production runs where diagnostic overhead must be minimised, but they hide
short-horizon phenomena.

## 7. Distinguishing "interference" from "noise"

A central methodological challenge: the framework's signals must be
distinguishable from random mini-batch noise. We use three complementary
approaches.

### 7.1 Calibration via no-interference reference

Run a "gold standard" config (full-batch GD, or very large mini-batches) on
the same problem. By definition there is no mini-batch noise and no
inter-step interference at the limit. Forgetting event rates and wasted-work
ratios on this reference give the "zero-interference" baseline. Anything
above it in stochastic training is candidate interference.

### 7.2 Synthetic experiments with controlled conflict

- **Quadratic loss with controlled cross-direction conflict.** Loss
  $L(\theta) = \tfrac{1}{2} \theta^\top H \theta - b^\top \theta$ with diagonal
  $H$ of controllable per-direction curvature. Mini-batches are constructed to
  alternate which direction the gradient emphasises, injecting controlled
  inter-step interference. Verify that the metrics detect it.
- **Class-disjoint mini-batches.** On CIFAR-10, force mini-batch $t$ to
  contain only classes $\{1, 2, 3\}$ and mini-batch $t{+}1$ only classes
  $\{8, 9, 10\}$ — maximum inter-batch class disjointness. Measure per-class
  forgetting and trajectory ratio, expecting elevated values. Then interleave
  normally and expect baseline values. The contrast tells us how sensitive
  the metrics are.
- **Permuted-class sequence.** All-class-$A$ batches followed by
  all-class-$B$ followed by all-class-$C$ — a continual-learning-style setup
  within a single dataset. Lets us watch forgetting accumulate and recover.

### 7.3 Statistical baselines

For each metric, compute its expected value and variance under a null model
(e.g. independent random Gaussian gradients with empirically observed norm
distribution). Anything beyond a few standard deviations from the null is
flagged as structural interference rather than sampling noise.

## 8. Empirical findings — first validation reference run

The framework was first exercised on a controlled validation run:
**SmallCNN trained on CIFAR-10 for one epoch (391 steps, batch size 128)**,
with full diagnostics at `log_every=1` and per-class probe at `probe_every=1`.
Four configurations were run with paired data ordering (same shuffled batch
sequence under all four):

1. **vanilla SGD** (lr=0.05, no momentum)
2. **SGD+momentum** (lr=0.05, $\mu$=0.9)
3. **BoGrad on vanilla SGD** (gradient-stage projection, $K{=}8$, asymmetric)
4. **BoGrad on SGD+momentum** (update-stage projection, $K{=}32$, asymmetric)

The run produced four findings that informed the subsequent versions of the
framework, the metric implementations, and the open-questions list. Final
test accuracies (one-epoch): vanilla 0.262, momentum 0.446, BoGrad+vanilla
0.380, BoGrad+momentum 0.498.

### Finding F1 — Momentum induces structural gradient correlation

| Run | $\rho^g_t = \cos(g_t, g_{t-1})$ | $\rho^u_t = \cos(u_t, u_{t-1})$ |
|---|---|---|
| vanilla SGD | $-0.039$ | $-0.039$ |
| SGD+momentum | $+0.257$ | $+0.773$ |
| BoGrad+vanilla | $-0.053$ | $+0.212$ |
| BoGrad+momentum | $+0.298$ | $+0.868$ |

Vanilla SGD has near-zero gradient correlation between consecutive steps —
successive $g_t$ are essentially decorrelated, consistent with mini-batch
noise dominating any structure (cf. §2: under iid sampling, $\xi_t$ is
independent across $t$). SGD+momentum, by contrast, shows
$\rho^g_t = +0.26$.

This is a non-trivial finding. Momentum, by its construction, only modifies
*updates* — it does not modify the gradient computation itself. The positive
consecutive-gradient correlation must therefore arise from a *trajectory
effect*: momentum drives parameters along a coherent direction, and gradients
sampled along that coherent trajectory carry shared signal from the loss
surface beyond what mini-batch noise alone provides. Random-walk-style
vanilla SGD does not exhibit this because each step disturbs the parameters
in a different direction, decorrelating successive sampling locations.

The framework distinguishes the two regimes cleanly. Importantly, this
distinction was *invisible* at `log_every=10`: the corresponding lag-10
$\cos(g_t, g_{t-10})$ for SGD+momentum was $-0.022$ in an earlier run,
indistinguishable from vanilla SGD at lag 10. The structural gradient
correlation introduced by momentum is short-horizon — it survives at lag 1
but decays by lag 10. **This validates `log_every=1` as the appropriate
cadence for observing the phenomenon.**

### Finding F2 — Wasted-work ratio orders methods by trajectory efficiency

| Run | $\overline{\mathrm{WW}_{K=32}}$ |
|---|---|
| vanilla SGD | $0.298$ |
| SGD+momentum | $0.406$ |
| BoGrad+vanilla | $0.533$ |
| BoGrad+momentum | $0.612$ |

The wasted-work ratio cleanly ranks the four methods by trajectory
efficiency. The ranking matches the predicted mechanism:

- Vanilla SGD's gradient-only steps zigzag (random-walk-like) → low ratio.
- Momentum smooths the trajectory → ratio improves $+0.11$.
- BoGrad on vanilla SGD removes destructive interference at the gradient
  level, exceeding even momentum's smoothing → $+0.13$ over momentum-baseline.
- BoGrad on SGD+momentum compounds both mechanisms → highest ratio.

Note the ranking does *not* match final accuracy ranking
(vanilla 0.26 < BoGrad+vanilla 0.38 < momentum 0.45 < BoGrad+momentum 0.50).
BoGrad+vanilla has higher wasted-work ratio than momentum-baseline but lower
accuracy. This indicates that **trajectory efficiency is a necessary but not
sufficient condition for fast training**: the magnitude of net displacement
matters as well, and BoGrad+vanilla makes smaller-magnitude steps than
momentum despite being more efficient. A method scoring high on
$\mathrm{WW}_K$ with low $\mathcal{D}_{\text{epoch}}$ is moving slowly but
accurately; a method scoring lower on $\mathrm{WW}_K$ but with higher net
displacement may still cover more ground.

This motivates reporting **net displacement per epoch**
$\mathcal{D}_{\text{epoch}} = \lVert \theta_{\text{end}} - \theta_{\text{start}}\rVert$
alongside $\mathrm{WW}_K$ when characterising a method (added to §4.3).

### Finding F3 — Methods exhibit distinct trajectory-lag decay profiles

| Run | $\rho^u_{t,1}$ | $\rho^u_{t,4}$ | $\rho^u_{t,16}$ |
|---|---|---|---|
| vanilla SGD | $-0.04$ | $+0.09$ | $+0.10$ |
| SGD+momentum | $+0.77$ | $+0.15$ | $+0.12$ |
| BoGrad+vanilla | $+0.21$ | $+0.28$ | $+0.26$ |
| BoGrad+momentum | $+0.87$ | $+0.44$ | $+0.24$ |

The decay of $\rho^u_{t,k}$ as a function of lag $k$ characterises the
*temporal memory* of the optimisation trajectory.

- **Vanilla SGD** has effectively no memory: lag-1 alignment is near zero
  and remains low across all lags.
- **Momentum** has strong short-horizon memory ($\rho^u_{t,1} = 0.77$) that
  decays sharply ($\rho^u_{t,4} = 0.15$). The effective horizon matches the
  theoretical EMA window $1/(1-\mu) \approx 10$.
- **BoGrad+vanilla** has moderate but *persistent* memory: lag-1, lag-4,
  lag-16 are all in the 0.2–0.3 range. The $K{=}8$ buffer creates correlation
  that extends beyond the immediate next step.
- **BoGrad+momentum** combines both effects: peak at lag-1 (0.87), elevated
  at lag-4 (0.44 vs. momentum's 0.15), trailing at lag-16.

The lag-decay profile is a *new diagnostic dimension* the framework reveals.
It shows that **momentum and BoGrad operate on different temporal scales**
(short vs. medium-to-long horizon respectively) **and combine compositionally
rather than redundantly**.

### Finding F4 — Forgetting magnitude is step-magnitude-confounded

| Run | events / step | $\overline{\sum_c \Delta_{t,c}}$ |
|---|---|---|
| vanilla SGD | 4.94 | $0.30$ |
| SGD+momentum | 4.82 | $0.65$ |
| BoGrad+vanilla | 4.92 | $0.45$ |
| BoGrad+momentum | 4.85 | $0.52$ |

The number of forgetting *events* per step is essentially constant across
methods ($\sim 4.85$, i.e. about half the classes regress every step
regardless of method). This is the noise floor of the forgetting metric.

The forgetting *magnitude* per step varies substantially. SGD+momentum has
$2.2\times$ the magnitude of vanilla SGD ($0.65$ vs $0.30$) — but achieves
higher final accuracy. BoGrad+momentum reduces magnitude by 20% relative to
momentum-baseline ($0.65 \to 0.52$) and improves accuracy further.
BoGrad+vanilla *increases* magnitude over vanilla ($0.30 \to 0.45$) but
improves accuracy substantially.

The pattern indicates **forgetting magnitude is confounded with step
magnitude**. Methods that take larger per-step parameter movements naturally
produce larger per-step changes in per-class loss, in both directions.
Direct comparison of $\sum \Delta$ across methods with different
$\lVert u\rVert$ is unfair to large-step methods. To recover a comparable
metric, we introduce the normalised variants of §4.2:
$\Delta^{\text{per-step}}_t$ and $\Delta^{\text{per-progress}}$.

These two metrics are now part of the framework module
([`common/diagnostics/interference.py`](../../common/diagnostics/interference.py));
all subsequent runs report them.

### Implications for the framework

The validation run confirms three structural choices:

(a) **All three formulations (geometric, per-class, trajectory) provide
non-redundant signal.** No single metric ranks the methods identically;
combining them reveals more than any one alone.

(b) **`log_every=1` is the appropriate cadence** for detecting structural
gradient-correlation effects (Finding F1). Coarser cadences hide phenomena
that are short-horizon.

(c) **Forgetting metrics require step-magnitude normalisation** to be
cross-method-comparable (Finding F4). The framework module is updated
accordingly.

The validation run also raised three new questions that are added to §9.

### Finding F5 — Detection ≠ slowdown (synthetic-quadratic experiment)

A controlled synthetic experiment ([`synthetic_quadratic.py`](synthetic_quadratic.py))
injected structured cross-step interference into a quadratic loss and
measured whether the framework's metrics detected it AND whether the
detected interference corresponded to actual training slowdown.

Setup: $L(\theta) = \tfrac{1}{2} \theta^\top H \theta - b^\top \theta$ with
$p = 100$, log-spaced eigenvalues $\in [1, 100]$. Mini-batch gradient
$g_t = (H\theta - b) + \xi_t + \iota_t$ with $\xi_t \sim \mathcal{N}(0, \sigma^2 I)$
iid noise and structured interference $\iota_t = (-1)^t \cdot a \cdot e_J$
on a fixed coordinate subset $J$. Three variants run for 500 steps each:
no-interference ($a = 0$), with-interference ($a = 2.0$), and
with-interference + BoGrad ($K = 8$, gradient-stage, asymmetric).

Result:

| Variant | $\overline{\rho^g_t}$ | $\overline{\mathrm{WW}_{K=32}}$ | $\overline{\rho^u_t}$ | $\lVert\theta - \theta^\star\rVert$ |
|---|---|---|---|---|
| no-interference | $+0.408$ | $0.599$ | $+0.408$ | $0.050$ |
| with-interference | $-0.873$ | $0.113$ | $-0.873$ | $0.059$ |
| with-interference + BoGrad | $-0.858$ | $0.661$ | $+0.024$ | $0.619$ |

The geometric and trajectory metrics detect the interference cleanly:
$\rho^g_t$ collapses from $+0.41$ to $-0.87$, $\mathrm{WW}_{K=32}$ from
$0.60$ to $0.11$. BoGrad attenuates the update-level signal (cos$_u$ from
$-0.87$ to $+0.02$) and recovers $\mathrm{WW}_{K=32}$ to $0.66$. Detection
is therefore validated.

But **the injected interference does not slow vanilla SGD**: clean and
with-interference converge to within $0.06$ of $\theta^\star$. The
alternating $\pm a \cdot e_J$ has zero mean across consecutive pairs;
SGD's natural averaging absorbs it. Worse, **BoGrad applied to this setup
actively hurts convergence** (distance $0.06 \to 0.62$): the projection
removes the alternating component from the gradient, but along $J$ that
component contains *useful* expected descent signal that SGD would have
integrated correctly. BoGrad strips the signal along with the noise.

The conclusion: **cross-step pattern detection does not imply training
slowdown**, and methods that reduce metric-measured interference may *hurt*
convergence when the interference is zero-mean and SGD-tractable.

Implications for the wasted-work hypothesis (§5):

- (W) is **not validated** by zero-mean alternating noise. The hypothesis
  predicts orthogonalisation should improve convergence by reducing
  interference; here it does the opposite because the "interference" is
  actually averageable noise, not destructive interference.
- (W) requires a *different* synthetic test: one where the injected
  cross-step pattern is **biased** (does not cancel across pairs) or
  **curvature-driven** (each step diverges further from the optimum).
  Both are queued as follow-up experiments
  ([`synthetic_quadratic_biased.py`](synthetic_quadratic_biased.py),
  TBD).
- The CIFAR-10 class-disjoint experiment ([§7.2 second bullet](#72-synthetic-experiments-with-controlled-conflict))
  is a more direct test of (W): forgetting events in a class-permuted
  training stream are *not* zero-mean, and orthogonalisation is expected
  to provide genuine speedups.

Finding F5 also tightens the framework's interpretation: a method that
reduces $\rho^g$ or improves $\mathrm{WW}_K$ has demonstrated detector
sensitivity, not training-speed improvement. The two are linked by
hypothesis (W), but the link must be tested separately, not assumed.

### Finding F6 — Per-class forgetting magnitude is dominated by step-noise, not interference (class-disjoint experiment)

A second synthetic experiment ([`synthetic_class_disjoint.py`](synthetic_class_disjoint.py))
tested whether *biased* per-class interference (each batch contains only
one class, classes cycled) produces measurable per-class forgetting
*relative to* normal interleaved training, and whether orthogonalisation
mitigates the resulting slowdown.

Setup: SmallCNN on CIFAR-10, lr=0.05, momentum=0.9, batch=128, 1 epoch,
100 batches per regime. Two batch regimes (interleaved, class-disjoint
cycling 0→1→…→9), each with and without BoGrad (update-stage K=32 neg).

Result:

| Variant | final acc | $\Delta^{\text{tot}}$ | $\Delta^{\text{tot}}/\sum\lVert u\rVert$ | $\overline{\mathrm{WW}_{K=32}}$ | $\overline{\rho^u_t}$ |
|---|---|---|---|---|---|
| interleaved baseline | $0.331$ | $9.50$ | $1.34$ | $0.60$ | $+0.42$ |
| disjoint baseline | $0.159$ | $0.68$ | $0.10$ | $0.30$ | $+0.86$ |
| interleaved + BoGrad | $0.308$ | $8.56$ | $1.11$ | $0.68$ | $+0.41$ |
| disjoint + BoGrad | $0.187$ | $1.76$ | $0.30$ | $0.60$ | $+0.77$ |

Two predictions held:
- **P2 PASS** — disjoint training is markedly slower than interleaved
  ($0.331 \to 0.159$, more than half the accuracy lost).
- **P4 PASS** — BoGrad recovers some accuracy in the disjoint regime
  ($0.159 \to 0.187$, $+2.8$ pts relative).

But the framework metrics produced unexpected readings:
- **P1 FAIL** — disjoint produces *less* total forgetting magnitude than
  interleaved ($0.68$ vs $9.50$, a $14\times$ ratio). The expected
  "disjoint = high interference = high forgetting" relationship is
  inverted.
- **P3 FAIL** — BoGrad *increases* forgetting magnitude in the disjoint
  case ($0.68 \to 1.76$).

The diagnosis is that **per-class forgetting magnitude is dominated by
step-by-step parameter fluctuation, not by structured interference**. In
interleaved training, every step changes parameters in a direction
informed by all classes; per-class losses fluctuate substantially in
both directions and the cumulative $\sum_t \sum_c \max(0, \Delta L_c)$ is
large. In disjoint training, consecutive batches push parameters along
class-specific gradient directions which are highly aligned (cos$_u$ =
$0.86$) — the trajectory is *smooth* (in a sense) and per-class loss
fluctuates less per step. Even though the trajectory cycles through
class-specific directions, the per-class-loss-increase metric does not
detect the rotation because it measures *magnitude of regression*, not
*structure of regression*.

The trajectory-level metric $\mathrm{WW}_{K=32}$ behaves correctly:
disjoint baseline drops to $0.30$ (vs interleaved $0.60$), correctly
detecting trajectory inefficiency. BoGrad recovers it to $0.60$ in the
disjoint case, mirroring the accuracy improvement.

**Implications:**

1. **The simple per-class forgetting magnitude $\sum_t \sum_c \Delta_{t,c}$
   is NOT a clean interference indicator.** It conflates random per-class
   loss fluctuation with structural cross-class interference.
2. **The trajectory metric $\mathrm{WW}_K$ is more robust** at detecting
   the kind of interference that produces actual training slowdown
   (P2 + WW correlation in this experiment).
3. **A refined per-class metric is needed.** Candidate: *out-of-batch
   forgetting* — for each step, distinguish forgetting events on classes
   that were *not* in the batch from those that were. The former is the
   "interference proper" channel: a regression on a class the step had no
   information about cannot be due to noise on that class's loss
   estimate; it must be due to parameter drift caused by other classes'
   gradient. Adding this metric is queued as a framework refinement
   ([§9 question 8](#9-open-questions)).

The class-disjoint experiment partially validates Hypothesis (W):
trajectory inefficiency correlates with slowdown, and orthogonalisation
recovers both. But it shows the *forgetting-magnitude* operationalisation
of (W) is too noisy to be useful directly; a refined operationalisation
(out-of-batch forgetting) is required.

### Finding F7 — Out-of-batch forgetting cleanly isolates the interference channel

A refined per-class metric was added to the framework:

> **Out-of-batch forgetting magnitude.** For each step $t$ with mini-batch
> $B_t$ containing classes $C(B_t)$, define
> $$\Delta^{\text{OOB}}_t = \sum_{c \notin C(B_t)} \max\bigl(0,\; L_c(\theta_t) - L_c(\theta_{t-1})\bigr).$$
> Run-cumulative: $\Delta^{\text{OOB,tot}} = \sum_t \Delta^{\text{OOB}}_t$.

The motivation is that forgetting on classes that were *not* in the batch
cannot be attributed to noise on those classes' loss estimates (the step's
gradient saw nothing of class $c$). It must be parameter drift caused by
the gradient of *other* classes — i.e. genuine cross-class interference.

Re-running the class-disjoint experiment with this metric:

| Variant | $\Delta^{\text{tot}}$ | $\Delta^{\text{in-batch}}$ | $\Delta^{\text{OOB}}$ |
|---|---|---|---|
| interleaved baseline | $9.50$ | $9.50$ | $\mathbf{0.000}$ |
| disjoint baseline | $0.68$ | $0.13$ | $\mathbf{0.552}$ |
| interleaved + BoGrad | $8.56$ | $8.56$ | $\mathbf{0.000}$ |
| disjoint + BoGrad | $1.76$ | $0.40$ | $\mathbf{1.362}$ |

The OOB metric correctly distinguishes the two regimes: it is exactly $0$
in the interleaved case (no class is ever out-of-batch by construction),
and substantial in the disjoint case ($0.552$ for the baseline). **The
interference signature that was masked by step-magnitude noise in
$\Delta^{\text{tot}}$ is now visible at the per-class level.**

Prediction outcomes after the refinement:
- **P1 PASS** (was FAIL): disjoint OOB ($0.552$) $\gg$ interleaved OOB
  ($0.000$). The clean test confirms biased per-class interference is what
  separates the regimes.
- **P2 PASS** (unchanged): disjoint slows convergence.
- **P3 FAIL** (was FAIL): BoGrad applied to disjoint *increases* OOB
  forgetting ($0.552 \to 1.362$). Diagnosis: BoGrad's projection enlarges
  per-step parameter movement (cos$_u$ drops from $0.86$ to $0.77$,
  trajectory less smooth), which mechanically increases per-class loss
  fluctuations. The OOB metric is therefore still partially confounded by
  per-step magnitude, just less so than the un-decomposed total.
- **P4 PASS** (unchanged): BoGrad recovers some accuracy in the disjoint
  regime ($0.158 \to 0.187$).

**Implications:**

(a) The OOB metric is the right per-class diagnostic for cross-class
interference. It belongs in the headline framework metrics (alongside
$\mathrm{WW}_K$).

(b) But P3 reveals a remaining confound: even OOB forgetting depends on
step magnitude. A method that legitimately fixes interference (closing
the gap between disjoint and interleaved accuracy) may still produce
larger per-step OOB forgetting if its steps are simply bigger. The
"interference fixed at expectation, larger fluctuations per step" pattern
is real and the framework should report a per-progress version of OOB
forgetting:

$$
\Delta^{\text{OOB-per-progress}} = \frac{\Delta^{\text{OOB,tot}}}{\lvert L^{\text{train}}_{\text{init}} - L^{\text{train}}_{\text{final}}\rvert}
$$

(c) The strongest single-number summary of "this method handles
interference well" is now *the combination of high $\mathrm{WW}_K$ and
low $\Delta^{\text{OOB-per-progress}}$*.

The class-disjoint experiment, with the OOB refinement, validates
hypothesis (W) at the conceptual level (interference exists, slows
training, orthogonalisation partly fixes it) while highlighting that the
metric definitions still need to control for step magnitude on a
per-method basis.

### Finding F8 — Pairwise alignment statistics over the K-buffer reveal regime structure

The framework was extended with a $K$-window pairwise alignment tracker
(`PairwiseAlignmentTracker` in
[`common/diagnostics/interference.py`](../../common/diagnostics/interference.py)).
For each logged step $t$, it computes
$$
\rho_{t,k} = \cos\bigl(x_t,\, x_{t-k}\bigr)
\quad \text{for all } k = 1, \ldots, K,
$$
where $x$ is either the gradient or the applied update. It aggregates per
step:
- `frac_positive`, `frac_negative`: fraction of the $K$ pairs with $\cos > 0$ or $< 0$;
- `mean_positive_cos`, `mean_negative_cos`: average $\cos$ within each subset;
- `min_cos`, `max_cos`, `mean_abs_cos`: distributional summaries.

Run-level summary then averages each across all logged steps. This
generalises the lag-1-only metrics $\rho^g_t$ and $\rho^u_t$ to the full
$K$-step window — the same window BoGrad orthogonalises against.

Empirical signature on the class-disjoint experiment:

| Regime | $\overline{\text{frac}^g_+}$ | $\overline{\text{frac}^g_-}$ | $\overline{\langle \cos^g_+\rangle}$ | $\overline{\langle \cos^g_-\rangle}$ |
|---|---|---|---|---|
| interleaved baseline | 41.6% | 58.4% | $+0.174$ | $-0.158$ |
| disjoint baseline | **100%** | **0%** | $+0.802$ | n/a |
| disjoint + BoGrad neg | 100% | 0% | $+0.751$ | n/a |

The disjoint regime exhibits a *qualitatively different* gradient
geometry: every pair in the K-buffer is positively aligned, with strong
mean cosine. Interleaved training has roughly balanced positive and
negative pairs with small magnitudes, consistent with mini-batch noise
dominating the structure.

This is a new diagnostic dimension: **the sign distribution of pairwise
alignments is itself a regime fingerprint**. It distinguishes
"random-noise-dominated" training from "structured-direction-dominated"
training in a way no single-pair metric can.

### Finding F9 — BoGrad-mode comparison on the disjoint regime confirms negative is the right default

Adding a third projection mode `"positive"` to BoGrad (subtracts overlap
when $\langle g, b\rangle > 0$ — the complement of `"negative"`), all
three modes were run on the class-disjoint regime:

| Mode | final acc | Δ vs no BoGrad | Δ$^{\text{OOB,tot}}$ |
|---|---|---|---|
| no BoGrad | $0.158$ | $+0.000$ | $0.55$ |
| BoGrad **full** | $0.100$ | $\mathbf{-0.059}$ | $45.8$ |
| BoGrad **negative** | $0.187$ | $\mathbf{+0.028}$ | $1.36$ |
| BoGrad **positive** | $0.100$ | $\mathbf{-0.059}$ | $318.5$ |

Both `full` and `positive` collapse training to chance accuracy. Reading
this with F8: the disjoint regime has 100% positive pairwise alignment
with strong cosine ($+0.80$). `Full` mode subtracts the entire projection
component on every pair; `positive` mode does so only on positive-aligned
pairs (which is *all* of them). Either way, the projection effectively
removes the entire buffered direction — which is also the direction the
optimiser is supposed to be moving in. Training collapses.

`Negative` mode skips when $\cos > 0$, so it almost never fires on the
gradient buffer (where 100% of pairs are positive). It still helps —
$+2.8$pts — by intervening on the rare pairs where $\cos$ does dip
below zero (after BoGrad's projection of past updates produces some
4% negative-pair fraction in the *update* buffer), and by very slightly
modifying the trajectory.

**Conclusion: the default `projection_mode="negative"` is empirically
correct for setups where buffered directions are highly aligned (the
typical training regime for momentum or class-cycled setups). Full and
positive modes are too aggressive in those regimes. Full is only safe
when buffered directions span diverse angles; positive should be
considered only as a research probe, not a production setting.**

This finding closes a ambiguity that the framework's earlier work left
open: the choice among full/negative/positive is *not* a knob to tune
per task — it is determined by the pairwise-alignment distribution of
the buffer, which the framework now measures directly.

### Finding F10 — Drift-biased interference is BoGrad-incompatible

A follow-up to F5 ([`synthetic_quadratic_biased.py`](synthetic_quadratic_biased.py))
tested whether *biased* (non-cancelling) interference produces real
training slowdown that BoGrad can fix. Three bias modes were tested:
*drift* (constant pull), *rotating* (slow rotation), *decaying* (fading initial bias).

For drift bias (constant pull on $J$ coordinates with amplitude $0.5$):

| Variant | $\lVert \theta - \theta^\star \rVert$ | $\overline{\rho^g_t}$ | $\overline{\text{frac}^g_+}$ |
|---|---|---|---|
| clean | $0.050$ | $+0.408$ | 88% |
| biased baseline | $0.599$ | $+0.449$ | 91% |
| biased + BoGrad full | $1.250$ | $+0.987$ | **100%** |
| biased + BoGrad neg | $0.605$ | $+0.450$ | 91% |
| biased + BoGrad pos | $1.250$ | $+0.987$ | **100%** |

Drift bias slows training (P2 PASS: $0.05 \to 0.60$), but **no BoGrad
mode recovers it**:

- `Negative` mode is essentially a no-op (the bias makes $\rho^g_t$
  positive, neg-mode skips).
- `Full` and `positive` mode hurt training further, doubling the
  distance to the optimum.

This is structurally the same failure mode as F9: when bias is
constant-direction, the entire $K$-buffer aligns with it ($\rho^g \to
0.99$), and removing buffered directions removes the descent signal.
Drift bias is a kind of "interference" that orthogonalisation
cannot, in principle, distinguish from useful descent.

**Implications:**

1. The wasted-work hypothesis (W) holds in the abstract — biased
   interference does slow training (P2). But (W) says "orthogonalisation
   can fix it" only when the interference is *separable* from the
   descent signal in the buffered subspace. Drift bias fails this
   separability, so BoGrad cannot help.
2. **Hypothesis (W'): orthogonalisation helps only when the
   interference is anti-aligned with descent (negative-mode fires) OR
   when buffered directions are mixed-sign enough that subtracting the
   span doesn't strip descent.** This is a refinement that needs
   further synthetic tests to confirm.
3. The class-disjoint result (F9, +2.8 pts from BoGrad neg) is
   consistent with (W'): in the disjoint regime, even though the
   gradient-buffer is 100% positive-aligned, the *update* buffer
   produces some negative-aligned pairs after the projection cycles,
   and BoGrad-neg fires on those. The intervention is small but real.

### Finding F11 — Pairwise alignment characterises optimisers under standard CIFAR-10

A multi-trial pairwise alignment study
([`pairwise_alignment_study.py`](pairwise_alignment_study.py)) characterises
gradient and update buffer geometry of six configurations under
*standard* random-shuffle CIFAR-10 batching (no class-disjoint or contrived
schedules). 3 trials × 1 epoch × full CIFAR-10. Pairwise stats over a $K = 32$
window. Reported as mean $\pm$ std across trials.

**Gradient-buffer alignment** ($\cos(g_t, g_{t-k})$ for $k = 1..32$):

| Variant | acc | %pos$^g$ | %neg$^g$ | $\langle \cos^g_+\rangle$ | $\langle \cos^g_-\rangle$ |
|---|---|---|---|---|---|
| SGD vanilla | $0.261 \pm 0.026$ | $63.3\% \pm 0.6$ | $36.7\% \pm 0.6$ | $+0.31 \pm 0.02$ | $-0.23 \pm 0.01$ |
| SGD+momentum | $0.435 \pm 0.006$ | $\mathbf{50.0\% \pm 0.1}$ | $\mathbf{50.0\% \pm 0.1}$ | $+0.30 \pm 0.01$ | $-0.30 \pm 0.01$ |
| Adam | $0.421 \pm 0.019$ | $\mathbf{49.7\% \pm 0.3}$ | $\mathbf{50.3\% \pm 0.3}$ | $+0.29 \pm 0.01$ | $-0.29 \pm 0.01$ |
| SGD-vanilla + BoGrad | $0.381 \pm 0.004$ | $60.8\% \pm 0.7$ | $39.2\% \pm 0.7$ | $+0.30 \pm 0.02$ | $-0.24 \pm 0.01$ |
| SGD+mom + BoGrad | $0.502 \pm 0.011$ | $50.1\% \pm 0.3$ | $49.9\% \pm 0.3$ | $+0.27 \pm 0.00$ | $-0.27 \pm 0.01$ |
| Adam + BoGrad | $0.485 \pm 0.016$ | $49.7\% \pm 0.2$ | $50.3\% \pm 0.2$ | $+0.26 \pm 0.01$ | $-0.25 \pm 0.01$ |

**The single most striking finding: SGD+momentum and Adam, under standard
CIFAR-10 training, produce gradient samples that are exactly $50/50$ positive
vs negative across the K=32 window — within 0.5% of perfect balance, with
matching magnitudes ($\langle \cos^g_+\rangle \approx -\langle \cos^g_-\rangle \approx 0.30$).**

This is essentially a *random-walk fingerprint* in gradient space: under
momentum/Adam, the trajectory visits enough varied regions of the loss surface
that gradient samples 1–32 steps apart are equally likely to align as
disagree. Momentum and Adam decorrelate the gradient sequence over the
K-window, even though F1 showed they correlate the *adjacent* gradients
($\rho^g_t = +0.26$ for momentum at lag 1).

Vanilla SGD shows a different fingerprint: 63/37 split — modestly more
positive than balanced. Smaller steps mean less parameter drift over 32
steps, which means more correlated sampling locations. The expected
"random-walk" 50/50 limit isn't reached.

BoGrad does not change the gradient-buffer alignment for momentum/Adam
(50/50 in both cases) — confirming that BoGrad operates on updates, not
gradients, when configured for update-stage projection.

**Update-buffer alignment** ($\cos(u_t, u_{t-k})$ for $k = 1..32$):

| Variant | $\overline{\mathrm{WW}_{32}}$ | %pos$^u$ | %neg$^u$ | $\langle \cos^u_+\rangle$ | $\langle \cos^u_-\rangle$ |
|---|---|---|---|---|---|
| SGD vanilla | $0.29 \pm 0.01$ | $63\% \pm 0.6$ | $37\% \pm 0.6$ | $+0.31$ | $-0.23$ |
| SGD+momentum | $0.40 \pm 0.01$ | $66\% \pm 0.7$ | $34\% \pm 0.7$ | $+0.28$ | $-0.14$ |
| Adam | $0.50 \pm 0.00$ | $80\% \pm 1.0$ | $20\% \pm 1.0$ | $+0.30$ | $-0.09$ |
| SGD-vanilla + BoGrad | $0.52 \pm 0.01$ | $88\% \pm 0.9$ | $12\% \pm 0.9$ | $+0.28$ | $-0.08$ |
| SGD+mom + BoGrad | $0.61 \pm 0.00$ | $\mathbf{99\% \pm 0.3}$ | $1\% \pm 0.3$ | $+0.31$ | $-0.01$ |
| Adam + BoGrad | $0.70 \pm 0.01$ | $\mathbf{100\% \pm 0.0}$ | $0\% \pm 0.0$ | $+0.41$ | $0.00$ |

The update-buffer alignment monotonically increases:
vanilla SGD ($63\%$) $\to$ SGD+mom ($66\%$) $\to$ Adam ($80\%$) $\to$
BoGrad-vanilla ($88\%$) $\to$ BoGrad+mom ($99\%$) $\to$ BoGrad+Adam ($100\%$).
The wasted-work ratio increases in lockstep ($0.29 \to 0.40 \to 0.50 \to
0.52 \to 0.61 \to 0.70$).

For BoGrad-wrapped variants the update buffer approaches or reaches $100\%$
positive alignment. This is mechanistic: BoGrad's negative-mode projection
in update-stage subtracts the part of any update that is anti-aligned with
buffered ones, so what remains in the buffer is necessarily positive-aligned
across all pairs.

**Three takeaways:**

1. Under momentum/Adam in standard training, the K=32 gradient-buffer is a
   **clean 50/50 random walk** — no consistent directional bias, equal
   magnitudes positive vs negative.
2. The transformation from gradient-buffer to update-buffer is what each
   optimiser is *for*: momentum shifts $50/50 \to 66/34$, Adam shifts to
   $80/20$, BoGrad pushes further. Each optimiser narrows the alignment
   distribution toward more positive.
3. **The pairwise alignment distribution is a stable, regime-specific
   fingerprint** with very tight cross-trial std ($\leq 1\%$ for fractions,
   $\leq 0.02$ for cosine magnitudes). It cleanly distinguishes optimisers
   in a way that can be reported as a single quantitative profile.

### Finding F12 — Perpendicular bias is BoGrad-RECOVERABLE: Hypothesis (W') validated

The (W') validation experiment
([`synthetic_quadratic_perpendicular.py`](synthetic_quadratic_perpendicular.py))
is the direct test of the refined hypothesis raised by F10:

> orthogonalisation helps only when the interference is meaningfully
> separable from the descent direction.

**Setup.** The parameter space is split into two subspaces:
- **Active** ($n_{\text{active}} = 50$ dims): full eigenvalue spectrum
  $[1, 100]$, $b$ chosen so $\theta^\star_{\text{active}}$ is non-zero —
  descent is active here.
- **Inactive** ($p - n_{\text{active}} = 50$ dims): tiny eigenvalue
  ($10^{-3}$), $b = 0$ so $\theta^\star_{\text{inactive}} = 0$ — descent
  has near-zero component here.

Bias is injected ONLY on inactive dims (drift, amplitude $0.5$). Because
descent has no meaningful component on inactive dims, the bias is
geometrically perpendicular to descent — the F10 failure mode (bias
parallel to descent) is excluded by construction.

**Result** ($p = 100$, 500 steps):

| Variant | dist$_{\text{total}}$ | dist$_{\text{active}}$ | dist$_{\text{inactive}}$ |
|---|---|---|---|
| clean baseline | $3.06$ | $0.07$ | $3.06$ |
| biased baseline | $9.48$ | $0.07$ | $\mathbf{9.48}$ |
| **biased + BoGrad FULL** | $\mathbf{3.82}$ | $1.88$ | $\mathbf{3.32}$ |
| biased + BoGrad NEG | $9.48$ | $0.07$ | $9.48$ |
| **biased + BoGrad POS** | $\mathbf{3.82}$ | $1.88$ | $\mathbf{3.32}$ |

The inactive dim doesn't converge in 500 steps even in the clean baseline
(tiny eigenvalue), so dist$_{\text{inactive}} = 3.06$ is the
no-interference floor. Bias drives it to $9.48$ ($3.1\times$ floor).

**BoGrad FULL and POS recover the inactive subspace almost perfectly**:
$9.48 \to 3.32$, a $96\%$ recovery toward the clean baseline (recovered
$6.16$ units out of $6.42$ units of bias-induced damage).

**BoGrad NEG does nothing** — exactly as predicted from F10/F11. The bias
makes pairwise alignment $100\%$ positive (all gradients share the bias
direction), and neg-mode skips when $\cos > 0$.

**This validates Hypothesis (W'):**

> *Orthogonalisation (BoGrad full or positive mode) recovers convergence
> when interference is separable from descent (bias on inactive subspace)
> but cannot recover when interference is parallel to descent (drift bias
> on the active subspace, F10).*

The pairwise alignment fingerprint (frac_positive = 100% in both F10 drift
and this F12 perpendicular case) does NOT distinguish the two regimes.
What distinguishes them is whether the dominant aligned direction in the
buffer is the descent direction or perpendicular to it. **The framework's
pairwise metric is necessary but not sufficient to predict BoGrad's
effect** — it tells us "all buffered directions are aligned" but not
"aligned with what".

**Implications:**

(a) **(W') is empirically validated.** Orthogonalisation works when
interference is separable; it fails when interference is parallel to
descent.

(b) The minor degradation of dist$_{\text{active}}$ ($0.07 \to 1.88$)
under BoGrad full/pos is the expected side-effect: removing the buffered
direction perturbs even active-subspace gradients slightly. The
inactive-recovery dominates the active-degradation in this setup, but
they are not free. A method that surgically removes only the inactive
component would do better.

(c) **For the thesis:** the cleanest framing of when BoGrad helps is
*"when the dominant aligned direction in the gradient buffer is
perpendicular to the loss-decreasing direction."* In typical training,
this happens when mini-batch gradients are noisy and the noise has
structure (e.g. class-cycled batches, label-correlated batches), but the
true descent direction is a different subspace.

(d) **Hypothesis (W'') — predictive criterion:** If we could measure
$\cos(\text{buffer mean direction}, \text{full-batch gradient})$ at any
step, that quantity would predict whether BoGrad helps or hurts at that
step. Low $\cos$ = perpendicular = BoGrad helps; high $\cos$ = parallel
= BoGrad hurts. This is testable using the existing
$\cos(g_t, g_t^{\text{full}})$ metric in the framework. Worth running.

### Finding F13 — Forgetting reduction without accuracy recovery (permuted-classes regime)

The permuted-classes synthetic experiment
([`synthetic_permuted_classes.py`](synthetic_permuted_classes.py)) runs
sequential class blocks (10 classes × 15 batches × 2 cycles, with class
transitions at every block boundary). This is a continual-learning-style
schedule within a single dataset.

Adjusted with a per-class-loss cap of 50 (so individual probe-loss spikes
don't dominate $\Delta^{\text{tot}}$):

| Variant | acc | $\Delta^{\text{OOB,tot}}$ |
|---|---|---|
| Interleaved baseline | $0.331$ | $0$ |
| Blocks baseline | $0.0996$ | $1378$ |
| Blocks + BoGrad neg | $0.0996$ | $\mathbf{579}$ |
| Blocks + BoGrad full | $0.0996$ | $574$ |
| Blocks + BoGrad pos | $0.0996$ | $728$ |

In this regime *no method recovers training* — the schedule is too extreme
for the SmallCNN within 300 batches: by the time class 9 is trained, the
model has overfit hard on it and predicts class 9 for everything (yielding
exactly $1/10 = 0.0996$ accuracy). All four block-trained variants land at
this floor.

But **the framework metrics still detect a real difference**:
BoGrad-negative reduces total OOB forgetting magnitude by **58%**
($1378 \to 579$). BoGrad does what it claims — reduces per-class forgetting
along the trajectory — even when the macro outcome (accuracy) is unaffected.
The micro-level evidence that BoGrad reduces interference is dissociable
from the macro-level question of whether reduced interference is enough
to fix the training failure.

This is a useful disambiguation. Earlier findings (F9, F10) showed cases
where BoGrad-full and BoGrad-positive collapsed training; F13 shows that
even in regimes where *all* BoGrad modes hit the same macro floor, the
*forgetting reduction* signal still differentiates them
($579 < 574 \ll 728$ — full slightly best, positive worst). The framework
remains diagnostic even when the methods themselves don't fix the problem.

**Implication for thesis framing:** the framework's per-class forgetting
metric reliably detects what BoGrad does (reduce per-class loss
fluctuations), but whether that detection translates to accuracy gains
depends on the regime. In normal training, it does (modest); in extreme
CL-style regimes, it doesn't (the failure mode is too severe). The
*existence* of the BoGrad effect is well-established by the framework even
where the *magnitude* of its accuracy benefit is contested.

## 9. Open questions

The framework is **deliberately broad**. As empirical evidence comes in,
some metrics will become headline measurements and others demoted to
appendix sanity-checks. The thesis-relevant subset will be decided through
the synthetic experiments (§7.2) and continued empirical capture.

Open questions originally identified:

1. **Which interference axis correlates best with training speed?**
   Geometric, per-class, or trajectory? The synthetic experiments and
   real-data captures will rank them.
2. **What does positive alignment indicate in practice?** The framework does
   not commit to "positive = good" or "positive = stale" (§3.3, P1). Empirical
   investigation: track positive-alignment events and check whether they
   correlate with productive learning (loss decrease) or unproductive
   stalling.
3. **Does the wasted-work ratio actually predict end-of-training accuracy?**
   §8 Finding F2 shows it ranks methods plausibly but is not a perfect
   predictor; the relationship requires more data.
4. **Are forgetting events power-law-distributed?** Most steps may have
   tiny $\Delta_{t,c}$; rare steps may have large $\Delta_{t,c}$. The
   "thick-tail" structure (or lack of it) determines whether mean,
   median, or max is the right summary.

New questions raised by the validation run (§8):

5. **Why is the noise-floor forgetting event count so stable** ($\sim 50\%$
   of classes regress every step) **across methods?** Is this the inherent
   noise of mini-batch SGD, or a feature of the SmallCNN architecture? Test
   on a different model or with full-batch GD to disambiguate.
6. **Does the lag-decay profile $\rho^u_{t,k}$ have an analytical form** for
   each optimiser? Specifically, can we derive the momentum decay analytically
   from $\mu$, and the BoGrad decay from $K$? An analytical prediction we can
   verify experimentally would strengthen the framework.
7. **Why does BoGrad+vanilla have higher $\mathrm{WW}_K$ but lower accuracy
   than momentum-baseline?** Is there a clean
   "trajectory efficiency × step magnitude = useful progress" decomposition
   that explains this, and can we predict accuracy from the framework metrics?

8. **Out-of-batch forgetting as a refined per-class metric** (raised by
   F6). For each step, distinguish forgetting events on classes that were
   *not* in the batch from those that were. The former is the
   "interference proper" channel — regression on a class the step had no
   gradient signal about can only be parameter drift from other classes'
   updates. Implementation requires the batch sampler to expose class
   membership per batch; modest infrastructure change.

9. **Is $\mathrm{WW}_K$ the most predictive single metric?** F6 suggests
   yes — it correctly tracked accuracy in the class-disjoint experiment
   while forgetting magnitude failed. F2 also showed it ranks methods
   plausibly. Worth elevating to the headline diagnostic, with per-class
   metrics as supporting structure rather than primary.

10. **Hypothesis (W'): refined wasted-work claim.** F10 showed that
    orthogonalisation cannot fix interference that is *aligned with*
    the descent direction (drift bias). The refined hypothesis is:
    *orthogonalisation helps only when the buffered interference
    component is meaningfully separable from the descent component
    — operationalised as a non-trivial fraction of negative-aligned
    pairs in the buffer (`frac_negative` $> \epsilon$) or a mixed-sign
    pairwise distribution.* Needs synthetic tests with carefully tuned
    bias-vs-descent angles to verify.

11. **What characterises a regime where positive-mode helps?** F9 / F10
    both showed positive-mode hurts. Is there *any* regime where
    removing redundant overlap is the right intervention? Plausibly:
    very late training (when momentum is stale and most overlap is
    re-application of past steps) or transitioning between phases.
    Worth a probe.

## 10. Connections to existing work

- **Continual learning forgetting** (Kirkpatrick et al., EWC; Lopez-Paz &
  Ranzato, GEM; Aljundi et al., MAS). Our per-class forgetting metric
  ($\Delta_{t,c}$) is inspired by but distinct from continual-learning
  forgetting — we measure *intra-task* per-class regressions, not *inter-task*
  forgetting.
- **Multi-task gradient surgery** (Yu et al., PCGrad; Wang et al., GradVac;
  Liu et al., MGDA). PCGrad's "remove the conflicting component" is exactly
  COSGD-style projection, applied across tasks rather than across batches.
  Our framework lets us apply their concept to single-task training.
- **Lookahead optimiser** (Zhang et al., 2019). Lookahead averages weights
  along the optimisation trajectory; the wasted-work ratio is closely related
  to what Lookahead is implicitly correcting.
- **Implicit gradient regularisation** (Smith et al., 2021; Geiping et al.).
  The systematic understanding of how SGD's stochasticity acts as a
  regulariser is a related lineage; our framework provides direct
  measurements that this literature has historically reasoned about
  analytically.
- **Conjugate-gradient theory.** The wasted-work ratio for a quadratic loss
  is bounded above by the rate at which CG iterations explore independent
  Krylov-subspace directions. BoGrad with $K$-sized buffer approximates a
  conjugate-direction-style optimiser; this connection suggests an analytical
  bound on $\mathrm{WW}_K$ achievable by orthogonalisation.

## 11. Status & next steps

Done:

- [x] Framework module implemented:
      [`common/diagnostics/interference.py`](../../common/diagnostics/interference.py),
      [`common/diagnostics/per_class_probe.py`](../../common/diagnostics/per_class_probe.py).
- [x] Validation reference run produced
      ([`results/baseline_reference/run_20260504_154438`](results/baseline_reference)).
- [x] Step-magnitude-normalised forgetting metrics added (§4.2).
- [x] Net-displacement-per-epoch metric added (§4.3).
- [x] Out-of-batch forgetting metric added (§4.2 / F7).
- [x] **Pairwise alignment tracker** added (`PairwiseAlignmentTracker`, F8).
- [x] **`positive` projection mode** added to BoGrad (F9, F10).
- [x] [`synthetic_quadratic.py`](synthetic_quadratic.py) — controlled
      cross-step interference in quadratic loss (§7.2 first bullet).
      Findings: F5 (detection ≠ slowdown).
- [x] [`synthetic_class_disjoint.py`](synthetic_class_disjoint.py) —
      class-disjoint vs interleaved mini-batches (§7.2 second bullet).
      Findings: F6 (per-class forgetting magnitude is noisy), F7 (OOB metric
      isolates interference), F8 (pairwise alignment reveals regime
      structure), F9 (BoGrad-neg is the right default).
- [x] [`synthetic_quadratic_biased.py`](synthetic_quadratic_biased.py) —
      drift / rotating / decaying biased interference. Finding F10
      (drift-biased interference is BoGrad-incompatible).
- [x] [`pairwise_alignment_study.py`](pairwise_alignment_study.py) —
      multi-trial pairwise alignment fingerprinting across 6 configurations,
      standard CIFAR-10. Finding F11 (clean 50/50 random-walk fingerprint
      in gradient buffer for momentum/Adam; pairwise distribution is a
      stable regime-specific fingerprint).
- [x] [`synthetic_permuted_classes.py`](synthetic_permuted_classes.py) —
      sequential class blocks. Finding F13 (BoGrad reduces forgetting along
      the trajectory even when macro accuracy doesn't recover in extreme CL
      regimes).
- [x] [`synthetic_quadratic_perpendicular.py`](synthetic_quadratic_perpendicular.py) —
      (W') validation experiment with perpendicular bias. Finding F12
      (Hypothesis W' VALIDATED — BoGrad recovers when bias is separable
      from descent; ~95% recovery of bias-induced damage).
- [x] Per-class loss-cap added to `ClassProbeSet.evaluate` (default 50)
      to prevent astronomical-loss spikes in continual-learning-style
      regimes from corrupting forgetting-magnitude metrics.
- [x] Per-progress OOB forgetting metric added to summary
      (`forgetting_oob_per_progress`).
- [x] Findings F1–F13 documented in §8.
- [x] Multi-trial validation reference run (3-trial pairwise study completed).

Pending:

- [ ] **Hypothesis (W'') predictive criterion** — test whether
      $\cos(g_t, g_t^{\text{full}})$ predicts BoGrad's per-step effect
      (high cos = parallel = BoGrad hurts; low cos = perpendicular =
      BoGrad helps).
- [ ] Statistical-baseline implementation (§7.3) — define the noise floor
      under a Gaussian-noise null model so we can identify "real" structural
      interference.

Each item produces a results subfolder and a short Markdown writeup that
this document will eventually link to.

---

*Draft v0.2. Updated 2026-05-04 with Findings F1–F6 from the first
validation reference run and the synthetic-quadratic + synthetic-class-disjoint
experiments. Next revision will fold in the out-of-batch forgetting metric
and the biased-interference quadratic experiment.*
