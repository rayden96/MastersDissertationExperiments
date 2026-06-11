# Dissertation — Experimental Chapters: Questions & Subsections

Working spine for writing the experimental chapters of *Batch Orthogonalised Gradient
Descent for Single-Task Classification* (BoGrad & COSGD), reframed as a study of
single-task gradient interference. Each subsection is framed as a **research question
(RQ)** with: the design choice / claim, the theoretical motivation, the prediction, the
interference metric that explains it ("the why"), and the current data status.

Status legend: ✅ data exists · ♻️ re-run/extend as we write · ➕ new experiment · ❓ open decision.

Convention note (set during the consolidation pass): the first-order deficit `D_t` is now
**positive = hurt** (≥ 0, larger = more first-order descent lost to interference). "Less
interference" metrics (I_inter, I_between_K) therefore correlate **negatively** with `D_t`.

---

## CHAPTER A — The Interference Framework (the coined metric)

The conceptual chapter that defines the lens. Canonical definitions: `PreDiscovery/FocusedWork/01–04`.
Implementation: `PaperReadyExperiments/interference/`.

### A.1 What single-task gradient interference is (and isn't)
- **RQ-A1:** What does it mean for gradients to *interfere* within a single classification
  task, and how is that different from multi-task / continual-learning interference?
- Define interference as **directional cancellation beyond what i.i.d. mini-batch noise
  would produce**. Position against the multi-task literature (PCGrad, GradDrop) — same
  geometry, single-task setting.
- ❓ **Open (A1):** the definition is reference-relative ("beyond i.i.d. noise") but the
  framework currently reports *absolute* levels with no i.i.d. null subtracted. Decide:
  add a Gaussian/shuffled-label null baseline, or frame absolutes + deltas-vs-baseline as
  the operational definition.

### A.2 Two types of interference
- **RQ-A2:** Is interference within a batch (across per-class subgradients) mechanistically
  distinct from interference across batches (across successive applied updates)?
- Define **inter-batch** (Type A, within-batch per-class — COSGD's target) and
  **between-batch** (Type B, across-update trajectory — BoGrad's target). State the
  separability as an empirical claim to be tested (→ delivered by the COSGD↔BoGrad
  dissociation, §B.8/§C.9).
- Terminology footnote: "inter-batch" here means *within one batch* (per-class), the
  opposite of the natural reading — flag explicitly for the examiner.

### A.3 The interference index and its companions
- **RQ-A3:** What scalar(s) capture each interference type, and what does each isolate?
- Inter-batch: cancellation index `I_inter = ‖Σ_c g_c‖ / Σ_c‖g_c‖`; pairwise cosine
  (`frac_neg`, `mean_cos`); magnitude stats (`mean/std norm`, `max/min ratio` — the PCGrad
  magnitude criterion); useful/wasted-mass decomposition vs the reference direction.
- Between-batch: `I_between_K = ‖θ_t − θ_{t−K}‖ / Σ‖u_i‖`; window pairwise cosine;
  useful/wasted **path**; `useful_path_frac` (the curvature-vs-cancellation disambiguator).
- **RQ-A3b:** can we attribute interference to **angle** vs **magnitude**? The methods act
  on angle only; the framework tracks both — state the angle-only working hypothesis.

### A.4 First-order loss-decrease accounting (the "training-hurt" scalar)
- **RQ-A4:** Can we turn geometric cancellation into a scalar that measures how much a step
  *hurt training*, independent of optimiser?
- Define `D_t = ⟨g̃,u_t⟩ + η‖g̃‖² ≥ 0` (positive = hurt) recovered from the applied update
  `u_t`; cumulative `Σ_t D_t` as the headline training-hurt total.
- **RQ-A4b (preconditioning):** for Adam/RMSprop, `u_t` is preconditioned, so the SGD-yardstick
  `D_t` conflates cancellation with rescaling. Define `D_t_precond` against the optimiser's
  own full-batch response to `g̃` to isolate cancellation. ➕ write `PRECONDITIONED_DEFICIT.md`
  (currently documented only in code).

### A.5 Measurement protocol
- **RQ-A5:** How are these measured without perturbing training, and how stable are they?
- Reference set (balanced subsample), refresh cadence, per-step logging, ≥3 seeds.
- Limitations to state: reference staleness between refreshes; subsample variance; the
  unweighted-vs-frequency-weighted batch-gradient approximation in `useful_descent_frac`
  (negligible for class-balanced batches, flagged for imbalanced).

### A.6 Validation on a controlled problem — `01_small_2d` ✅♻️
- **RQ-A6:** When interference is *dialled in* on a 2D Gaussian-mixture problem, do the
  metrics move exactly as theory predicts (I_inter low→high as θ→centroid; mean cosine
  rises; `D_t` falls)?
- The sanity check that the metric *means what we say it means*.

### A.7 Behaviour at scale — `02_medium_cifar10`, `03_large_cifar100` ✅♻️
- **RQ-A7:** Do the metrics stay interpretable on real CNNs/ResNets, where high
  dimensionality drives random cosines toward 0?
- State the **dimensionality-sensitivity** limitation: `I_inter` is mechanically lower for
  more classes; cosines compress in high-d. ❓ **Open (A7):** add a #subgroup / dimension
  normalization for cross-dataset comparison, or restrict cross-dataset claims to deltas.

### A.8 Are the metrics *predictive* (not just descriptive)? → bridge to synthesis
- **RQ-A8 (the examiner question):** Does a reduction in an interference metric *predict* a
  gain in accuracy across the suite? Within-run, the framework correlates each metric with
  `D_t`; across-run prediction of **accuracy** is delivered in the synthesis chapter (M4 /
  `40_synthesis`). ➕ not yet run.

### A.9 Limitations of the framework
- i.i.d. null not yet subtracted (A1); cross-dataset normalization (A7); first-order
  sufficiency check `Σ D_t^meas` vs `Σ D_t` not yet computed; headline-K fixed at 32 rather
  than data-driven; momentum/precond handling. These become the honest "threats to validity".

---

## CHAPTER B — COSGD: Class-Orthogonalised Gradient Descent (inter-batch)

Narrative climbs the **dimensionality** ladder and ends at the O(n²) cost wall that
motivates BoGrad. Testbed: low-dim ladder iris(4)→wine(13)→breast_cancer(30)→digits(64)
+ image check (MNIST, CIFAR-10). Headline "why" metric = `I_inter` (should rise) +
`inter_mean_cos` (toward 0). Canonical "reclaimed" config: `gram_schmidt_normal` + desc +
`combine="sum"` + `combine_norm_cap=2.0`.

### B.0 What COSGD is and the mechanism it targets
- Per-class subgradients → Gram-Schmidt orthogonalise → combine → base optimiser. It edits
  the **angles** among `{g_c}`, not magnitudes. The honest framing up front: COSGD reliably
  *reduces measured inter-batch cancellation*; whether that converts to accuracy/speed is
  regime-dependent (the chapter's throughline).

### B.1 Combine rule + norm cap — `20_05_combine` ✅ (central scrutiny axis)
- **RQ-B1:** Where does COSGD's acceleration actually come from — the orthogonalisation, or
  the way the orthogonalised per-class gradients are *combined* into a step?
- Motivation: `combine="sum"` produces a bigger, well-directed step (the paper's speed-up);
  `mean`/`freq` shrink it back toward a normal averaged update; raw `sum` over-scales and
  diverges at higher dim → `combine_norm_cap` rescues it.
- Prediction: sum/sum+cap win on low-dim; raw sum breaks on digits while sum+cap holds;
  mean/freq sit near baseline. **Why:** `I_inter` + observed effective step-norm across the
  ladder. (Lead with this — it explains *why* COSGD works before refining *how*.)

### B.2 Gram-Schmidt variant — `20_01_gs_variant` ✅
- **RQ-B2:** How should the per-class conflict be removed — classical vs modified GS,
  full vs negative-only projection?
- Motivation: modified GS = numerically stabler in high-d; `negative` removes only
  destructive (anti-aligned) overlap (softer) vs `normal` removing all overlap.
- Prediction (F14): `modified_gs_negative` cuts the anti-aligned class-pair fraction and
  flips mean cosine toward 0 while improving accuracy. **Why:** `I_inter`, `inter_mean_cos`.
- ➕ **Extend:** fold `pcgrad` (symmetric, order-independent projection) into this axis to
  close the GS order-dependence loop opened by B.3.

### B.3 Class (magnitude) ordering — `20_02_class_order` ✅♻️
- **RQ-B3:** GS is order-dependent (the first vector is preserved exactly). Does processing
  classes in **descending gradient magnitude** — preserving confident, well-represented
  classes — beat ascending/random/fixed, or is it marginal?
- **Why:** per-class useful mass + `I_inter`. ♻️ extend to image datasets (currently ladder-only).

### B.4 Pre-normalisation before GS — `20_03_prenormalize` ✅♻️
- **RQ-B4:** In high dimension, raw-magnitude vectors are not naturally near-orthogonal
  (only *unit* vectors are). Does unit-normalising per-class subgradients before GS change
  what's removed, and does it stabilise high-d?
- Note: in the §05 data, prenorm was the lever that most rescued digits — worth foregrounding.
- **Why:** `I_inter` + per-class useful/wasted; pair with the synthetic cosine-vs-dim curves.

### B.5 Step method / BatchNorm handling — `20_04_step_method` ✅♻️
- **RQ-B5:** Per-class forward passes corrupt BN running statistics. Does the BN-frozen
  variant matter, and only for BN models?
- Prediction: near-tie on no-BN CIFAR-10 CNN; BN-frozen wins on BN ResNet. **Why:** accuracy
  gap between variants on BN vs no-BN models. (Implementation-correctness axis.)

### B.6 Base optimizer × COSGD — `20_06_base_optimizer` ✅
- **RQ-B6:** Does per-class orthogonalisation still help once a preconditioner (RMSprop/Adam)
  or momentum already smooths the update — or does the base absorb the within-batch conflict?
- Observed: COSGD+adaptive is robust on the ladder; COSGD+plain-SGD collapses on higher-dim
  at un-retuned LR. **Why:** `I_inter` + per-optimiser `D_t_precond`. ❓ ties to the LR
  decision (D.1).

### B.7 Class-count scalability — `20_07_scalability` ➕ (the O(n²) wall)
- **RQ-B7:** How do wall-clock/step and peak memory scale with the number of classes n?
- Motivation: per-class GS is O(n²) in dot products and stores n vectors. This is the
  **narrative hinge to BoGrad** (O(K), fixed buffer, independent of n).
- **Why:** sec/step + peak MB vs n (synthetic CIFAR-10 {2..100} + real CIFAR-10/EMNIST-47/CIFAR-100).
  ➕ not yet in the master table — run it; it is the bridge to Chapter C.

### B.8 Synthesis + COSGD↔BoGrad mechanism contrast — `20_08_cross_summary` ✅
- **RQ-B8:** Does COSGD move the **inter-batch** axis (ΔI_inter > 0) while leaving the
  **between-batch** axis roughly untouched — and BoGrad the reverse?
- The dissertation's central dissociation. Now symmetric (Δ-vs-Δ) after the aggregator fix.

### B.9 Knob gaps to consider (symmetry with BoGrad)
- ❓ direction-vs-magnitude control for COSGD (`preserve_magnitude`/`orth_strength`) — the
  COSGD analogue of BoGrad §C.4; currently no axis.
- ❓ explicit conflict-gate demonstration axis (gate exists in code, never demonstrated).

---

## CHAPTER C — BoGrad: Batch-Orthogonalised Gradient Descent (between-batch)

Narrative crosses the **optimizer** family and answers COSGD's cost wall with an O(K)
fixed buffer. Workhorse: CIFAR-10 / small CNN (no BN); all four bases {SGD, SignSGD,
RMSprop, Adam}. **Fixed by design (not ablated):** `project_stage="update"` — the
between-batch framework sums applied updates `u_t`, so projecting the applied update is the
principled target (and correct for momentum/preconditioned bases). Headline "why" =
`I_between_K` + `cos(u_t,u_{t−1})` + `D_t`.

### C.0 What BoGrad is and the mechanism it targets
- Project the applied update against a buffer of K recent updates (soft sequential
  Gram-Schmidt), removing recent-direction overlap. It edits the **trajectory**, not the
  within-batch class structure.

### C.1 Buffer size K — `10_01_buffer_K` ✅
- **RQ-C1:** Recent updates are correlated, so small K already captures most destructive
  overlap while large K over-strips useful descent. Is the optimum **finite and per-optimizer**?
- Prediction: interior optimum, distinct per base (SGD~32, RMSprop~16, Adam≥128, SignSGD~64).
  **Why:** `I_between_K` rises then saturates/over-corrects; the accuracy-optimal K coincides
  with the K that best cuts cancellation without inflating `D_t`. (The knob that defines
  BoGrad; seeds best-K downstream.)

### C.2 Learning-rate retune — `10_02_lr_retune` ✅
- **RQ-C2:** BoGrad shrinks the effective step, so is its best LR *lower* than the same
  optimiser's baseline best — and does every later comparison need each method at its own LR?
- The methodological prerequisite, not a design choice. **Why:** LR-vs-accuracy basin shift.
  ❗ Caveat to state: same-LR comparisons inflate/deflate the gap (see the inflated Δ in the
  current master table). Ties to D.1.

### C.3 Projection mode + strength — `10_03_projection_mode` ✅
- **RQ-C3:** *Which* overlap should be removed — destructive only (`negative`), all (`full`),
  or redundant only (`positive`) — and is partial projection (α) ever better than full?
- Prediction: `negative` is the safe default; `full`/`positive` collapse when buffered
  directions align with descent (they strip signal). **Why:** pairwise-alignment fingerprint,
  `cos(g, g̃)`. Helps iff interference is *separable* from descent.

### C.4 Direction vs magnitude — `10_06_magnitude` ✅
- **RQ-C4:** Is BoGrad's benefit from the **direction** change, or the incidental step-size
  reduction? (2×2: real-vs-random buffer × magnitude free-vs-preserved.)
- Prediction: real-buffer ≫ random-projection control; `preserve_magnitude` ≈ default
  (magnitude is inert). **Why:** `g_ratio = ‖g̃‖/‖g‖` + accuracy gap vs the random control.
  (Placed early — it justifies the premise before piling on knobs.)

### C.5 Orthogonalisation method — `10_04_orth_method` ✅
- **RQ-C5:** Is the *soft* sequential subtraction (not a true projection) better than a
  *true-orthogonal* projector (QR/Householder), and why?
- Prediction: sequential-negative wins; true-orth is too aggressive and strips descent.
  **Why:** orthogonality residual `‖B̂ᵀg̃‖` (≈0 true-orth, larger sequential) + accuracy +
  ms/step. Note: qr ≡ householder (same projector) — report as one "true-orthogonal" arm.

### C.6 Projection scope — `10_05_projection_scope` ✅
- **RQ-C6:** Per-tensor (local per-layer history) vs global (one joint cross-layer subspace)
  — does the joint subspace buy accuracy, at what memory/time cost?
- **Why:** `I_between_K`, ms/step, peak memory. (Architecture/cost refinement.)

### C.7 Momentum × BoGrad — `10_07_momentum_2x2` ✅
- **RQ-C7:** Momentum and BoGrad both shape the trajectory — momentum induces *positive*
  `cos(g_t,g_{t−1})` (short-horizon smoothing); BoGrad removes recent-direction overlap
  (persistent). Does BoGrad help *on top of* momentum (do they compose)?
- 2×2 {μ∈0,0.9}×{BoGrad off,on}, SGD/SignSGD/RMSprop (Adam excluded — no clean μ-off).
  **Why:** `cos(u_t,u_{t−1})` **lag-decay profile** + `I_between_K`. ➕ the lag-decay profile
  needs adding to the meter (currently the window cosine is pooled, not lag-resolved) — see D.4.

### C.8 Batch size × K — `10_08_batch_K` ✅
- **RQ-C8:** Smaller (noisier) batches have more between-batch cancellation. Does BoGrad's
  benefit and best-K grow as batch shrinks?
- Prediction: largest gain at small batch, moderate K (heatmap). **Why:** `I_between_K` as a
  function of batch size. ♻️ extend beyond SGD-only / add EMNIST.

### C.9 Cross-optimizer / cross-architecture summary — `10_09_cross_summary` ✅
- **RQ-C9:** Across optimisers and architectures, *when* does BoGrad help, with *what*
  settings, and *which* interference metric moved? The master "when/what/why" table (now
  with epoch/wall speed-up + Δ-I_between).

### C.10 Knob gaps (mention as held-at-default)
- `store_normalised`, `nesterov`, `min_projection_dim`, base-optimiser internals
  (weight_decay/betas/alpha) held at defaults; `project_stage` fixed to "update" by design.

---

## CHAPTER D — Open decisions to resolve while writing  ❓

These shape what we re-run as we write each subsection. Resolve early.

- **D.1 — LR policy for ablations. ✅ DECIDED:** keep the fixed-LR cells for clean knob
  isolation **and** add one per-method tuned-LR **headline row** per key axis (buffer-K,
  combine, base-optimizer), so the "does it help" claim is fair while isolation stays clean.
  (Fixed-LR alone confounds the deltas — COSGD's high-dim collapse and BoGrad's inflated
  adaptive deltas are both LR-mismatch artifacts.)
- **D.2 — Headline metric. ✅ DECIDED:** lead with convergence **speed-up** (epochs- and
  wall-clock-to-baseline-target, now computed by the aggregators from existing curves) and
  report final accuracy alongside. Speed is the thesis's stated point.
- **D.3 — i.i.d. null + cross-dataset normalization** (A1/A7): add, or frame as deltas only.
- **D.4 — Lag-resolved cosine** in the meter (needed for C.7's stated analysis): add the
  `cos(u_t,u_{t−k})` vs k profile.
- **D.5 — Metric→accuracy predictiveness** (A8): run the synthesis regression (M4).
- **D.6 — Symmetry axes:** COSGD direction-vs-magnitude (B.9), conflict-gate demo (B.9),
  pcgrad into B.2.

---

## CHAPTER E — Final comparison (the main result, after both method chapters)

The 6-dataset × 4-optimizer × 5-arm bakeoff (`30_main_comparison`): baseline vs COSGD vs
BoGrad vs dropout vs GradDrop, independently tuned, ≥5 seeds, paired data order, speed +
final accuracy + accuracy-vs-wall-clock Pareto. Uses the Chapter B/C winners to bound tuning.
Then the synthesis chapter (`40_synthesis`): does interference reduction predict the gain (A8)?
