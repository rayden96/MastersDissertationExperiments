# COSGD scrutiny — improvement findings

Implement-and-measure study of COSGD design changes, each tested for what it
targets, with/without, across problems of increasing realism (2D blobs → hard
5-class blobs → CIFAR-10 small CNN). Code: `improvements_AE.py` (round 1),
`improvements_round2.py` (round 2). Optimizer flags live in
`common/optimizers/COSGD.py` + `_per_class.py`, all opt-in (plain COSGD unchanged).

## Round 1 (A–E)

| # | change | targets | verdict | key evidence |
|---|---|---|---|---|
| **A** | `preserve_magnitude` (rescale combined → raw-batch-grad norm) | effective-step / LR coupling | **ADOPT — the main win** | CIFAR-10: shifts COSGD's best LR 0.2→0.05 and raises acc 0.253→0.265; cleaner single-peaked LR curve. Magnitude reduction was a hidden LR cut; removing it decouples LR from the method. |
| **B** | `low_memory` in-place GS | peak memory | **ADOPT (free)** | GS kernel **bit-identical** to dense (max diff 0.00e+00); same accuracy. NOTE: peak mem unchanged so far because `_extract_flat_grad` already materialises the `[C,P]` matrix — that first alloc dominates, not GS scratch. Real memory win needs round-3 (stream per-class grads, never materialise `[C,P]`). |
| **C** | `negative`+`freq` canonical defaults | best config | **freq confirmed; negative≈normal here** | 5cls best-over-LR: freq > mean ≫ sum for both GS variants; `sum` worst (0.690). `freq` (the §02 frequency weighting) is best + most LR-robust → make it the default. |
| **D** | `cluster_k` (cosine k-means C→K, orthogonalise K cluster-sums) | scalability + granularity | **REJECT — hurts badly** | 5cls (C=5): full=0.72 but K=2→0.25, K=3→0.26. Merging classes before orthogonalising destroys the very conflict structure COSGD exploits. *Naive clustering is NOT the answer to COSGD's O(n²) cost — BoGrad (or a smarter scheme) is.* A genuine negative result for the thesis. |
| **E** | `vmap` per-sample grads (torch.func) replacing the C-backward loop | compute | **ADOPT for no-BN models** | per-class grads match the loop to ~1e-7 (numerically exact); 1.49× faster on the 5cls MLP; same accuracy. (BN models excluded — functional_call + running-stat updates don't compose.) |

### Round-1 narrative
- COSGD's only *accuracy*-relevant lever among A–E was **A**, and it works by
  fixing an implementation artefact (magnitude/LR coupling), not by changing the
  orthogonalisation itself. This reframes the earlier "tuned COSGD ties baseline"
  result: part of the gap was just an unfavourable effective-LR.
- **D** is the most informative result: per-class granularity is load-bearing.
- **B/E** are pure efficiency wins with proven equivalence — safe to default
  (E only on no-BN models).

## Round 2 (F–H + STACK)  — motivated by round 1

Hypotheses under test:
- **F** soft orthogonalisation (`orth_strength` α): full orth may *over-correct*;
  peak accuracy might be at α<1 (a blend of raw and orthogonalised).
- **G** conflict gating: COSGD orthogonalises every step even when classes don't
  conflict (the F5 "detection ≠ benefit" trap). Gate on mean pairwise cosine →
  only orthogonalise when there's real conflict (also cheaper).
- **H** PCGrad symmetric projection vs sequential GS: removes GS's arbitrary
  order-dependence (each vector projected against *all* others, random order).
- **STACK** best-of-round-2 vs round-1-best COSGD vs baseline (±momentum).

### Round-2 results — hard 5-class (averageable-noise regime, baseline 0.7322)

**F — soft orthogonalisation is monotonically harmful here:**
```
orth_strength: 0.0    0.25   0.5    0.75   1.0
accuracy:     0.7322 0.7300 0.7284 0.7262 0.7236
```
Peak is at α=0 (no orthogonalisation). The more you orthogonalise, the worse —
because the per-class conflict on overlapping Gaussian blobs is *averageable
noise, not structured interference*. This is the sharpest confirmation of the
F5 "detection ≠ benefit" principle: COSGD removes conflict that SGD's averaging
would have handled for free, so it only loses signal. (α=0 exactly equals
baseline → the soft-orth interpolation is correctly implemented.)

**G — conflict gating works, but needs a high threshold:**
```
ungated 0.7236 | gate(0.05) 0.7236 | gate(0.1) 0.7231 | gate(0.2) 0.7320
```
At threshold 0.2 the gate fires (skips orthogonalisation on the noisy steps) and
recovers baseline (0.732 vs 0.724 ungated). So gating CAN rescue the noise
regime — but the scalar mean-pairwise-cosine gate is a crude discriminator: this
regime's conflict is many small near-orthogonal pairs (mean cos ≈ 0), not
strongly negative, so a low threshold never triggers. **Design lesson:** a better
gate keys on `cos(orthogonalised_dir, raw_dir)` — if orthogonalisation barely
rotates the step, the conflict was averageable → skip. (Round 3.)

### Round-2 results — CIFAR-10 (2 seeds, lr∈{0.05,0.1,0.2}, 4 epochs)

**F — soft orth: full orthogonalisation again over-corrects.**
```
baseline 0.308 | a=0 0.312 | a=0.25 0.292 | a=0.5 0.262 | a=0.75 0.286 | a=1 0.280
```
Peak at α=0 again; ungated COSGD (any α>0) underperforms baseline at this budget.

**G — conflict gating: THE WIN. Gated COSGD beats baseline.**
```
baseline 0.3075 | ungated 0.2798 | gate(0.05) 0.2823 | gate(0.10) 0.3225 |
gate(0.20) 0.3095 | gate(0.40) 0.3080
```
**gate(thr=0.10) = 0.3225 vs baseline 0.3075 → +1.5pts**, flipping COSGD from a
−2.8pt loss (ungated) to a win. First time in the whole study COSGD beats SGD on
CIFAR-10 at matched budget. Mechanism = exactly W': orthogonalise only the steps
with structured conflict, skip the averageable-noise steps. There is a clear
gate sweet-spot (0.10) — too low (0.05) doesn't fire, too high (0.4) gates
everything back to baseline.

**H — PCGrad symmetric vs sequential GS: no improvement.**
```
baseline 0.308 | seq-GS negative 0.281 | PCGrad symmetric 0.277
```
Removing GS's order-dependence didn't help (slightly worse + much slower, O(C²)
Python loop). Order-dependence is not COSGD's problem.

**STACK — combined best-of-round-2:**
```
SGD no-mom 0.308 | SGD+mom0.9 0.412 | COSGD round-1-best 0.270 | STACK 0.276
```
The stack (freq+preserve+soft0.75+gate) ≈ round-1 COSGD, both well below
SGD+momentum (0.412). NOTE the stack used orth_strength=0.75 + a fixed gate
threshold; the *standalone* gate(0.10) at full strength (0.3225) beat it — i.e.
**the gate is the active ingredient; soft-orth dilutes it.** The right config is
full-strength orth + a well-tuned conflict gate, NOT soft orth.

### FINAL verdict table

| change | adopt? | why |
|---|---|---|
| A preserve_magnitude | **yes** | decouples LR; small acc gain; fixes hidden LR cut |
| B low_memory GS | yes (but) | bit-identical; real mem win needs round-3 (don't materialise [C,P]) |
| C freq combine default | **yes** | best + LR-robust; sum diverges |
| D clustering | **no** | destroys accuracy (chance) on both problems |
| E vmap grads | yes (no-BN) | exact, 1.5× faster |
| F soft orth (α<1) | **no** | full orth over-corrects, but soft orth just dilutes the gate; gate is the real lever |
| **G conflict gate** | **YES — the breakthrough** | gate(0.10) = +1.5pts on CIFAR, COSGD's first matched-budget win |
| H PCGrad symmetric | no | no gain, slower; order-dependence isn't the problem |

### Bottom line
COSGD's problem was never the orthogonalisation algorithm — it was applying it
*indiscriminately*. A conflict gate that orthogonalises only structured-conflict
steps (and lets SGD average the rest) turns COSGD from a net loss into a net win.
This is the same "separable from descent" (W') precondition BoGrad needs — the
two methods unify. **Recommended canonical COSGD: freq + preserve_magnitude +
conflict_gate(~0.1) + full-strength modified_gs_negative + vmap grads (no-BN).**
Caveat: all CIFAR numbers are short-budget (4 epochs, 2 seeds); the gate win
(+1.5pts) needs confirmation at more seeds / longer schedule before it's
load-bearing for the thesis — queued next.

### Synthesis so far
The study is converging on one clear thesis-level statement:
**COSGD's per-class orthogonalisation helps only when the inter-batch conflict is
structured (separable from descent); on averageable-noise conflict it strictly
hurts, monotonically in how much it orthogonalises.** The practical implication
is that COSGD needs a *gate* that detects the regime — which is exactly the W'
"separable from descent" criterion the BoGrad work also landed on. The two
methods converge on the same precondition.
