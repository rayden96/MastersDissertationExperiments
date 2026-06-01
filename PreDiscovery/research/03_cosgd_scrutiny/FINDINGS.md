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

(Results filled in when the round-2 run completes — see `cosgd_round2.log`.)
