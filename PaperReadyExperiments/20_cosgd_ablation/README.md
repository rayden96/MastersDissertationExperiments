# 20 — COSGD Investigation (Gap 2)

A thorough, theory-backed + empirical ablation of every COSGD design decision.
Goal: **when does per-class orthogonalisation help, with what settings, and why**
— explained with the inter-batch metric `I_inter` (within-batch per-class
cancellation), which is exactly what COSGD targets.

> **Realigned to the scrutiny study** (`PreDiscovery/research/03_cosgd_scrutiny/
> FINDINGS.md`) + the conference paper. The canonical COSGD is the **reclaimed**
> config: the paper's algorithm — full classical Gram-Schmidt, **descending-
> magnitude sort, `combine="sum"`** — plus a **`combine_norm_cap`** (default 2.0)
> that preserves `sum`'s big-step speed-up while preventing high-dim divergence.
> `sum` IS the speed mechanism; `mean`/`freq` and the conflict gate flatten COSGD
> to baseline and are tested here only to *demonstrate* that, not as defaults.

## Protocol

- **Primary testbed = the low-dimensional ladder** where COSGD's effect is large
  and interpretable: **iris (4) → wine (13) → breast_cancer (30) → digits (64)**,
  small MLP. This is COSGD's home turf (paper: 5.6× on iris; reproduced).
- **Image check:** MNIST + CIFAR-10 (small CNN) confirm the win carries to conv
  nets (reclaimed COSGD: 3.33× on both — see scrutiny `reclaim_img_results.log`).
- **Baseline-to-beat in every axis:** SGD *and* reclaimed COSGD (so each knob is
  judged against the known-good config, not the broken mean/gate one).
- **Base optimizers:** SGD for mechanism axes; 20.06 crosses all four (wrapper).
- **Seeds:** 3, paired order, mean ± std.
- **The "why":** every run carries the `InterferenceMeter`; COSGD should raise
  `I_inter` and flip mean per-class cosine toward 0.

## Axes

| # | Folder | Knob | Sweep | Scrutiny expectation |
|---|---|---|---|---|
| 20.01 | `01_gs_variant` | GS variant | classical/modified × normal/negative | full classical (paper) is the reference |
| 20.02 | `02_class_order` | magnitude ordering | desc/asc/random/fixed | desc (paper) preserves the big confident-class step |
| 20.03 | `03_prenormalize` | pre-normalisation | on/off (+ synthetic dim sweep) | off (paper); norm changes the sum scale |
| 20.04 | `04_step_method` | per-class fwd / BN | single/multi/multi_with_BN | BN-frozen matters only for BN models |
| 20.05 | `05_combine` | **combine + norm-cap** | sum / sum+cap{2,3,4} / mean / freq | **sum is the win; mean/freq flatten it; cap rescues high-dim** |
| 20.06 | `06_base_optimizer` | base × COSGD | sgd/signsgd/rmsprop/adam | COSGD overlaps with adaptive/momentum smoothing |
| 20.07 | `07_scalability` | class count | synthetic {2..100} + real | O(n²) cost wall (the BoGrad motivation) |
| 20.08 | `08_cross_summary` | master table | best per (opt×dataset) + COSGD↔BoGrad | inter- vs between-batch contrast |

## The combine finding (central, characterised)

`combine="sum"` sums the orthogonalised per-class gradients into a **bigger,
well-directed step** — that magnitude is COSGD's acceleration mechanism, not a
bug. It does over-scale and diverge on the higher-dim digits MLP; the
`combine_norm_cap` (cap × mean per-class norm) fixes exactly that case while
leaving the low-dim/few-class win untouched. `mean`/`freq` shrink the step back
to ~SGD and remove the speed-up. 20.05 maps this directly across the dim ladder.

## How to run

```bash
cd PaperReadyExperiments/20_cosgd_ablation
python 05_combine/run.py --smoke                       # quick shake-out
python run_all.py --datasets iris wine breast_cancer digits --epochs 30
python run_all.py --datasets mnist cifar10 --epochs 10  # image check
python 08_cross_summary/run.py
```

Colab: `colab_launcher.ipynb`. Resumable via the per-cell `JobManager`.
