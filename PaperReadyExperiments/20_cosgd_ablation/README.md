# 20 — COSGD Investigation (Gap 2)

A thorough, theory-backed + empirical ablation of every COSGD design decision,
across base optimizers, problems, and architectures. Goal: **when does per-class
orthogonalisation help, with what settings, and why** — explained with the
inter-batch interference metric `I_inter` (within-batch per-class cancellation),
which is exactly what COSGD targets.

## Protocol

- **Workhorse:** CIFAR-10 + `small_cifar_cnn` (10 classes, cheap per-class
  backward, no BN). Class-count and BN axes use EMNIST-47 / CIFAR-100.
- **Base optimizers:** SGD by default for the mechanism axes (cleanest); the
  base-optimizer axis (20.06) crosses all four — COSGD is now a wrapper, so it
  composes with SGD/SignSGD/RMSprop/Adam.
- **Seeds:** 3 (paired data order), mean ± std.
- **The "why":** every run carries the `InterferenceMeter`. COSGD should *raise*
  `I_inter` (less cancellation) and flip the mean per-class pairwise cosine
  toward 0 (prior finding F14: modified_gs_negative cut IB_%neg 70%→54%).

## Axes

| # | Folder | Knob | Sweep |
|---|---|---|---|
| 20.01 | `01_gs_variant` | GS variant | {classical,modified} × {normal,negative} |
| 20.02 | `02_class_order` | magnitude ordering | desc/asc/random/fixed |
| 20.03 | `03_prenormalize` | pre-normalisation | on/off (+ synthetic dim sweep) |
| 20.04 | `04_step_method` | per-class forward / BN | single/multi/multi_with_BN |
| 20.05 | `05_combine` | combine rule | sum/mean/freq |
| 20.06 | `06_base_optimizer` | base × COSGD | sgd/signsgd/rmsprop/adam |
| 20.07 | `07_scalability` | class count | synthetic {2..100} + real |
| 20.08 | `08_cross_summary` | master table | best per (opt×dataset) + COSGD↔BoGrad |

## Known interaction (characterised, not a bug)

`combine="sum"` sums the orthogonalised per-class gradients, so the effective
step scales ≈ (number of classes) × LR — this destabilises an untuned LR on
small problems (seen in early smoke tests). 20.05 maps this directly; `mean` and
`freq` (the §02 class-frequency weighting) are the stable alternatives.

## How to run

```bash
cd PaperReadyExperiments/20_cosgd_ablation
python 01_gs_variant/run.py --smoke           # quick shake-out
python run_all.py --epochs 10 --seeds 2026 2027 2028
python 08_cross_summary/run.py                # master table + COSGD<->BoGrad contrast
```

Colab: use the M1 `colab_launcher.ipynb` as a template (point it at this folder),
or extend `run_all.py`. Resumable via the per-cell `JobManager`.
