# 10 — BoGrad Investigation (Gap 1)

A thorough, theory-backed + empirical ablation of every BoGrad hyperparameter
and design decision, across multiple base optimizers, problems, and
architectures. The goal is to answer **when BoGrad helps, with what settings,
and why** — with no gaps — and to explain each accuracy result with the
interference metric that moves underneath it.

## Protocol

- **Workhorse:** CIFAR-10 + `small_cifar_cnn` (fast, no BN, clean gradient
  geometry). Most axes are characterised here first.
- **Generalisation set:** MNIST (grayscale CNN), CIFAR-100 (ResNet-18, has BN),
  Covertype (MLP). Used to confirm the workhorse winner transfers — a couple of
  configs, not a full re-sweep.
- **Base optimizers:** SGD, SignSGD, RMSprop, Adam — every axis is swept across
  all four unless noted (the "for which optimizer" half of the question).
- **Seeds:** 3 for sweeps (paired data order), mean ± std reported.
- **The "why":** every run carries an `InterferenceMeter`, so each axis summary
  pairs accuracy with `I_between_K`, `cos(u_t,u_{t-1})`, the per-step deficit
  `D_t`, and (for the adaptive bases) the preconditioned deficit `D_t_precond`.

## Fixed by design (NOT swept)

`project_stage="update"`. The between-batch framework (FocusedWork §04) sums the
**applied updates** `u_t`, so the applied update is the principled projection
target; it is also the correct stage for momentum/preconditioned bases (project
after the preconditioner, not the raw gradient). Stated and justified in the
chapter, not ablated.

## Axes

| # | Folder | Knob | Sweep |
|---|---|---|---|
| 10.01 | `01_buffer_K` | buffer size K | {0,1,2,4,8,16,32,64,128} |
| 10.02 | `02_lr_retune` | learning rate per (opt,K) | log grid |
| 10.03 | `03_projection_mode` | mode + strength α | full/negative/positive × α∈[0,1] |
| 10.04 | `04_orth_method` | orthogonalisation | sequential / QR / Householder |
| 10.05 | `05_projection_scope` | scope | per_tensor / global |
| 10.06 | `06_magnitude` | direction vs magnitude | preserve_magnitude × random_projection |
| 10.07 | `07_momentum_2x2` | momentum × BoGrad | μ∈{0,0.9} × BoGrad∈{off,on} |
| 10.08 | `08_batch_K` | batch size × K | {32..512} × {0,2,8,32} |
| 10.09 | `09_cross_summary` | master table | best config per (opt × dataset/arch) |

## Conventions

- Each axis is a thin `run.py` that declares a sweep grid and calls
  `run_bograd_sweep(...)` from `_ablation.py` (shared harness — no copy-paste).
- `run.py` produces data only; `plot.py` reads `results/run_<id>/summary.json`
  and renders. Outputs under `<axis>/results/run_<id>/` (gitignored).
- Resumable: a completed cell's `results.json` is skipped, so a dead Colab
  session re-launches and continues the sweep.

## How to run (Colab or local)

```bash
python PaperReadyExperiments/10_bograd_ablation/01_buffer_K/run.py            # full sweep
python PaperReadyExperiments/10_bograd_ablation/01_buffer_K/run.py --smoke    # 1 base, tiny, fast
python PaperReadyExperiments/10_bograd_ablation/01_buffer_K/plot.py
```
