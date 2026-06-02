# 30 — Main Comparison / Bakeoff (Gap 3)

The headline empirical chapter: **baseline vs COSGD vs BoGrad vs dropout vs
GradDrop**, across 4 base optimizers and 6 datasets (3 modalities), with
independent HP tuning and multiple seeds. Same architecture + schedule per
dataset across all arms.

## Execution model: tune-then-run

Unlike the ablations, every cell is **tuned on the validation split, reported on
the held-out test split**:

1. `_bakeoff.run_bakeoff_cell` calls `common.tuning.tune_cell` over a *small* HP
   grid bounded by the M1/M2 ablation winners (lr always; K for bograd;
   dropout_p for dropout; GS variant + combine for cosgd; leak for graddrop).
   Cached in `tune/<base+method>/best.json`.
2. The tuned config runs `n_seeds` (default 5) times with **paired data order**
   (`common.seeding.paired_shuffle` — identical batches across all methods within
   a seed), evaluating on test, with the interference meter attached.

This produces one **canonical record set** under `_core/results/run_<id>/`. The
view axes below all read it — they never retrain.

## Datasets (confirmed)

`mnist` (10), `cifar10` (10), `cifar100` (100), `emnist_balanced` (47),
`covertype` (tabular, 7), `yahoo_answers` (text, 10).

## Axes

**Views over the canonical record set** (`views.py`):

| # | Output | Thesis |
|---|---|---|
| **30.00** | **convergence speed-up (epochs-to-target + speed-up factor + epoch-1 acc) — THE HEADLINE** | 5.x |
| 30.01 | test-accuracy trajectories per dataset | 5.1 |
| 30.02 | fixed-budget tables (10/50/100% epochs), bold best | 5.2 |
| 30.03 | final-accuracy summary bars | 5.3 |
| 30.09 | accuracy-vs-wall-clock Pareto per dataset | 5.9 |

> **The primary question is convergence SPEED, not final accuracy.** 30.00 reports,
> per (dataset × base optimizer), how many epochs each method needs to reach the
> baseline's final accuracy, the speed-up factor (`baseline_epochs / method_epochs`),
> a `wall_speedup` that folds in per-step cost, and epoch-1 accuracy (early
> progress). `--target-frac 0.95` measures time-to-95%-of-baseline instead.
> Early evidence (MNIST, no momentum): BoGrad ≈ **1.8× (SGD), 1.67× (RMSprop),
> 1.25× (Adam)**; COSGD ≈ 1.0× (no speed-up). BoGrad accelerates training; COSGD
> mainly reduces within-batch interference without accelerating convergence.

**Focused sweeps** (own runs):

| # | Output | Script | Thesis |
|---|---|---|---|
| 30.04 | LR sensitivity (CIFAR-10/100) | `sensitivity.py --kind lr` | 5.4 |
| 30.05 | K sensitivity, BoGrad (per dataset) | `sensitivity.py --kind K` | 5.5 |
| 30.06 | batch-size sensitivity | `sensitivity.py --kind batch` | 5.6 |
| 30.07 | model-scale (width CNN + ResNet-18/34) | `scale_gradstats.py --kind scale` | 5.7 |
| 30.08 | gradient statistics during training | `scale_gradstats.py --kind gradstats` | 5.8 |

## How to run

```bash
cd PaperReadyExperiments/30_main_comparison
python run.py --smoke                                  # 1 ds, 2 methods, tiny
python run.py --datasets cifar10 --seeds 2026 2027 2028  # one dataset
python run.py                                          # full bakeoff (600 tuned runs!)
python views.py                                        # all figures + tables from newest campaign
python sensitivity.py --kind lr --datasets cifar10 cifar100
python scale_gradstats.py --kind scale
```

Budget: the full grid is **6 × 4 × 5 × 5 = 600** tuned final runs (plus tuning
trials). Run **per-dataset across Colab sessions** — every cell is resumable
(JobManager skips completed `results.json`), and tuning is cached.

## Rigor controls

Paired data order across arms within a seed (audited via `order_hash`); val-only
tuning, test-only reporting; identical schedule/architecture per dataset;
`git_sha` + resolved `hp` recorded per run.
