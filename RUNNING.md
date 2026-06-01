# RUNNING — how to execute the experimental program

This is the operational guide: what runs where, in what order, how long it takes,
how results flow between phases, and the exact commands. For *what each
experiment means*, see each phase's `README.md`; for the overall design, see
`PLAN.md` / the plan file.

---

## The model in one picture

```
M0  shared infra + optimizers   (DONE, in common/ + tests/)         <- no runs
        |
M1  10_bograd_ablation/   9 axes  --\
M2  20_cosgd_ablation/    8 axes  ---> ablation "winners" (best hp per knob)
        |                              |
        v                              v
M3  30_main_comparison/   tune-then-run bakeoff  (uses M1/M2 winners to bound tuning)
        |                              produces the canonical record set
        v
M4  40_synthesis/   reads M3 records -> "does interference predict accuracy" + W'
```

Each phase writes JSON under its own `results/` (gitignored). Plots/tables are
regenerated from JSON by the `plot.py` / `views.py` / `analyze.py` scripts —
**training and plotting are always separate**.

---

## Where things run

| | Laptop (RTX 3050, 4 GB) | Colab / cloud GPU |
|---|---|---|
| Dev, smoke tests (`--smoke`) | ✓ fast | ✓ |
| Synthetic-only scripts (20.03 dim-sweep) | ✓ instant | ✓ |
| One ablation axis, CIFAR-10 proxy | ✓ slow (~3h/axis) | ✓ preferred |
| Full M1/M2 sweeps (all bases) | ✗ too slow | ✓ |
| M3 bakeoff (600 tuned runs) | ✗ | ✓ per-dataset across sessions |
| CIFAR-100 / ResNet, COSGD@100-classes | ✗ (OOM/slow) | ✓ |

**Rule of thumb:** iterate and smoke-test locally; run real sweeps on Colab via
each phase's `colab_launcher.ipynb`. Everything is resumable, so Colab
disconnects are harmless.

---

## Resume semantics (important)

- Every training cell writes `results.json`; a `JobManager` **skips any cell whose
  `results.json` exists**. Re-running a sweep continues where it stopped.
- The `Trainer` also checkpoints `last.pt`/`best.pt` mid-run, so an *interrupted*
  cell resumes from its last epoch (same config_hash required).
- HP tuning caches `best.json` per cell — tuning is done once.
- On Colab, results are symlinked onto Drive (the launchers do this), so they
  survive runtime resets.

So the safe loop is always: **re-run the same command; it picks up.**

---

## Order of operations

### 0. One-time sanity (anywhere)
```bash
python tests/test_optimizer_hardening.py
python tests/test_cosgd_refactor.py
python tests/test_graddrop.py
python tests/test_deficit_extension.py
python tests/test_trainer.py          # trains a tiny CIFAR-10 + resume check
python tests/test_phase0_smoke.py     # all 5 methods x 2 bases, schema-valid
```

### 1. M1 — BoGrad ablation
```bash
cd PaperReadyExperiments/10_bograd_ablation
python run_all.py --smoke --axes 01 04            # shake-out (~2 min/axis)
python run_all.py --epochs 10 --seeds 2026 2027 2028   # full (Colab: hours)
python 09_cross_summary/run.py                    # master "when/what/why" table
# or per axis:  python 01_buffer_K/run.py ; python 01_buffer_K/plot.py
```
Colab: open `10_bograd_ablation/colab_launcher.ipynb`, run top to bottom.

### 2. M2 — COSGD ablation
```bash
cd PaperReadyExperiments/20_cosgd_ablation
python run_all.py --smoke --axes 01 05
python run_all.py --epochs 10 --seeds 2026 2027 2028
python 07_scalability/run.py                      # synthetic O(n^2) wall  (the BoGrad motivation)
python 07_scalability/run.py --real               # CIFAR-10 / EMNIST-47 / CIFAR-100
python 03_prenormalize/synthetic_dim_sweep.py     # thesis 3.1/3.2 (no GPU)
python 08_cross_summary/run.py                    # COSGD table + COSGD<->BoGrad contrast
```
Colab: `20_cosgd_ablation/colab_launcher.ipynb`.

### 3. M3 — main bakeoff  (uses M1/M2 winners; see `_bakeoff.hp_axes_for`)
```bash
cd PaperReadyExperiments/30_main_comparison
python run.py --smoke && python views.py          # prove the tune->run->view flow
# then ONE DATASET AT A TIME (Colab, resumable):
python run.py --datasets mnist    --seeds 2026 2027 2028 2029 2030
python run.py --datasets cifar10  --seeds 2026 2027 2028 2029 2030
python run.py --datasets emnist_balanced --seeds 2026 2027 2028 2029 2030
python run.py --datasets covertype --seeds 2026 2027 2028 2029 2030
python run.py --datasets yahoo_answers --seeds 2026 2027 2028 2029 2030
python run.py --datasets cifar100 --seeds 2026 2027 2028   # heaviest
# sensitivity / scale / grad-stats:
python sensitivity.py --kind lr    --datasets cifar10 cifar100
python sensitivity.py --kind K     --datasets cifar10 mnist emnist_balanced
python sensitivity.py --kind batch --datasets cifar10 emnist_balanced
python scale_gradstats.py --kind scale
python scale_gradstats.py --kind gradstats
python views.py                                   # all figures + tables
```
Colab: `30_main_comparison/colab_launcher.ipynb` (per-dataset cells).

### 4. M4 — synthesis (no training; after M3 records exist)
```bash
python PaperReadyExperiments/40_synthesis/analyze.py
```

---

## Budget reality

- **Per training step** (CIFAR-10 small CNN, with interference meter): ~14 ms
  train + ~1.6 ms amortised measurement (meter logs 1-in-50 steps). Add
  `--no-measure` in M3 for accuracy-only runs.
- **One ablation cell** (8–10 epoch proxy, CIFAR-10): a few minutes on a T4.
- **M1 full**: 9 axes × ~tens of cells × 4 bases × 3 seeds — a Colab day, chunked.
- **M3 full**: 6 ds × 4 opt × 5 method × 5 seed = 600 tuned runs + tuning — run
  per-dataset across several Colab sessions; CIFAR-100 dominates.
- COSGD cost grows with class count (the 20.07 finding); on CIFAR-100 it is the
  slowest cell by far — budget accordingly or run it last.

## Tuning the budget down

- Fewer seeds (`--seeds 2026 2027 2028`) for exploration; 5 for headline numbers.
- Shorter `--epochs` (proxy schedule) for ablations — they *rank* settings, they
  don't need full convergence.
- `--no-measure` (M3) when you only need accuracy/Pareto, not the interference
  series.
- `--bases sgd` to restrict an axis to the cleanest optimizer first.

---

## Outputs map

| Phase | Canonical data | Figures/tables from it |
|---|---|---|
| M1 | `10_*/<axis>/results/run_*/summary.json` | `<axis>/plot.py`; `09_cross_summary/master_table.json` |
| M2 | `20_*/<axis>/results/run_*/summary.json` | `08_cross_summary/{master_table,mechanism_contrast}.json` |
| M3 | `30_main_comparison/_core/results/run_*/all_rows.json` | `views.py` -> 30.01/02/03/09; `sensitivity.py`, `scale_gradstats.py` |
| M4 | `40_synthesis/synthesis_40_0*.json` | `40_synthesis/{findings.md, *.png}` |
```
