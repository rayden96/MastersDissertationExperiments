# MastersDissertationExperiments

Experiments for the dissertation *Batch Orthogonalised Gradient Descent for Single-Task Classification* (BOGrad & COSGD).

## Start here

- **[TESTS_OVERVIEW.md](TESTS_OVERVIEW.md)** — single-page navigator across all experiments and findings (F1–F7, D1–D4). **Read this first if you're trying to remember what's been done.**
- [research/01_interference_framework/framework.md](research/01_interference_framework/framework.md) — formal write-up of the interference framework (the current research focus).
- [thesis_experiment_plan.md](thesis_experiment_plan.md) — the full list of thesis-chapter experiments and the plots each produces.
- [docs/experiment_design.md](docs/experiment_design.md) — conventions every experiment follows (logging, saving, checkpointing, plotting).
- [docs/results_schema.md](docs/results_schema.md) — on-disk JSON shapes.
- [CLAUDE.md](CLAUDE.md) — instructions for AI assistants working in this repo.

## Layout

```
common/                     shared python (optimizers, diagnostics)
research/                   stream-level investigations (interference framework, BoGrad / COSGD scrutiny, implicit-method comparisons)
testing/                    tactical tests for specific BoGrad-implementation questions
discoveryPhase2/            historical scratch sweeps (reference only — do not extend)
experiments/chN/NN_name/    final thesis-chapter experiments (queued)
docs/                       design docs and conventions
```

## Running

Notebooks are designed to run in either local Jupyter or Google Colab. `common/storage.get_results_root()` transparently returns a Drive-mounted path in Colab and `./results` locally — no per-environment branching in notebooks.
