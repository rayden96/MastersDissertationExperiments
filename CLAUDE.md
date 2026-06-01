# CLAUDE.md

This repo contains the experimental codebase for the dissertation *Batch Orthogonalised Gradient Descent for Single-Task Classification* (BOGrad & COSGD), reframed as a study of single-task gradient interference.

## Folder structure

```
PaperReadyExperiments/      paper-ready experiments — new work happens here
PreDiscovery/               all earlier discovery / exploratory work, archived
common/                     shared infrastructure: optimizers, models, datasets
data/                       datasets (gitignored)
docs/                       general design conventions
```

The dissertation itself lives in a **sibling folder** outside this repo (theoretical sections, paper-shaped writing). The repo holds only experimental code and results.

## What `PreDiscovery/` contains

Everything generated before the paper-ready phase. Each subfolder is a self-contained discovery campaign — not all of it is current. Useful for tracing how findings were reached but no new work should add to it.

Notable subfolders:

- `FocusedWork/` — the interference framework reformulation (definition, two types, measurement protocols for §03 / §04). These four markdown docs are the canonical conceptual definitions of the interference metrics that `PaperReadyExperiments/` is testing.
- `research/01_interference_framework/framework.md` — earlier long-form version of the framework (superseded by `FocusedWork/`).
- `HANDOFF.md`, `TESTS_OVERVIEW.md`, `thesis_experiment_plan.md` — running historical notes from earlier sessions.
- `discovery/`, `discoveryPhase2/`, `testing/`, `stage1/`, `MoGrad/`, `BoGradExperimentsLive/` — earlier method studies, ablations, sweeps.

## What `PaperReadyExperiments/` contains

The work that will end up in the dissertation. Different conventions from PreDiscovery: higher rigor on reproducibility, multi-trial reporting, and clean plots.

Read its own `README.md` for current state and conventions.

## Shared infrastructure

- `common/optimizers/` — production `BoGrad` and `COSGD` implementations.
- `common/diagnostics/` — older diagnostics; superseded for new work by the metrics module inside `PaperReadyExperiments/`.
- `docs/experiment_design.md` — general conventions (logging, checkpointing, plot regeneration).
- `docs/results_schema.md` — on-disk JSON schema for results.

## Non-negotiables (for `PaperReadyExperiments/`)

- **Training and plotting are separate.** Training writes data; plotting reads it. Never plot from in-memory state during a training run.
- **Every run writes to `PaperReadyExperiments/<experiment>/results/run_<id>/`.** Nothing critical lives only in memory.
- **Multi-trial by default.** Paper-ready experiments report mean ± std across at least 3 seeds unless explicitly justified.
- **Shared code goes in `common/`** or in a sibling utility module under `PaperReadyExperiments/`. Do not paste helpers between experiments.

## When adding a new experiment to `PaperReadyExperiments/`

1. Decide which conceptual claim it supports (interference definition, metric behavior, method effect, etc.).
2. Create a subfolder `PaperReadyExperiments/NN_name/` with at minimum:
   - `README.md` — claim being tested, design, expected outcome.
   - `run.py` — produces JSON results.
   - `plot.py` — reads results, produces figures.
3. Reuse the interference metrics module rather than re-implementing measurements.

If a request conflicts with what's in `PreDiscovery/FocusedWork/`, surface the conflict before silently diverging — those four docs are the working definitions.
