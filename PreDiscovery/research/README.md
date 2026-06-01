# Research streams

Stream-level investigations supporting the dissertation. Each subfolder
captures the work for one research stream defined in the plan
([on-the-questions-1-tidy-goblet.md](../../../Users/rayde/.claude/plans/on-the-questions-1-tidy-goblet.md)).

Distinct from `testing/`, which holds tactical tests, this folder holds
foundational research artifacts that may not all appear in the thesis but
inform the methodology.

## Streams

| # | Folder | Goal | Status |
|---|---|---|---|
| 01 | [01_interference_framework/](01_interference_framework/) | Build a rigorous, quantitative framework for measuring interference | In progress (framework drafted; validation pending) |
| 02 | [02_bograd_scrutiny/](02_bograd_scrutiny/) | Theoretical and empirical scrutiny of BoGrad | Not started |
| 03 | [03_cosgd_scrutiny/](03_cosgd_scrutiny/) | Same depth for COSGD | Not started |
| 04 | [04_implicit_comparisons/](04_implicit_comparisons/) | Compare BoGrad/COSGD vs dropout, GradDrop, Lookahead, etc. | Not started |

## Where things live

- **Framework definitions** → `<stream>/framework.md` or `<stream>/theory.md`.
- **Code** → may live in `common/` (if re-usable) or in the stream folder
  (if one-off).
- **Run scripts** → `<stream>/<experiment>.py`.
- **Results** → `<stream>/results/<run_id>/`.

## Key infrastructure shared across streams

- [`common/diagnostics/`](../common/diagnostics/) — InterferenceTracker,
  ClassProbeSet, ForgettingTracker, WastedWorkTracker.
- [`common/optimizers/`](../common/optimizers/) — BoGrad, plus eventually
  COSGD, Complement* variants etc.
- [`testing/_common.py`](../testing/_common.py) — SmallCNN, ResNet8,
  ResNet18CIFAR, train_run, evaluate. Reuse for non-research-specific glue.

## Findings index

A running record so we don't re-litigate (mirrors §9 of the plan):

- Sequential GS subtraction > QR-based orthogonal projection.
- Magnitude reduction in BoGrad is inert; direction modification is the active
  ingredient. (testing/01_attribution settled this.)
- BoGrad's optimal LR is lower than baseline's, even with magnitude preserved.
- Optimal K varies per optimiser: SGD+mom: 32, Adam: ≥128, RMSprop: ~16,
  SignSGD: ~64.
- BoGrad fights momentum at gradient stage; update-stage works.
- Same-LR comparisons systematically inflate BoGrad's reported gain.
