# 40 — Synthesis (the dissertation payoff)

Pure analysis over the M1/M2/M3 records — trains nothing. Answers the two
framework-level questions the whole program exists to settle.

## 40.01 — Does interference reduction *predict* the accuracy gain?

For every (dataset × base optimizer), compare each orthogonalising method to its
baseline and regress

  Δ(test accuracy) on Δ(the interference index that method targets)

across the whole suite — `I_inter` for COSGD/GradDrop (inter-batch), `I_between_K`
for BoGrad (between-batch). A **positive, significant slope** means the framework's
metric is *predictive* of training benefit, not merely descriptive — the central
empirical claim that the interference framework is the right lens.

## 40.02 — W′ at scale

Tests the refined hypothesis from PreDiscovery (W′): **orthogonalisation helps iff
the direction it removes is separable from descent.** Relates each method's
Δaccuracy to the baseline conflict cosine it faced — cells with anti-aligned /
mixed-sign conflict (separable) should show gains; cells where conflict is aligned
with descent (inseparable) should show flat or negative gains. Confirms whether
the prior synthetic-only finding holds across real datasets, optimizers, and
architectures.

## How to run

```bash
# after the M3 bakeoff has produced records:
python PaperReadyExperiments/40_synthesis/analyze.py
```

Auto-discovers the newest `30_main_comparison/_core/results/run_*` campaign (or
pass `--campaign`). If no records exist yet it writes a `no_data` stub and tells
you to run M3 first. Outputs `synthesis_40_01.{json,png}`,
`synthesis_40_02.{json,png}`, and `findings.md`.
