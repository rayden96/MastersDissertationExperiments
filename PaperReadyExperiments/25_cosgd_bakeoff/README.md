# 25 — COSGD optimiser bakeoff on the conference study's ladder

## Claim being tested

Section 4.4 of the dissertation predicts that COSGD's scope for acting shrinks
as input dimension and class count grow: the per-class subgradients approach
mutual orthogonality, so there is less conflict for the projection to remove.
The ablation of Section 4.6 tests the knobs on CIFAR-10 alone and finds parity
with a tuned baseline.

This study tests the other end of that prediction. If the geometry argument is
right, COSGD's advantage should be visible on low-dimensional, few-class
problems and should narrow as the ladder is climbed. A monotone narrowing is
the outcome that supports Section 4.4; an advantage that is absent everywhere,
or present everywhere, would not.

## Design

Five optimiser arms per dataset, each at its own tuned learning rate:

| arm | what it is |
|---|---|
| COSGD | per-class Gram-Schmidt wrapping SGD, canonical config |
| SGD | plain stochastic gradient descent |
| Adam | adaptive, first and second moment |
| RMSProp | adaptive, second moment only |
| SignSGD | sign of the gradient, constant step magnitude |

Datasets, architectures, batch sizes and per-optimiser learning rates follow
Tables I and II of the COSGD conference paper. Its rates came from a grid
search over {0.1, 0.01, 0.001} per optimiser, so every arm runs at a rate
chosen for itself rather than at a shared one. That matters here: a shared rate
is what invalidated the first two attempts at the Chapter 4 ablation.

| dataset | features / classes | batch | epochs | lr COSGD | lr SGD | lr other |
|---|---|---|---|---|---|---|
| Iris | 4 / 3 | 12 | 30 | 0.1 | 0.1 | 0.001 |
| Titanic | 13 / 2 | 8 | 15 | 0.01 | 0.01 | 0.001 |
| MNIST | 784 / 10 | 128 | 15 | 0.01 | 0.1 | 0.001 |
| Fashion-MNIST | 784 / 10 | 128 | 15 | 0.01 | 0.1 | 0.001 |
| CIFAR-10 | 3072 / 10 | 128 | 15 | 0.01 | 0.1 | 0.001 |

COSGD is held at the configuration the ablation fixed: classical Gram-Schmidt,
descending magnitude order, summed combine, no norm cap. Nothing is re-tuned
here, which is the point of running the ablation first.

## Deviations from the conference study

Both are stated in the chapter rather than absorbed silently.

- **Three seeds, not 10 to 30.** COSGD costs 8 to 10 backward passes per step,
  and the budget does not stretch to the paper's trial count. Spreads are
  correspondingly wider and are reported.
- **15 epochs on the image sets, not 7.** Ranking arms on epochs-to-target
  needs enough resolution to separate them; seven integer epochs does not give
  it. The tabular budgets are the paper's.
- **BloodMNIST is not included.** It needs the `medmnist` package and is the
  one dataset in the paper's set that adds no rung to the dimensional ladder
  the section reads along.

## Reporting

Consistent with the rest of the dissertation, arms are ranked on epochs to
reach a target, not on final accuracy. The target is 99% of the best final
accuracy any arm attains on that dataset, so all five are measured against one
common bar. An arm whose seeds never reach it is charged the budget plus one
epoch and marked censored, which is a statement that it did not arrive rather
than a speed.

## Running

```
python run.py                            # all five datasets, ~2 h on a T4
python run.py --datasets iris titanic    # the cheap end, ~1 min
python run.py --smoke
python plot.py                           # figure + table, reads results only
```

Resumable: `JobManager` skips completed cells, and the check compares the
recorded hyperparameters, so changing a rate re-runs rather than silently
reusing.

## Outputs

- `results/25_cosgd_bakeoff/<dataset>/run_<id>/cells/<base>__<arm>__seed<n>/`
- `chapters/cosgd/figures/cosgd_bakeoff.pdf` — one panel per dataset
- `bakeoff_table.json` — epochs-to-target and final accuracy per arm
