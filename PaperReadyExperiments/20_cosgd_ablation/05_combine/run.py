"""
20.05 — Combine rule.

Claim. After orthogonalising the per-class subgradients, COSGD folds them back
into one update. The rule sets the effective step scale:
  - sum:  add the C orthogonalised vectors -> effective step ~ C x a single
          class's contribution; destabilises an untuned LR as C grows (seen in
          early smoke: train_loss blew up). This is the legacy default.
  - mean: average -> scale-stable across C.
  - freq: class-frequency weighted -> reproduces the FocusedWork S02 batch-
          gradient definition (sum_c (n_c/|B|) g_c) exactly. The principled one.

Sweep. combine in {sum, mean, freq} + baseline, on CIFAR-10 + SGD, at a FIXED lr
so the scale effect is visible (not hidden by per-cell LR tuning). Explained via
effective step norm (u_norm) and accuracy: sum should need a lower LR than
mean/freq for the same stability.

Run: python run.py [--smoke] [--lr 0.05]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AXIS = _HERE.parent
if str(_AXIS) not in sys.path:
    sys.path.insert(0, str(_AXIS))

from _ablation import run_cosgd_sweep  # noqa: E402


def build_cells(lr):
    cells = [{"label": "baseline", "method": "baseline", "hp": {"lr": lr}}]
    for combine in ("sum", "mean", "freq"):
        cells.append({"label": f"combine_{combine}", "method": "cosgd",
                      "hp": {"cosgd_method": "modified_gs_negative",
                             "combine": combine, "lr": lr}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 1; args.train_subset = 2000
    run_cosgd_sweep(axis_name="20_05_combine", cells=build_cells(args.lr),
                    bases=args.bases, dataset=args.dataset, seeds=args.seeds,
                    epochs=args.epochs, out_root=_HERE / "results",
                    train_subset=args.train_subset)


if __name__ == "__main__":
    main()
