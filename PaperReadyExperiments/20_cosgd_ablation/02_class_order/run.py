"""
20.02 — Magnitude (class) ordering.

Claim. Gram-Schmidt is order-dependent: the first vector is preserved exactly,
later vectors lose their projections onto earlier ones. Processing per-class
subgradients in DESCENDING magnitude order should therefore preserve the
high-magnitude (confident, well-represented) classes and orthogonalise the
smaller ones against them (thesis 3.3). Tests whether descending order beats
ascending / random / fixed, or whether it's marginal.

Sweep. class_order in {desc, asc, random, fixed} + baseline, on MNIST / CIFAR-10
/ EMNIST-Balanced (run --dataset per set). modified_gs_negative, combine=mean.
Explained via per-class useful mass (inter_useful_mass) and I_inter.

Run: python run.py [--smoke] [--dataset cifar10|mnist|emnist_balanced]
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


def build_cells():
    cells = [{"label": "baseline", "method": "baseline", "hp": {}}]
    for order in ("desc", "asc", "random", "fixed"):
        cells.append({"label": f"order_{order}", "method": "cosgd",
                      "hp": {"cosgd_method": "modified_gs_negative",
                             "class_order": order, "combine": "mean"}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 1; args.train_subset = 2000
    run_cosgd_sweep(axis_name=f"20_02_class_order_{args.dataset}", cells=build_cells(),
                    bases=args.bases, dataset=args.dataset, seeds=args.seeds,
                    epochs=args.epochs, out_root=_HERE / "results" / args.dataset,
                    train_subset=args.train_subset)


if __name__ == "__main__":
    main()
