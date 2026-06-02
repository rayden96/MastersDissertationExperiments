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


# Reclaimed FOLD held fixed (sum+cap, full classical GS); only class_order varies.
RECLAIM = dict(cosgd_method="gram_schmidt_normal", combine="sum", combine_norm_cap=2.0)


def build_cells():
    cells = [{"label": "baseline", "method": "baseline", "hp": {}}]
    for order in ("desc", "asc", "random", "fixed"):
        cells.append({"label": f"order_{order}", "method": "cosgd",
                      "hp": {**RECLAIM, "class_order": order}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "digits"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 5; args.datasets = ["iris"]
    for ds in args.datasets:
        run_cosgd_sweep(axis_name=f"20_02_class_order_{ds}", cells=build_cells(),
                        bases=args.bases, dataset=ds, seeds=args.seeds,
                        epochs=args.epochs, out_root=_HERE / "results" / ds,
                        train_subset=args.train_subset)


if __name__ == "__main__":
    main()
