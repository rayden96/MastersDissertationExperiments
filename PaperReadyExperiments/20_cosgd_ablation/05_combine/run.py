"""
20.05 — Combine rule + norm cap (the central scrutiny axis).

Scrutiny finding (PreDiscovery/research/03_cosgd_scrutiny/FINDINGS.md): COSGD's
acceleration comes from `combine="sum"` — summing the orthogonalised per-class
gradients yields a bigger, well-directed step. `mean`/`freq` shrink that step back
toward a normal averaged update and remove the speed-up. The only weakness of
`sum` is that it over-scales and diverges at higher dim (digits, dim 64); the
`combine_norm_cap` fixes exactly that (caps ||combined|| at cap*mean-per-class-
norm), preserving the low-dim win and rescuing high-dim.

This axis maps it directly across the dimensionality ladder:
  sum (paper)  |  sum+cap{2,3,4} (reclaimed)  |  mean  |  freq
vs SGD baseline, at a FIXED lr so the scale effect is visible (not hidden by
per-cell LR tuning). Expected: sum/sum+cap win on low-dim; raw sum breaks on
digits while sum+cap holds; mean/freq sit near baseline everywhere.

Run:
    python run.py --datasets iris wine breast_cancer digits   # the ladder
    python run.py --datasets mnist cifar10 --epochs 10        # image check
    python run.py --smoke
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

PAPER = dict(cosgd_method="gram_schmidt_normal", class_order="desc")


def build_cells(lr):
    return [
        {"label": "baseline", "method": "baseline", "hp": {"lr": lr}},
        {"label": "sum(paper)", "method": "cosgd", "hp": {**PAPER, "combine": "sum", "lr": lr}},
        {"label": "sum+cap2", "method": "cosgd", "hp": {**PAPER, "combine": "sum", "combine_norm_cap": 2.0, "lr": lr}},
        {"label": "sum+cap3", "method": "cosgd", "hp": {**PAPER, "combine": "sum", "combine_norm_cap": 3.0, "lr": lr}},
        {"label": "mean", "method": "cosgd", "hp": {**PAPER, "combine": "mean", "lr": lr}},
        {"label": "freq", "method": "cosgd", "hp": {**PAPER, "combine": "freq", "lr": lr}},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "digits"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 5; args.datasets = ["iris", "digits"]

    for ds in args.datasets:
        run_cosgd_sweep(axis_name=f"20_05_combine_{ds}", cells=build_cells(args.lr),
                        bases=args.bases, dataset=ds, seeds=args.seeds,
                        epochs=args.epochs, out_root=_HERE / "results" / ds,
                        train_subset=args.train_subset)


if __name__ == "__main__":
    main()
