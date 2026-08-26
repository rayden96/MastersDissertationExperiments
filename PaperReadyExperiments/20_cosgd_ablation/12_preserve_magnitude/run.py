"""
20.12 — Magnitude preservation.

Summing C mutually orthogonal per-class subgradients produces a step longer
than the averaged baseline step by a factor that grows with the class count.
`preserve_magnitude` rescales the combined step back to the length of the raw
batch gradient, clipped at `max_rescale`, which removes that enlargement while
leaving the direction the orthogonalisation produced untouched.

That makes it the sharpest available test of the chapter's central claim. If
COSGD's behaviour is governed by step length, restoring the baseline's length
should recover the baseline's behaviour and the orthogonalisation should add
nothing on top. If instead the orthogonalised direction is worth something in
its own right, the rescaled arm should beat the baseline. The equivalent
control on the between-batch axis (Section 5.6.6) found the direction, not the
length, was carrying the benefit.

Sweep. preserve_magnitude {off, on} against the baseline, at the learning rate
the method actually operates at.

Run: python run.py --datasets covertype cifar10 --lr 0.005
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

CANONICAL = dict(cosgd_method="gram_schmidt_normal", class_order="desc",
                 combine="sum", combine_norm_cap=0.0)


def build_cells(lr, max_rescale, lr_baseline=None):
    return [
        {"label": "baseline", "method": "baseline",
         "hp": {"lr": lr_baseline if lr_baseline is not None else lr}},
        {"label": "preserve0", "method": "cosgd",
         "hp": {**CANONICAL, "preserve_magnitude": False, "lr": lr}},
        {"label": "preserve1", "method": "cosgd",
         "hp": {**CANONICAL, "preserve_magnitude": True,
                "max_rescale": max_rescale, "lr": lr}},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["covertype", "cifar10"])
    ap.add_argument("--lr", type=float, required=True,
                    help="the rate this method operates at; see axis 20.10")
    ap.add_argument("--lr_baseline", type=float, default=None,
                    help="rate for the baseline arm; defaults to --lr. The two "
                         "arms do not share an optimum, so a single rate "
                         "handicaps whichever arm did not choose it")
    ap.add_argument("--max_rescale", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train_subset", type=int, default=None)
    args = ap.parse_args()

    for ds in args.datasets:
        run_cosgd_sweep(axis_name=f"20_12_preserve_magnitude_{ds}",
                        cells=build_cells(args.lr, args.max_rescale, args.lr_baseline),
                        bases=args.bases, dataset=ds, seeds=args.seeds,
                        epochs=args.epochs, out_root=_HERE / "results" / ds,
                        train_subset=args.train_subset)


if __name__ == "__main__":
    main()
