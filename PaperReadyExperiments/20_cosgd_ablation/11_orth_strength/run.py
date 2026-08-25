"""
20.11 — Orthogonalisation strength.

COSGD's combined step is built from fully orthogonalised per-class
subgradients. `orth_strength` interpolates between that and the plain batch
gradient: at 1.0 the combined step is the fully orthogonalised one, at 0.0 it
is the raw average, and in between it is a convex blend. This is the exact
analogue of BOGrad's projection strength, where a partial projection at 0.75
separated from the baseline sooner than complete removal in most panels
(Section 5.6.3), and it has never been tested for COSGD.

The motivation is the same in both cases. The per-class conflict estimate from
a single batch is noisy, and full orthogonalisation acts on all of it; shrinking
the correction trades a little of the intended effect for a lot less variance.
For COSGD there is a second reason: a partial blend also shortens the combined
step, which is the variable every other axis in Chapter 4 turns out to act on.

Sweep. orth_strength in {0.25, 0.5, 0.75, 1.0} against the baseline, at the
learning rate the method actually operates at.

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


def _tag(a: float) -> str:
    return ("%g" % a).replace(".", "p")


def build_cells(strengths, lr):
    cells = [{"label": "baseline", "method": "baseline", "hp": {"lr": lr}}]
    for a in strengths:
        cells.append({"label": f"orth{_tag(a)}", "method": "cosgd",
                      "hp": {**CANONICAL, "orth_strength": a, "lr": lr}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["covertype", "cifar10"])
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--lr", type=float, required=True,
                    help="the rate this method operates at; see axis 20.10")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train_subset", type=int, default=None)
    args = ap.parse_args()

    for ds in args.datasets:
        run_cosgd_sweep(axis_name=f"20_11_orth_strength_{ds}",
                        cells=build_cells(args.strengths, args.lr),
                        bases=args.bases, dataset=ds, seeds=args.seeds,
                        epochs=args.epochs, out_root=_HERE / "results" / ds,
                        train_subset=args.train_subset)


if __name__ == "__main__":
    main()
