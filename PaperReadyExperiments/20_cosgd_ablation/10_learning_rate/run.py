"""
20.10 — Learning rate.

The axis that the chapter's scale analysis makes unavoidable. Summing C
orthogonalised subgradients produces a step whose length exceeds the averaged
baseline step by a factor that grows with the class count, bounded but not
removed by the norm cap. Every other axis in this chapter holds the learning
rate fixed at the baseline's value, which is the right choice for isolating a
knob but leaves one question open: whether COSGD is worse than the baseline, or
merely operating at an effective step size the baseline's learning rate was
never chosen for.

Sweep. lr over a decade and a half on a log grid, with a matched baseline at
every value, so the two arms can be compared at their own best settings rather
than at a setting chosen for one of them. If the scale reading is right, COSGD's
accuracy peak sits at a learning rate several times smaller than the baseline's,
and the height of that peak is the method's honest score.

Run: python run.py --datasets covertype cifar10
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
                 combine="sum", combine_norm_cap=2.0)


def _tag(lr: float) -> str:
    return ("%g" % lr).replace(".", "p").replace("-", "m")


def build_cells(lrs):
    cells = []
    for lr in lrs:
        cells.append({"label": f"lr{_tag(lr)}_baseline", "method": "baseline",
                      "hp": {"lr": lr}})
        cells.append({"label": f"lr{_tag(lr)}_cosgd", "method": "cosgd",
                      "hp": {**CANONICAL, "lr": lr}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["covertype", "cifar10"])
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [2026]; args.epochs = 2; args.lrs = [0.01, 0.1]
        args.datasets = ["covertype"]; args.train_subset = 5000

    for ds in args.datasets:
        run_cosgd_sweep(axis_name=f"20_10_learning_rate_{ds}",
                        cells=build_cells(args.lrs),
                        bases=args.bases, dataset=ds, seeds=args.seeds,
                        epochs=args.epochs, out_root=_HERE / "results" / ds,
                        train_subset=args.train_subset)


if __name__ == "__main__":
    main()
