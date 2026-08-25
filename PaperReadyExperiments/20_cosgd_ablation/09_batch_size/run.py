"""
20.09 — Batch size.

Not a design axis of the method: a training hyperparameter that the geometry
COSGD acts on depends on directly. The number of samples per class in a batch
is N/C, so the batch size sets how well each per-class subgradient is estimated
and therefore how much of the measured within-batch conflict is real structure
rather than sampling noise. Small batches give noisy per-class subgradients
whose pairwise angles are dominated by that noise; large batches give cleaner
ones and fewer steps per epoch.

Sweep. batch_size in {32, 64, 128, 256, 512}, with a matched baseline at every
batch size so the effect can be attributed to the method rather than to the
batch size itself. Learning rate held fixed across the sweep: rescaling it with
the batch size would confound the two.

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

# Canonical configuration; only the batch size moves.
CANONICAL = dict(cosgd_method="gram_schmidt_normal", class_order="desc",
                 combine="sum", combine_norm_cap=0.0)


def build_cells(batch_sizes, lr):
    cells = []
    for bs in batch_sizes:
        cells.append({"label": f"bs{bs}_baseline", "method": "baseline",
                      "hp": {"lr": lr, "batch_size": bs}})
        cells.append({"label": f"bs{bs}_cosgd", "method": "cosgd",
                      "hp": {**CANONICAL, "lr": lr, "batch_size": bs}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["covertype", "cifar10"])
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [2026]; args.epochs = 2; args.batch_sizes = [128, 256]
        args.datasets = ["covertype"]; args.train_subset = 5000

    for ds in args.datasets:
        run_cosgd_sweep(axis_name=f"20_09_batch_size_{ds}",
                        cells=build_cells(args.batch_sizes, args.lr),
                        bases=args.bases, dataset=ds, seeds=args.seeds,
                        epochs=args.epochs, out_root=_HERE / "results" / ds,
                        train_subset=args.train_subset)


if __name__ == "__main__":
    main()
