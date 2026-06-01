"""
10.08 — Batch size x K interaction.

Claim. Smaller batches have noisier, more temporally-variable gradients, so they
have more between-batch cancellation for BoGrad to remove — BoGrad's benefit
(and the best K) should grow as batch size shrinks. Large batches already average
out much of the noise, leaving less for projection to help with.

Sweep. batch in {32,64,128,256,512} x K in {0(baseline),2,8,32}. The result is a
(batch x K) grid of final accuracy -> heatmap; the headline is where in the grid
BoGrad's gain over K=0 is largest (expected: small batch, moderate K).

Explained via. I_between_K as a function of batch size (should fall as batch
grows), tying the accuracy heatmap to the trajectory-cancellation mechanism.

Workhorse CIFAR-10 (+ EMNIST transfer). Default base SGD (the cleanest momentum
story); add --bases for more.

Run: python run.py [--smoke] [--bases sgd] [--batches 32 128 512] [--K 0 8 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AXIS = _HERE.parent
if str(_AXIS) not in sys.path:
    sys.path.insert(0, str(_AXIS))

from _ablation import run_bograd_sweep  # noqa: E402

BATCHES = [32, 64, 128, 256, 512]
K_GRID = [0, 2, 8, 32]


def build_cells(K_grid):
    cells = []
    for K in K_grid:
        if K == 0:
            cells.append({"label": "baseline_K0", "method": "baseline", "hp": {}})
        else:
            cells.append({"label": f"K{K}", "method": "bograd",
                          "hp": {"K": K, "projection_mode": "negative"}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    ap.add_argument("--K", type=int, nargs="+", default=K_GRID)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 1
        args.train_subset = 2000; args.batches = [64, 256]; args.K = [0, 8]

    # batch_size is a Trainer-level arg -> one harness call per batch size,
    # tagged into the axis name so each (batch) grid lands in its own results dir.
    for bs in args.batches:
        run_bograd_sweep(
            axis_name=f"10_08_batch_K_bs{bs}",
            cells=build_cells(args.K), bases=args.bases, dataset=args.dataset,
            seeds=args.seeds, epochs=args.epochs, batch_size=bs,
            out_root=_HERE / "results" / f"bs{bs}", train_subset=args.train_subset,
        )


if __name__ == "__main__":
    main()
