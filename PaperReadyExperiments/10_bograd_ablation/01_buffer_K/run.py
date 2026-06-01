"""
10.01 — Buffer size K.

Claim. BoGrad's buffer size K trades off how much of the recent-direction
subspace is removed. Because recent updates are correlated, small K already
captures most destructive overlap; large K over-strips useful descent signal.
The optimum is therefore finite and **per base optimizer** (prior discovery D3:
SGD+momentum ~32, RMSprop ~16, Adam >=128, SignSGD+momentum ~64).

Sweep. K in {0,1,2,4,8,16,32,64,128} (K=0 == baseline) x {sgd, signsgd, rmsprop,
adam}. project_stage="update", projection_mode="negative" (the defaults the rest
of the chapter justifies). Workhorse: CIFAR-10 + small CNN.

Explained via. between-batch cancellation index I_between_K (should rise then
saturate / over-correct as K grows), cos(u_t, u_{t-1}), and the per-step deficit
D_t / D_t_precond. The K that maximises test accuracy should coincide with the K
that best reduces between-batch cancellation without inflating the deficit.

Run:
    python run.py                       # full sweep (4 bases x 9 K x 3 seeds)
    python run.py --smoke               # sgd only, tiny subset, 1 seed
    python run.py --bases sgd adam --seeds 2026
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

K_GRID = [0, 1, 2, 4, 8, 16, 32, 64, 128]


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
    ap.add_argument("--bases", nargs="+",
                    default=["sgd", "signsgd", "rmsprop", "adam"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--K", type=int, nargs="+", default=K_GRID)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="sgd only, 2k-sample subset, 1 epoch, 1 seed, K in {0,8,32}")
    args = ap.parse_args()

    if args.smoke:
        args.bases = ["sgd"]
        args.seeds = [2026]
        args.epochs = 1
        args.train_subset = 2000
        args.K = [0, 8, 32]

    run_bograd_sweep(
        axis_name="10_01_buffer_K",
        cells=build_cells(args.K),
        bases=args.bases,
        dataset=args.dataset,
        seeds=args.seeds,
        epochs=args.epochs,
        out_root=_HERE / "results",
        train_subset=args.train_subset,
    )


if __name__ == "__main__":
    main()
