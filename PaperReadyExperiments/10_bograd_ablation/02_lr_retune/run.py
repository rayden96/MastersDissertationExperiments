"""
10.02 — Learning-rate retune per (optimizer, K).

Claim. BoGrad's optimal LR differs from the baseline's — prior discovery D2:
projecting out recent-direction overlap shrinks the effective step, so the best
LR is typically *lower* than the same optimizer's baseline best. Every later
BoGrad number must therefore be taken at its OWN best LR, not the baseline's, or
the comparison is confounded (same-LR comparisons inflate/deflate the gap).

Sweep. For each base optimizer: baseline and BoGrad(best-K from 10.01) over a log
LR grid. The output is the LR-vs-accuracy basin per (optimizer, method); the
headline is the shift in the peak between baseline and BoGrad.

LR grids are per-family (SGD/SignSGD higher, RMSprop/Adam lower). K defaults to a
mid value per optimizer (override with --K after 10.01 lands its winners).

Run:
    python run.py
    python run.py --smoke
    python run.py --bases sgd --K 32
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

# Per-family LR grids (log-spaced). Tune these as 10.01/early runs inform.
LR_GRID = {
    "sgd":     [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    "signsgd": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "rmsprop": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "adam":    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}
# Default best-K per optimizer (prior D3; refine from 10.01).
DEFAULT_K = {"sgd": 32, "signsgd": 64, "rmsprop": 16, "adam": 128}


def build_cells_for_base(base, K):
    cells = []
    for lr in LR_GRID[base]:
        cells.append({"label": f"baseline_lr{lr:g}", "method": "baseline", "hp": {"lr": lr}})
        cells.append({"label": f"bograd_K{K}_lr{lr:g}", "method": "bograd",
                      "hp": {"lr": lr, "K": K, "projection_mode": "negative"}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd", "signsgd", "rmsprop", "adam"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--K", type=int, default=None, help="override best-K (else per-optimizer default)")
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 1
        args.train_subset = 2000
        global LR_GRID
        LR_GRID = {"sgd": [0.05, 0.1]}

    # Each base has its own LR grid + K, so run the harness per base.
    for base in args.bases:
        K = args.K if args.K is not None else DEFAULT_K[base]
        run_bograd_sweep(
            axis_name=f"10_02_lr_retune_{base}",
            cells=build_cells_for_base(base, K),
            bases=[base],
            dataset=args.dataset,
            seeds=args.seeds,
            epochs=args.epochs,
            out_root=_HERE / "results" / base,
            train_subset=args.train_subset,
        )


if __name__ == "__main__":
    main()
