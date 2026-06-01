"""
10.03 — Projection mode + strength.

Claim. `projection_mode` decides *which* overlap with the buffer is removed:
  - negative: only destructive overlap (the safe default);
  - full:     all overlap (destructive + redundant);
  - positive: only redundant overlap.
Prior findings F9/F10: `full`/`positive` collapse training when buffered
directions are highly aligned with descent (removing them strips the descent
signal). `projection_strength` alpha in [0,1] interpolates baseline<->full so we
see the continuum, not just the discrete modes.

Sweep. {negative, full, positive} at alpha=1, plus a negative-mode alpha ladder
{0.25,0.5,0.75,1.0}, x 4 bases, fixed mid-K. Workhorse (+ CIFAR-100 transfer).

Explained via. the pairwise-alignment fingerprint (frac of buffer pairs that are
+/-) and cos(g, g_tilde): modes help iff interference is separable from descent.

Run: python run.py [--smoke] [--bases ...] [--K 32]
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

DEFAULT_K = {"sgd": 32, "signsgd": 64, "rmsprop": 16, "adam": 128}


def build_cells(K):
    cells = [{"label": "baseline", "method": "baseline", "hp": {}}]
    for mode in ("negative", "full", "positive"):
        cells.append({"label": f"mode_{mode}", "method": "bograd",
                      "hp": {"K": K, "projection_mode": mode}})
    for a in (0.25, 0.5, 0.75):  # 1.0 == mode_negative above
        cells.append({"label": f"neg_alpha{a:g}", "method": "bograd",
                      "hp": {"K": K, "projection_mode": "negative", "projection_strength": a}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd", "signsgd", "rmsprop", "adam"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 1; args.train_subset = 2000

    for base in args.bases:
        K = args.K if args.K is not None else DEFAULT_K[base]
        run_bograd_sweep(
            axis_name=f"10_03_projection_mode_{base}",
            cells=build_cells(K), bases=[base], dataset=args.dataset,
            seeds=args.seeds, epochs=args.epochs,
            out_root=_HERE / "results" / base, train_subset=args.train_subset,
        )


if __name__ == "__main__":
    main()
