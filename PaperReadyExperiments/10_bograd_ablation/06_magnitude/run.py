"""
10.06 — Magnitude controls (is it direction or magnitude?).

Claim. BoGrad's benefit is attributable to the *direction* change, not the
incidental step-magnitude reduction (prior discovery D1). Two controls test this:
  - preserve_magnitude: rescale g_tilde back to ||g|| after projection -> isolates
    the pure direction effect (if accuracy holds, magnitude was inert);
  - random_projection: project against random unit vectors instead of the buffer
    -> a magnitude-matched control that carries NO temporal information; it MUST
    fail to match real BoGrad if the specific projection direction is what matters.

Sweep. baseline; bograd(default); bograd+preserve_magnitude; bograd+random_
projection -> a 2x2 of {real vs random buffer} x {magnitude free vs preserved},
x 4 bases, fixed mid-K. Workhorse.

Explained via. g_ratio = ||g_tilde||/||g|| (how much magnitude each variant
removes) and the accuracy gap vs the random-projection control.

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
    base_hp = {"K": K, "projection_mode": "negative"}
    return [
        {"label": "baseline", "method": "baseline", "hp": {}},
        {"label": "bograd", "method": "bograd", "hp": dict(base_hp)},
        {"label": "bograd_preserve_mag", "method": "bograd",
         "hp": {**base_hp, "preserve_magnitude": True, "max_rescale": 5.0}},
        {"label": "random_proj", "method": "bograd",
         "hp": {**base_hp, "random_projection": True}},
        {"label": "random_proj_preserve_mag", "method": "bograd",
         "hp": {**base_hp, "random_projection": True, "preserve_magnitude": True, "max_rescale": 5.0}},
    ]


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
            axis_name=f"10_06_magnitude_{base}",
            cells=build_cells(K), bases=[base], dataset=args.dataset,
            seeds=args.seeds, epochs=args.epochs,
            out_root=_HERE / "results" / base, train_subset=args.train_subset,
        )


if __name__ == "__main__":
    main()
