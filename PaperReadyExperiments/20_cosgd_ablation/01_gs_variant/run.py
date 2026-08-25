"""
20.01 — Gram-Schmidt variant.

Claim. The GS variant controls how COSGD removes per-class conflict:
  - classical vs modified GS: modified is more numerically stable in high dim;
  - normal vs negative: `negative` removes only destructive (anti-aligned)
    overlap (softer), `normal` removes all overlap.
Prior finding F14: `modified_gs_negative` cut the within-batch anti-aligned
class-pair fraction (IB_%neg) from ~70% to ~54% and flipped mean cosine from
-0.08 to +0.01 while improving accuracy.

Sweep. {gram_schmidt, modified_gs} x {normal, negative} + baseline, on CIFAR-10
+ SGD. Explained via I_inter (should rise) and inter_mean_cos (toward 0).

Run: python run.py [--smoke] [--bases sgd] [--seeds ...]
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

VARIANTS = ["gram_schmidt_normal", "gram_schmidt_negative",
            "modified_gs_normal", "modified_gs_negative"]


# Hold the reclaimed FOLD fixed (sum+cap+desc) so this axis isolates the GS
# variant only, not a confound with the combine rule.
RECLAIM = dict(class_order="desc", combine="sum", combine_norm_cap=0.0)


def build_cells(lr):
    cells = [{"label": "baseline", "method": "baseline", "hp": {"lr": lr}}]
    for v in VARIANTS:
        cells.append({"label": v, "method": "cosgd",
                      "hp": {**RECLAIM, "cosgd_method": v, "lr": lr}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "digits"])
    ap.add_argument("--lr", type=float, required=True,
                    help="the rate this method operates at; see axis 20.10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 5; args.datasets = ["iris"]
    for ds in args.datasets:
        run_cosgd_sweep(axis_name=f"20_01_gs_variant_{ds}", cells=build_cells(args.lr),
                        bases=args.bases, dataset=ds, seeds=args.seeds,
                        epochs=args.epochs, out_root=_HERE / "results" / ds,
                        train_subset=args.train_subset)


if __name__ == "__main__":
    main()
