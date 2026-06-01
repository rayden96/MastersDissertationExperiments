"""
20.03 — Pre-normalisation before Gram-Schmidt.

Claim. In high dimension, raw-magnitude vectors are not naturally near-orthogonal
(only *unit* vectors are; the cosine concentration result is about normalised
vectors). Unit-normalising the per-class subgradients before GS therefore changes
what gets removed and how much each vector is perturbed (thesis 3.1/3.2): without
normalisation a few high-magnitude classes dominate the orthogonalisation.

Sweep. prenormalize {off, on} x {modified_gs_negative, modified_gs_normal} +
baseline, on CIFAR-10 + SGD. Explained via I_inter and the per-class
useful/wasted decomposition.

(The synthetic cosine-vs-dim and L2-change-vs-dim curves that motivate this — old
thesis Exp 3.1/3.2 — are produced by synthetic_dim_sweep.py in this folder, which
needs no training.)

Run: python run.py [--smoke]
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


def build_cells():
    cells = [{"label": "baseline", "method": "baseline", "hp": {}}]
    for gs in ("modified_gs_negative", "modified_gs_normal"):
        for pn in (False, True):
            cells.append({"label": f"{gs}_prenorm{int(pn)}", "method": "cosgd",
                          "hp": {"cosgd_method": gs, "prenormalize": pn, "combine": "mean"}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd"]; args.seeds = [2026]; args.epochs = 1; args.train_subset = 2000
    run_cosgd_sweep(axis_name="20_03_prenormalize", cells=build_cells(),
                    bases=args.bases, dataset=args.dataset, seeds=args.seeds,
                    epochs=args.epochs, out_root=_HERE / "results",
                    train_subset=args.train_subset)


if __name__ == "__main__":
    main()
