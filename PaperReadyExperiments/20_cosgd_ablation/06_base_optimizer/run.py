"""
20.06 — Base optimizer x COSGD.

Claim. COSGD is now a wrapper: it orthogonalises the per-class subgradients then
hands the combined vector to a base optimizer. Does per-class orthogonalisation
still help once a preconditioner (RMSprop/Adam) or momentum is in play, or does
the base optimizer already absorb the within-batch conflict? This axis is the
COSGD analogue of BoGrad's 10.07/10.09 cross-optimizer view.

Sweep. For each base in {sgd, signsgd, rmsprop, adam}: baseline vs COSGD
(modified_gs_negative, combine=mean). Explained via I_inter (does COSGD still
raise it under each base?) and the per-optimizer deficit D_t_precond (does the
reduced cancellation translate to less first-order training-hurt?).

Run: python run.py [--smoke] [--bases sgd adam]
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
    # Reclaimed COSGD (full GS + sum + cap + desc) wrapped on each base optimizer.
    return [
        {"label": "baseline", "method": "baseline", "hp": {}},
        {"label": "cosgd", "method": "cosgd",
         "hp": {"cosgd_method": "gram_schmidt_normal", "class_order": "desc",
                "combine": "sum", "combine_norm_cap": 2.0}},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd", "signsgd", "rmsprop", "adam"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["wine", "digits"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd", "adam"]; args.seeds = [2026]; args.epochs = 5; args.datasets = ["wine"]
    # one harness call per (dataset, base) so each base's baseline+cosgd land together
    for ds in args.datasets:
        for base in args.bases:
            run_cosgd_sweep(axis_name=f"20_06_base_optimizer_{ds}_{base}", cells=build_cells(),
                            bases=[base], dataset=ds, seeds=args.seeds,
                            epochs=args.epochs, out_root=_HERE / "results" / ds / base,
                            train_subset=args.train_subset)


if __name__ == "__main__":
    main()
