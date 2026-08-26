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


def build_cells(lr=None, lr_baseline=None):
    """Canonical COSGD (full GS, descending order, sum, no cap) on one base.

    `lr` is applied to both arms when given. Only plain SGD needs it: the
    adaptive bases default to 1e-3, which sits inside COSGD's usable band
    already, whereas SGD's 0.05 default is far above it and was what made the
    original sweep of this axis a comparison at a rate chosen for the baseline.
    """
    lr_b = lr_baseline if lr_baseline is not None else lr
    hp_base = {"lr": lr_b} if lr_b is not None else {}
    hp_cos = {"cosgd_method": "gram_schmidt_normal", "class_order": "desc",
              "combine": "sum", "combine_norm_cap": 0.0}
    if lr is not None:
        hp_cos["lr"] = lr
    return [
        {"label": "baseline", "method": "baseline", "hp": hp_base},
        {"label": "cosgd", "method": "cosgd", "hp": hp_cos},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["sgd", "signsgd", "rmsprop", "adam"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--datasets", nargs="+", default=["wine", "digits"])
    ap.add_argument("--lr_sgd_baseline", type=float, default=None,
                    help="rate for the SGD baseline arm; defaults to --lr_sgd")
    ap.add_argument("--lr_sgd", type=float, default=None,
                    help="rate for the SGD base only; the adaptive bases "
                         "default to 1e-3, already inside COSGD's band")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.bases = ["sgd", "adam"]; args.seeds = [2026]; args.epochs = 5; args.datasets = ["wine"]
    # one harness call per (dataset, base) so each base's baseline+cosgd land together
    for ds in args.datasets:
        for base in args.bases:
            run_cosgd_sweep(axis_name=f"20_06_base_optimizer_{ds}_{base}", cells=build_cells(args.lr_sgd if base == "sgd" else None,
                                                              args.lr_sgd_baseline if base == "sgd" else None),
                            bases=[base], dataset=ds, seeds=args.seeds,
                            epochs=args.epochs, out_root=_HERE / "results" / ds / base,
                            train_subset=args.train_subset)


if __name__ == "__main__":
    main()
