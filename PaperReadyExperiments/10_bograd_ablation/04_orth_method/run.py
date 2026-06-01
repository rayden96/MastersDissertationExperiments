"""
10.04 — Orthogonalisation method.

Claim. The sequential ("soft") Gram-Schmidt subtraction is not a true orthogonal
projection onto span(buffer)^perp, and that is a feature: it preserves more of the
descent signal than the true-orthogonal QR/Householder projectors, which are more
aggressive and tend to hurt training (matches the BoGrad docstring + IJCNN
experience). The three differ in accuracy, wall-clock, and orthogonality residual.

Sweep. orth_method in {sequential, qr, householder} x 4 bases, fixed mid-K. Modes
qr/householder only realise a *true* projection under projection_mode="full"
(they defer to sequential under negative/positive), so this axis uses
projection_mode="full" for an apples-to-apples projector comparison, plus the
sequential-negative default as the production reference.

Explained via. orthogonality residual ||B_hat^T g_tilde|| (BoGrad.orthogonality_
residual): ~0 for qr/householder, larger for sequential — the quantitative
"soft vs hard" contrast — alongside accuracy and ms/step.

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
    cells = [{"label": "baseline", "method": "baseline", "hp": {}},
             # production reference: sequential + negative
             {"label": "sequential_negative", "method": "bograd",
              "hp": {"K": K, "orth_method": "sequential", "projection_mode": "negative"}}]
    # true-projector comparison under full mode
    for om in ("sequential", "qr", "householder"):
        cells.append({"label": f"{om}_full", "method": "bograd",
                      "hp": {"K": K, "orth_method": om, "projection_mode": "full"}})
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
            axis_name=f"10_04_orth_method_{base}",
            cells=build_cells(K), bases=[base], dataset=args.dataset,
            seeds=args.seeds, epochs=args.epochs,
            out_root=_HERE / "results" / base, train_subset=args.train_subset,
        )


if __name__ == "__main__":
    main()
