"""
10.05 — Projection scope.

Claim. `projection_scope="per_tensor"` orthogonalises each parameter tensor
against its own per-layer history (cheap, local subspace); `"global"`
orthogonalises one concatenated vector against a single joint history (captures
cross-layer structure, but one big buffer). Per-tensor is the default; this axis
asks whether the global joint subspace buys accuracy and at what memory/time cost.

Sweep. {per_tensor, global} x 4 bases, fixed mid-K. Workhorse (+ ResNet-18
transfer, where the per-layer-count difference is largest).

Explained via. I_between_K (does the scope change trajectory cancellation?),
ms/step, and peak GPU memory.

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
    for scope in ("per_tensor", "global"):
        cells.append({"label": f"scope_{scope}", "method": "bograd",
                      "hp": {"K": K, "projection_mode": "negative", "projection_scope": scope}})
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
            axis_name=f"10_05_projection_scope_{base}",
            cells=build_cells(K), bases=[base], dataset=args.dataset,
            seeds=args.seeds, epochs=args.epochs,
            out_root=_HERE / "results" / base, train_subset=args.train_subset,
        )


if __name__ == "__main__":
    main()
