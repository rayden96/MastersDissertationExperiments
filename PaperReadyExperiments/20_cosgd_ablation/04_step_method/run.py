"""
20.04 — Step method / BatchNorm handling.

Claim. COSGD computes a separate forward+backward per class. How that interacts
with BatchNorm matters:
  - single_forward: one forward on the full batch, per-class losses sliced from
    shared logits (cheapest; BN sees the full batch once);
  - multi_forward: a separate forward per class (BN sees only that class's
    samples -> corrupts running stats / uses class-only batch stats);
  - multi_forward_with_BN: update BN running stats on the full batch first, then
    do per-class forwards in eval mode (BN frozen) -> the principled choice for
    BN models.
On a no-BN model the three should be near-identical; on a BN model
(CIFAR-100/ResNet) the BN-frozen variant should win.

Sweep. {single_forward, multi_forward, multi_forward_with_BN} + baseline. Run on
CIFAR-10 (no BN -> expect ~tie) and CIFAR-100 (ResNet, BN -> expect BN variant
best). Explained via accuracy gap between the variants on BN vs no-BN models.

Run: python run.py [--smoke] [--dataset cifar10|cifar100]
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
    for sm in ("single_forward", "multi_forward", "multi_forward_with_BN"):
        cells.append({"label": f"step_{sm}", "method": "cosgd",
                      "hp": {"cosgd_method": "modified_gs_negative",
                             "step_method": sm, "combine": "mean"}})
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
    run_cosgd_sweep(axis_name=f"20_04_step_method_{args.dataset}", cells=build_cells(),
                    bases=args.bases, dataset=args.dataset, seeds=args.seeds,
                    epochs=args.epochs, out_root=_HERE / "results" / args.dataset,
                    train_subset=args.train_subset)


if __name__ == "__main__":
    main()
