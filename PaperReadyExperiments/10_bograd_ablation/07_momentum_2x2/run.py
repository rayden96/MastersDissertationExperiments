"""
10.07 — Momentum x BoGrad (2x2 decomposition).

Claim. Momentum and BoGrad both act on between-batch trajectory structure, but
differently: momentum induces POSITIVE cos(g_t, g_{t-1}) (a short-horizon
smoothing; prior finding F1), while BoGrad removes recent-direction overlap
(persistent across horizons; F3). The 2x2 {momentum off/on} x {BoGrad off/on}
decomposes their separate and combined contributions — directly answering the
user's "does momentum help, and does it help on top of BoGrad" question, per base.

Scope. Only bases with a momentum knob: SGD and SignSGD (Signum). RMSprop's
momentum is optional (included), Adam's is baked into beta1 (excluded — no clean
off state), so Adam is not part of this axis.

Sweep. 4 cells per base: (mu=0, BoGrad off), (mu=0, BoGrad on), (mu=0.9, BoGrad
off), (mu=0.9, BoGrad on), fixed mid-K. Workhorse.

Explained via. cos(u_t,u_{t-1}) lag-decay profile and WW_K (= I_between_K): the
2x2 should show momentum and BoGrad each raising trajectory efficiency, and the
combined cell highest (multiplicative composition, F3).

Run: python run.py [--smoke] [--bases sgd signsgd rmsprop] [--K 32]
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

MOMENTUM_BASES = ["sgd", "signsgd", "rmsprop"]   # Adam excluded (no clean mu-off)
DEFAULT_K = {"sgd": 32, "signsgd": 64, "rmsprop": 16}


def build_cells(K):
    cells = []
    for mu in (0.0, 0.9):
        cells.append({"label": f"mu{mu:g}_bogradOff", "method": "baseline",
                      "hp": {"momentum": mu}})
        cells.append({"label": f"mu{mu:g}_bogradOn", "method": "bograd",
                      "hp": {"momentum": mu, "K": K, "projection_mode": "negative"}})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=MOMENTUM_BASES)
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
        K = args.K if args.K is not None else DEFAULT_K.get(base, 32)
        run_bograd_sweep(
            axis_name=f"10_07_momentum_2x2_{base}",
            cells=build_cells(K), bases=[base], dataset=args.dataset,
            seeds=args.seeds, epochs=args.epochs,
            out_root=_HERE / "results" / base, train_subset=args.train_subset,
        )


if __name__ == "__main__":
    main()
