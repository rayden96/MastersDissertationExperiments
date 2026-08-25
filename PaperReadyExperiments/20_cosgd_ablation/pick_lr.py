"""
Read the learning-rate axis and report the rate each arm actually operates at.

Wave 2 of the COSGD re-run has to be launched at a rate inside the method's
usable band, and that band is only known once wave 1 (axis 20.10, uncapped)
has run. This reads those results and prints the answer, so the wave-2 cells
configure themselves rather than relying on a number typed by hand.

The rate chosen is the one whose curve reaches the highest accuracy at the end
of the budget, which for a sweep of this shape is also the one that climbs
fastest; ties are broken toward the smaller rate, which is the safer side for
a method whose failure mode is an over-long step.

    python pick_lr.py                 # both anchors
    python pick_lr.py --json          # machine-readable, for a launcher
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage            # noqa: E402
from _curves import load_axis         # noqa: E402

ANCHORS = ["covertype", "cifar10"]


def _lr(tag: str) -> float:
    return float(tag.lstrip("lr").replace("p", ".").replace("m", "-"))


def best_rates(anchor: str):
    root = storage.get_results_root() / "20_cosgd_ablation"
    recs = []
    for d in sorted(root.glob(f"20_10_learning_rate_{anchor}*")):
        recs.extend(load_axis(d))
    recs = [r for r in recs if r.dataset == anchor and r.acc]
    arms = {}
    for r in recs:
        tag, _, arm = r.cell.rpartition("_")
        arms.setdefault(arm, {}).setdefault(tag, []).append(r.acc)
    out = {}
    for arm, by_lr in arms.items():
        rows = [(_lr(t), float(np.mean([c[-1] for c in cs]))) for t, cs in by_lr.items()]
        if not rows:
            continue
        top = max(r[1] for r in rows)
        # ties within half a point go to the smaller rate
        cands = [r for r in rows if r[1] >= top - 0.005]
        out[arm] = {"lr": min(c[0] for c in cands), "final": top,
                    "grid": sorted(r[0] for r in rows)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    result = {a: best_rates(a) for a in ANCHORS}
    if args.json:
        print(json.dumps(result)); return
    for a, arms in result.items():
        print(f"=== {a}")
        if not arms:
            print("    no learning-rate results yet — run wave 1 first")
            continue
        for arm, d in sorted(arms.items()):
            print(f"    {arm:<10} best lr = {d['lr']:<8g} (grid {d['grid']})")


if __name__ == "__main__":
    main()
