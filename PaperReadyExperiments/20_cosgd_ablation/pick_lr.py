"""
Read the learning-rate axis and report the rate each arm actually operates at.

Wave 2 of the COSGD re-run has to be launched at a rate inside the method's
usable band, and that band is only known once wave 1 (axis 20.10, uncapped)
has run. This reads those results and prints the answer, so the wave-2 cells
configure themselves rather than relying on a number typed by hand.

The rate chosen is the one that reaches the arm's target *soonest*, the target
being 99% of the best final accuracy that arm attains anywhere on the grid. The
study measures time to a useful accuracy, not the accuracy itself, so the rate
has to be selected on the same footing; ranking on final accuracy instead picks
a rate that arrives at the same place several epochs later, which is precisely
the quantity under test. A rate that never reaches the target is charged the
whole budget plus one epoch, so it sorts behind every rate that does. Ties go
to the smaller rate, the safer side for a method whose failure mode is an
over-long step.

Both arms are reported. They do not share a rate: the baseline's optimum sits
far above COSGD's usable band, and running the pair at one rate handicaps
whichever arm did not choose it.

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

# Matches the chapter-wide convention: "reached" means within 1% of the target.
TARGET_FRAC = 0.99


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
        rows = [(_lr(t), float(np.mean([c[-1] for c in cs])), cs)
                for t, cs in by_lr.items()]
        if not rows:
            continue
        top = max(r[1] for r in rows)
        target = top * TARGET_FRAC
        scored = []
        for lr, fin, cs in rows:
            eps = []
            for c in cs:
                hit = next((i + 1 for i, v in enumerate(c) if v >= target), None)
                eps.append(hit if hit is not None else len(c) + 1)   # censored
            scored.append((lr, fin, float(np.mean(eps))))
        fastest = min(s[2] for s in scored)
        # ties within half an epoch go to the smaller rate
        cands = [s for s in scored if s[2] <= fastest + 0.5]
        pick = min(cands, key=lambda s: s[0])
        out[arm] = {"lr": pick[0], "epochs": pick[2], "final": pick[1],
                    "target": target, "best_final": top,
                    "grid": sorted(r[0] for r in rows),
                    "curve": {str(s[0]): {"final": s[1], "epochs": s[2]}
                              for s in sorted(scored)}}
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
            print(f"    {arm:<10} lr = {d['lr']:<8g} reaches {d['target']:.4f} "
                  f"in {d['epochs']:.1f} epochs  (grid {d['grid']})")
            for lr, v in sorted(d["curve"].items(), key=lambda kv: float(kv[0])):
                star = " *" if float(lr) == d["lr"] else "  "
                print(f"       {star} lr={float(lr):<7g} final={v['final']:.4f} "
                      f"epochs={v['epochs']:5.1f}")


if __name__ == "__main__":
    main()
