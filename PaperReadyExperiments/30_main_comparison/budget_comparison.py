"""
30.12 — Budget-matched comparison: the speed-up story told properly.

The question this dissertation asks is whether COSGD and BOGrad make training
FASTER, not whether they reach a higher ceiling. A single "epochs to reach the
baseline's final accuracy" number answers that only at one point on the curve,
and it hides two things: that a method's advantage varies with how demanding
the target is, and that steps saved are not the same as time saved once the
per-step cost differs.

This view reports the comparison in both directions, against the baseline's own
budget:

  1. ACCURACY AT A FRACTION OF THE BASELINE'S BUDGET, at 20%, 50% and 100%,
     measured twice:
       - by STEPS  (what you get for the same number of updates), and
       - by WALL-CLOCK (what you get for the same amount of time, which charges
         the method for its per-step overhead).
  2. COST TO REACH THE BASELINE'S BEST ACCURACY, in steps and in wall-clock,
     as a ratio (below 1 = cheaper than the baseline).

Reads only persisted records, trains nothing:
  - accuracy curves from the bakeoff cells (history.epoch_test_acc, per seed);
  - per-step cost from 30.11 timing.json, which measures every arm back-to-back
    in one process. The per-cell mean_step_wall_time_s recorded during training
    is NOT used: those come from different sessions on different devices and are
    not comparable across arms.

Because steps-per-epoch is shared by every arm within a dataset, a fraction of
the baseline's steps is the same fraction of its epochs, so the step-budget
columns need no timing data at all. Only the wall-clock columns do.

Run:
    python budget_comparison.py
    python budget_comparison.py --campaign _core/results/main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json, write_json_atomic, get_results_root  # noqa: E402

BASES = ["sgd", "signsgd", "rmsprop", "adam"]
METHODS = ["cosgd", "bograd", "graddrop", "dropout"]
FRACTIONS = [0.2, 0.5, 1.0]


def _acc_at_epoch(curve: List[float], epoch: float) -> Optional[float]:
    """Accuracy after `epoch` epochs, linearly interpolated between the two
    surrounding logged epochs. `epoch` is 1-based and may be fractional, which
    it will be whenever a wall-clock budget lands mid-epoch. Beyond the end of
    the curve the final value is carried forward, since the run stopped there
    and no further progress was made."""
    if not curve:
        return None
    if epoch <= 1:
        return curve[0]
    if epoch >= len(curve):
        return curve[-1]
    lo = int(np.floor(epoch)) - 1
    frac = epoch - np.floor(epoch)
    return curve[lo] + frac * (curve[lo + 1] - curve[lo])


def _epochs_to_reach(curve: List[float], target: float) -> Optional[float]:
    """First (fractional) epoch at which `curve` reaches `target`, interpolating
    within the epoch that crosses it; None if never reached."""
    for i, a in enumerate(curve):
        if a is not None and a == a and a >= target:
            if i == 0:
                return 1.0
            prev = curve[i - 1]
            if a == prev:
                return float(i + 1)
            return i + (target - prev) / (a - prev)
    return None


def _step_times(timing: Optional[dict]) -> Dict[tuple, float]:
    """{(dataset, base, method): median seconds/step} from 30.11."""
    out: Dict[tuple, float] = {}
    for r in (timing or {}).get("rows", []):
        s = r.get("median_step_s")
        if isinstance(s, (int, float)) and s == s:
            out[(r.get("dataset"), r.get("base"), r.get("method"))] = s
    return out


def _curves(cell_path: Path) -> List[List[float]]:
    try:
        rows = read_json(cell_path).get("rows", [])
    except Exception:
        return []
    return [r["epoch_test_acc"] for r in rows
            if isinstance(r.get("epoch_test_acc"), list) and r["epoch_test_acc"]]


def build(campaign: Path, timing: Optional[dict]) -> Dict[str, Any]:
    st = _step_times(timing)
    out: Dict[str, Any] = {"cells": [], "fractions": FRACTIONS,
                           "timing_available": sorted({k[0] for k in st})}

    for ds_dir in sorted(p for p in campaign.glob("*") if p.is_dir()):
        ds = ds_dir.name
        for base in BASES:
            bcur = _curves(ds_dir / f"cell_{base}__baseline.json")
            if not bcur:
                continue
            n_ep = min(len(c) for c in bcur)
            # The baseline's own budget and what it achieved with it.
            b_best = float(np.mean([max(c) for c in bcur]))
            b_final = float(np.mean([c[-1] for c in bcur]))
            b_step = st.get((ds, base, "baseline"))

            for method in METHODS:
                mcur = _curves(ds_dir / f"cell_{base}__{method}.json")
                if not mcur:
                    continue
                m_step = st.get((ds, base, method))
                rec: Dict[str, Any] = {
                    "dataset": ds, "base": base, "method": method,
                    "baseline_epochs": n_ep,
                    "baseline_best_acc": round(b_best, 4),
                    "baseline_final_acc": round(b_final, 4),
                    "step_time_ratio": (round(m_step / b_step, 3)
                                        if b_step and m_step else None),
                }

                # --- accuracy at a fraction of the baseline's budget ---------
                for f in FRACTIONS:
                    tag = f"{int(f * 100)}"
                    # same number of STEPS = same number of epochs
                    rec[f"acc_at_{tag}pct_steps"] = round(float(np.mean(
                        [_acc_at_epoch(c, f * n_ep) for c in mcur])), 4)
                    rec[f"baseline_acc_at_{tag}pct_steps"] = round(float(np.mean(
                        [_acc_at_epoch(c, f * n_ep) for c in bcur])), 4)
                    # same WALL-CLOCK: a costlier step buys fewer of them
                    if b_step and m_step:
                        m_epochs = f * n_ep * (b_step / m_step)
                        rec[f"acc_at_{tag}pct_time"] = round(float(np.mean(
                            [_acc_at_epoch(c, m_epochs) for c in mcur])), 4)
                    else:
                        rec[f"acc_at_{tag}pct_time"] = None

                # --- cost to reach the baseline's BEST accuracy -------------
                eps = [_epochs_to_reach(c, b_best) for c in mcur]
                beps = [_epochs_to_reach(c, b_best) for c in bcur]
                reached = [e for e in eps if e is not None]
                if reached and len(reached) == len(eps):
                    m_e = float(np.mean(eps))
                    b_e = float(np.mean([e for e in beps if e is not None])) or n_ep
                    rec["steps_ratio_to_baseline_best"] = round(m_e / b_e, 3)
                    rec["time_ratio_to_baseline_best"] = (
                        round((m_e * m_step) / (b_e * b_step), 3)
                        if b_step and m_step else None)
                    rec["reached_baseline_best"] = True
                else:
                    rec["steps_ratio_to_baseline_best"] = None
                    rec["time_ratio_to_baseline_best"] = None
                    rec["reached_baseline_best"] = False
                out["cells"].append(rec)
    return out


def _fmt(v, spec="{:.3f}", na="--"):
    return spec.format(v) if isinstance(v, (int, float)) else na


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None)
    args = ap.parse_args()

    campaign = Path(args.campaign) if args.campaign else \
        (_HERE / "_core" / "results" / "main")
    if not campaign.exists():
        raise SystemExit(f"campaign not found: {campaign}")

    timing = None
    for c in (get_results_root() / "30_main_comparison" / "timing" / "timing.json",
              _HERE / "timing.json"):
        if c.exists():
            timing = read_json(c)
            break
    if timing is None:
        print("!! no timing.json — wall-clock columns will be blank")

    res = build(campaign, timing)
    print(f"\ntiming available for: {res['timing_available'] or 'NONE'}")

    print("\n=== Accuracy at a fraction of the baseline's budget ===")
    print("(steps = same number of updates; time = same wall-clock, "
          "so a costlier step buys fewer updates)")
    hdr = (f"{'dataset':<16}{'base':<9}{'method':<10}"
           f"{'20%st':>8}{'50%st':>8}{'100%st':>8}"
           f"{'20%t':>8}{'50%t':>8}{'100%t':>8}{'base100':>9}")
    print(hdr); print("-" * len(hdr))
    for c in res["cells"]:
        print(f"{c['dataset']:<16}{c['base']:<9}{c['method']:<10}"
              f"{_fmt(c['acc_at_20pct_steps']):>8}{_fmt(c['acc_at_50pct_steps']):>8}"
              f"{_fmt(c['acc_at_100pct_steps']):>8}"
              f"{_fmt(c['acc_at_20pct_time']):>8}{_fmt(c['acc_at_50pct_time']):>8}"
              f"{_fmt(c['acc_at_100pct_time']):>8}"
              f"{_fmt(c['baseline_acc_at_100pct_steps']):>9}")

    print("\n=== Cost to reach the baseline's BEST accuracy (ratio, <1 = cheaper) ===")
    hdr2 = (f"{'dataset':<16}{'base':<9}{'method':<10}"
            f"{'steps':>8}{'wall':>8}{'step cost':>11}{'reached':>9}")
    print(hdr2); print("-" * len(hdr2))
    for c in res["cells"]:
        print(f"{c['dataset']:<16}{c['base']:<9}{c['method']:<10}"
              f"{_fmt(c['steps_ratio_to_baseline_best']):>8}"
              f"{_fmt(c['time_ratio_to_baseline_best']):>8}"
              f"{_fmt(c['step_time_ratio'], '{:.2f}x'):>11}"
              f"{('yes' if c['reached_baseline_best'] else 'NO'):>9}")

    for d in (get_results_root() / "30_main_comparison" / "budget", _HERE):
        Path(d).mkdir(parents=True, exist_ok=True)
        write_json_atomic(Path(d) / "budget_comparison.json", res)
    print(f"\nwrote budget_comparison.json ({len(res['cells'])} cells)")


if __name__ == "__main__":
    main()
