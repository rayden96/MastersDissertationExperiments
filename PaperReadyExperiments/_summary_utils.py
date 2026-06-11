"""
Shared ablation-summary utilities (BoGrad 10.09 + COSGD 20.08 aggregators).

Two concerns both master-table builders share:

  1. Layout dedup. Runs persist under either the NEW prefixed Drive layout
     ('10_04_orth_method_sgd/...') or the LEGACY bare layout ('04_orth_method/...').
     When an axis was re-run after the layout change, BOTH exist and the bare one
     is stale. `prefer_new_layout` keeps, per (folder, base, dataset), only the
     rows from the highest-priority source (prefixed=1 > bare=0), so stale legacy
     duplicates drop out without manual deletion. Rows sharing a key at the SAME
     priority are all kept, so multi-run axes (10.08 batch x K: several batch
     sizes under one key) survive. Assumption: a re-run replaces a whole axis,
     not a subset of a multi-run axis.

  2. Convergence speed-up — the dissertation's headline metric. Every cell
     persists results.json with history.epoch_test_acc; `speedup_for_run` reads
     those curves and computes, per (base, cell), epochs-to-target and the
     speed-up vs the axis baseline (same convention as
     30_main_comparison/views.view_speedup), so the master tables can report
     'does it train faster', not only final accuracy. No re-running needed.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent          # PaperReadyExperiments/
_REPO = _HERE.parent
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.storage import read_json  # noqa: E402


# --- layout dedup ----------------------------------------------------------
def prefer_new_layout(collected: List[Tuple[tuple, int, dict]]) -> List[dict]:
    """collected: list of (key, priority, row). Keep only rows at the max
    priority seen for each key (prefixed=1 beats legacy bare=0); rows that share
    a key at the same priority are all preserved (multi-run axes)."""
    maxprio: Dict[tuple, int] = {}
    for key, prio, _row in collected:
        maxprio[key] = max(maxprio.get(key, -1), prio)
    return [row for key, prio, row in collected if prio == maxprio[key]]


# --- convergence speed-up --------------------------------------------------
def epochs_to_target(curve, target) -> Optional[int]:
    """First 1-based epoch at which `curve` reaches `target`, else None."""
    for i, a in enumerate(curve):
        if a is not None and a == a and a >= target:
            return i + 1
    return None


def pick_baseline_label(labels) -> Optional[str]:
    """The axis baseline cell label: a cell named 'baseline', or (momentum 2x2)
    the all-off corner 'mu0_bogradOff' (preferring mu=0 over mu=0.9)."""
    for l in labels:
        if "baseline" in l.lower():
            return l
    offs = [l for l in labels if "bogradoff" in l.lower().replace("_", "")]
    if offs:
        mu0 = [l for l in offs if l.lower().startswith("mu0_")]
        return mu0[0] if mu0 else offs[0]
    return None


def _parse_cell_key(name: str):
    """'base__label__seedN' -> (base, label, seed). Label may itself contain
    single underscores; only the double '__' separates the three fields."""
    parts = name.split("__")
    if len(parts) < 3:
        return None
    return parts[0], "__".join(parts[1:-1]), parts[-1]


def read_cell_results(run_dir: Path) -> Dict[tuple, Dict[str, list]]:
    """{(base,label): {'curves':[per-seed epoch_test_acc], 'step_times':[...],
    'finals':[...]}} from each COMPLETED cell's results.json under run_dir/cells/."""
    out: Dict[tuple, Dict[str, list]] = defaultdict(
        lambda: {"curves": [], "step_times": [], "finals": []})
    cells_dir = Path(run_dir) / "cells"
    if not cells_dir.exists():
        return out
    for rj in cells_dir.glob("*/results.json"):
        pk = _parse_cell_key(rj.parent.name)
        if pk is None:
            continue
        base, label, _seed = pk
        try:
            res = read_json(rj)
        except Exception:
            continue
        if res.get("status") != "completed":
            continue
        curve = (res.get("history", {}) or {}).get("epoch_test_acc") or []
        sc = res.get("scalars", {}) or {}
        rec = out[(base, label)]
        if curve:
            rec["curves"].append(curve)
        st = sc.get("mean_step_wall_time_s")
        if isinstance(st, (int, float)) and st == st:
            rec["step_times"].append(st)
        fa = sc.get("final_test_acc")
        if isinstance(fa, (int, float)) and fa == fa:
            rec["finals"].append(fa)
    return out


def speedup_for_run(run_dir: Path, target_frac: float = 1.0) -> Dict[tuple, dict]:
    """{(base,label): speed-record} for every non-baseline cell vs the axis
    baseline, from persisted epoch curves. epoch_speedup = baseline_epochs /
    method_epochs to reach `target_frac` x baseline final acc (>1 = faster);
    wall_speedup folds in mean per-step cost. A method that never reaches the
    target is charged len(curve)+1 epochs (same convention as views.view_speedup)."""
    import numpy as np
    data = read_cell_results(run_dir)
    out: Dict[tuple, dict] = {}
    for base in sorted({b for (b, _l) in data}):
        labels = [l for (b, l) in data if b == base]
        bl = pick_baseline_label(labels)
        if bl is None:
            continue
        brec = data[(base, bl)]
        bcurves = [c for c in brec["curves"] if c]
        if not bcurves:
            continue
        bfinal = float(np.mean([c[-1] for c in bcurves]))
        tgt = target_frac * bfinal
        b_ep = float(np.mean([epochs_to_target(c, tgt) or (len(c) + 1) for c in bcurves]))
        b_spt = float(np.mean(brec["step_times"])) if brec["step_times"] else None
        for label in labels:
            if label == bl:
                continue
            mrec = data[(base, label)]
            mcurves = [c for c in mrec["curves"] if c]
            if not mcurves:
                continue
            m_ep = float(np.mean([epochs_to_target(c, tgt) or (len(c) + 1) for c in mcurves]))
            sp = (b_ep / m_ep) if m_ep > 0 else None
            m_spt = float(np.mean(mrec["step_times"])) if mrec["step_times"] else None
            wall = (sp * (b_spt / m_spt)) if (sp and b_spt and m_spt) else None
            out[(base, label)] = {
                "baseline_cell": bl,
                "baseline_epochs": round(b_ep, 3),
                "method_epochs": round(m_ep, 3),
                "epoch_speedup": round(sp, 3) if sp else None,
                "wall_speedup": round(wall, 3) if wall else None,
                "method_epoch1_acc": round(float(np.mean([c[0] for c in mcurves])), 4),
            }
    return out


__all__ = ["prefer_new_layout", "epochs_to_target", "pick_baseline_label",
           "read_cell_results", "speedup_for_run"]
