"""
10.09 — Cross-optimizer / cross-architecture master summary.

Not a sweep — an aggregator. Reads the summary.json files produced by axes
10.01-10.08 and assembles the BoGrad "when / what / why" master table: for each
(base optimizer x axis), the best cell, its test accuracy vs the axis baseline,
and the interference metric that moved (I_between_K32, mean cos, deficit).

This is the artefact that answers Gap 1's headline question in one place. It
reads whatever axis runs exist (so it is useful incrementally as axes land) and
writes master_table.json + a printed table.

Run:
    python run.py                      # scan all axis results under 10_bograd_ablation/
    python run.py --axis 01_buffer_K   # restrict to one axis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_AXIS_ROOT = _HERE.parent
_PRE = _AXIS_ROOT.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json, write_json_atomic, get_results_root  # noqa: E402
from _summary_utils import prefer_new_layout, speedup_for_run  # noqa: E402

# Axes persist under get_results_root() (Drive on Colab) as
# 10_bograd_ablation/<axis_name>/...  where axis_name is e.g. '10_01_buffer_K'.
_RESULTS_BASE = get_results_root() / "10_bograd_ablation"

# Axis folder -> human label
AXES = {
    "01_buffer_K": "buffer size K",
    "02_lr_retune": "learning rate",
    "03_projection_mode": "projection mode/strength",
    "04_orth_method": "orthogonalisation method",
    "05_projection_scope": "projection scope",
    "06_magnitude": "direction vs magnitude",
    "07_momentum_2x2": "momentum x BoGrad",
    "08_batch_K": "batch size x K",
}


def _run_rank(run_dir: Path) -> tuple:
    """Rank a run for 'best per parent': prefer MOST cells (most complete), then
    most recent mtime. Robust to mixed run-dir naming (old timestamps vs new
    config-hash names) where a plain string sort would mis-order. With one
    complete run + several partial re-runs, the complete one always wins."""
    try:
        n_cells = len(read_json(run_dir / "summary.json").get("cells", []))
    except Exception:
        n_cells = 0
    try:
        mtime = (run_dir / "summary.json").stat().st_mtime
    except Exception:
        mtime = 0.0
    return (n_cells, mtime)


def _latest_runs(axis_dir: Path) -> List[tuple]:
    """(run_dir, summary) per parent dir, choosing the most-complete/most-recent
    run, so duplicate re-runs under one parent never need manual deletion."""
    out = []
    if not axis_dir.exists():
        return out
    by_parent: Dict[Path, Path] = {}
    for s in axis_dir.rglob("summary.json"):
        parent = s.parent.parent  # the results/<...>/ dir containing run_<id>/
        if parent not in by_parent or _run_rank(s.parent) > _run_rank(by_parent[parent]):
            by_parent[parent] = s.parent
    for run_dir in by_parent.values():
        try:
            out.append((run_dir, read_json(run_dir / "summary.json")))
        except Exception:
            pass
    return out


def _axis_runs(folder: str) -> List[tuple]:
    """All (run_dir, summary, priority) for an axis, from the NEW prefixed Drive
    layout ('10_<folder>*', priority 1) and the LEGACY bare layout ('<folder>',
    priority 0). prefer_new_layout() later drops stale bare duplicates."""
    prefix = f"10_{folder}"
    out = []
    for base_dir in (_RESULTS_BASE, _AXIS_ROOT):
        if not base_dir.exists():
            continue
        for d in base_dir.glob(f"{prefix}*"):
            out += [(rd, s, 1) for rd, s in _latest_runs(d)]
        out += [(rd, s, 0) for rd, s in _latest_runs(base_dir / folder)]
    return out


def _baseline_cell(cells: List[Dict]) -> Optional[Dict]:
    """The no-intervention reference cell for an axis. Usually a cell named
    'baseline'; the momentum 2x2 (07_momentum_2x2) has no 'baseline' cell, so the
    all-off corner (mu=0, BoGrad off) 'mu0_bogradOff' is the true reference —
    fall back to it (preferring mu=0 over mu=0.9)."""
    for e in cells:
        if "baseline" in e["cell"].lower():
            return e
    offs = [e for e in cells if "bogradoff" in e["cell"].lower().replace("_", "")]
    if offs:
        mu0 = [e for e in offs if e["cell"].lower().startswith("mu0_")]
        return mu0[0] if mu0 else offs[0]
    return None


def _baseline_acc(cells: List[Dict]) -> Optional[float]:
    e = _baseline_cell(cells)
    if e is None:
        return None
    a = e["metrics"]["final_test_acc"]["mean"]
    return a if a == a else None


def _best_cell(cells: List[Dict]):
    best = None
    for e in cells:
        a = e["metrics"]["final_test_acc"]["mean"]
        if a == a and (best is None or a > best["metrics"]["final_test_acc"]["mean"]):
            best = e
    return best


def build_master(axis_filter: Optional[str]) -> Dict[str, Any]:
    # collect (key, priority, row); prefer_new_layout() then drops stale legacy
    # (bare-folder) duplicates per (folder, base, dataset).
    collected = []
    cell_collected = []   # per-cell speed-up, every (base, cell), for the reframed tables
    for folder, label in AXES.items():
        if axis_filter and axis_filter not in folder:
            continue
        for run_dir, summary, prio in _axis_runs(folder):
            cells = summary.get("cells", [])
            dataset = summary.get("dataset")
            sp = speedup_for_run(run_dir)   # {(base, cell): speed record} from curves
            for (cb, ccell), rec in sp.items():
                em = next((e["metrics"] for e in cells
                           if e["base"] == cb and e["cell"] == ccell), {})
                fa = em.get("final_test_acc", {}) if isinstance(em, dict) else {}
                cell_collected.append(((folder, cb, dataset, ccell), prio, {
                    "axis": label, "folder": folder, "base": cb, "dataset": dataset,
                    "cell": ccell,
                    "epoch_speedup": rec.get("epoch_speedup"),
                    "wall_speedup": rec.get("wall_speedup"),
                    "baseline_epochs": rec.get("baseline_epochs"),
                    "method_epochs": rec.get("method_epochs"),
                    "final_test_acc": (round(fa["mean"], 4)
                                       if isinstance(fa, dict)
                                       and isinstance(fa.get("mean"), (int, float)) else None),
                }))
            for base in sorted({e["base"] for e in cells}):
                bcells = [e for e in cells if e["base"] == base]
                base_cell = _baseline_cell(bcells)
                base_acc = _baseline_acc(bcells)
                base_ibtw = (_g(base_cell["metrics"], "I_between_K32_mean")
                             if base_cell is not None else None)
                best = _best_cell(bcells)
                if best is None:
                    continue
                m = best["metrics"]
                best_ibtw = _g(m, "I_between_K32_mean")
                spd = sp.get((base, best["cell"]), {})
                collected.append(((folder, base, dataset), prio, {
                    "axis": label, "folder": folder, "base": base,
                    "dataset": dataset,
                    "best_cell": best["cell"],
                    "best_acc": round(m["final_test_acc"]["mean"], 4),
                    "best_acc_std": round(m["final_test_acc"]["std"], 4),
                    "baseline_acc": round(base_acc, 4) if base_acc is not None else None,
                    "delta_vs_baseline": (round(m["final_test_acc"]["mean"] - base_acc, 4)
                                          if base_acc is not None else None),
                    "epoch_speedup": spd.get("epoch_speedup"),
                    "wall_speedup": spd.get("wall_speedup"),
                    "baseline_epochs": spd.get("baseline_epochs"),
                    "method_epochs": spd.get("method_epochs"),
                    "I_between_K32": best_ibtw,
                    "baseline_I_between_K32": base_ibtw,
                    "delta_I_between_K32": (round(best_ibtw - base_ibtw, 4)
                                           if best_ibtw is not None and base_ibtw is not None
                                           else None),
                    "between_K32_mean_cos": _g(m, "between_K32_mean_cos_mean"),
                    "mean_deficit_per_step": _g(m, "mean_deficit_per_step"),
                }))
    rows = prefer_new_layout(collected)
    cell_rows = prefer_new_layout(cell_collected)
    return {"rows": rows, "n": len(rows), "speedup_cells": cell_rows}


def _g(metrics, key):
    v = metrics.get(key, {})
    m = v.get("mean") if isinstance(v, dict) else None
    return round(m, 4) if isinstance(m, (int, float)) and m == m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default=None, help="restrict to one axis folder substring")
    args = ap.parse_args()

    master = build_master(args.axis)
    # Write to BOTH the persistent (Drive) location and repo-local so the master
    # table survives a Colab session end.
    persist_dir = _RESULTS_BASE / "09_cross_summary"
    persist_dir.mkdir(parents=True, exist_ok=True)
    for d in (persist_dir, _HERE):
        write_json_atomic(d / "master_table.json",
                          {"rows": master["rows"], "n": master["n"]})
        write_json_atomic(d / "speedup_cells.json",
                          {"cells": master["speedup_cells"],
                           "n": len(master["speedup_cells"])})

    print(f"\n=== BoGrad master table ({master['n']} rows) ===")
    hdr = (f"{'axis':<26}{'base':<9}{'best cell':<22}{'acc':>8}{'d-base':>8}"
           f"{'spd':>7}{'I_btwn32':>10}{'cos':>8}")
    print(hdr); print("-" * len(hdr))
    for r in master["rows"]:
        d = r["delta_vs_baseline"]; sp = r.get("epoch_speedup")
        print(f"{r['axis']:<26}{r['base']:<9}{r['best_cell']:<22}"
              f"{r['best_acc']:>8.3f}{(f'{d:+.3f}' if d is not None else '   n/a'):>8}"
              f"{(f'{sp:.2f}x' if sp is not None else '  n/a'):>7}"
              f"{(r['I_between_K32'] if r['I_between_K32'] is not None else float('nan')):>10.3f}"
              f"{(r['between_K32_mean_cos'] if r['between_K32_mean_cos'] is not None else float('nan')):>8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
