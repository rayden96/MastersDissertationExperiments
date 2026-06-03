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


def _latest_summaries(axis_dir: Path) -> List[Dict[str, Any]]:
    """All summary.json under an axis folder's results/** (newest run per leaf)."""
    out = []
    if not axis_dir.exists():
        return out
    # axes may nest results/<base>/run_* or results/run_*; collect newest per parent
    by_parent: Dict[Path, Path] = {}
    for s in axis_dir.rglob("summary.json"):
        parent = s.parent.parent  # the results/<...>/ dir containing run_<id>/
        if parent not in by_parent or s.parent.name > by_parent[parent].name:
            by_parent[parent] = s.parent
    for run_dir in by_parent.values():
        try:
            out.append(read_json(run_dir / "summary.json"))
        except Exception:
            pass
    return out


def _baseline_acc(cells: List[Dict]) -> Optional[float]:
    for e in cells:
        if "baseline" in e["cell"].lower():
            a = e["metrics"]["final_test_acc"]["mean"]
            if a == a:
                return a
    return None


def _best_cell(cells: List[Dict]):
    best = None
    for e in cells:
        a = e["metrics"]["final_test_acc"]["mean"]
        if a == a and (best is None or a > best["metrics"]["final_test_acc"]["mean"]):
            best = e
    return best


def build_master(axis_filter: Optional[str]) -> Dict[str, Any]:
    rows = []
    for folder, label in AXES.items():
        if axis_filter and axis_filter not in folder:
            continue
        # Gather from BOTH the persistent (Drive) location and repo-local. The
        # harness names per-run dirs by full axis_name ('10_<folder>'), and some
        # axes nest per-base/per-dataset subdirs, so glob by the '10_<folder>'
        # prefix and also the bare folder for older layouts.
        prefix = f"10_{folder}"
        summaries = []
        for base_dir in (_RESULTS_BASE, _AXIS_ROOT):
            if not base_dir.exists():
                continue
            for d in base_dir.glob(f"{prefix}*"):
                summaries += _latest_summaries(d)
            summaries += _latest_summaries(base_dir / folder)
        for summary in summaries:
            cells = summary.get("cells", [])
            # group by base
            bases = sorted({e["base"] for e in cells})
            for base in bases:
                bcells = [e for e in cells if e["base"] == base]
                base_acc = _baseline_acc(bcells)
                best = _best_cell(bcells)
                if best is None:
                    continue
                m = best["metrics"]
                rows.append({
                    "axis": label, "folder": folder, "base": base,
                    "dataset": summary.get("dataset"),
                    "best_cell": best["cell"],
                    "best_acc": round(m["final_test_acc"]["mean"], 4),
                    "best_acc_std": round(m["final_test_acc"]["std"], 4),
                    "baseline_acc": round(base_acc, 4) if base_acc is not None else None,
                    "delta_vs_baseline": (round(m["final_test_acc"]["mean"] - base_acc, 4)
                                          if base_acc is not None else None),
                    "I_between_K32": _g(m, "I_between_K32_mean"),
                    "between_K32_mean_cos": _g(m, "between_K32_mean_cos_mean"),
                    "mean_deficit_per_step": _g(m, "mean_deficit_per_step"),
                })
    return {"rows": rows, "n": len(rows)}


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
    out = _HERE / "master_table.json"
    for d in (persist_dir, _HERE):
        write_json_atomic(d / "master_table.json", master)

    print(f"\n=== BoGrad master table ({master['n']} rows) ===")
    hdr = f"{'axis':<26}{'base':<9}{'best cell':<22}{'acc':>8}{'d-base':>8}{'I_btwn32':>10}{'cos':>8}"
    print(hdr); print("-" * len(hdr))
    for r in master["rows"]:
        d = r["delta_vs_baseline"]
        print(f"{r['axis']:<26}{r['base']:<9}{r['best_cell']:<22}"
              f"{r['best_acc']:>8.3f}{(f'{d:+.3f}' if d is not None else '   n/a'):>8}"
              f"{(r['I_between_K32'] if r['I_between_K32'] is not None else float('nan')):>10.3f}"
              f"{(r['between_K32_mean_cos'] if r['between_K32_mean_cos'] is not None else float('nan')):>8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
