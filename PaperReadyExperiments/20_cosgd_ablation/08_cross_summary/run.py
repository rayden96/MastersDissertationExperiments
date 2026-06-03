"""
20.08 — COSGD master summary + COSGD <-> BoGrad mechanism contrast.

Aggregator (no training). Two artefacts:

  1. COSGD "when / what / why" master table: best COSGD cell per (base x axis),
     its accuracy vs the axis baseline, and the inter-batch metric that moved
     (I_inter, inter_mean_cos) — the COSGD analogue of 10.09.

  2. The mechanism contrast that is the dissertation's central claim: COSGD acts
     on INTER-batch cancellation (raises I_inter, flips inter cosine toward 0)
     while leaving BETWEEN-batch trajectory cancellation roughly untouched; BoGrad
     does the opposite. Reads the COSGD axis summaries here and the BoGrad master
     table (10_bograd_ablation/09_cross_summary/master_table.json) and prints a
     side-by-side: delta-I_inter vs delta-I_between for each method.

Run:
    python run.py
"""

from __future__ import annotations

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

# Axes now persist under get_results_root() (Drive on Colab). Scan there; fall
# back to the repo-local axis folders for any results produced before this change.
_RESULTS_BASE = get_results_root() / "20_cosgd_ablation"

AXES = {
    "01_gs_variant": "GS variant",
    "02_class_order": "class order",
    "03_prenormalize": "pre-normalisation",
    "04_step_method": "step method / BN",
    "05_combine": "combine rule",
    "06_base_optimizer": "base optimizer",
}


def _latest_summaries(axis_dir: Path) -> List[Dict[str, Any]]:
    out = []
    if not axis_dir.exists():
        return out
    by_parent: Dict[Path, Path] = {}
    for s in axis_dir.rglob("summary.json"):
        parent = s.parent.parent
        if parent not in by_parent or s.parent.name > by_parent[parent].name:
            by_parent[parent] = s.parent
    for run_dir in by_parent.values():
        try:
            out.append(read_json(run_dir / "summary.json"))
        except Exception:
            pass
    return out


def _g(metrics, key):
    v = metrics.get(key, {})
    m = v.get("mean") if isinstance(v, dict) else None
    return round(m, 4) if isinstance(m, (int, float)) and m == m else None


def _baseline(cells, base):
    for e in cells:
        if e["base"] == base and "baseline" in e["cell"].lower():
            return e
    return None


def _axis_summaries(folder: str):
    """Summaries for an axis from BOTH the persistent (Drive) location and the
    repo-local one. The harness persists each run under a dir named by the full
    axis_name, e.g. '20_05_combine_iris' (= '20_' + folder + '_<ds>'), so we glob
    by that prefix rather than the bare AXES-key folder name."""
    prefix = f"20_{folder}"          # e.g. 05_combine -> 20_05_combine
    out = []
    for base in (_RESULTS_BASE, _AXIS_ROOT):
        if not base.exists():
            continue
        for d in base.glob(f"{prefix}*"):   # 20_05_combine, 20_05_combine_iris, ...
            out += _latest_summaries(d)
        # also the bare folder (older layout / repo-local results dir)
        out += _latest_summaries(base / folder)
    return out


def build():
    rows = []
    for folder, label in AXES.items():
        for summary in _axis_summaries(folder):
            cells = summary.get("cells", [])
            for base in sorted({e["base"] for e in cells}):
                bcells = [e for e in cells if e["base"] == base]
                bl = _baseline(bcells, base)
                best = None
                for e in bcells:
                    if "baseline" in e["cell"].lower():
                        continue
                    a = e["metrics"]["final_test_acc"]["mean"]
                    if a == a and (best is None or a > best["metrics"]["final_test_acc"]["mean"]):
                        best = e
                if best is None:
                    continue
                bm, blm = best["metrics"], (bl["metrics"] if bl else {})
                base_acc = _g(blm, "final_test_acc")
                base_Iinter = _g(blm, "I_inter_mean")
                best_Iinter = _g(bm, "I_inter_mean")
                rows.append({
                    "axis": label, "folder": folder, "base": base,
                    "dataset": summary.get("dataset"),
                    "best_cell": best["cell"],
                    "best_acc": _g(bm, "final_test_acc"),
                    "baseline_acc": base_acc,
                    "delta_acc": (round(_g(bm, "final_test_acc") - base_acc, 4)
                                  if base_acc is not None and _g(bm, "final_test_acc") is not None else None),
                    "I_inter": best_Iinter,
                    "delta_I_inter": (round(best_Iinter - base_Iinter, 4)
                                      if best_Iinter is not None and base_Iinter is not None else None),
                    "inter_mean_cos": _g(bm, "inter_mean_cos_mean"),
                    "I_between_K32": _g(bm, "I_between_K32_mean"),
                })
    return {"rows": rows, "n": len(rows)}


def contrast():
    """COSGD vs BoGrad: which interference axis each one moves."""
    # prefer the persistent (Drive) BoGrad master table, fall back to repo-local
    bograd_mt = get_results_root() / "10_bograd_ablation" / "09_cross_summary" / "master_table.json"
    if not bograd_mt.exists():
        bograd_mt = _REPO / "PaperReadyExperiments" / "10_bograd_ablation" / "09_cross_summary" / "master_table.json"
    out = {"cosgd_moves_I_inter": None, "bograd_moves_I_between": None}
    cos = build()
    cos_di = [r["delta_I_inter"] for r in cos["rows"] if r.get("delta_I_inter") is not None]
    if cos_di:
        out["cosgd_moves_I_inter"] = round(sum(cos_di) / len(cos_di), 4)
    if bograd_mt.exists():
        try:
            bg = read_json(bograd_mt)
            ibs = [r["I_between_K32"] for r in bg.get("rows", []) if r.get("I_between_K32") is not None]
            if ibs:
                out["bograd_I_between_observed"] = round(sum(ibs) / len(ibs), 4)
        except Exception:
            pass
    return out


def main():
    # Write to BOTH the persistent (Drive) location and the repo-local folder so
    # the master tables survive a Colab session end.
    out_dir = _RESULTS_BASE / "08_cross_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    master = build()
    ctr = contrast()
    for d in (out_dir, _HERE):
        write_json_atomic(d / "master_table.json", master)
        write_json_atomic(d / "mechanism_contrast.json", ctr)
    print(f"\n(also persisted to {out_dir})", flush=True)

    print(f"\n=== COSGD master table ({master['n']} rows) ===")
    hdr = f"{'axis':<22}{'base':<9}{'best cell':<26}{'acc':>8}{'d-acc':>8}{'I_inter':>9}{'dI_int':>8}{'cos':>8}"
    print(hdr); print("-" * len(hdr))
    for r in master["rows"]:
        da = r["delta_acc"]; di = r["delta_I_inter"]; cos = r["inter_mean_cos"]
        print(f"{r['axis']:<22}{r['base']:<9}{r['best_cell']:<26}"
              f"{(r['best_acc'] if r['best_acc'] is not None else float('nan')):>8.3f}"
              f"{(f'{da:+.3f}' if da is not None else '   n/a'):>8}"
              f"{(r['I_inter'] if r['I_inter'] is not None else float('nan')):>9.3f}"
              f"{(f'{di:+.3f}' if di is not None else '   n/a'):>8}"
              f"{(cos if cos is not None else float('nan')):>8.3f}")

    print("\n=== COSGD <-> BoGrad mechanism contrast ===")
    print(f"  COSGD mean delta-I_inter (raises within-batch alignment): {ctr.get('cosgd_moves_I_inter')}")
    print(f"  BoGrad observed I_between_K32 (its between-batch axis):    {ctr.get('bograd_I_between_observed')}")
    print("  Expectation: COSGD raises I_inter (inter-batch) while BoGrad acts on I_between.")
    print(f"\nwrote {_HERE/'master_table.json'} and {_HERE/'mechanism_contrast.json'}")


if __name__ == "__main__":
    main()
