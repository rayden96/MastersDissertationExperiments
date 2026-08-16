"""
40 — Synthesis (the dissertation payoff). Pure analysis; reads M3 bakeoff records
(and optionally M1/M2 summaries), trains nothing.

Two questions the whole program exists to answer:

  40.01  Does interference reduction PREDICT the accuracy gain?
         For each (dataset x base), compute, per method m vs its baseline:
            delta_acc      = acc(m) - acc(baseline)
            delta_I_inter  = I_inter(m) - I_inter(baseline)        (COSGD/GradDrop axis)
            delta_I_between= I_between_K32(m) - I_between_K32(baseline) (BoGrad axis)
         Then regress delta_acc on the axis-appropriate delta-interference across
         the suite. A positive, significant slope = the framework's metric is
         PREDICTIVE, not merely descriptive. Reported per method family + pooled.

  40.02  W' at scale: orthogonalisation helps IFF the removed direction is
         separable from descent. Using the bakeoff rows, relate each
         orthogonalising method's delta_acc to its mean pairwise-alignment /
         cosine signal: cells where buffered/averaged directions are mixed-sign
         (separable) should show gains; cells that are ~100% aligned with descent
         (inseparable) should show flat/negative gains. Reads the BoGrad mode
         findings (10.03) + COSGD inter cosine where available.

Inputs (auto-discovered, newest first):
  - 30_main_comparison/_core/results/run_*/all_rows.json   (primary)
  - falls back to per-cell cell_*.json
Outputs: synthesis_40_01.json/.png, synthesis_40_02.json/.png + a short
  findings.md.

Usage:
    python analyze.py
    python analyze.py --campaign /path/to/run_<id>
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json, write_json_atomic   # noqa: E402

ORTHO_METHODS = ("cosgd", "bograd", "graddrop")
# which interference axis each method primarily targets
AXIS_OF = {"cosgd": "I_inter_mean", "graddrop": "I_inter_mean", "bograd": "I_between_K32_mean"}


def _find_campaign(arg: Optional[str]) -> Optional[Path]:
    if arg:
        return Path(arg)
    root = _REPO / "PaperReadyExperiments" / "30_main_comparison" / "_core" / "results"
    runs = sorted(root.glob("run_*")) if root.exists() else []
    return runs[-1] if runs else None


def _load_rows(campaign: Optional[Path]) -> List[Dict[str, Any]]:
    """Aggregate the per-cell records, exactly as views.py does.

    all_rows.json must NOT be preferred: _bakeoff writes it as a per-session
    convenience snapshot and each session overwrites it, so on a campaign built
    across several sessions it holds only the last one. Reading it made this
    analysis see 20 cells of a 6-dataset campaign. The cell_*.json files never
    collide, so they are the authoritative source; all_rows.json is kept only as
    a fallback for a campaign that has no per-cell files.
    """
    if campaign is None:
        return []
    rows: List[Dict[str, Any]] = []
    for c in campaign.rglob("cell_*.json"):
        try:
            rows.extend(read_json(c).get("rows", []))
        except Exception:
            pass
    if rows:
        return rows
    allp = campaign / "all_rows.json"
    return read_json(allp) if allp.exists() else []


def _agg_cell(rows):
    """(dataset, base, method) -> mean of each numeric field across seeds."""
    g: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["base"], r["method"])].append(r)
    out = {}
    for k, rs in g.items():
        agg = {}
        for f in ("final_test_acc", "I_inter_mean", "inter_mean_cos_mean",
                  "I_between_K32_mean", "between_K32_mean_cos_mean",
                  "mean_deficit_per_step"):
            vals = [r.get(f) for r in rs if isinstance(r.get(f), (int, float)) and r.get(f) == r.get(f)]
            if vals:
                agg[f] = float(np.mean(vals))
        out[k] = agg
    return out


def _linfit(xs, ys) -> Dict[str, float]:
    if len(xs) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "n": len(xs)}
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.std() == 0 or ys.std() == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "n": len(xs)}
    slope, intercept = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "r": r, "n": len(xs)}


def analyze_predicts(cell_agg) -> Dict[str, Any]:
    """40.01 — delta_acc vs delta_interference, per method family + pooled."""
    per_method: Dict[str, Dict[str, List[float]]] = {m: {"dx": [], "dy": []} for m in ORTHO_METHODS}
    for (ds, base, method), agg in cell_agg.items():
        if method not in ORTHO_METHODS:
            continue
        base_agg = cell_agg.get((ds, base, "baseline"))
        if not base_agg:
            continue
        axis = AXIS_OF[method]
        if "final_test_acc" not in agg or "final_test_acc" not in base_agg:
            continue
        if axis not in agg or axis not in base_agg:
            continue
        dacc = agg["final_test_acc"] - base_agg["final_test_acc"]
        dI = agg[axis] - base_agg[axis]   # higher I = LESS cancellation (good)
        per_method[method]["dx"].append(dI)
        per_method[method]["dy"].append(dacc)

    fits = {m: _linfit(d["dx"], d["dy"]) for m, d in per_method.items()}
    pooled_dx = [v for d in per_method.values() for v in d["dx"]]
    pooled_dy = [v for d in per_method.values() for v in d["dy"]]
    fits["pooled"] = _linfit(pooled_dx, pooled_dy)
    return {"per_method": per_method, "fits": fits}


def analyze_w_prime(cell_agg) -> Dict[str, Any]:
    """40.02 — gain vs separability signal (inter cosine / between cosine).

    Hypothesis (W'): orthogonalisation helps when the directions it removes are
    separable from descent. Proxy separability signal: a cosine near/below 0
    (mixed-sign / anti-aligned, i.e. separable conflict) vs near +1 (aligned with
    descent, inseparable). Relate delta_acc to that cosine across cells.
    """
    pts = []
    for (ds, base, method), agg in cell_agg.items():
        if method not in ORTHO_METHODS:
            continue
        base_agg = cell_agg.get((ds, base, "baseline"))
        if not base_agg or "final_test_acc" not in agg or "final_test_acc" not in base_agg:
            continue
        dacc = agg["final_test_acc"] - base_agg["final_test_acc"]
        cos_key = "inter_mean_cos_mean" if method in ("cosgd", "graddrop") else "between_K32_mean_cos_mean"
        cos = base_agg.get(cos_key)   # the conflict structure the method faced
        if cos is None:
            continue
        pts.append({"dataset": ds, "base": base, "method": method,
                    "baseline_cos": cos, "delta_acc": dacc})
    xs = [p["baseline_cos"] for p in pts]
    ys = [p["delta_acc"] for p in pts]
    return {"points": pts, "fit": _linfit(xs, ys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None)
    args = ap.parse_args()

    campaign = _find_campaign(args.campaign)
    rows = _load_rows(campaign)
    if not rows:
        print("No bakeoff records found yet. Run 30_main_comparison/run.py first, "
              "then re-run this. (M4 is pure analysis over M3 outputs.)")
        # still write empty stubs so downstream tooling has a shape
        write_json_atomic(_HERE / "synthesis_40_01.json", {"status": "no_data"})
        write_json_atomic(_HERE / "synthesis_40_02.json", {"status": "no_data"})
        return

    cell_agg = _agg_cell(rows)
    print(f"Synthesis from {campaign} — {len(rows)} rows, {len(cell_agg)} (ds,base,method) cells")

    p = analyze_predicts(cell_agg)
    w = analyze_w_prime(cell_agg)
    write_json_atomic(_HERE / "synthesis_40_01.json", p)
    write_json_atomic(_HERE / "synthesis_40_02.json", w)

    print("\n=== 40.01  Does interference reduction predict accuracy gain? ===")
    print("  (slope>0 & r>0 => the framework metric is predictive)")
    for m, fit in p["fits"].items():
        print(f"  {m:<10} slope={fit['slope']:+.3f}  r={fit['r']:+.3f}  n={fit['n']}")

    print("\n=== 40.02  W' at scale (gain vs baseline conflict cosine) ===")
    f = w["fit"]
    print(f"  delta_acc vs baseline_cos: slope={f['slope']:+.3f} r={f['r']:+.3f} n={f['n']}")
    print("  (W' predicts: more anti-aligned/ mixed-sign baseline conflict -> bigger gain)")

    _write_findings(p, w)
    _maybe_plot(p, w)


def _write_findings(p, w):
    lines = ["# 40 — Synthesis findings\n",
             "\n## 40.01 Does interference reduction predict accuracy gain?\n",
             "| method | slope (Δacc per ΔI) | Pearson r | n cells |", "|---|---|---|---|"]
    for m, fit in p["fits"].items():
        lines.append(f"| {m} | {fit['slope']:+.3f} | {fit['r']:+.3f} | {fit['n']} |")
    lines += ["\n## 40.02 W' at scale\n",
              f"Δacc vs baseline conflict cosine: slope {w['fit']['slope']:+.3f}, "
              f"r {w['fit']['r']:+.3f}, n {w['fit']['n']}.\n",
              "W' predicts a negative slope (more anti-aligned baseline conflict -> larger gain).\n"]
    (_HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {_HERE/'findings.md'}")


def _maybe_plot(p, w):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sys.path.insert(0, str(_PRE))
        from common.plotting import apply_thesis_rcparams, PALETTE
        apply_thesis_rcparams()

        fig, ax = plt.subplots(figsize=(7, 5))
        for m in ORTHO_METHODS:
            d = p["per_method"][m]
            if d["dx"]:
                ax.scatter(d["dx"], d["dy"], label=m, s=50, alpha=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$\Delta$ interference index (method - baseline)")
        ax.set_ylabel(r"$\Delta$ test accuracy")
        ax.set_title("40.01  interference reduction vs accuracy gain")
        ax.legend()
        fig.savefig(_HERE / "synthesis_40_01.png", bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for pt in w["points"]:
            ax.scatter(pt["baseline_cos"], pt["delta_acc"], s=50, alpha=0.8,
                       color=PALETTE.get(pt["base"]))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("baseline conflict cosine (lower = more separable)")
        ax.set_ylabel(r"$\Delta$ test accuracy")
        ax.set_title("40.02  W' — gain vs conflict separability")
        fig.savefig(_HERE / "synthesis_40_02.png", bbox_inches="tight"); plt.close(fig)
        print(f"wrote {_HERE/'synthesis_40_01.png'} and synthesis_40_02.png")
    except Exception as e:
        print(f"(plots skipped: {e})")


if __name__ == "__main__":
    main()
