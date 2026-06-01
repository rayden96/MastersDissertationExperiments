"""
30 — bakeoff VIEWS (pure presentation; reads the canonical record set, never trains).

Produces, from a bakeoff campaign's all_rows.json (or per-cell cell_*.json):

  30.01  test-accuracy trajectories per dataset  (line plot per dataset; one line
         per (base x method), mean +/- std band across seeds)
  30.02  fixed-budget accuracy tables            (acc at 10/50/100% of epochs;
         rows = dataset x base, cols = method; mean +/- std; markdown + json)
  30.03  final-accuracy summary bars             (grouped bars per dataset)
  30.09  accuracy-vs-wall-clock Pareto           (scatter per dataset; colour=base,
         marker=method; Pareto frontier overlaid)

Usage:
    python views.py --campaign _core/results/run_<id>            # all views
    python views.py --campaign ... --only 01 03
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json, write_json_atomic   # noqa: E402
from common.plotting import apply_thesis_rcparams, PALETTE, METHOD_STYLE  # noqa: E402

METHOD_ORDER = ["baseline", "cosgd", "bograd", "graddrop", "dropout"]


def _resolve_campaign(arg: Optional[str]) -> Path:
    root = _HERE / "_core" / "results"
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else (_HERE / arg)
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise SystemExit(f"No bakeoff campaign under {root} (run.py first)")
    return runs[-1]


def _load_rows(campaign: Path) -> List[Dict[str, Any]]:
    allp = campaign / "all_rows.json"
    if allp.exists():
        return read_json(allp)
    rows: List[Dict[str, Any]] = []
    for cell in campaign.rglob("cell_*.json"):
        rows.extend(read_json(cell).get("rows", []))
    return rows


def _group(rows):
    """(dataset) -> (base, method) -> list of seed rows."""
    g: Dict[str, Dict[tuple, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[r["dataset"]][(r["base"], r["method"])].append(r)
    return g


def _mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)


# ---- 30.01 trajectories ----------------------------------------------------
def view_trajectories(campaign, grouped):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")
    for dataset, cells in grouped.items():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (base, method), rs in sorted(cells.items()):
            curves = [r.get("epoch_test_acc", []) for r in rs if r.get("epoch_test_acc")]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            M = np.stack([np.asarray(c[:n]) for c in curves], 0)
            x = np.arange(1, n + 1)
            mean, std = M.mean(0), M.std(0)
            ax.plot(x, mean, color=PALETTE.get(base), linestyle=METHOD_STYLE.get(method),
                    label=f"{base}+{method}")
            ax.fill_between(x, mean - std, mean + std, color=PALETTE.get(base), alpha=0.12)
        ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
        ax.set_title(f"30.01  {dataset} — test-accuracy trajectories")
        ax.legend(fontsize=7, ncol=2)
        out = campaign / f"30_01_trajectories_{dataset}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


# ---- 30.02 fixed-budget tables --------------------------------------------
def view_budget_tables(campaign, grouped):
    fractions = {"10pct": 0.1, "50pct": 0.5, "100pct": 1.0}
    out_md = ["# 30.02 Fixed-budget accuracy tables\n"]
    out_json: Dict[str, Any] = {}
    for frac_name, frac in fractions.items():
        out_md.append(f"\n## At {frac_name} of epochs\n")
        for dataset, cells in sorted(grouped.items()):
            bases = sorted({b for b, _ in cells})
            out_md.append(f"\n**{dataset}**\n")
            out_md.append("| base | " + " | ".join(METHOD_ORDER) + " |")
            out_md.append("|" + "---|" * (len(METHOD_ORDER) + 1))
            for base in bases:
                cellvals = []
                for method in METHOD_ORDER:
                    rs = cells.get((base, method), [])
                    accs = []
                    for r in rs:
                        c = r.get("epoch_test_acc", [])
                        if c:
                            idx = max(0, int(round(frac * len(c))) - 1)
                            accs.append(c[idx])
                    m, s, n = _mean_std(accs)
                    cellvals.append((method, m, s, n))
                    out_json[f"{frac_name}/{dataset}/{base}/{method}"] = {"mean": m, "std": s, "n": n}
                best = max((v for v in cellvals if v[1] == v[1]), key=lambda v: v[1], default=None)
                cells_str = []
                for method, m, s, n in cellvals:
                    txt = f"{m:.3f}±{s:.3f}" if n else "—"
                    if best and method == best[0] and n:
                        txt = f"**{txt}**"
                    cells_str.append(txt)
                out_md.append(f"| {base} | " + " | ".join(cells_str) + " |")
    (campaign / "30_02_budget_tables.md").write_text("\n".join(out_md), encoding="utf-8")
    write_json_atomic(campaign / "30_02_budget_tables.json", out_json)
    print(f"  wrote {campaign/'30_02_budget_tables.md'}")


# ---- 30.03 final-accuracy bars --------------------------------------------
def view_final_bars(campaign, grouped):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")
    for dataset, cells in grouped.items():
        bases = sorted({b for b, _ in cells})
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.15
        x = np.arange(len(bases))
        for mi, method in enumerate(METHOD_ORDER):
            means, stds = [], []
            for base in bases:
                m, s, _ = _mean_std([r.get("final_test_acc") for r in cells.get((base, method), [])])
                means.append(m); stds.append(s)
            ax.bar(x + (mi - 2) * width, means, width, yerr=stds, capsize=2,
                   label=method, linestyle=METHOD_STYLE.get(method))
        ax.set_xticks(x); ax.set_xticklabels(bases)
        ax.set_ylabel("final test accuracy"); ax.set_title(f"30.03  {dataset} — final accuracy")
        ax.legend(fontsize=8, ncol=3)
        out = campaign / f"30_03_final_bars_{dataset}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


# ---- 30.09 accuracy vs wall-clock Pareto ----------------------------------
def view_pareto(campaign, grouped):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")
    markers = {"baseline": "o", "cosgd": "s", "bograd": "^", "graddrop": "D", "dropout": "v"}
    for dataset, cells in grouped.items():
        pts = []
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (base, method), rs in cells.items():
            acc, _, _ = _mean_std([r.get("final_test_acc") for r in rs])
            spt, _, _ = _mean_std([r.get("mean_step_wall_time_s") for r in rs])
            if acc != acc or spt != spt:
                continue
            pts.append((spt, acc))
            ax.scatter(spt, acc, color=PALETTE.get(base), marker=markers.get(method, "o"),
                       s=70, edgecolor="black", linewidth=0.4, label=f"{base}+{method}")
        # Pareto frontier (max acc for min time)
        if pts:
            pts_sorted = sorted(pts)
            frontier, best_acc = [], -1
            for spt, acc in pts_sorted:
                if acc > best_acc:
                    frontier.append((spt, acc)); best_acc = acc
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, "k--", alpha=0.5, label="Pareto frontier")
        ax.set_xscale("log"); ax.set_xlabel("sec / step (log)"); ax.set_ylabel("final test accuracy")
        ax.set_title(f"30.09  {dataset} — accuracy vs wall-clock")
        ax.legend(fontsize=6, ncol=2)
        out = campaign / f"30_09_pareto_{dataset}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--only", nargs="+", default=["01", "02", "03", "09"])
    args = ap.parse_args()

    campaign = _resolve_campaign(args.campaign)
    rows = _load_rows(campaign)
    if not rows:
        raise SystemExit(f"No rows in {campaign}")
    grouped = _group(rows)
    print(f"Views from {campaign} — {len(rows)} rows, {len(grouped)} datasets")

    if "01" in args.only: view_trajectories(campaign, grouped)
    if "02" in args.only: view_budget_tables(campaign, grouped)
    if "03" in args.only: view_final_bars(campaign, grouped)
    if "09" in args.only: view_pareto(campaign, grouped)


if __name__ == "__main__":
    main()
