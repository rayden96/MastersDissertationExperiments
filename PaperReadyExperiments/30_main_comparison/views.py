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
    main = root / "main"
    if main.exists():
        return main
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise SystemExit(f"No bakeoff campaign under {root} (run.py first)")
    return runs[-1]


def _load_rows(campaign: Path) -> List[Dict[str, Any]]:
    # Aggregate the per-cell files first: they never collide across concurrent
    # per-dataset sessions, whereas a campaign-level all_rows.json is overwritten
    # per session. Fall back to all_rows.json only if no cell files exist.
    rows: List[Dict[str, Any]] = []
    for cell in sorted(campaign.rglob("cell_*.json")):
        rows.extend(read_json(cell).get("rows", []))
    if rows:
        return rows
    allp = campaign / "all_rows.json"
    return read_json(allp) if allp.exists() else []


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


# ---- 30.00 CONVERGENCE SPEED-UP (the headline metric) ---------------------
def _epochs_to(curve, target):
    """First (1-based) epoch at which curve reaches `target`, or None."""
    for i, a in enumerate(curve):
        if a is not None and a == a and a >= target:
            return i + 1
    return None


def view_speedup(campaign, grouped, target_frac=1.0):
    """For each (dataset, base): how many epochs each method needs to reach the
    BASELINE's target accuracy, and the speed-up factor (baseline_epochs /
    method_epochs). This is the primary 'does it train faster' result.

    target = `target_frac` x baseline's final mean accuracy (1.0 = match baseline's
    end-of-training accuracy; e.g. 0.95 = reach 95% of it). Also reports epoch-1
    accuracy (early-progress) and the steps-to-target speed-up via mean step time.
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")

    out_md = [f"# 30.00 Convergence speed-up (target = {target_frac:g}x baseline final acc)\n",
              "Speed-up = baseline_epochs_to_target / method_epochs_to_target. "
              ">1 means faster. 'wall speed-up' also folds in per-step cost.\n"]
    out_json: Dict[str, Any] = {}

    for dataset, cells in sorted(grouped.items()):
        bases = sorted({b for b, _ in cells})
        out_md.append(f"\n## {dataset}\n")
        out_md.append("| base | metric | " + " | ".join(METHOD_ORDER) + " |")
        out_md.append("|" + "---|" * (len(METHOD_ORDER) + 2))
        # bar fig: speed-up per (base, method)
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.15; x = np.arange(len(bases))
        for dataset_method_i, method in enumerate(METHOD_ORDER):
            speedups = []
            for base in bases:
                base_rows = cells.get((base, "baseline"), [])
                meth_rows = cells.get((base, method), [])
                base_curves = [r.get("epoch_test_acc", []) for r in base_rows if r.get("epoch_test_acc")]
                meth_curves = [r.get("epoch_test_acc", []) for r in meth_rows if r.get("epoch_test_acc")]
                if not base_curves or not meth_curves:
                    speedups.append(np.nan); continue
                bmean = np.mean([c[-1] for c in base_curves])
                tgt = target_frac * bmean
                b_ep = np.mean([_epochs_to(c, tgt) or len(c) + 1 for c in base_curves])
                m_ep = np.mean([_epochs_to(c, tgt) or len(c) + 1 for c in meth_curves])
                sp = b_ep / m_ep if m_ep > 0 else np.nan
                speedups.append(sp)
                # wall speed-up: fold in per-step cost
                b_spt = _mean_std([r.get("mean_step_wall_time_s") for r in base_rows])[0]
                m_spt = _mean_std([r.get("mean_step_wall_time_s") for r in meth_rows])[0]
                wall_sp = sp * (b_spt / m_spt) if (m_spt and b_spt and m_spt == m_spt) else None
                out_json[f"{dataset}/{base}/{method}"] = {
                    "epochs_to_target": float(m_ep), "baseline_epochs": float(b_ep),
                    "epoch_speedup": float(sp) if sp == sp else None,
                    "wall_speedup": float(wall_sp) if wall_sp else None,
                    "epoch1_acc": float(np.mean([c[0] for c in meth_curves])),
                }
            ax.bar(x + (dataset_method_i - 2) * width, [s if s == s else 0 for s in speedups],
                   width, label=method, linestyle=METHOD_STYLE.get(method))
        ax.axhline(1.0, color="black", lw=0.8, ls="--", label="baseline (1.0x)")
        ax.set_xticks(x); ax.set_xticklabels(bases)
        ax.set_ylabel("epoch speed-up vs baseline"); ax.set_title(f"30.00  {dataset} — convergence speed-up")
        ax.legend(fontsize=8, ncol=3)
        out = campaign / f"30_00_speedup_{dataset}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")
        # markdown rows: epoch_speedup + epoch1_acc
        for base in bases:
            for metric_key, label in [("epoch_speedup", "epoch speed-up"), ("epoch1_acc", "epoch-1 acc")]:
                cellstr = []
                for method in METHOD_ORDER:
                    d = out_json.get(f"{dataset}/{base}/{method}", {})
                    v = d.get(metric_key)
                    if v is None:
                        cellstr.append("—")
                    elif metric_key == "epoch_speedup":
                        cellstr.append(f"**{v:.2f}x**" if v > 1.01 else f"{v:.2f}x")
                    else:
                        cellstr.append(f"{v:.3f}")
                out_md.append(f"| {base} | {label} | " + " | ".join(cellstr) + " |")

    (campaign / "30_00_speedup.md").write_text("\n".join(out_md), encoding="utf-8")
    write_json_atomic(campaign / "30_00_speedup.json", out_json)
    print(f"  wrote {campaign/'30_00_speedup.md'}")


# ---- 30.10 FOGO-style master results table --------------------------------
def _fmt(m, s, nd=3):
    if m is None or m != m:
        return "—"
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}" if (s == s and s > 0) else f"{m:.{nd}f}"


def _acc_at(rs, frac):
    vals = []
    for r in rs:
        c = r.get("epoch_test_acc", [])
        if c:
            idx = 0 if frac <= 0 else max(0, int(round(frac * len(c))) - 1)
            vals.append(c[idx])
    return _mean_std(vals)


def _latex_cell(c: str) -> str:
    """Escape one table cell for LaTeX. _fmt already emits $\\pm$; here we fix
    percent signs and turn a trailing speed-up 'x' into $\\times$."""
    c = c.replace("%", r"\%").replace("→", r"$\to$")
    body = c[:-1].replace(".", "").replace("-", "")
    if c.endswith("x") and body.isdigit():
        c = c[:-1] + r"$\times$"
    return c


def _write_fogo_latex(campaign, dataset, header, rows):
    safe_ds = dataset.replace("_", r"\_")
    spec = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
             r"  \begin{tabular}{" + spec + "}", r"  \hline",
             "  " + " & ".join(_latex_cell(h) for h in header) + r" \\",
             r"  \hline"]
    for row in rows:
        cells = [(c.replace("_", r"\_") if i == 0 else _latex_cell(c)) for i, c in enumerate(row)]
        lines.append("  " + " & ".join(cells) + r" \\")
    lines += [r"  \hline", r"  \end{tabular}",
              rf"  \caption{{Main comparison on {safe_ds}: per-method cost and "
              r"convergence (mean $\pm$ std across seeds). PeakMem is the clean "
              r"training peak GPU; Buf is the persistent BOGrad buffer; step is the "
              r"per-step training time; ep$\to$base is epochs to the baseline's "
              r"final accuracy; speed-up is baseline/method epochs.}",
              rf"  \label{{tab:experiments:fogo_{dataset}}}", r"\end{table}"]
    (campaign / f"30_10_fogo_table_{dataset}.tex").write_text("\n".join(lines), encoding="utf-8")


def view_fogo_table(campaign, grouped):
    """30.10 — the headline FOGO-style table. Per (dataset x base x method),
    mean +/- std across seeds of: param count, clean peak training memory,
    persistent buffer memory, clean per-step time, total wall-clock, early and
    final accuracy, epochs-to-baseline, and convergence speed-up. Markdown + JSON
    + a drop-in LaTeX table per dataset."""
    header = ["base+method", "Params(M)", "PeakMem(MB)", "Buf(MB)", "step(ms)",
              "Time(min)", "acc@1", "acc@50%", "best acc", "final acc", "ep→base", "speed-up"]
    out_md = ["# 30.10 FOGO-style results table\n",
              "Mean$\\pm$std across seeds. PeakMem = clean training peak GPU; "
              "Buf = persistent BOGrad buffer (K x params x 4B); step = clean "
              "per-step training time; Time = total wall-clock; ep->base = epochs "
              "to reach the base's baseline final accuracy; speed-up = "
              "baseline_epochs / method_epochs.\n"]
    out_json: Dict[str, Any] = {}

    for dataset, cells in sorted(grouped.items()):
        bases = sorted({b for b, _ in cells})
        out_md.append(f"\n## {dataset}\n")
        out_md.append("| " + " | ".join(header) + " |")
        out_md.append("|" + "---|" * len(header))
        latex_rows: List[List[str]] = []
        for base in bases:
            base_rows = cells.get((base, "baseline"), [])
            base_curves = [r.get("epoch_test_acc", []) for r in base_rows if r.get("epoch_test_acc")]
            base_final = float(np.mean([c[-1] for c in base_curves])) if base_curves else float("nan")
            b_ep = (float(np.mean([_epochs_to(c, base_final) or len(c) + 1 for c in base_curves]))
                    if base_curves else float("nan"))
            for method in METHOD_ORDER:
                rs = cells.get((base, method), [])
                if not rs:
                    continue
                pm = _mean_std([r.get("n_params") for r in rs])
                peak = _mean_std([r.get("peak_mem_mb") for r in rs])
                step = _mean_std([(r.get("mean_train_step_s") or r.get("mean_step_wall_time_s"))
                                  for r in rs])
                tmin = _mean_std([(r.get("total_wall_time_s") or float("nan")) / 60.0 for r in rs])
                best = _mean_std([r.get("best_test_acc") for r in rs])
                final = _mean_std([r.get("final_test_acc") for r in rs])
                a1 = _acc_at(rs, 0.0)
                a50 = _acc_at(rs, 0.5)
                m_eps = [(_epochs_to(c, base_final) or len(c) + 1)
                         for c in (r.get("epoch_test_acc", []) for r in rs)
                         if c and base_final == base_final]
                m_ep = float(np.mean(m_eps)) if m_eps else float("nan")
                speed = (b_ep / m_ep) if (m_ep and m_ep == m_ep and b_ep == b_ep and m_ep > 0) else float("nan")
                buf = 0.0
                if method == "bograd":
                    K = (rs[0].get("hp") or {}).get("K")
                    if K and pm[0] == pm[0]:
                        buf = K * pm[0] * 4 / 1e6
                params_M = pm[0] / 1e6 if pm[0] == pm[0] else float("nan")
                step_ms = (step[0] * 1000, step[1] * 1000) if step[0] == step[0] else (float("nan"), 0.0)
                row = [
                    f"{base}+{method}",
                    f"{params_M:.2f}" if params_M == params_M else "—",
                    _fmt(peak[0], peak[1], 1),
                    f"{buf:.1f}" if buf else "0",
                    _fmt(step_ms[0], step_ms[1], 2),
                    _fmt(tmin[0], tmin[1], 1),
                    _fmt(a1[0], a1[1]),
                    _fmt(a50[0], a50[1]),
                    _fmt(best[0], best[1]),
                    _fmt(final[0], final[1]),
                    f"{m_ep:.1f}" if m_ep == m_ep else "—",
                    f"{speed:.2f}x" if speed == speed else "—",
                ]
                out_md.append("| " + " | ".join(row) + " |")
                latex_rows.append(row)
                spe = _mean_std([(r.get("total_steps") or 0) / (r.get("total_epochs") or 1)
                                 for r in rs])[0]
                steps_to_base = (m_ep * spe) if (m_ep == m_ep and spe == spe) else float("nan")
                out_json[f"{dataset}/{base}/{method}"] = {
                    "params_M": params_M, "peak_mem_mb": peak[0], "buf_mb": buf,
                    "step_ms": step_ms[0], "time_min": tmin[0], "acc_ep1": a1[0],
                    "acc_50pct": a50[0], "best_acc": best[0], "final_acc": final[0],
                    "epochs_to_base": m_ep, "steps_per_epoch": spe,
                    "steps_to_base": steps_to_base,
                    "epoch_speedup": speed, "n_seeds": final[2],
                }
        _write_fogo_latex(campaign, dataset, header, latex_rows)
    (campaign / "30_10_fogo_table.md").write_text("\n".join(out_md), encoding="utf-8")
    write_json_atomic(campaign / "30_10_fogo_table.json", out_json)
    print(f"  wrote {campaign/'30_10_fogo_table.md'} (+ per-dataset .tex)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--only", nargs="+", default=["00", "01", "02", "03", "09", "10"])
    ap.add_argument("--target-frac", type=float, default=1.0,
                    help="speed-up target as fraction of baseline final acc (e.g. 0.95)")
    args = ap.parse_args()

    campaign = _resolve_campaign(args.campaign)
    rows = _load_rows(campaign)
    if not rows:
        raise SystemExit(f"No rows in {campaign}")
    grouped = _group(rows)
    print(f"Views from {campaign} — {len(rows)} rows, {len(grouped)} datasets")

    if "00" in args.only: view_speedup(campaign, grouped, args.target_frac)
    if "01" in args.only: view_trajectories(campaign, grouped)
    if "02" in args.only: view_budget_tables(campaign, grouped)
    if "03" in args.only: view_final_bars(campaign, grouped)
    if "09" in args.only: view_pareto(campaign, grouped)
    if "10" in args.only: view_fogo_table(campaign, grouped)


if __name__ == "__main__":
    main()
