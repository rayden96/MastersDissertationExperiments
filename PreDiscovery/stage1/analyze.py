"""
Analyse and plot Stage 1 (3-method) comparison results.

Reads `results/<run_id>/{baseline,bograd,cosgd}/{result.json,
inter_batch_history.json, summary.json}` and produces:

  - acc_summary.png        : final accuracy bar chart
  - ib_axis.png            : IB_%neg per method (mean over training)
  - ww_axis.png            : WW_K per method
  - axes_scatter.png       : IB_%neg vs WW_K, point per method
  - ib_over_time.png       : IB_%neg trace, methods overlaid
  - summary.md             : markdown summary table

Usage
-----
    python stage1/analyze.py
        (analyses the most recent run)
    python stage1/analyze.py --run run_20260507_105741
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure stdout can handle unicode on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "stage1" / "results"

ORDER = ["baseline", "bograd", "cosgd"]
LABELS = {
    "baseline": "SGD+momentum baseline",
    "bograd": "+ BoGrad upd-K=32 neg",
    "cosgd": "COSGD GS",
}
COLOURS = {
    "baseline": "#1f77b4",
    "bograd": "#d62728",
    "cosgd": "#ff7f0e",
}


def find_latest_run() -> Optional[Path]:
    if not RESULTS_ROOT.exists():
        return None
    runs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()
                   and p.name.startswith("run_")])
    return runs[-1] if runs else None


def load_run(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        result_p = sub / "result.json"
        if not result_p.exists():
            continue
        with result_p.open() as fh:
            res = json.load(fh)
        ib_p = sub / "inter_batch_history.json"
        ibh: List[Dict[str, Any]] = []
        if ib_p.exists():
            with ib_p.open() as fh:
                ibh = json.load(fh)
        sm_p = sub / "summary.json"
        sm: Dict[str, Any] = {}
        if sm_p.exists():
            with sm_p.open() as fh:
                sm = json.load(fh)
        out[sub.name] = {
            "result": res,
            "inter_batch_history": ibh,
            "summary": sm,
        }
    return out


def order_keys(keys) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for k in ORDER:
        if k in keys and k not in seen:
            ordered.append(k)
            seen.add(k)
    for k in keys:
        if k not in seen:
            ordered.append(k)
    return ordered


def aggregate_metrics(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in order_keys(data.keys()):
        d = data[name]
        res = d["result"]
        ibh = d["inter_batch_history"]
        sm = d["summary"]
        non_empty = [r for r in ibh if r.get("n_pairs", 0) > 0]
        ib_pct_neg = sum(r["frac_negative"] for r in non_empty) / len(non_empty) \
            if non_empty else float("nan")
        ib_mean_cos = sum(r["mean_cos"] for r in non_empty) / len(non_empty) \
            if non_empty else float("nan")
        ww = sm.get("ww_wasted_work_ratio_mean", float("nan"))
        oob_total = res["forgetting"].get("out_of_batch_total_magnitude", float("nan"))
        rows.append({
            "name": name,
            "label": res.get("label", LABELS.get(name, name)),
            "final_acc": res.get("final_acc", float("nan")),
            "ib_pct_neg": ib_pct_neg,
            "ib_mean_cos": ib_mean_cos,
            "ww_k": ww,
            "oob_total": oob_total,
        })
    return rows


def plot_bar(rows, key, title, ylabel, out_path, *, baseline=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["name"] for r in rows]
    values = [r[key] for r in rows]
    colours = [COLOURS.get(n, "#bbbbbb") for n in names]
    labels = [LABELS.get(n, n) for n in names]
    bars = ax.bar(labels, values, color=colours, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, values):
        if v != v:
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if baseline is not None and baseline == baseline:
        ax.axhline(baseline, linestyle="--", color="black",
                   linewidth=1, alpha=0.4,
                   label=f"baseline = {baseline:.3f}")
        ax.legend(loc="best", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_axes_scatter(rows, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in rows:
        x = r["ib_pct_neg"]
        y = r["ww_k"]
        if x != x or y != y:
            continue
        c = COLOURS.get(r["name"], "#bbbbbb")
        ax.scatter(x, y, s=200, c=c, edgecolor="black", linewidths=0.7, zorder=3)
        ax.annotate(
            f"{LABELS.get(r['name'], r['name'])}\nacc={r['final_acc']:.3f}",
            (x, y), xytext=(8, 8), textcoords="offset points",
            fontsize=9, ha="left", va="bottom",
        )
    ax.set_xlabel("IB_%neg  (inter-batch class-pair conflict; lower = COSGD-like)")
    ax.set_ylabel("WW_K  (between-batch trajectory efficiency; higher = BoGrad-like)")
    ax.set_title("Two interference axes — Stage 1 headline")
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ib_over_time(data, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in order_keys(data.keys()):
        ibh = data[name]["inter_batch_history"]
        non_empty = [r for r in ibh if r.get("n_pairs", 0) > 0]
        if not non_empty:
            continue
        steps = [r["step"] for r in non_empty]
        vals = [r["frac_negative"] for r in non_empty]
        ax.plot(steps, vals, label=LABELS.get(name, name),
                color=COLOURS.get(name, "#bbbbbb"),
                linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("IB_%neg (frac. of class-pair gradient cosines < 0)")
    ax.set_title("Inter-batch interference over training")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary_md(rows, run_dir):
    md = ["# Stage 1 — three-method comparison summary",
          "",
          f"_Run directory: `{run_dir.relative_to(ROOT)}`_",
          ""]
    md += ["| method | final_acc | IB_%neg | IB_<cos> | WW_K | OOB_total |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| **{r['label']}** | {r['final_acc']:.4f} | {r['ib_pct_neg']:.3f} "
            f"| {r['ib_mean_cos']:+.3f} | {r['ww_k']:.3f} | {r['oob_total']:.3f} |"
        )
    md += ["",
           "## Reading guide",
           "",
           "- **IB_%neg**: fraction of class-pair gradient cosines < 0. "
           "Lower = less inter-batch class conflict (COSGD target).",
           "- **IB_<cos>**: mean class-pair cosine. Sign-flip across "
           "methods is the headline finding.",
           "- **WW_K**: wasted-work ratio over K=32 step window. Higher "
           "= better trajectory efficiency (BoGrad target).",
           "- **OOB_total**: structurally 0 in standard CIFAR-10 batching."]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--run-path", type=str, default=None)
    args = ap.parse_args()

    if args.run_path:
        run_dir = Path(args.run_path)
    elif args.run:
        run_dir = RESULTS_ROOT / args.run
    else:
        latest = find_latest_run()
        if latest is None:
            print("No run directories found.", file=sys.stderr)
            sys.exit(1)
        run_dir = latest

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {run_dir}")
    data = load_run(run_dir)
    if not data:
        print("No variant subfolders found.", file=sys.stderr)
        sys.exit(1)

    rows = aggregate_metrics(data)
    baseline_acc = next((r["final_acc"] for r in rows
                         if r["name"] == "baseline"), None)
    baseline_ib = next((r["ib_pct_neg"] for r in rows
                        if r["name"] == "baseline"), None)
    baseline_ww = next((r["ww_k"] for r in rows
                        if r["name"] == "baseline"), None)

    print("\nMethods loaded:")
    for r in rows:
        print(f"  {r['name']:14s}  acc={r['final_acc']:.4f}  "
              f"IB%neg={r['ib_pct_neg']:.3f}  WW_K={r['ww_k']:.3f}")

    plot_bar(rows, "final_acc", "Final test accuracy",
             "test acc", run_dir / "acc_summary.png", baseline=baseline_acc)
    plot_bar(rows, "ib_pct_neg", "Inter-batch interference",
             "IB_%neg", run_dir / "ib_axis.png", baseline=baseline_ib)
    plot_bar(rows, "ww_k", "Wasted-work ratio (K=32)",
             "WW_K", run_dir / "ww_axis.png", baseline=baseline_ww)
    plot_axes_scatter(rows, run_dir / "axes_scatter.png")
    plot_ib_over_time(data, run_dir / "ib_over_time.png")
    write_summary_md(rows, run_dir)

    print(f"\nWrote plots and summary.md to {run_dir}")


if __name__ == "__main__":
    main()
