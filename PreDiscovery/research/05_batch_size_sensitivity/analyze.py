"""
Analyse and plot results from research/05_batch_size_sensitivity/run_sensitivity.py.

Reads `results/<run_id>/batch_<NNNN>/{result.json, inter_batch_history.json,
summary.json}` for every batch-size present, builds the comparison
table and the IB-vs-batch and WW-vs-batch plots.

Usage
-----
    python research/05_batch_size_sensitivity/analyze.py
        (analyses the most recent run)
    python research/05_batch_size_sensitivity/analyze.py --run run_20260507_140000
"""

from __future__ import annotations

import argparse
import json
import re
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

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = ROOT / "research" / "05_batch_size_sensitivity" / "results"

_RX = re.compile(r"^batch_(\d+)$")


def find_latest_run() -> Optional[Path]:
    if not RESULTS_ROOT.exists():
        return None
    runs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()
                   and p.name.startswith("run_")])
    return runs[-1] if runs else None


def load_run(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        m = _RX.match(sub.name)
        if not m:
            continue
        bs = int(m.group(1))
        result_p = sub / "result.json"
        if not result_p.exists():
            continue
        with result_p.open() as fh:
            res = json.load(fh)
        ibh: List[Dict[str, Any]] = []
        ib_p = sub / "inter_batch_history.json"
        if ib_p.exists():
            with ib_p.open() as fh:
                ibh = json.load(fh)
        sm: Dict[str, Any] = {}
        sm_p = sub / "summary.json"
        if sm_p.exists():
            with sm_p.open() as fh:
                sm = json.load(fh)
        out[bs] = {"result": res, "inter_batch_history": ibh, "summary": sm}
    return out


def aggregate_metrics(data: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bs in sorted(data.keys()):
        d = data[bs]
        res = d["result"]
        ibh = d["inter_batch_history"]
        summary = d["summary"]
        non_empty = [r for r in ibh if r.get("n_pairs", 0) > 0]
        ib_pct_neg = sum(r["frac_negative"] for r in non_empty) / len(non_empty) \
            if non_empty else float("nan")
        ib_mean_cos = sum(r["mean_cos"] for r in non_empty) / len(non_empty) \
            if non_empty else float("nan")
        ww = summary.get("ww_wasted_work_ratio_mean", float("nan"))
        rows.append({
            "batch_size": bs,
            "lr": res.get("lr", float("nan")),
            "final_acc": res.get("final_acc", float("nan")),
            "ib_pct_neg": ib_pct_neg,
            "ib_mean_cos": ib_mean_cos,
            "ww_k": ww,
            "n_steps": res.get("n_steps", 0),
        })
    return rows


def plot_axis_vs_batch(rows: List[Dict[str, Any]], key: str, ylabel: str,
                       title: str, out_path: Path, log_x: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [r["batch_size"] for r in rows]
    ys = [r[key] for r in rows]
    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.4)
    for x, y in zip(xs, ys):
        if y != y:
            continue
        ax.annotate(f"{y:.3f}", (x, y), xytext=(4, 6),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ib_over_time(data: Dict[int, Dict[str, Any]], out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("viridis")
    keys = sorted(data.keys())
    for i, bs in enumerate(keys):
        ibh = data[bs]["inter_batch_history"]
        non_empty = [r for r in ibh if r.get("n_pairs", 0) > 0]
        if not non_empty:
            continue
        steps = [r["step"] for r in non_empty]
        vals = [r["frac_negative"] for r in non_empty]
        c = cmap(i / max(len(keys) - 1, 1))
        ax.plot(steps, vals, label=f"batch={bs}", color=c, linewidth=1.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("IB_%neg")
    ax.set_title("Inter-batch interference over training, by batch size")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary_md(rows: List[Dict[str, Any]], run_dir: Path):
    md = ["# Batch-size sensitivity — summary",
          "",
          f"_Run directory: `{run_dir.relative_to(ROOT)}`_",
          ""]
    md += ["| batch | lr | final_acc | IB_%neg | IB_<cos> | WW_K | n_steps |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['batch_size']} | {r['lr']:.4f} | {r['final_acc']:.4f} "
            f"| {r['ib_pct_neg']:.3f} | {r['ib_mean_cos']:+.3f} "
            f"| {r['ww_k']:.3f} | {r['n_steps']} |"
        )
    md += ["",
           "## Reading guide",
           "",
           "- LR scaled linearly: lr = lr_base × batch / 128.",
           "- IB_%neg: averaged over inter-batch measurements during training.",
           "- WW_K: wasted-work ratio over a 32-step window."]
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
        print("No batch_NNNN/ subfolders found.", file=sys.stderr)
        sys.exit(1)

    rows = aggregate_metrics(data)
    for r in rows:
        print(f"  batch={r['batch_size']:>5d}  lr={r['lr']:.4f}  "
              f"acc={r['final_acc']:.4f}  IB%neg={r['ib_pct_neg']:.3f}  "
              f"WW_K={r['ww_k']:.3f}")

    plot_axis_vs_batch(rows, "ib_pct_neg", "IB_%neg",
                       "Inter-batch interference vs batch size",
                       run_dir / "ib_vs_batch.png")
    plot_axis_vs_batch(rows, "ww_k", "WW_K",
                       "Wasted-work ratio (K=32) vs batch size",
                       run_dir / "ww_vs_batch.png")
    plot_axis_vs_batch(rows, "final_acc", "test acc",
                       "Final test accuracy vs batch size",
                       run_dir / "acc_vs_batch.png")
    plot_ib_over_time(data, run_dir / "ib_over_time.png")
    write_summary_md(rows, run_dir)

    print(f"\nWrote plots and summary.md to {run_dir}")


if __name__ == "__main__":
    main()
