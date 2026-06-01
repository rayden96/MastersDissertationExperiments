"""
Analyse and plot results from research/04_implicit_comparisons/run_comparison.py.

Reads `results/<run_id>/<variant>/{result.json, inter_batch_history.json,
history.json, summary.json}` for every variant present, builds a comparison
table and a small set of plots that surface the two interference axes.

Plots produced (saved as PNG into the run directory):
  - acc_summary.png             : final accuracy bar chart, methods sorted
  - ib_axis.png                 : IB_%neg per method (mean over training)
  - ww_axis.png                 : WW_K per method
  - axes_scatter.png            : IB_%neg vs WW_K, point per method, label
                                  with accuracy
  - ib_over_time.png            : IB_%neg trace per measurement step,
                                  every method overlaid

Also writes a `summary.md` markdown table.

Usage
-----
    python research/04_implicit_comparisons/analyze.py
        (analyses the most recent run)
    python research/04_implicit_comparisons/analyze.py --run run_20260507_133012
    python research/04_implicit_comparisons/analyze.py --run-path /path/to/run
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

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = ROOT / "research" / "04_implicit_comparisons" / "results"


# Stable display order. Method names not in this list are appended at the end.
ORDER = [
    "sgd_vanilla",
    "sgd_momentum",
    "sgd_mom_dropout01",
    "sgd_mom_dropout03",
    "sgd_mom_clip1",
    "sgd_mom_clip5",
    "adam",
    "adam_lr05",
    "bograd",
    "cosgd",
]

# Pretty labels and colours for plots.
LABELS = {
    "sgd_vanilla": "SGD vanilla",
    "sgd_momentum": "SGD+momentum",
    "sgd_mom_dropout01": "SGD+mom +Drop(0.1)",
    "sgd_mom_dropout03": "SGD+mom +Drop(0.3)",
    "sgd_mom_clip1": "SGD+mom +Clip(1.0)",
    "sgd_mom_clip5": "SGD+mom +Clip(5.0)",
    "adam": "Adam (1e-3)",
    "adam_lr05": "Adam (0.05)",
    "bograd": "+BoGrad K=32 neg",
    "cosgd": "COSGD GS",
}

COLOURS = {
    "sgd_vanilla": "#7f7f7f",
    "sgd_momentum": "#1f77b4",
    "sgd_mom_dropout01": "#aec7e8",
    "sgd_mom_dropout03": "#5778a4",
    "sgd_mom_clip1": "#9467bd",
    "sgd_mom_clip5": "#c5b0d5",
    "adam": "#2ca02c",
    "adam_lr05": "#98df8a",
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
    """Load all variant subfolders into a dict keyed by variant name."""
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
        ib_records: List[Dict[str, Any]] = []
        if ib_p.exists():
            with ib_p.open() as fh:
                ib_records = json.load(fh)
        summary_p = sub / "summary.json"
        summary: Dict[str, Any] = {}
        if summary_p.exists():
            with summary_p.open() as fh:
                summary = json.load(fh)
        out[sub.name] = {
            "result": res,
            "inter_batch_history": ib_records,
            "summary": summary,
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
        summary = d["summary"]
        non_empty = [r for r in ibh if r.get("n_pairs", 0) > 0]
        ib_pct_neg = sum(r["frac_negative"] for r in non_empty) / len(non_empty) \
            if non_empty else float("nan")
        ib_mean_cos = sum(r["mean_cos"] for r in non_empty) / len(non_empty) \
            if non_empty else float("nan")
        ww = summary.get("ww_wasted_work_ratio_mean", float("nan"))
        oob_total = res["forgetting"].get("out_of_batch_total_magnitude", float("nan"))
        rows.append({
            "name": name,
            "label": res.get("label", LABELS.get(name, name)),
            "final_acc": res.get("final_acc", float("nan")),
            "ib_pct_neg": ib_pct_neg,
            "ib_mean_cos": ib_mean_cos,
            "ww_k": ww,
            "oob_total": oob_total,
            "wall_clock_s": res.get("wall_clock_s", float("nan")),
            "lag1": summary.get("cos_u_lag1_mean", float("nan")),
            "lag4": summary.get("cos_u_lag4_mean", float("nan")),
            "lag16": summary.get("cos_u_lag16_mean", float("nan")),
            "pw_grad_pos": summary.get("pairwise_grad_frac_positive_mean", float("nan")),
            "pw_update_pos": summary.get("pairwise_update_frac_positive_mean", float("nan")),
        })
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_bar(rows: List[Dict[str, Any]], key: str, title: str,
             ylabel: str, out_path: Path, *, baseline: Optional[float] = None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [r["name"] for r in rows]
    values = [r[key] for r in rows]
    colours = [COLOURS.get(n, "#bbbbbb") for n in names]
    labels = [LABELS.get(n, n) for n in names]
    bars = ax.bar(labels, values, color=colours, edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, values):
        if v != v:  # NaN
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    if baseline is not None and baseline == baseline:
        ax.axhline(baseline, linestyle="--", color="black",
                   linewidth=1, alpha=0.4,
                   label=f"baseline = {baseline:.3f}")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_axes_scatter(rows: List[Dict[str, Any]], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        x = r["ib_pct_neg"]
        y = r["ww_k"]
        if x != x or y != y:
            continue
        c = COLOURS.get(r["name"], "#bbbbbb")
        ax.scatter(x, y, s=180, c=c, edgecolor="black", linewidths=0.6, zorder=3)
        ax.annotate(
            f"{LABELS.get(r['name'], r['name'])}\nacc={r['final_acc']:.3f}",
            (x, y), xytext=(7, 7), textcoords="offset points",
            fontsize=8, ha="left", va="bottom",
        )
    ax.set_xlabel("IB_%neg  (inter-batch class-pair conflict, lower = COSGD-like)")
    ax.set_ylabel("WW_K  (between-batch trajectory efficiency, higher = BoGrad-like)")
    ax.set_title("Two interference axes — implicit vs explicit methods")
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trajectory_cosines(rows: List[Dict[str, Any]], out_path: Path):
    """Per-method bar chart with bars for lag-1, lag-4, lag-16 update cosines."""
    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(rows)
    width = 0.27
    xs = list(range(n))
    lag1 = [r["lag1"] for r in rows]
    lag4 = [r["lag4"] for r in rows]
    lag16 = [r["lag16"] for r in rows]
    ax.bar([x - width for x in xs], lag1, width, label="lag-1",
           color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(xs, lag4, width, label="lag-4",
           color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.bar([x + width for x in xs], lag16, width, label="lag-16",
           color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(r["name"], r["name"]) for r in rows],
                       rotation=30, ha="right", fontsize=8)
    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_ylabel("cos(u_t, u_{t-k})")
    ax.set_title("Trajectory cosines (update self-correlation at lag 1, 4, 16)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pairwise_pos(rows: List[Dict[str, Any]], out_path: Path):
    """Bar chart of pairwise frac_positive in gradient buffer and update buffer."""
    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(rows)
    width = 0.4
    xs = list(range(n))
    pw_g = [r["pw_grad_pos"] for r in rows]
    pw_u = [r["pw_update_pos"] for r in rows]
    ax.bar([x - width / 2 for x in xs], pw_g, width, label="gradient buffer",
           color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar([x + width / 2 for x in xs], pw_u, width, label="update buffer",
           color="#d62728", edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(r["name"], r["name"]) for r in rows],
                       rotation=30, ha="right", fontsize=8)
    ax.axhline(0.5, linewidth=0.8, linestyle="--", color="black", alpha=0.4,
               label="50/50 (random walk)")
    ax.set_ylabel("frac. of pairwise cosines > 0  over K=32 buffer")
    ax.set_title("Pairwise alignment over the K=32 buffer (positive vs negative)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ib_over_time(data: Dict[str, Dict[str, Any]], out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    keys = order_keys(data.keys())
    for name in keys:
        ibh = data[name]["inter_batch_history"]
        non_empty = [r for r in ibh if r.get("n_pairs", 0) > 0]
        if not non_empty:
            continue
        steps = [r["step"] for r in non_empty]
        vals = [r["frac_negative"] for r in non_empty]
        ax.plot(steps, vals, label=LABELS.get(name, name),
                color=COLOURS.get(name, "#bbbbbb"),
                linewidth=1.4, alpha=0.85)
    ax.set_xlabel("training step")
    ax.set_ylabel("IB_%neg  (frac. of class-pair gradient cosines < 0)")
    ax.set_title("Inter-batch interference over training")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def write_summary_md(rows: List[Dict[str, Any]], run_dir: Path):
    md = ["# Implicit-method comparison — summary",
          "",
          f"_Run directory: `{run_dir.relative_to(ROOT)}`_",
          ""]
    md += ["| method | label | final_acc | IB_%neg | IB_<cos> | WW_K | OOB_total | wall_s |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| `{r['name']}` | {r['label']} "
            f"| {r['final_acc']:.4f} | {r['ib_pct_neg']:.3f} "
            f"| {r['ib_mean_cos']:+.3f} | {r['ww_k']:.3f} "
            f"| {r['oob_total']:.3f} | {r['wall_clock_s']:.0f} |"
        )
    md += ["",
           "## Trajectory cosines (update self-correlation)",
           "",
           "| method | lag-1 | lag-4 | lag-16 | pw_grad+ | pw_upd+ |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| `{r['name']}` "
            f"| {r['lag1']:+.3f} | {r['lag4']:+.3f} | {r['lag16']:+.3f} "
            f"| {r['pw_grad_pos']:.3f} | {r['pw_update_pos']:.3f} |"
        )
    md += ["",
           "## Reading guide",
           "",
           "- **IB_%neg** — fraction of class-pair gradient cosines < 0 within a batch.",
           "  Lower = less inter-batch conflict (what COSGD targets).",
           "- **IB_<cos>** — mean class-pair cosine. Positive = aligned.",
           "- **WW_K** — wasted-work ratio, K=32. 1 = perfectly efficient",
           "  (every step contributes net forward progress); lower = more cancellation.",
           "  Higher = less between-batch interference (what BoGrad targets).",
           "- **OOB_total** — out-of-batch forgetting. Structurally 0 in standard",
           "  iid CIFAR-10 batching; non-zero only under non-iid regimes.",
           "- **lag-k** — `cos(u_t, u_{t-k})`, the update-direction self-correlation",
           "  at lag k. Higher = trajectory is smoothed at that horizon.",
           "  log_every=10 here so lag-1 means cos at 10-step real spacing.",
           "- **pw_grad+** / **pw_upd+** — fraction of pairwise cosines > 0 over",
           "  the K=32-step buffer of gradients / updates. 0.5 = random walk;",
           "  > 0.5 = persistent direction; ~1.0 = all aligned (BoGrad target)."]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="Run id (directory name under results/). Default: latest.")
    ap.add_argument("--run-path", type=str, default=None,
                    help="Absolute path to a run directory (overrides --run).")
    args = ap.parse_args()

    if args.run_path:
        run_dir = Path(args.run_path)
    elif args.run:
        run_dir = RESULTS_ROOT / args.run
    else:
        latest = find_latest_run()
        if latest is None:
            print("No run directories found under results/", file=sys.stderr)
            sys.exit(1)
        run_dir = latest

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {run_dir}")
    data = load_run(run_dir)
    if not data:
        print("No variant result files found in this run directory.", file=sys.stderr)
        sys.exit(1)

    rows = aggregate_metrics(data)
    baseline_acc = next((r["final_acc"] for r in rows
                         if r["name"] == "sgd_momentum"), None)
    baseline_ib = next((r["ib_pct_neg"] for r in rows
                        if r["name"] == "sgd_momentum"), None)
    baseline_ww = next((r["ww_k"] for r in rows
                        if r["name"] == "sgd_momentum"), None)

    print("\nMethods loaded:")
    for r in rows:
        print(f"  {r['name']:24s}  acc={r['final_acc']:.4f}  "
              f"IB%neg={r['ib_pct_neg']:.3f}  WW_K={r['ww_k']:.3f}")

    plot_bar(rows, "final_acc", "Final test accuracy",
             "test acc", run_dir / "acc_summary.png", baseline=baseline_acc)
    plot_bar(rows, "ib_pct_neg", "Inter-batch interference (mean over training)",
             "IB_%neg", run_dir / "ib_axis.png", baseline=baseline_ib)
    plot_bar(rows, "ww_k", "Wasted-work ratio (K=32)",
             "WW_K", run_dir / "ww_axis.png", baseline=baseline_ww)
    plot_axes_scatter(rows, run_dir / "axes_scatter.png")
    plot_ib_over_time(data, run_dir / "ib_over_time.png")
    plot_trajectory_cosines(rows, run_dir / "trajectory_cosines.png")
    plot_pairwise_pos(rows, run_dir / "pairwise_positive.png")
    write_summary_md(rows, run_dir)

    print(f"\nWrote plots and summary.md to {run_dir}")


if __name__ == "__main__":
    main()
