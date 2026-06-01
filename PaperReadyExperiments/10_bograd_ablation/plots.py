"""
Generic plotter for any BoGrad-ablation axis summary.json.

Most axes (10.02, 10.03, 10.05, 10.06, 10.07) share the same shape: a set of
cells per base optimizer, each with a mean+/-std test accuracy and the headline
interference metric. This one script renders them all, so we don't paste a
near-identical plot.py into every axis folder (CLAUDE.md: no copy-paste).

Special-shaped axes have their own scripts:
  10.01 buffer_K -> 01_buffer_K/plot.py        (accuracy + I_between vs K)
  10.04 orth_method -> 04_orth_method/plot.py   (accuracy + orthogonality residual)
  10.08 batch_K -> 08_batch_K/plot.py           (heatmap)
  10.09 cross_summary -> the master table (text), no figure

Usage:
    python plots.py --axis 03_projection_mode
    python plots.py --axis 02_lr_retune --base sgd     # axes that nest results/<base>/
    python plots.py --axis 06_magnitude --metric I_between_K32_mean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json                       # noqa: E402
from common.plotting import apply_thesis_rcparams, PALETTE  # noqa: E402


def _latest_summary(axis_dir: Path, base: Optional[str]) -> Path:
    """Find the newest summary.json under an axis folder (axes may nest
    results/<base>/run_* or results/run_*)."""
    search_root = axis_dir / "results"
    if base and (search_root / base).exists():
        search_root = search_root / base
    summaries = sorted(search_root.rglob("summary.json"))
    if not summaries:
        raise SystemExit(f"No summary.json under {search_root} — run the axis first.")
    return summaries[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, help="axis folder name, e.g. 03_projection_mode")
    ap.add_argument("--base", default=None, help="for axes that nest results/<base>/")
    ap.add_argument("--metric", default="I_between_K32_mean",
                    help="headline interference metric to show alongside accuracy")
    args = ap.parse_args()
    apply_thesis_rcparams("dense")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    axis_dir = _HERE / args.axis
    summary_path = _latest_summary(axis_dir, args.base)
    summary = read_json(summary_path)
    cells = summary["cells"]

    # group by base; x = cell label
    bases = sorted({e["base"] for e in cells})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(max(10, 1.2 * len(cells)), 5))
    for base in bases:
        bcells = [e for e in cells if e["base"] == base]
        labels = [e["cell"] for e in bcells]
        acc = [e["metrics"]["final_test_acc"]["mean"] for e in bcells]
        accsd = [e["metrics"]["final_test_acc"]["std"] for e in bcells]
        met = [e["metrics"].get(args.metric, {}).get("mean", float("nan")) for e in bcells]
        x = np.arange(len(labels))
        c = PALETTE.get(base)
        axA.errorbar(x, acc, yerr=accsd, marker="o", label=base, color=c, capsize=3)
        axB.plot(x, met, marker="s", label=base, color=c)
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axA.set_ylabel("final test accuracy"); axA.set_title(f"{args.axis} — accuracy")
    axB.set_ylabel(args.metric); axB.set_title(f"{args.axis} — {args.metric} (the why)")
    axA.legend(title="base", fontsize=8)
    fig.suptitle(f"BoGrad ablation {args.axis} — {summary.get('dataset','')}", y=1.02)

    out = summary_path.parent / f"{args.axis}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
