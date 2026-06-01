"""
Generic plotter for any COSGD-ablation axis summary.json.

Renders the shared-shape axes (20.01, 20.02, 20.04, 20.05, 20.06): cells per base
optimizer with mean+/-std test accuracy and the headline INTER-batch metric
(I_inter — COSGD's target). Special axes have their own scripts:
  20.03 prenormalize -> synthetic_dim_sweep.py already produces its figure
  20.07 scalability  -> 07_scalability/plot.py (the O(n^2) curve)
  20.08 cross_summary -> master table (text)

Usage:
    python plots.py --axis 01_gs_variant
    python plots.py --axis 02_class_order --base sgd --dataset cifar10
    python plots.py --axis 06_base_optimizer --base adam --metric I_inter_mean
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


def _latest_summary(axis_dir: Path, sub: Optional[str]) -> Path:
    root = axis_dir / "results"
    if sub and (root / sub).exists():
        root = root / sub
    s = sorted(root.rglob("summary.json"))
    if not s:
        raise SystemExit(f"No summary.json under {root} — run the axis first.")
    return s[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, help="e.g. 01_gs_variant")
    ap.add_argument("--base", default=None)
    ap.add_argument("--dataset", default=None, help="for axes that nest results/<dataset>/")
    ap.add_argument("--metric", default="I_inter_mean")
    args = ap.parse_args()
    apply_thesis_rcparams("dense")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    axis_dir = _HERE / args.axis
    summary_path = _latest_summary(axis_dir, args.dataset or args.base)
    summary = read_json(summary_path)
    cells = summary["cells"]
    bases = sorted({e["base"] for e in cells})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(max(10, 1.2 * len(cells)), 5))
    for base in bases:
        bcells = [e for e in cells if e["base"] == base]
        labels = [e["cell"] for e in bcells]
        acc = [e["metrics"]["final_test_acc"]["mean"] for e in bcells]
        accsd = [e["metrics"]["final_test_acc"]["std"] for e in bcells]
        met = [e["metrics"].get(args.metric, {}).get("mean", float("nan")) for e in bcells]
        x = np.arange(len(labels)); c = PALETTE.get(base)
        axA.errorbar(x, acc, yerr=accsd, marker="o", label=base, color=c, capsize=3)
        axB.plot(x, met, marker="s", label=base, color=c)
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axA.set_ylabel("final test accuracy"); axA.set_title(f"{args.axis} — accuracy")
    axB.set_ylabel(args.metric); axB.set_title(f"{args.axis} — {args.metric} (the why)")
    axA.legend(title="base", fontsize=8)
    fig.suptitle(f"COSGD ablation {args.axis} — {summary.get('dataset','')}", y=1.02)

    out = summary_path.parent / f"{args.axis}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
