"""
10.04 plot — orthogonalisation method comparison.

Two panels per base optimizer's run: final test accuracy and wall-clock per step
for {baseline, sequential_negative, sequential_full, qr_full, householder_full}.
The expected story (already visible in smoke): the soft sequential_negative
recipe wins; the true-orthogonal qr/householder_full projectors are more
aggressive and hurt, and qr == householder (same projector).

The orthogonality residual ||B^T g~|| that distinguishes soft vs true-orth is a
per-step quantity exposed by BoGrad.orthogonality_residual(); it is reported in
the unit test (householder ~4e-7 vs sequential ~0.37) and quoted in the chapter
text rather than re-logged per training cell.

Usage: python plot.py [--base sgd]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parents[2]
_REPO = _HERE.parents[3]
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json                       # noqa: E402
from common.plotting import apply_thesis_rcparams, PALETTE  # noqa: E402


def _latest(base):
    root = _HERE / "results"
    if base and (root / base).exists():
        root = root / base
    s = sorted(root.rglob("summary.json"))
    if not s:
        raise SystemExit(f"No summary.json under {root} — run 04_orth_method/run.py first.")
    return s[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sgd")
    args = ap.parse_args()
    apply_thesis_rcparams()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    summary = read_json(_latest(args.base))
    cells = [e for e in summary["cells"] if e["base"] == args.base]
    order = ["baseline", "sequential_negative", "sequential_full", "qr_full", "householder_full"]
    cells.sort(key=lambda e: order.index(e["cell"]) if e["cell"] in order else 99)
    labels = [e["cell"] for e in cells]
    acc = [e["metrics"]["final_test_acc"]["mean"] for e in cells]
    accsd = [e["metrics"]["final_test_acc"]["std"] for e in cells]
    ms = [e["metrics"].get("mean_step_wall_time_s", {}).get("mean", float("nan")) * 1000 for e in cells]
    x = np.arange(len(labels))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))
    axA.bar(x, acc, yerr=accsd, capsize=3, color=PALETTE.get(args.base, "tab:blue"), alpha=0.85)
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axA.set_ylabel("final test accuracy"); axA.set_title("10.04 — accuracy by orth method")
    axB.bar(x, ms, capsize=3, color="tab:gray", alpha=0.85)
    axB.set_xticks(x); axB.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axB.set_ylabel("ms / step"); axB.set_title("wall-clock per step")
    fig.suptitle(f"BoGrad orthogonalisation method — {args.base} / {summary.get('dataset','')}", y=1.02)

    out = _latest(args.base).parent / "10_04_orth_method.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
