"""
10.08 plot — batch size x K heatmap.

The axis writes one results/bs<N>/run_* per batch size, each with cells
baseline_K0, K2, K8, K32. This assembles a (batch x K) grid of final test
accuracy and renders a heatmap; the headline is where BoGrad's gain over the K=0
baseline column is largest (expected: small batch, moderate K).

Usage: python plot.py [--base sgd]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parents[2]
_REPO = _HERE.parents[3]
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json               # noqa: E402
from common.plotting import apply_thesis_rcparams   # noqa: E402


def _k_of(cell):
    if cell.startswith("baseline"):
        return 0
    m = re.match(r"K(\d+)", cell)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sgd")
    args = ap.parse_args()
    apply_thesis_rcparams()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    results_root = _HERE / "results"
    bs_dirs = sorted([d for d in results_root.glob("bs*") if d.is_dir()],
                     key=lambda d: int(d.name[2:]))
    if not bs_dirs:
        raise SystemExit(f"No bs* dirs under {results_root} — run 08_batch_K/run.py first.")

    grid = {}  # (bs, K) -> acc
    batches, Ks = set(), set()
    for bsd in bs_dirs:
        bs = int(bsd.name[2:])
        summ = sorted(bsd.rglob("summary.json"))
        if not summ:
            continue
        summary = read_json(summ[-1])
        for e in summary["cells"]:
            if e["base"] != args.base:
                continue
            K = _k_of(e["cell"])
            if K is None:
                continue
            grid[(bs, K)] = e["metrics"]["final_test_acc"]["mean"]
            batches.add(bs); Ks.add(K)

    batches = sorted(batches); Ks = sorted(Ks)
    M = np.full((len(batches), len(Ks)), np.nan)
    for i, bs in enumerate(batches):
        for j, K in enumerate(Ks):
            if (bs, K) in grid:
                M[i, j] = grid[(bs, K)]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_yticks(range(len(batches))); ax.set_yticklabels([f"bs={b}" for b in batches])
    for i in range(len(batches)):
        for j in range(len(Ks)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmean(M) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="final test accuracy")
    ax.set_title(f"10.08 — batch x K final accuracy ({args.base})")

    out = results_root / "10_08_batch_K_heatmap.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
