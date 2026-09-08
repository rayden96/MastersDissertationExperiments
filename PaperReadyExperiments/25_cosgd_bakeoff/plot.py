"""Figures and speed table for the COSGD optimiser bakeoff (study 25).

One panel per dataset, five arms per panel, test accuracy against epoch. The
ladder runs from four features and three classes (Iris) to three colour
channels and ten classes (CIFAR-10), so the panels are ordered by input
dimension and read left to right as the geometry of Section 4.4 predicts.

Speed is reported the same way as the rest of the dissertation: epochs to reach
99% of the best final accuracy any arm on that dataset attains, so all five are
measured against one common target rather than each against itself. An arm
whose seeds never reach it is charged the budget plus one and marked censored.

    python plot.py
    python plot.py --outdir some/other/dir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import storage
from _curves import load_axis, group, curve_stats, epochs_to_target

# ordered by input dimension, which is the axis the section reads along
ORDER = ["iris", "titanic", "mnist", "fashion_mnist", "cifar10"]
TITLE = {"iris": "Iris (4 features, 3 classes)",
         "titanic": "Titanic (13 features, 2 classes)",
         "mnist": "MNIST (784, 10)",
         "fashion_mnist": "Fashion-MNIST (784, 10)",
         "cifar10": "CIFAR-10 (3072, 10)"}
ARMS = ["cosgd", "sgd", "adam", "rmsprop", "signsgd"]
PRETTY = {"cosgd": "COSGD", "sgd": "SGD", "adam": "Adam",
          "rmsprop": "RMSProp", "signsgd": "SignSGD"}
COLOUR = {"cosgd": "#d62728", "sgd": "#1f77b4", "adam": "#2ca02c",
          "rmsprop": "#ff7f0e", "signsgd": "#9467bd"}

TARGET_FRAC = 0.99


def _load(name):
    root = storage.get_results_root() / "25_cosgd_bakeoff" / name
    recs = []
    for d in sorted(root.glob("run_*")):
        recs.extend(load_axis(d))
    return [r for r in recs if r.acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (
        _REPO.parent / "dissertation" / "chapters" / "cosgd" / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from common.plotting import apply_thesis_rcparams
        apply_thesis_rcparams()
    except Exception:
        pass

    have = [(n, _load(n)) for n in ORDER]
    have = [(n, r) for n, r in have if r]
    if not have:
        print("no results found under", storage.get_results_root() / "25_cosgd_bakeoff")
        return

    ncols = min(3, len(have))
    nrows = int(np.ceil(len(have) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows),
                             squeeze=False)
    table = {}

    for i, (name, recs) in enumerate(have):
        ax = axes[i // ncols][i % ncols]
        cur = group(recs)

        # common target: 99% of the best final accuracy any arm reaches here
        finals = {a: float(curve_stats(cur[a]["acc"])[0][-1])
                  for a in ARMS if a in cur and cur[a]["acc"]}
        best = max(finals.values())
        target = best * TARGET_FRAC

        row = {}
        for a in ARMS:
            if a not in cur or not cur[a]["acc"]:
                continue
            m, s = curve_stats(cur[a]["acc"])
            x = np.arange(1, len(m) + 1)
            ax.plot(x, m, color=COLOUR[a], lw=1.7, label=PRETTY[a])
            ax.fill_between(x, m - s, m + s, color=COLOUR[a], alpha=0.15, lw=0)

            eps, reached = [], 0
            for c in cur[a]["acc"]:
                hit = epochs_to_target(c, target)
                eps.append(hit if hit is not None else len(c) + 1)
                reached += hit is not None
            row[a] = {"epochs": float(np.mean(eps)), "reached": reached,
                      "n": len(cur[a]["acc"]), "final": finals[a]}

        ax.axhline(target, color="0.6", ls=":", lw=1.0, zorder=0)
        ax.set_title(TITLE.get(name, name), fontsize=9)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel("test accuracy", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7.5, loc="lower right")
        table[name] = {"target": target, "best_final": best, "arms": row}

    for j in range(len(have), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out = outdir / "cosgd_bakeoff.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    (_HERE / "bakeoff_table.json").write_text(json.dumps(table, indent=1))
    print(f"\n{'dataset':16s} {'target':>7s}  " +
          "  ".join(f"{PRETTY[a]:>16s}" for a in ARMS))
    for name, d in table.items():
        cells = []
        for a in ARMS:
            v = d["arms"].get(a)
            if not v:
                cells.append(f"{'--':>16s}"); continue
            mark = "" if v["reached"] == v["n"] else f" c{v['reached']}/{v['n']}"
            cells.append(f"{v['epochs']:6.1f}ep {v['final']:.3f}{mark}"[:16].rjust(16))
        print(f"{name:16s} {d['target']:7.3f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
