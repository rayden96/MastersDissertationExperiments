"""
10.01 plot — reads results/run_<id>/summary.json (+ rows.json), renders:

  fig A: test accuracy vs K, one line per base optimizer (baseline K0 marked).
  fig B: between-batch cancellation I_between_K32 vs K, same lines — the "why".

Pure presentation: no training, no checkpoint reads (docs/experiment_design.md §8).

    python plot.py                 # newest run under results/
    python plot.py --run <run_id>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
_AXIS = _HERE.parent
_PRE = _AXIS.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json          # noqa: E402
from common.plotting import apply_thesis_rcparams, PALETTE  # noqa: E402


def _resolve_run(run_id):
    root = _HERE / "results"
    if run_id:
        return root / f"run_{run_id}"
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise SystemExit(f"No runs under {root}")
    return runs[-1]


def _k_of(cell_label):
    if cell_label.startswith("baseline"):
        return 0
    m = re.match(r"K(\d+)", cell_label)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    apply_thesis_rcparams()

    run_dir = _resolve_run(args.run)
    summary = read_json(run_dir / "summary.json")

    # base -> sorted [(K, acc_mean, acc_std, Ibetween_mean)]
    by_base = {}
    for e in summary["cells"]:
        K = _k_of(e["cell"])
        if K is None:
            continue
        acc = e["metrics"]["final_test_acc"]
        ib = e["metrics"].get("I_between_K32_mean", {})
        by_base.setdefault(e["base"], []).append(
            (K, acc["mean"], acc["std"], ib.get("mean", float("nan"))))
    for base in by_base:
        by_base[base].sort(key=lambda r: r[0])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    for base, rows in by_base.items():
        Ks = [r[0] for r in rows]
        accs = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        ib = [r[3] for r in rows]
        c = PALETTE.get(base, None)
        xs = [max(k, 0.5) for k in Ks]  # K=0 placed at 0.5 on log axis
        axA.errorbar(xs, accs, yerr=stds, marker="o", label=base, color=c, capsize=3)
        axB.plot(xs, ib, marker="s", label=base, color=c)

    for ax in (axA, axB):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("buffer size $K$ (K=0 = baseline)")
    axA.set_ylabel("final test accuracy")
    axA.set_title("10.01  Accuracy vs buffer size $K$")
    axB.set_ylabel(r"$I_{\mathrm{between},K=32}$")
    axB.set_title("Between-batch cancellation vs $K$  (the why)")
    axA.legend(title="base optimizer")
    fig.suptitle(f"BoGrad buffer-size ablation — {summary['dataset']}", y=1.02)

    out = run_dir / "10_01_buffer_K.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(run_dir / "10_01_buffer_K.pdf", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
