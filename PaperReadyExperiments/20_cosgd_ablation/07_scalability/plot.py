"""
20.07 plot — COSGD class-count scalability (the O(n^2) wall).

Reads scalability_synthetic.json and/or scalability_real.json (written by run.py)
and renders:
  - sec/step vs number of classes (COSGD vs flat SGD baseline) with a fitted
    quadratic reference on the synthetic curve;
  - peak GPU memory vs number of classes.

This is the figure that motivates BoGrad (O(K) buffer, independent of class count).

Usage: python plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent           # 07_scalability
_REPO = _HERE.parents[2]                            # repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.storage import read_json               # noqa: E402
from common.plotting import apply_thesis_rcparams   # noqa: E402


def main():
    apply_thesis_rcparams()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    res = _HERE / "results"
    syn_p = res / "scalability_synthetic.json"
    real_p = res / "scalability_real.json"
    if not syn_p.exists() and not real_p.exists():
        raise SystemExit(f"No scalability_*.json under {res} — run 07_scalability/run.py first.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    if syn_p.exists():
        d = read_json(syn_p)
        pts = sorted(d["points"], key=lambda p: p["n_classes"])
        n = [p["n_classes"] for p in pts]
        cos = [p["cosgd_sec_per_step"] * 1000 for p in pts]
        base = [p["baseline_sec_per_step"] * 1000 for p in pts]
        axA.plot(n, cos, "o-", color="tab:red", label="COSGD")
        axA.plot(n, base, "s-", color="tab:blue", label="SGD baseline")
        # quadratic reference fitted to COSGD
        if len(n) >= 3:
            coef = np.polyfit(n, cos, 2)
            xs = np.linspace(min(n), max(n), 100)
            axA.plot(xs, np.polyval(coef, xs), "k--", alpha=0.5, label="quadratic fit")
        axA.set_xlabel("number of classes in batch"); axA.set_ylabel("ms / step")
        axA.set_title("20.07 — per-step cost (synthetic CIFAR-10 sub-classes)")
        axA.legend()
        cosm = [p["cosgd_peak_mb"] for p in pts]
        basem = [p["baseline_peak_mb"] for p in pts]
        axB.plot(n, cosm, "o-", color="tab:red", label="COSGD")
        axB.plot(n, basem, "s-", color="tab:blue", label="SGD baseline")
        axB.set_xlabel("number of classes in batch"); axB.set_ylabel("peak GPU memory (MB)")
        axB.set_title("peak memory vs class count"); axB.legend()

    out = res / "20_07_scalability.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    if real_p.exists():
        d = read_json(real_p)
        fig2, ax = plt.subplots(figsize=(8, 5))
        pts = sorted(d["points"], key=lambda p: p["n_classes"])
        names = [f"{p['dataset']}\n({p['n_classes']})" for p in pts]
        x = np.arange(len(pts)); w = 0.35
        ax.bar(x - w / 2, [p["baseline_sec_per_step"] * 1000 for p in pts], w,
               label="SGD", color="tab:blue")
        ax.bar(x + w / 2, [p["cosgd_sec_per_step"] * 1000 for p in pts], w,
               label="COSGD", color="tab:red")
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("ms / step")
        ax.set_title("20.07 — per-step cost on real datasets"); ax.legend()
        out2 = res / "20_07_scalability_real.png"
        fig2.savefig(out2, bbox_inches="tight"); plt.close(fig2)
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()
