"""
30.04-30.08 plots — read the sensitivity / scale / grad-stats JSON and render.

  --kind lr     30.04  acc vs learning rate (log x), one line per method, per dataset
  --kind K      30.05  acc vs BoGrad buffer K (log2 x), per dataset
  --kind batch  30.06  acc vs batch size (log x), baseline/bograd/cosgd, per dataset
  --kind scale  30.07  acc + ms/step vs params (log x), per method
  --kind gradstats 30.08  per-step series (u_norm, I_between, cos, D_t) per method

Reads the JSON written by sensitivity.py / scale_gradstats.py. Pure presentation.

Usage:
    python plot_sensitivity.py --kind lr
    python plot_sensitivity.py --kind scale
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json                       # noqa: E402
from common.plotting import apply_thesis_rcparams, METHOD_STYLE  # noqa: E402


def _plt():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    return plt


def plot_lr_or_batch(kind):
    plt = _plt(); apply_thesis_rcparams()
    data = read_json(_HERE / "_sensitivity" / f"sensitivity_{kind}.json")
    xkey = "lr" if kind == "lr" else "batch"
    for ds, points in data["datasets"].items():
        by_method = defaultdict(list)
        for p in points:
            by_method[p["method"]].append((p[xkey], p["acc"]))
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, pts in by_method.items():
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", linestyle=METHOD_STYLE.get(method, "-"), label=method)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate" if kind == "lr" else "batch size")
        ax.set_ylabel("final test accuracy")
        ax.set_title(f"{'30.04' if kind=='lr' else '30.06'}  {ds} — {kind} sensitivity")
        ax.legend()
        out = _HERE / "_sensitivity" / f"{kind}_{ds}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")


def plot_K():
    plt = _plt(); apply_thesis_rcparams()
    data = read_json(_HERE / "_sensitivity" / "sensitivity_K.json")
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, points in data["datasets"].items():
        points = sorted(points, key=lambda p: p["K"])
        ax.plot([p["K"] for p in points], [p["acc"] for p in points], marker="o", label=ds)
    ax.set_xscale("log", base=2); ax.set_xlabel("BoGrad buffer K")
    ax.set_ylabel("final test accuracy"); ax.set_title("30.05  K sensitivity per dataset")
    ax.legend()
    out = _HERE / "_sensitivity" / "K_sensitivity.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def plot_scale():
    plt = _plt(); apply_thesis_rcparams()
    data = read_json(_HERE / "_scale" / "scale_30_07.json")
    by_method = defaultdict(list)
    for p in data["points"]:
        by_method[p["method"]].append((p["params"], p["final_test_acc"], p["sec_per_step"]))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    for method, pts in by_method.items():
        pts.sort()
        params = [p[0] for p in pts]
        axA.plot(params, [p[1] for p in pts], marker="o",
                 linestyle=METHOD_STYLE.get(method, "-"), label=method)
        axB.plot(params, [p[2] * 1000 for p in pts], marker="s",
                 linestyle=METHOD_STYLE.get(method, "-"), label=method)
    for ax in (axA, axB):
        ax.set_xscale("log"); ax.set_xlabel("parameters (log)")
    axA.set_ylabel("final test accuracy"); axA.set_title("30.07 — accuracy vs model scale")
    axB.set_ylabel("ms / step"); axB.set_title("overhead vs model scale")
    axA.legend()
    out = _HERE / "_scale" / "30_07_model_scale.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def plot_gradstats():
    plt = _plt(); apply_thesis_rcparams("dense")
    data = read_json(_HERE / "_gradstats" / "gradstats_30_08.json")
    keys = [("u_norm", "effective step ||u||"), ("I_between_K32", "$I_{between,32}$"),
            ("between_K32_mean_cos", "mean cos within K=32"), ("D_t", "per-step deficit $D_t$")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axs.flatten(), keys):
        for method, blob in data["methods"].items():
            series = blob["series"].get(key, [])
            if series:
                steps, vals = zip(*series)
                ax.plot(steps, vals, label=method, linestyle=METHOD_STYLE.get(method, "-"))
        ax.set_title(title); ax.set_xlabel("step")
    axs.flatten()[0].legend(fontsize=8)
    fig.suptitle("30.08 — gradient statistics during training (CIFAR-10)", y=1.0)
    out = _HERE / "_gradstats" / "30_08_gradstats.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["lr", "K", "batch", "scale", "gradstats"])
    args = ap.parse_args()
    if args.kind in ("lr", "batch"):
        plot_lr_or_batch(args.kind)
    elif args.kind == "K":
        plot_K()
    elif args.kind == "scale":
        plot_scale()
    else:
        plot_gradstats()


if __name__ == "__main__":
    main()
