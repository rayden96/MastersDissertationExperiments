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


# The dissertation inputs these by fixed name (chapters/experiments/main.tex),
# one figure per axis, so emit exactly those and put the per-dataset views in
# panels rather than in separate files.
FIGNAME = {
    "lr":        "30_04_sensitivity_lr.png",
    "K":         "30_05_sensitivity_K.png",
    "batch":     "30_06_sensitivity_batch.png",
    "scale":     "30_07_scale.png",
    "gradstats": "30_08_gradstats.png",
}


def plot_lr_or_batch(kind):
    plt = _plt(); apply_thesis_rcparams()
    data = read_json(_HERE / "_sensitivity" / f"sensitivity_{kind}.json")
    xkey = "lr" if kind == "lr" else "batch"
    datasets = list(data["datasets"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5),
                             squeeze=False)
    for ax, ds in zip(axes.flatten(), datasets):
        by_method = defaultdict(list)
        for p in data["datasets"][ds]:
            by_method[p["method"]].append((p[xkey], p["acc"]))
        for method, pts in sorted(by_method.items()):
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", linestyle=METHOD_STYLE.get(method, "-"),
                    label=method)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate" if kind == "lr" else "batch size")
        ax.set_ylabel("final test accuracy")
        ax.set_title(ds)
        ax.legend()
    fig.suptitle(f"{'30.04' if kind == 'lr' else '30.06'} — "
                 f"{'learning-rate' if kind == 'lr' else 'batch-size'} sensitivity", y=1.02)
    out = _HERE / "_sensitivity" / FIGNAME[kind]
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
    ax.set_ylabel("final test accuracy"); ax.set_title("30.05 — K sensitivity per dataset")
    ax.legend()
    out = _HERE / "_sensitivity" / FIGNAME["K"]
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def plot_scale():
    plt = _plt(); apply_thesis_rcparams()
    data = read_json(_HERE / "_scale" / "scale_30_07.json")
    by_method = defaultdict(list)
    for p in data["points"]:
        # Points recorded as OOM carry no measurements, and a crashed sweep can
        # leave a point without params or timing; skip rather than raise.
        if p.get("oom"):
            continue
        if p.get("params") is None or p.get("final_test_acc") is None:
            continue
        by_method[p["method"]].append((p["params"], p["final_test_acc"],
                                       p.get("sec_per_step")))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    for method, pts in sorted(by_method.items()):
        pts.sort()
        axA.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                 linestyle=METHOD_STYLE.get(method, "-"), label=method)
        timed = [(p[0], p[2]) for p in pts if isinstance(p[2], (int, float))]
        if timed:
            axB.plot([t[0] for t in timed], [t[1] * 1000 for t in timed], marker="s",
                     linestyle=METHOD_STYLE.get(method, "-"), label=method)
    for ax in (axA, axB):
        ax.set_xscale("log"); ax.set_xlabel("parameters (log)")
    axA.set_ylabel("final test accuracy"); axA.set_title("30.07 — accuracy vs model scale")
    # Per-step times here come from separate training runs on whatever device was
    # allocated, so they are indicative only; 30.11 (timing.py) is the citable
    # measurement, taken back-to-back in one process.
    axB.set_ylabel("ms / step (indicative)"); axB.set_title("step cost vs model scale")
    axA.legend()
    out = _HERE / "_scale" / FIGNAME["scale"]
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
    out = _HERE / "_gradstats" / FIGNAME["gradstats"]
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


KINDS = ["lr", "K", "batch", "scale", "gradstats"]


def _plot_one(kind):
    if kind in ("lr", "batch"):
        plot_lr_or_batch(kind)
    elif kind == "K":
        plot_K()
    elif kind == "scale":
        plot_scale()
    else:
        plot_gradstats()


def main():
    ap = argparse.ArgumentParser()
    # Defaults to every axis: five separate invocations is an easy step to get
    # half-right, and a missing figure silently leaves a PENDING box in the
    # chapter rather than failing loudly.
    ap.add_argument("--kind", nargs="+", default=KINDS, choices=KINDS)
    args = ap.parse_args()
    failed = []
    for kind in args.kind:
        try:
            _plot_one(kind)
        except FileNotFoundError:
            print(f"!! {kind}: source JSON not found, skipped")
            failed.append(kind)
        except Exception as e:
            print(f"!! {kind}: {type(e).__name__}: {e}")
            failed.append(kind)
    done = [k for k in args.kind if k not in failed]
    print(f"\n{len(done)}/{len(args.kind)} figures written"
          + (f"; missing: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
