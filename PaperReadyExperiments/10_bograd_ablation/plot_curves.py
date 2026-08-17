"""
Chapter 5 figures: one per BOGrad ablation axis, as learning curves.

Same convention as 20_cosgd_ablation/plot_curves.py, adapted to this
chapter's shape: BOGrad's axes are swept across four base optimisers as
well as the two anchors, so each axis becomes a grid of panels (rows are
the anchors, columns the base optimisers) rather than the two-column
accuracy/loss layout Chapter 4 uses. An accuracy grid and a training-loss
grid are written for every axis; the chapter includes the loss grid only
where the argument turns on it.

Reads the per-run results.json files already on disk — nothing is re-run.

Usage:
    python plot_curves.py                       # every axis with results
    python plot_curves.py --axes buffer_K mode
    python plot_curves.py --outdir some/other/dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                    # noqa: E402
from _curves import load_axis, group, final_means, plot_grid   # noqa: E402

ANCHORS = ["covertype", "cifar10"]
ANCHOR_SHORT = {"covertype": "Covertype", "cifar10": "CIFAR-10"}
BASES = ["sgd", "adam", "rmsprop", "signsgd"]


def _K_pretty(label: str) -> str:
    return "baseline" if label.startswith("baseline") else f"$K = {label[1:]}$"


# axis key -> how to find and lay out its cells
AXES: Dict[str, dict] = {
    "buffer_K": dict(
        stem="10_01_buffer_K", fig="ax_buffer_K",
        # nine buffer sizes is too many lines for a panel this size; the
        # subset spans the range at every doubling of the exponent.
        keep=["baseline_K0", "K1", "K4", "K16", "K64", "K128"],
        order=["baseline_K0", "K1", "K4", "K16", "K64", "K128"],
        pretty={"baseline_K0": "baseline", "K1": "$K = 1$", "K4": "$K = 4$",
                "K16": "$K = 16$", "K64": "$K = 64$", "K128": "$K = 128$"},
        baseline="baseline_K0",
    ),
    "mode": dict(
        stem="10_03_projection_mode", fig="ax_mode",
        order=["baseline", "mode_negative", "mode_full", "mode_positive",
               "neg_alpha0.25", "neg_alpha0.5", "neg_alpha0.75"],
        pretty={"mode_negative": "negative-only", "mode_full": "full",
                "mode_positive": "positive-only",
                "neg_alpha0.25": r"negative, $\alpha = 0.25$",
                "neg_alpha0.5": r"negative, $\alpha = 0.5$",
                "neg_alpha0.75": r"negative, $\alpha = 0.75$"},
        # full and positive-only collapse to chance or below in almost every
        # cell; autoscaling to include them squashes the band where the
        # remaining five settings actually differ.
        focus=True,
    ),
    "orth": dict(
        stem="10_04_orth_method", fig="ax_orth",
        order=["baseline", "sequential_negative", "sequential_full",
               "qr_full", "householder_full"],
        pretty={"sequential_negative": "sequential, negative",
                "sequential_full": "sequential, full",
                "qr_full": "QR (exact)", "householder_full": "Householder (exact)"},
    ),
    "scope": dict(
        stem="10_05_projection_scope", fig="ax_scope",
        order=["baseline", "scope_global", "scope_per_tensor"],
        pretty={"scope_global": "global", "scope_per_tensor": "per-tensor"},
    ),
    "magnitude": dict(
        stem="10_06_magnitude", fig="ax_magnitude",
        order=["baseline", "bograd", "bograd_preserve_mag",
               "random_proj", "random_proj_preserve_mag"],
        pretty={"bograd": "BOGrad", "bograd_preserve_mag": "BOGrad, magnitude preserved",
                "random_proj": "random projection",
                "random_proj_preserve_mag": "random, magnitude preserved"},
    ),
    "momentum": dict(
        stem="10_07_momentum_2x2", fig="ax_momentum",
        bases=["sgd", "rmsprop", "signsgd"],
        order=["mu0_bogradOff", "mu0_bogradOn", "mu0.9_bogradOff", "mu0.9_bogradOn"],
        pretty={"mu0_bogradOff": r"$\mu = 0$, no BOGrad",
                "mu0_bogradOn": r"$\mu = 0$, BOGrad",
                "mu0.9_bogradOff": r"$\mu = 0.9$, no BOGrad",
                "mu0.9_bogradOn": r"$\mu = 0.9$, BOGrad"},
        baseline="mu0_bogradOff",
    ),
}


def _results_root() -> Path:
    return storage.get_results_root() / "10_bograd_ablation"


def _default_outdir() -> Path:
    d = _REPO.parent / "dissertation" / "chapters" / "bograd" / "figures"
    return d if d.exists() else (_HERE / "figures")


def _records(stem: str, base: Optional[str], dataset: str) -> List:
    """Every run for (axis, base, dataset). Axes launched per base have one
    directory each; 10.01 holds all four bases in one."""
    root = _results_root()
    pattern = f"{stem}_{base}*" if base else f"{stem}*"
    recs = []
    for d in sorted(root.glob(pattern)):
        if d.is_dir():
            recs.extend(load_axis(d))
    return [r for r in recs if r.dataset == dataset and (base is None or r.base == base)]


def do_axis(key: str, outdir: Path) -> Optional[dict]:
    spec = AXES[key]
    bases = spec.get("bases", BASES)
    keep = set(spec["keep"]) if spec.get("keep") else None
    cells, stats = [], {}

    for ds in ANCHORS:
        for base in bases:
            # per-base directories first; fall back to the single-directory form
            recs = _records(spec["stem"], base, ds) or _records(spec["stem"], None, ds)
            recs = [r for r in recs if r.base == base]
            if keep:
                recs = [r for r in recs if r.cell in keep]
            if not recs:
                continue
            cells.append((f"{ANCHOR_SHORT.get(ds, ds)} — {base}", group(recs)))
            stats[f"{ds}/{base}"] = final_means(recs)

    if not cells:
        print(f"  {key}: no results under {_results_root()} — skipped")
        return None

    kw = dict(order=spec["order"], pretty=spec["pretty"],
              baseline_key=spec.get("baseline", "baseline"))
    plot_grid(cells, outdir / spec["fig"], ncols=len(bases), which="acc",
              focus_on_baseline=spec.get("focus", False), **kw)
    plot_grid(cells, outdir / (spec["fig"] + "_loss"), ncols=len(bases),
              which="loss", **kw)
    return stats


def do_batch_K(outdir: Path) -> Optional[dict]:
    """10.08 varies the batch size as well as K, on SGD only: rows are the
    anchors, columns the batch sizes."""
    sizes = [32, 64, 128, 256, 512]
    order = ["baseline_K0", "K2", "K8", "K32"]
    pretty = {"baseline_K0": "baseline", "K2": "$K = 2$",
              "K8": "$K = 8$", "K32": "$K = 32$"}
    cells, stats = [], {}
    for ds in ANCHORS:
        for bs in sizes:
            recs = _records(f"10_08_batch_K_bs{bs}", None, ds)
            if not recs:
                continue
            cells.append((f"{ANCHOR_SHORT.get(ds, ds)} — batch {bs}", group(recs)))
            stats[f"{ds}/bs{bs}"] = final_means(recs)
    if not cells:
        print("  batch_K: no results — skipped")
        return None
    plot_grid(cells, outdir / "ax_batch_K", ncols=len(sizes), which="acc",
              order=order, pretty=pretty, baseline_key="baseline_K0")
    return stats


def do_lr(outdir: Path) -> Optional[dict]:
    """10.02 sweeps the learning rate for a baseline arm and a BOGrad arm.

    Two figures: the basin (best accuracy against learning rate, which is
    the natural summary of a tuning sweep and the one the chapter's
    argument turns on), and the curves of each arm at its own best rate,
    which is the only fair curve comparison this axis supports.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotting import apply_thesis_rcparams
    from _curves import curve_stats, BASELINE_STYLE

    apply_thesis_rcparams("dense")
    stats: Dict[str, dict] = {}
    basin: Dict[tuple, Dict[str, Dict[float, float]]] = {}
    best_curves = []

    for ds in ANCHORS:
        for base in BASES:
            recs = _records("10_02_lr_retune", base, ds)
            if not recs:
                continue
            stats[f"{ds}/{base}"] = final_means(recs)
            arms: Dict[str, Dict[float, List]] = {"baseline": {}, "bograd": {}}
            for r in recs:
                if "_lr" not in r.cell:
                    continue
                head, _, lr_s = r.cell.rpartition("_lr")
                try:
                    lr = float(lr_s)
                except ValueError:
                    continue
                arm = "baseline" if head.startswith("baseline") else "bograd"
                arms[arm].setdefault(lr, []).append(r)
            basin[(ds, base)] = {
                arm: {lr: float(np.mean([max(x.acc) if x.acc else float("nan")
                                         for x in rs]))
                      for lr, rs in sorted(by_lr.items())}
                for arm, by_lr in arms.items()}
            # curves at each arm's own best rate
            panel = {}
            for arm, by_lr in arms.items():
                if not by_lr:
                    continue
                best_lr = max(by_lr, key=lambda lr: basin[(ds, base)][arm][lr])
                panel[f"{arm} (lr {best_lr:g})"] = {
                    "acc": [x.acc for x in by_lr[best_lr] if x.acc],
                    "loss": [x.loss for x in by_lr[best_lr] if x.loss]}
            best_curves.append((f"{ANCHOR_SHORT.get(ds, ds)} — {base}", panel))

    if not basin:
        print("  lr: no results — skipped")
        return None

    # (a) the basin
    keys = [k for k in basin if basin[k]]
    nrows, ncols = len(ANCHORS), len(BASES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows),
                             squeeze=False)
    for i, ds in enumerate(ANCHORS):
        for j, base in enumerate(BASES):
            ax = axes[i][j]
            d = basin.get((ds, base))
            if not d:
                ax.axis("off"); continue
            for arm, style in (("baseline", BASELINE_STYLE),
                               ("bograd", dict(color="#1f77b4", linewidth=2.0))):
                pts = d.get(arm) or {}
                if pts:
                    xs = sorted(pts)
                    ax.plot(xs, [pts[x] for x in xs], marker="o", markersize=4,
                            label="baseline" if arm == "baseline" else "BOGrad",
                            **style)
            ax.set_xscale("log")
            ax.set_title(f"{ANCHOR_SHORT.get(ds, ds)} — {base}", fontsize=11)
            ax.set_xlabel("learning rate"); ax.set_ylabel("best test accuracy")
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
    out = outdir / "ax_lr_basin"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")

    # (b) curves at each arm's own best rate
    order = sorted({k for _t, p in best_curves for k in p})
    plot_grid(best_curves, outdir / "ax_lr_best", ncols=len(BASES), which="acc",
              order=order, baseline_key="__none__")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+",
                    default=list(AXES) + ["lr", "batch_K"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else _default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"results root: {_results_root()}")
    print(f"figures ->    {outdir}\n")

    allstats: Dict[str, dict] = {}
    for key in args.axes:
        print(f"[{key}]")
        if key == "lr":
            s = do_lr(outdir)
        elif key == "batch_K":
            s = do_batch_K(outdir)
        else:
            s = do_axis(key, outdir)
        if s:
            allstats[key] = s

    out = _HERE / "curve_stats.json"
    out.write_text(json.dumps(allstats, indent=2))
    print(f"\nwrote {out}")
    for axis, per in allstats.items():
        print(f"\n=== {axis} (final test accuracy, mean +/- std) ===")
        for grp, cells in per.items():
            print(f"  {grp}")
            for cell, v in cells.items():
                print(f"    {cell:<30} {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
