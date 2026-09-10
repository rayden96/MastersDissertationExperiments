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

# CIFAR-10 is the sole anchor for the ablation chapters. Covertype is
# retained in the head-to-head comparison of Chapter 6, where it stands
# for the tabular regime, but carrying two anchors through every design
# axis doubled the grid without changing which setting of any knob won.
ANCHORS = ["cifar10"]
ANCHOR_SHORT = {"covertype": "Covertype", "cifar10": "CIFAR-10"}
BASES = ["sgd", "adam", "rmsprop", "signsgd"]


def _K_pretty(label: str) -> str:
    return "baseline" if label.startswith("baseline") else f"$K = {label[1:]}$"


# axis key -> how to find and lay out its cells
AXES: Dict[str, dict] = {
    "buffer_K": dict(
        stem="10_01_buffer_K", fig="ax_buffer_K",
        pretty={"baseline_K0": "baseline", "K1": "$K = 1$", "K2": "$K = 2$",
                "K4": "$K = 4$", "K8": "$K = 8$", "K16": "$K = 16$",
                "K32": "$K = 32$", "K64": "$K = 64$", "K128": "$K = 128$"},
        baseline="baseline_K0",
        # Nine buffer sizes on one panel is unreadable at this size. Split the
        # sweep at the middle of the range: the short buffers show benefit
        # appearing, the long ones show it turning over.
        subsets=[("short", ["baseline_K0", "K1", "K2", "K4", "K8"]),
                 ("long",  ["baseline_K0", "K16", "K32", "K64", "K128"])],
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
        # Two different questions live on this axis. Which components to
        # remove is a categorical comparison against two settings that do not
        # train; how completely to remove them is a continuum. They read far
        # better apart.
        subsets=[("which", ["baseline", "mode_negative", "mode_full",
                            "mode_positive"]),
                 ("strength", ["baseline", "neg_alpha0.25", "neg_alpha0.5",
                               "neg_alpha0.75", "mode_negative"])],
    ),
    "orth": dict(
        stem="10_04_orth_method", fig="ax_orth",
        order=["baseline", "sequential_negative", "sequential_full",
               "qr_full", "householder_full"],
        pretty={"sequential_negative": "sequential", "sequential_full": "sequential",
                "qr_negative": "QR (exact)", "qr_full": "QR (exact)",
                "householder_negative": "Householder (exact)",
                "householder_full": "Householder (exact)"},
        # The argument is that the method is irrelevant at matched mode and
        # the mode is everything. One figure per mode says that directly;
        # putting all six on one panel says it far less clearly.
        subsets=[("negative", ["baseline", "sequential_negative", "qr_negative",
                               "householder_negative"]),
                 ("full", ["baseline", "sequential_full", "qr_full",
                           "householder_full"])],
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

    bl = spec.get("baseline", "baseline")
    # Each subset becomes its own figure. A panel carrying more than four or
    # five lines is unreadable once the grid is scaled to the text width, and
    # the subsets are chosen so that each figure answers one question.
    subsets = spec.get("subsets") or [("", spec.get("order") or [])]
    for suffix, order in subsets:
        keep = set(order) if order else None
        sub = [(title, {k: v for k, v in curves.items() if not keep or k in keep})
               for title, curves in cells]
        stem = spec["fig"] + (f"_{suffix}" if suffix else "")
        kw = dict(order=order or None, pretty=spec["pretty"], baseline_key=bl)
        plot_grid(sub, outdir / stem, ncols=len(bases), which="acc",
                  focus_on_baseline=spec.get("focus", False), **kw)
        plot_grid(sub, outdir / (stem + "_loss"), ncols=len(bases),
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

    One figure, one row per base optimiser: test accuracy on the left and
    training loss on the right. Each panel carries a single baseline curve,
    drawn at the baseline's own best rate, against BOGrad at every rate in
    the grid. The question this axis answers is where BOGrad's usable band
    sits relative to a properly tuned baseline, and that reads directly off
    the curves. A basin of best accuracy against rate hides how quickly each
    rate gets there, which is the quantity the rest of the dissertation ranks
    on.

    The baseline's rate is chosen the same way every other comparison in the
    dissertation is ranked: fewest epochs to a common target, 99% of the best
    final accuracy any cell on that base reaches. Ties, including the case
    where no baseline rate reaches the target at all, are broken by the mean
    accuracy over the whole run, which is the closest measure of speed an
    all-censored sweep offers.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotting import apply_thesis_rcparams
    from _curves import BASELINE_STYLE, epochs_to_target

    apply_thesis_rcparams("dense")
    ds = "cifar10"
    pretty = {"sgd": "SGD", "adam": "Adam", "rmsprop": "RMSprop",
              "signsgd": "SignSGD"}
    stats: Dict[str, dict] = {}
    # the speed table the chapter quotes; kept out of `stats`, whose shape is
    # cell -> mean/std/n for main()'s printer
    summary: Dict[str, dict] = {}
    rows = []

    def seed_array(recs, key, width):
        seeds = [list(getattr(r, key)) for r in recs if getattr(r, key)]
        if not seeds:
            return None
        arr = np.full((len(seeds), width), np.nan)
        for i, s in enumerate(seeds):
            v = np.asarray(s[-width:], dtype=float)
            # a diverged seed logs inf or nan; it must not poison the mean of
            # the seeds that trained
            v[~np.isfinite(v)] = np.nan
            # A run resumed from a checkpoint logs only the epochs after the
            # resume point, and those are the LAST epochs of the run. Aligning
            # them to the start drew one Adam seed's epochs 14 to 20 as 1 to 7.
            arr[i, width - len(v):] = v
        return arr

    for base in BASES:
        recs = _records("10_02_lr_retune", base, ds)
        if not recs:
            continue
        stats[f"{ds}/{base}"] = final_means(recs)
        arms: Dict[str, Dict[float, List]] = {"baseline": {}, "bograd": {}}
        K = None
        for r in recs:
            head, _, lr_s = r.cell.rpartition("_lr")
            try:
                lr = float(lr_s)
            except ValueError:
                continue
            arm = "baseline" if head.startswith("baseline") else "bograd"
            if arm == "bograd" and head.startswith("bograd_K"):
                K = head[len("bograd_K"):]
            arms[arm].setdefault(lr, []).append(r)
        if not arms["baseline"] or not arms["bograd"]:
            continue

        width = max(len(r.acc) for r in recs if r.acc)
        acc = {a: {lr: seed_array(rs, "acc", width) for lr, rs in d.items()}
               for a, d in arms.items()}
        loss = {a: {lr: seed_array(rs, "loss", width) for lr, rs in d.items()}
                for a, d in arms.items()}

        # one common bar per base: 99% of the best final accuracy of any cell
        finals = {(a, lr): float(np.nanmean(v[:, -1]))
                  for a, d in acc.items() for lr, v in d.items()}
        target = 0.99 * max(finals.values())

        def speed(a, lr):
            per_seed = []
            for c in acc[a][lr]:
                # NaN marks an epoch this seed did not log. It is skipped, not
                # compacted out, so a resumed seed keeps its true epoch numbers
                hit = epochs_to_target(list(c), target)
                per_seed.append(hit if hit is not None else len(c) + 1)
            return float(np.mean(per_seed)), per_seed

        table = {}
        for a in ("baseline", "bograd"):
            table[a] = {}
            for lr in sorted(acc[a]):
                mean_ep, per_seed = speed(a, lr)
                table[a][lr] = dict(
                    epochs=mean_ep, per_seed=per_seed,
                    reached=sum(e <= acc[a][lr].shape[1] for e in per_seed),
                    final=finals[(a, lr)],
                    auc=float(np.nanmean(np.nanmean(acc[a][lr], axis=0))),
                    partial=int(np.isnan(acc[a][lr][:, 0]).sum()))
        base_lr = min(table["baseline"],
                      key=lambda lr: (table["baseline"][lr]["epochs"],
                                      -table["baseline"][lr]["auc"]))
        summary[base] = dict(
            target=target, K=K, baseline_lr=base_lr,
            baseline={f"{lr:g}": v for lr, v in table["baseline"].items()},
            bograd={f"{lr:g}": v for lr, v in table["bograd"].items()})
        rows.append((base, K, base_lr, acc, loss, table, target))

    if not rows:
        print("  lr: no results — skipped")
        return None

    fig, axes = plt.subplots(len(rows), 2, figsize=(10.5, 2.95 * len(rows)),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, (base, K, base_lr, acc, loss, table, target) in enumerate(rows):
        lrs = sorted(acc["bograd"])
        colour = {lr: cmap(0.05 + 0.82 * j / max(1, len(lrs) - 1))
                  for j, lr in enumerate(lrs)}
        name = f"{pretty.get(base, base)}, BOGrad $K = {K}$"
        for col, (curves, ylabel) in enumerate(((acc, "test accuracy"),
                                                (loss, "training loss"))):
            ax = axes[i][col]
            top = 0.0
            for lr in lrs:
                arr = curves["bograd"][lr]
                if arr is None:
                    continue
                m = np.nanmean(arr, axis=0)
                x = np.arange(1, len(m) + 1)
                ax.plot(x, m, color=colour[lr], linewidth=1.6,
                        label=f"BOGrad, lr {lr:g}")
                if col == 1 and np.isfinite(m).any():
                    top = max(top, float(np.nanmax(m)))
            arr = curves["baseline"][base_lr]
            m = np.nanmean(arr, axis=0)
            s = np.nanstd(arr, axis=0)
            x = np.arange(1, len(m) + 1)
            ax.plot(x, m, label=f"baseline, lr {base_lr:g}", **BASELINE_STYLE)
            ax.fill_between(x, m - s, m + s, color=BASELINE_STYLE["color"],
                            alpha=0.12, linewidth=0)
            if col == 0:
                ax.axhline(target, color="0.6", linestyle=":", linewidth=1.0,
                           zorder=0)
            else:
                # a diverging rate sends its loss off the top; hold the axis to
                # the range where the trained curves are distinguishable
                top = max(top, float(np.nanmax(m)))
                ax.set_ylim(0.0, min(top, 2.5) * 1.05)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if col == 1:
                # outside the axes: inside, it sat on top of the loss curves
                ax.legend(fontsize=7.5, loc="upper left",
                          bbox_to_anchor=(1.02, 1.0), frameon=False)

    fig.tight_layout()
    out = outdir / "ax_lr_curves"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")

    for base, K, base_lr, acc, loss, table, target in rows:
        print(f"\n  {base} (K={K})  target {target:.4f}  ideal baseline lr {base_lr:g}")
        for a in ("baseline", "bograd"):
            for lr, v in table[a].items():
                mark = "  <- drawn" if a == "baseline" and lr == base_lr else ""
                cens = "" if v["reached"] == len(v["per_seed"]) else "*"
                if v["partial"]:
                    mark += "  (%d seed resumed, early epochs unlogged)" % v["partial"]
                print(f"    {a:8s} lr {lr:<7g} {v['epochs']:5.1f}{cens:1s} "
                      f"{str(v['per_seed']):14s} final {v['final']:.4f} "
                      f"auc {v['auc']:.4f}{mark}")
    (_HERE / "lr_curves.json").write_text(json.dumps(summary, indent=2))
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
    # Merge rather than overwrite: plotting one axis used to erase every other
    # axis's statistics from this file.
    merged = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text())
        except Exception:
            merged = {}
    merged.update(allstats)
    out.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {out}")
    for axis, per in allstats.items():
        print(f"\n=== {axis} (final test accuracy, mean +/- std) ===")
        for grp, cells in per.items():
            print(f"  {grp}")
            for cell, v in cells.items():
                print(f"    {cell:<30} {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
