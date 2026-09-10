"""
Chapter 4 figures: one per COSGD ablation axis, as learning curves.

Each axis becomes a figure with one row per anchor dataset (Covertype,
CIFAR-10) and two columns: test accuracy and training loss against epoch,
mean over seeds with a +/-1 std band, and the axis baseline drawn as a
dashed grey reference. The knob is read off the curves; the comparative
timing question (wall-clock and steps to a target) belongs to Chapter 6.

Reads the per-run results.json files already on disk — nothing is re-run.

Usage:
    python plot_curves.py                       # every axis with results
    python plot_curves.py --axes combine prenorm
    python plot_curves.py --outdir some/other/dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                   # noqa: E402
from _curves import load_axis, group, final_means, plot_axis, plot_grid  # noqa: E402
from pick_lr import best_rates                              # noqa: E402

# The two anchors of Chapters 4 and 5. Everything else in results/ is from
# the earlier small-dataset sweeps and is deliberately not plotted.
# CIFAR-10 is the sole anchor for the ablation chapters. Covertype is
# retained in the head-to-head comparison of Chapter 6, where it stands
# for the tabular regime, but carrying two anchors through every design
# axis doubled the grid without changing which setting of any knob won.
ANCHORS = ["cifar10"]
ANCHOR_TITLE = {"covertype": "Covertype (7 classes)", "cifar10": "CIFAR-10 (10 classes)"}

BASES = ["sgd", "adam", "rmsprop", "signsgd"]

_RATE_CACHE: Dict[str, Optional[tuple]] = {}


def anchor_rates(dataset: str) -> Optional[tuple]:
    """The pair of rates this anchor's axes were launched at, from axis 20.10.

    Every fixed-knob axis runs its COSGD cells at one rate and its baseline at
    another, because the two arms' optima are more than a decade apart. Any
    cell on disk recorded at some other rate is left over from an earlier
    sweep, so the pair doubles as the filter that keeps those out: it needs no
    maintenance, unlike a list of superseded run ids.
    """
    if dataset not in _RATE_CACHE:
        try:
            arms = best_rates(dataset)
            rates = tuple(sorted({arms[a]["lr"] for a in ("cosgd", "baseline")
                                  if a in arms}))
        except Exception:
            rates = ()
        _RATE_CACHE[dataset] = rates or None
    return _RATE_CACHE[dataset]

# axis key -> (results-dir stem, figure stem, cell order, pretty labels)
AXES: Dict[str, dict] = {
    "combine": dict(
        stem="20_05_combine", fig="ax_combine",
        order=["baseline", "sum", "mean", "freq"],
        pretty={"sum": "sum", "mean": "mean", "freq": "frequency"},
    ),
    # Only the projection mode is reported. The classical and modified
    # Gram-Schmidt variants were both swept and are indistinguishable here
    # (13.00 epochs each in full mode), so the axis is shown on classical
    # alone and the two modes are compared without a second factor moving.
    "gs_variant": dict(
        stem="20_01_gs_variant", fig="ax_gs_variant",
        order=["baseline", "gram_schmidt_normal", "gram_schmidt_negative"],
        pretty={"gram_schmidt_normal": "full projection",
                "gram_schmidt_negative": "negative-only"},
    ),
    "class_order": dict(
        stem="20_02_class_order", fig="ax_class_order",
        order=["baseline", "order_desc", "order_asc", "order_random", "order_fixed"],
        pretty={"order_desc": "descending magnitude", "order_asc": "ascending magnitude",
                "order_random": "random", "order_fixed": "fixed (label)"},
    ),
    "prenorm": dict(
        stem="20_03_prenormalize", fig="ax_prenorm",
        order=["baseline", "prenorm0", "prenorm1"],
        pretty={"prenorm0": "pre-normalisation off", "prenorm1": "pre-normalisation on"},
    ),
    "orth_strength": dict(
        stem="20_11_orth_strength", fig="ax_orth_strength",
        order=["baseline", "orth0p25", "orth0p5", "orth0p75", "orth1"],
        pretty={"orth0p25": r"$\alpha = 0.25$", "orth0p5": r"$\alpha = 0.5$",
                "orth0p75": r"$\alpha = 0.75$", "orth1": r"$\alpha = 1$ (full)"},
    ),
    "preserve_magnitude": dict(
        stem="20_12_preserve_magnitude", fig="ax_preserve_magnitude",
        order=["baseline", "preserve0", "preserve1"],
        pretty={"preserve0": "magnitude free", "preserve1": "magnitude preserved"},
    ),
}

# The batch-size sweep varies the knob AND carries a matched baseline at every
# value, so each value gets its own panel. The learning-rate sweep has the same
# shape but is drawn by `do_learning_rate`, as a single panel.
HYPER_AXES: Dict[str, dict] = {
    "batch_size": dict(stem="20_09_batch_size", fig="ax_batch_size",
                       filter_lr=True,
                       label=lambda v: f"batch {v[2:]}"),
}


def _results_root() -> Path:
    return storage.get_results_root() / "20_cosgd_ablation"


def _default_outdir() -> Path:
    d = _REPO.parent / "dissertation" / "chapters" / "cosgd" / "figures"
    return d if d.exists() else (_HERE / "figures")


def _axis_dirs(stem: str, dataset: str) -> List[Path]:
    """Every results directory for (axis, dataset). Axes that were launched
    per-base have one directory each; the rest have a single one."""
    root = _results_root()
    hits = [p for p in sorted(root.glob(f"{stem}_{dataset}*")) if p.is_dir()]
    return hits


def _sorted_numeric(keys, prefix: str) -> List[str]:
    """Order cells like 'bs64','bs128','lr0p005' numerically, baseline first.

    The learning-rate labels encode the decimal point as 'p' and a minus as
    'm' (a dot is not safe in a directory name), so both have to be decoded
    before parsing or every rate below one collapses to the same key and the
    panels come out shuffled.
    """
    def k(s):
        body = s.lstrip("abcdefghijklmnopqrstuvwxyz_")
        body = body.replace("p", ".").replace("m", "-")
        try:
            return float(body)
        except ValueError:
            return float("inf")
    rest = sorted([x for x in keys if x != "baseline"], key=k)
    return (["baseline"] if "baseline" in keys else []) + rest


def do_axis(key: str, outdir: Path, quiet: bool = False) -> Optional[dict]:
    spec = AXES[key]
    drop = set(spec.get("exclude") or ())
    panels, stats = [], {}
    for ds in ANCHORS:
        recs = []
        for d in _axis_dirs(spec["stem"], ds):
            recs.extend(load_axis(d, require_lr=anchor_rates(ds)))
        recs = [r for r in recs if r.cell not in drop]
        if not recs:
            continue
        curves = group(recs)
        panels.append((ANCHOR_TITLE.get(ds, ds), curves))
        stats[ds] = final_means(recs)
    if not panels:
        if not quiet:
            print(f"  {key}: no results under {_results_root()} — skipped")
        return None

    order = list(spec["order"])
    if not order:                                    # numeric sweeps (bs, lr)
        keys = {k for _, c in panels for k in c}
        order = _sorted_numeric(keys, key)

    # One figure per subset. A panel carrying more than four or five lines is
    # unreadable once scaled to the text width, so axes that sweep many cells
    # are split into figures that each answer one question.
    for suffix, sub_order in (spec.get("subsets") or [("", order)]):
        keep = set(sub_order)
        sub = [(title, {k: v for k, v in curves.items() if k in keep})
               for title, curves in panels]
        stem = spec["fig"] + (f"_{suffix}" if suffix else "")
        plot_axis(sub, outdir / stem, order=sub_order, pretty=spec["pretty"])
    return stats


def do_hyper_axis(key: str, outdir: Path) -> Optional[dict]:
    """Cells are labelled '<value>_baseline' / '<value>_cosgd'. One panel per
    value, rows are the anchors, and each panel holds the matched pair."""
    spec = HYPER_AXES[key]
    cells, stats = [], {}
    by_anchor: Dict[str, list] = {ds: [] for ds in ANCHORS}
    for ds in ANCHORS:
        recs = []
        lr_filter = anchor_rates(ds) if spec.get("filter_lr") else None
        for d in _axis_dirs(spec["stem"], ds):
            recs.extend(load_axis(d, require_lr=lr_filter))
        if not recs:
            continue
        stats[ds] = final_means(recs)
        by_value: Dict[str, List] = {}
        for r in recs:
            value, _, arm = r.cell.rpartition("_")
            if not value:
                continue
            r.cell = arm
            by_value.setdefault(value, []).append(r)
        for value in _sorted_numeric(list(by_value), key):
            if value == "baseline" or value not in by_value:
                continue
            title = f"{ANCHOR_TITLE.get(ds, ds).split(' (')[0]} — {spec['label'](value)}"
            panel = (title, group(by_value[value]))
            cells.append(panel)
            by_anchor[ds].append(panel)
    if not cells:
        print(f"  {key}: no results — skipped")
        return None
    # One figure per anchor: a single grid holding both would be twelve panels
    # wide and unreadable at text width.
    for ds, panels in by_anchor.items():
        if not panels:
            continue
        plot_grid(panels, outdir / f"{spec['fig']}_{ds}", ncols=3, which="acc",
                  order=["baseline", "cosgd"], pretty={"cosgd": "COSGD"})
    return stats


def do_base_optimizer(outdir: Path) -> Optional[dict]:
    """Two things vary at once here, so it gets a grid of accuracy panels:
    rows are the anchors, columns the base optimisers.

    Only the SGD column takes the anchor's rates. The adaptive bases run at
    their own 1e-3 default, which already sits inside COSGD's usable band, so
    they were never mis-tuned and filtering them on SGD's rates would discard
    every one of them.
    """
    cells, stats = [], {}
    for ds in ANCHORS:
        for base in BASES:
            recs = []
            lr_filter = anchor_rates(ds) if base == "sgd" else None
            for d in sorted(_results_root().glob(f"20_06_base_optimizer_{ds}_{base}*")):
                recs.extend(load_axis(d, require_lr=lr_filter))
            if not recs:
                continue
            cells.append((f"{ANCHOR_TITLE.get(ds, ds).split(' (')[0]} — {base}",
                          group(recs)))
            stats[f"{ds}/{base}"] = final_means(recs)
    if not cells:
        print("  base_optimizer: no results — skipped")
        return None
    plot_grid(cells, outdir / "ax_base_optimizer", ncols=len(BASES), which="acc",
              order=["baseline", "cosgd"], pretty={"cosgd": "COSGD"})
    return stats


def do_learning_rate(outdir: Path) -> Optional[dict]:
    """20.10 sweeps the rate for both arms. One panel of test accuracy: the
    baseline at its own best rate against COSGD at every rate on the grid.

    The drawn baseline is the rate `pick_lr.best_rates` selects, which is the
    rate every fixed-knob axis of the chapter runs its baseline at, so the
    dashed curve is the same reference as on every earlier figure. The other
    baseline rates are left off: the question is where COSGD's usable band
    sits against a properly tuned baseline, and a second family of curves
    would bury it.

    Speed is measured as on every other axis, epochs to 99% of the drawn
    baseline's own final accuracy, with a seed that never arrives charged the
    budget plus one. The per-rate table goes to lr_curves.json.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotting import apply_thesis_rcparams
    from _curves import BASELINE_STYLE, epochs_to_target
    from pick_lr import TARGET_FRAC

    ds = "cifar10"
    recs = []
    for d in _axis_dirs("20_10_learning_rate", ds):
        recs.extend(load_axis(d))
    recs = [r for r in recs if r.dataset == ds and r.acc]
    if not recs:
        print("  learning_rate: no results — skipped")
        return None
    stats = {ds: final_means(recs)}

    arms: Dict[str, Dict[float, list]] = {}
    for r in recs:
        tag, _, arm = r.cell.rpartition("_")
        lr = float(tag[2:].replace("p", ".").replace("m", "-"))
        arms.setdefault(arm, {}).setdefault(lr, []).append(r)
    width = max(len(r.acc) for r in recs)

    def seed_array(rs):
        arr = np.full((len(rs), width), np.nan)
        for i, r in enumerate(rs):
            v = np.asarray(r.acc[-width:], dtype=float)
            v[~np.isfinite(v)] = np.nan
            # a run resumed from a checkpoint logs only the epochs after the
            # resume point, and those are the last epochs of the run
            arr[i, width - len(v):] = v
        return arr

    acc = {arm: {lr: seed_array(rs) for lr, rs in by.items()}
           for arm, by in arms.items()}
    base_lr = best_rates(ds)["baseline"]["lr"]
    ref = acc["baseline"][base_lr]
    target = TARGET_FRAC * float(np.nanmean(ref[:, -1]))

    table: Dict[str, Dict[str, dict]] = {}
    for arm in ("baseline", "cosgd"):
        table[arm] = {}
        for lr in sorted(acc[arm]):
            per_seed = []
            for c in acc[arm][lr]:
                hit = epochs_to_target(list(c), target)
                per_seed.append(hit if hit is not None else width + 1)
            table[arm][f"{lr:g}"] = dict(
                epochs=float(np.mean(per_seed)), per_seed=per_seed,
                reached=sum(e <= width for e in per_seed),
                final=float(np.nanmean(acc[arm][lr][:, -1])))
    base_ep = table["baseline"][f"{base_lr:g}"]["epochs"]
    for arm in table:
        for v in table[arm].values():
            v["speedup"] = base_ep / v["epochs"]
    (_HERE / "lr_curves.json").write_text(json.dumps(
        dict(dataset=ds, target=target, baseline_lr=base_lr, **table), indent=1))

    apply_thesis_rcparams("dense")
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    cmap = plt.get_cmap("viridis")
    lrs = sorted(acc["cosgd"])
    x = np.arange(1, width + 1)
    for j, lr in enumerate(lrs):
        ax.plot(x, np.nanmean(acc["cosgd"][lr], axis=0), linewidth=1.8,
                color=cmap(0.05 + 0.82 * j / max(1, len(lrs) - 1)),
                label=f"COSGD, lr {lr:g}")
    m, s = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
    ax.plot(x, m, label=f"baseline, lr {base_lr:g}", **BASELINE_STYLE)
    ax.fill_between(x, m - s, m + s, color=BASELINE_STYLE["color"],
                    alpha=0.12, linewidth=0)
    ax.axhline(target, color="0.6", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("CIFAR-10: learning rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True)
    out = outdir / "ax_learning_rate_cifar10"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")

    print(f"\n  {ds}  target {target:.4f}  drawn baseline lr {base_lr:g}")
    for arm in ("baseline", "cosgd"):
        for lr, v in table[arm].items():
            cens = "" if v["reached"] == len(v["per_seed"]) else "*"
            print(f"    {arm:8s} lr {lr:<6s} {v['epochs']:5.1f}{cens:1s} "
                  f"{str(v['per_seed']):14s} final {v['final']:.4f} "
                  f"speed-up {v['speedup']:.2f}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+",
                    default=list(AXES) + list(HYPER_AXES)
                    + ["learning_rate", "base_optimizer"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else _default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"results root: {_results_root()}")
    print(f"figures ->    {outdir}\n")

    allstats: Dict[str, dict] = {}
    for key in args.axes:
        print(f"[{key}]")
        if key == "base_optimizer":
            s = do_base_optimizer(outdir)
        elif key == "learning_rate":
            s = do_learning_rate(outdir)
        elif key in HYPER_AXES:
            s = do_hyper_axis(key, outdir)
        else:
            s = do_axis(key, outdir)
        if s:
            allstats[key] = s

    # Final accuracies per cell, so the prose can quote them without
    # re-deriving anything from the figures. Merged, so that redrawing one
    # axis does not wipe the others' entries.
    out = _HERE / "curve_stats.json"
    merged: Dict[str, dict] = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text())
        except ValueError:
            merged = {}
    merged.update(allstats)
    out.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {out}")
    for axis, per_ds in allstats.items():
        print(f"\n=== {axis} (final test accuracy, mean +/- std) ===")
        for ds, cells in per_ds.items():
            print(f"  {ds}")
            for cell, v in cells.items():
                print(f"    {cell:<26} {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
