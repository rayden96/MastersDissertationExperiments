"""
Shared per-epoch curve loading and rendering for the ablation chapters.

Chapters 4 and 5 report each design axis as *curves* rather than as a
scalar speed-up against the baseline: for every cell in the axis, test
accuracy and training loss against epoch, averaged over seeds, with the
axis baseline drawn as a reference line. The comparative timing question
(wall-clock and steps to a target) belongs to Chapter 6 and is deliberately
not answered here — these figures are for reading off which setting of a
knob is better, not for ranking the method against a baseline.

Every ablation run already writes what is needed:
    results.json -> history.epoch_test_acc, history.epoch_train_loss
so nothing has to be re-run to produce these figures.

Entry points
------------
    load_axis(root)                 -> list[CurveRecord]
    group(records, ...)             -> {cell_label: {"acc": [...], "loss": [...]}}
    plot_axis(panels, out_path)     -> writes .pdf and .png
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_PRE = Path(__file__).resolve().parent
_REPO = _PRE.parent
for _p in (str(_REPO), str(_PRE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.plotting import apply_thesis_rcparams  # noqa: E402


def curve_stats(seed_curves: Sequence[Sequence[float]]):
    """Mean and std across seeds, padding to the longest curve rather than
    truncating to the shortest.

    common.plotting.curve_mean_std truncates, which silently cuts a whole
    panel back to its shortest seed when one run was resumed and lost part of
    its history. Here the tail is averaged over whichever seeds reach it.
    """
    if not seed_curves:
        return np.array([]), np.array([])
    n = max(len(c) for c in seed_curves)
    M = np.full((len(seed_curves), n), np.nan)
    for i, c in enumerate(seed_curves):
        M[i, :len(c)] = np.asarray(c, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(M, axis=0), np.nan_to_num(np.nanstd(M, axis=0))

# The baseline is always drawn the same way, in every axis of both chapters,
# so a reader can find it without consulting the legend.
BASELINE_STYLE = dict(color="#333333", linestyle=(0, (4, 2)), linewidth=2.4, zorder=5)

# Qualitative colours for the non-baseline cells. Deliberately not the
# base-optimiser PALETTE: within one axis the varying thing is the knob,
# not the optimiser.
CELL_COLOURS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22",
]


@dataclass
class CurveRecord:
    base: str
    cell: str
    seed: int
    dataset: str
    acc: List[float]
    loss: List[float]
    scalars: Dict[str, Any] = field(default_factory=dict)


def load_axis(root: Path) -> List[CurveRecord]:
    """Every completed run beneath `root`, with its per-epoch history.

    Runs resumed from a checkpoint carry an empty history (the epoch lists
    accumulate only from the resume point), so they are kept for their final
    scalars but contribute no curve; `group` drops the empty ones.
    """
    out: List[CurveRecord] = []
    seen = set()
    for rj in sorted(Path(root).rglob("results.json")):
        # Archives unpacked twice leave 'seed2027 (1)' siblings; keep the first
        # of any (base, cell, seed) rather than double-counting the seed.
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        h = r.get("history") or {}
        acc, loss = h.get("epoch_test_acc") or [], h.get("epoch_train_loss") or []
        cfg = r.get("config", {})
        # cell label comes from the run-dir name: <base>__<cell>__seed<n>
        parts = rj.parent.name.split("__")
        cell = parts[1] if len(parts) >= 3 else r.get("label", rj.parent.name)
        base = cfg.get("base_optimizer", parts[0] if parts else "?")
        seed = int(cfg.get("seed", -1))
        if (base, cell, seed) in seen:
            continue
        seen.add((base, cell, seed))
        out.append(CurveRecord(
            base=base, cell=cell, seed=seed,
            dataset=cfg.get("dataset", "?"),
            acc=[float(v) for v in acc], loss=[float(v) for v in loss],
            scalars=r.get("scalars", {})))
    return out


def group(records: Iterable[CurveRecord],
          base: Optional[str] = None) -> Dict[str, Dict[str, List[List[float]]]]:
    """Collect the per-seed curves for each cell label."""
    out: Dict[str, Dict[str, List[List[float]]]] = {}
    for r in records:
        if base is not None and r.base != base:
            continue
        d = out.setdefault(r.cell, {"acc": [], "loss": []})
        if r.acc:
            d["acc"].append(r.acc)
        if r.loss:
            d["loss"].append(r.loss)
    return out


def final_means(records: Iterable[CurveRecord]) -> Dict[str, Dict[str, float]]:
    """Mean +/- std of final test accuracy per cell — for quoting in prose."""
    by: Dict[str, List[float]] = {}
    for r in records:
        v = r.scalars.get("final_test_acc")
        if isinstance(v, (int, float)) and v == v:
            by.setdefault(r.cell, []).append(float(v))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for k, v in sorted(by.items())}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _draw(ax, curves: Dict[str, Dict[str, List[List[float]]]], which: str,
          order: Sequence[str], pretty: Dict[str, str],
          baseline_key: str, band: bool) -> None:
    ci = 0
    for cell in order:
        seeds = curves.get(cell, {}).get(which, [])
        if not seeds:
            continue
        mean, std = curve_stats(seeds)
        x = np.arange(1, len(mean) + 1)
        if cell == baseline_key:
            style = dict(BASELINE_STYLE)
        else:
            style = dict(color=CELL_COLOURS[ci % len(CELL_COLOURS)], linewidth=1.9)
            ci += 1
        ax.plot(x, mean, label=pretty.get(cell, cell), **style)
        if band and len(seeds) > 1:
            ax.fill_between(x, mean - std, mean + std, alpha=0.13,
                            color=style["color"], linewidth=0)


def plot_axis(panels: Sequence[Tuple[str, Dict[str, Dict[str, List[List[float]]]]]],
              out_path: Path,
              *,
              order: Optional[Sequence[str]] = None,
              pretty: Optional[Dict[str, str]] = None,
              baseline_key: str = "baseline",
              band: bool = True,
              log_loss: bool = True,
              acc_ylim: Optional[Tuple[float, float]] = None,
              width: float = 11.0,
              row_height: float = 3.6) -> Path:
    """One row per panel (typically one per dataset), two columns:
    test accuracy and training loss against epoch.

    `panels` is [(row_title, curves_dict), ...] as returned by `group`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_thesis_rcparams("dense")
    pretty = pretty or {}
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(width, row_height * n), squeeze=False)

    for row, (title, curves) in enumerate(panels):
        keys = list(order) if order else sorted(curves)
        keys = [k for k in keys if k in curves]
        for k in sorted(curves):                      # anything the caller missed
            if k not in keys:
                keys.append(k)
        axA, axB = axes[row][0], axes[row][1]
        _draw(axA, curves, "acc", keys, pretty, baseline_key, band)
        _draw(axB, curves, "loss", keys, pretty, baseline_key, band)
        axA.set_ylabel("test accuracy")
        axB.set_ylabel("training loss")
        if log_loss:
            axB.set_yscale("log")
        if acc_ylim:
            axA.set_ylim(*acc_ylim)
        for ax in (axA, axB):
            ax.set_xlabel("epoch")
        axA.set_title(f"{title} — accuracy")
        axB.set_title(f"{title} — training loss")
        if row == 0:
            axA.legend(fontsize=9, framealpha=0.9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.with_suffix('.pdf')}")
    return out_path.with_suffix(".pdf")


def plot_grid(cells: Sequence[Tuple[str, Dict[str, Dict[str, List[List[float]]]]]],
              out_path: Path,
              *,
              ncols: int,
              which: str = "acc",
              order: Optional[Sequence[str]] = None,
              pretty: Optional[Dict[str, str]] = None,
              baseline_key: str = "baseline",
              band: bool = True,
              panel: float = 3.4) -> Path:
    """A free grid of single-metric panels — used where the axis varies two
    things at once (dataset x base optimiser) and the two-column layout of
    `plot_axis` would not fit on the page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_thesis_rcparams("dense")
    pretty = pretty or {}
    nrows = (len(cells) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows),
                             squeeze=False)
    for i, (title, curves) in enumerate(cells):
        ax = axes[i // ncols][i % ncols]
        keys = [k for k in (order or sorted(curves)) if k in curves]
        for k in sorted(curves):
            if k not in keys:
                keys.append(k)
        _draw(ax, curves, which, keys, pretty, baseline_key, band)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy" if which == "acc" else "training loss")
        if which == "loss":
            ax.set_yscale("log")
        if i == 0:
            ax.legend(fontsize=9)
    for j in range(len(cells), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.with_suffix('.pdf')}")
    return out_path.with_suffix(".pdf")


__all__ = ["CurveRecord", "load_axis", "group", "final_means", "curve_stats",
           "plot_axis", "plot_grid", "BASELINE_STYLE", "CELL_COLOURS"]
