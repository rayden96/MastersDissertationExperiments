"""
common.plotting — consistent thesis figure style + run loading/aggregation.

docs/experiment_design.md §8. Plotting code (`plot.py` in each experiment) is
pure presentation: it loads results.json files, aggregates across seeds, and
renders. It never trains and never reads checkpoints.

Entry points
------------
  apply_thesis_rcparams(density="normal")
  PALETTE, METHOD_STYLE                     consistent colours/linestyles
  load_runs(run_dir)            -> list[dict]   all results.json under a dir
  load_seed_logs(run_dir)       -> dict        per-seed interference logs
  curve_mean_std(seed_curves)   -> (mean, std)
  series_mean_std(seed_logs, key) -> (steps, mean, std)   aligned across seeds
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Base-optimizer colour, method linestyle — so a (base × method) line is
# identifiable by colour+style consistently across every figure.
PALETTE = {
    "sgd": "#1f77b4", "signsgd": "#2ca02c", "rmsprop": "#ff7f0e", "adam": "#d62728",
}
METHOD_STYLE = {
    "baseline": "-", "bograd": "--", "cosgd": ":", "graddrop": "-.", "dropout": (0, (3, 1, 1, 1)),
}


def apply_thesis_rcparams(density: str = "normal") -> None:
    """Set matplotlib rcParams for thesis-consistent figures. `density` scales
    font sizes down for dense multi-panel figures."""
    import matplotlib as mpl

    scale = {"normal": 1.0, "dense": 0.8, "sparse": 1.15}.get(density, 1.0)
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 16 * scale,
        "axes.titlesize": 20 * scale,
        "axes.labelsize": 18 * scale,
        "xtick.labelsize": 16 * scale,
        "ytick.labelsize": 16 * scale,
        "legend.fontsize": 14 * scale,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "figure.autolayout": True,
    })


def style_for(base: str, method: str) -> Dict[str, Any]:
    return {"color": PALETTE.get(base, "#555555"),
            "linestyle": METHOD_STYLE.get(method, "-")}


# ---------------------------------------------------------------------------
# Loading / aggregation
# ---------------------------------------------------------------------------
def load_runs(run_dir: Path) -> List[Dict[str, Any]]:
    """Load every results.json beneath `run_dir` (recursively)."""
    run_dir = Path(run_dir)
    out = []
    for rj in sorted(run_dir.rglob("results.json")):
        try:
            out.append(json.loads(rj.read_text()))
        except Exception:
            pass
    return out


def load_seed_logs(run_dir: Path) -> Dict[int, List[Dict]]:
    """Map seed -> interference_logs for runs under `run_dir` (those that have them)."""
    out: Dict[int, List[Dict]] = {}
    for r in load_runs(run_dir):
        if "interference_logs" in r:
            seed = r.get("config", {}).get("seed")
            out[seed] = r["interference_logs"]
    return out


def curve_mean_std(seed_curves: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std across seeds of equal-length per-epoch curves (truncates to the
    shortest if seeds differ in length)."""
    if not seed_curves:
        return np.array([]), np.array([])
    n = min(len(c) for c in seed_curves)
    M = np.stack([np.asarray(c[:n], dtype=float) for c in seed_curves], axis=0)
    return M.mean(0), M.std(0)


def series_mean_std(seed_logs: List[List[Dict]], key: str):
    """Align a per-step interference series across seeds (by common steps) and
    return (steps, mean, std). Generalises the helper in image_runner."""
    per_seed = []
    for logs in seed_logs:
        steps = [l["step"] for l in logs if key in l and l[key] == l[key]]
        vals = [l[key] for l in logs if key in l and l[key] == l[key]]
        per_seed.append((np.array(steps), np.array(vals, dtype=np.float64)))
    if not per_seed:
        return np.array([]), np.array([]), np.array([])
    common = set(per_seed[0][0].tolist())
    for s, _ in per_seed[1:]:
        common &= set(s.tolist())
    common = sorted(common)
    if not common:
        return np.array([]), np.array([]), np.array([])
    aligned = []
    for s, v in per_seed:
        idx = np.array([np.where(s == cs)[0][0] for cs in common])
        aligned.append(v[idx])
    M = np.stack(aligned, axis=0)
    return np.array(common), M.mean(0), M.std(0)


__all__ = [
    "apply_thesis_rcparams", "style_for", "PALETTE", "METHOD_STYLE",
    "load_runs", "load_seed_logs", "curve_mean_std", "series_mean_std",
]
