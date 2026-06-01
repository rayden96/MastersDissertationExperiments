"""
Aggregation helpers — convert a meter's raw log list into run-level summaries
and pairwise correlations between geometric metrics and the per-step deficit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _fvals(logs: Iterable[Dict], key: str) -> List[float]:
    return [
        l[key] for l in logs
        if key in l and isinstance(l[key], (int, float)) and math.isfinite(l[key])
    ]


def safe_mean(logs: Iterable[Dict], key: str) -> float:
    v = _fvals(logs, key)
    return float(np.mean(v)) if v else float("nan")


def safe_std(logs: Iterable[Dict], key: str) -> float:
    v = _fvals(logs, key)
    return float(np.std(v)) if len(v) > 1 else 0.0


def correlate(logs: Iterable[Dict], key_x: str, key_y: str) -> float:
    """Pearson correlation between two log fields across logged steps.
    Returns NaN if fewer than 5 valid pairs."""
    pairs = [
        (l[key_x], l[key_y]) for l in logs
        if key_x in l and key_y in l
        and isinstance(l[key_x], (int, float)) and isinstance(l[key_y], (int, float))
        and math.isfinite(l[key_x]) and math.isfinite(l[key_y])
    ]
    if len(pairs) < 5:
        return float("nan")
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def summarize_run(
    logs: List[Dict],
    calibration_logs: Optional[List[Dict]] = None,
    K_values: Iterable[int] = (4, 32, 128),
    cum_deficit: Optional[float] = None,
    cum_deficit_count: Optional[int] = None,
    cum_deficit_precond: Optional[float] = None,
    cum_deficit_precond_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert raw per-step logs into a flat summary dict with run-level stats
    and the standard correlations.

    `cum_deficit` is the SGD-yardstick first-order training-hurt total (the
    cross-method common scale). `cum_deficit_precond`, when provided, is the
    per-optimiser deficit that isolates cancellation from preconditioning
    (only meaningful when the problem implements `preconditioned_ideal`)."""

    summary: Dict[str, Any] = {
        "n_logs": len(logs),
        "n_steps_total": cum_deficit_count if cum_deficit_count is not None else len(logs),
        "cum_deficit": cum_deficit if cum_deficit is not None else float("nan"),
        "mean_deficit_per_step": (
            (cum_deficit / max(cum_deficit_count, 1))
            if (cum_deficit is not None and cum_deficit_count is not None)
            else float("nan")
        ),
        "cum_deficit_precond": (
            cum_deficit_precond if cum_deficit_precond is not None else float("nan")
        ),
        "mean_deficit_precond_per_step": (
            (cum_deficit_precond / max(cum_deficit_precond_count, 1))
            if (cum_deficit_precond is not None and cum_deficit_precond_count)
            else float("nan")
        ),

        # §03 inter-batch
        "I_inter_mean":               safe_mean(logs, "I_inter"),
        "I_inter_std":                safe_std(logs, "I_inter"),
        "inter_frac_neg_mean":        safe_mean(logs, "inter_frac_neg"),
        "inter_mean_cos_mean":        safe_mean(logs, "inter_mean_cos"),
        "inter_mean_grad_norm_mean":  safe_mean(logs, "inter_mean_grad_norm"),
        "inter_max_min_ratio_mean":   safe_mean(logs, "inter_max_min_ratio"),
        "inter_useful_mass_mean":     safe_mean(logs, "inter_useful_mass"),
        "inter_wasted_mass_mean":     safe_mean(logs, "inter_wasted_mass"),
        "inter_useful_descent_frac_mean": safe_mean(logs, "inter_useful_descent_frac"),
    }

    # §04 between-batch, per K
    for K in K_values:
        summary[f"I_between_K{K}_mean"] = safe_mean(logs, f"I_between_K{K}")
        summary[f"I_between_K{K}_std"]  = safe_std(logs,  f"I_between_K{K}")
        summary[f"between_K{K}_frac_neg_mean"] = safe_mean(logs, f"between_K{K}_frac_neg")
        summary[f"between_K{K}_mean_cos_mean"] = safe_mean(logs, f"between_K{K}_mean_cos")
        summary[f"between_K{K}_useful_path_frac_mean"] = safe_mean(
            logs, f"between_K{K}_useful_path_frac"
        )
        summary[f"between_K{K}_mean_u_norm_mean"] = safe_mean(
            logs, f"between_K{K}_mean_u_norm"
        )

    # Standard correlations: do geometric summaries predict per-step deficit?
    summary["corr_I_inter_vs_Dt"]            = correlate(logs, "I_inter", "D_t")
    summary["corr_inter_mean_cos_vs_Dt"]     = correlate(logs, "inter_mean_cos", "D_t")
    summary["corr_inter_max_min_ratio_vs_Dt"] = correlate(logs, "inter_max_min_ratio", "D_t")
    summary["corr_I_between_K32_vs_Dt"]      = correlate(logs, "I_between_K32", "D_t")
    summary["corr_between_K32_mean_cos_vs_Dt"] = correlate(logs, "between_K32_mean_cos", "D_t")

    # Loss-drop calibration if provided
    if calibration_logs and len(calibration_logs) >= 2:
        summary["ref_loss_initial"] = float(calibration_logs[0]["ref_loss"])
        summary["ref_loss_final"]   = float(calibration_logs[-1]["ref_loss"])
        summary["ref_loss_drop"]    = summary["ref_loss_initial"] - summary["ref_loss_final"]

    return summary
