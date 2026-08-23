"""
10.10 — Scale transfer: BOGrad vs baseline on CIFAR-100, read from the bakeoff.

Aggregator (no training). See README.md. For each base optimiser it reads the
30_main_comparison cell records cell_<base>__baseline.json and
cell_<base>__bograd.json for the target dataset and computes the Chapter 5
transfer row: final acc (mean +/- std), epochs-to-target, epoch speed-up and
wall speed-up (same convention as _summary_utils.speedup_for_run: target =
TARGET_FRAC x the baseline's mean final accuracy; a curve that never reaches
it is charged len(curve)+1 epochs).

Run:
    python run.py                     # campaign=main, dataset=cifar100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_AXIS_ROOT = _HERE.parent
_PRE = _AXIS_ROOT.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json, write_json_atomic, get_results_root  # noqa: E402
from _summary_utils import epochs_to_target, TARGET_FRAC  # noqa: E402

BASES = ["sgd", "signsgd", "rmsprop", "adam"]


def _candidate_dirs(campaign: str, dataset: str) -> List[Path]:
    """Where the bakeoff's per-dataset cell_*.json may live: the persistent
    (Drive) results root first, then the repo-local _core layout."""
    return [
        get_results_root() / "30_main_comparison" / "_core" / "results" / campaign / dataset,
        get_results_root() / "30_main_comparison" / campaign / dataset,
        _PRE / "30_main_comparison" / "_core" / "results" / campaign / dataset,
    ]


def _load_cell(dirs: List[Path], base: str, method: str) -> Optional[Dict[str, Any]]:
    for d in dirs:
        p = d / f"cell_{base}__{method}.json"
        if p.exists():
            try:
                return read_json(p)
            except Exception:
                pass
    return None


def _stats(rows: List[Dict[str, Any]]):
    """(final_mean, final_std, curves, mean_step_time) from a cell's seed rows."""
    finals = [r["final_test_acc"] for r in rows
              if isinstance(r.get("final_test_acc"), (int, float))]
    curves = [r.get("epoch_test_acc") or [] for r in rows]
    curves = [c for c in curves if c]
    sts = [r["mean_step_wall_time_s"] for r in rows
           if isinstance(r.get("mean_step_wall_time_s"), (int, float))]
    fm = float(np.mean(finals)) if finals else None
    fs = float(np.std(finals)) if finals else None
    st = float(np.mean(sts)) if sts else None
    return fm, fs, curves, st


def build(campaign: str, dataset: str) -> Dict[str, Any]:
    dirs = _candidate_dirs(campaign, dataset)
    out_rows: List[Dict[str, Any]] = []
    for base in BASES:
        bl = _load_cell(dirs, base, "baseline")
        bg = _load_cell(dirs, base, "bograd")
        if bl is None or bg is None:
            print(f"  [{base}] missing cell(s) — baseline={'ok' if bl else 'MISSING'} "
                  f"bograd={'ok' if bg else 'MISSING'}; skipped", flush=True)
            continue
        b_fm, b_fs, b_curves, b_st = _stats(bl.get("rows", []))
        m_fm, m_fs, m_curves, m_st = _stats(bg.get("rows", []))
        if b_fm is None or not b_curves or not m_curves:
            print(f"  [{base}] incomplete rows; skipped", flush=True)
            continue
        # TARGET_FRAC, not the baseline's mean exactly: an exact-match
        # target is unreachable for about half the baseline's own seeds,
        # which charges them the full budget and inflates every speed-up
        # measured against them (see Section 6.2.2).
        tgt = b_fm * TARGET_FRAC
        b_ep = float(np.mean([epochs_to_target(c, tgt) or (len(c) + 1) for c in b_curves]))
        m_ep = float(np.mean([epochs_to_target(c, tgt) or (len(c) + 1) for c in m_curves]))
        sp = (b_ep / m_ep) if m_ep > 0 else None
        wall = (sp * (b_st / m_st)) if (sp and b_st and m_st) else None
        hp = bg.get("best_hp", {})
        out_rows.append({
            "base": base,
            "baseline_acc": round(b_fm, 4), "baseline_acc_std": round(b_fs, 4),
            "bograd_acc": round(m_fm, 4), "bograd_acc_std": round(m_fs, 4),
            "delta_acc": round(m_fm - b_fm, 4),
            "baseline_epochs": round(b_ep, 2), "bograd_epochs": round(m_ep, 2),
            "epoch_speedup": round(sp, 3) if sp else None,
            "wall_speedup": round(wall, 3) if wall else None,
            "step_overhead_pct": (round(100.0 * (m_st / b_st - 1.0), 1)
                                  if b_st and m_st else None),
            "tuned_K": hp.get("K"), "tuned_lr": hp.get("lr"),
        })
    return {"campaign": campaign, "dataset": dataset,
            "rows": out_rows, "n": len(out_rows)}


def _latex(res: Dict[str, Any]) -> str:
    """A LaTeX tabular for \\input in chapters/bograd (styling stays minimal;
    the chapter provides the table environment and caption)."""

    def f(v, spec="{:.3f}", na="--"):
        return spec.format(v) if isinstance(v, (int, float)) else na

    lines = [
        r"\begin{tabular}{l r r r r r r}",
        r"\hline",
        r"\textbf{Base} & \textbf{Baseline acc} & \textbf{BOGrad acc} & "
        r"\textbf{Epoch speed-up} & \textbf{Wall speed-up} & "
        r"\textbf{Overhead (\%)} & \textbf{Tuned $K$} \\",
        r"\hline",
    ]
    x = "{:.2f}$\\times$"
    for r in res["rows"]:
        K = r["tuned_K"] if r["tuned_K"] is not None else "--"
        lines.append(
            f"{r['base']} & "
            f"${f(r['baseline_acc'])} \\pm {f(r['baseline_acc_std'])}$ & "
            f"${f(r['bograd_acc'])} \\pm {f(r['bograd_acc_std'])}$ & "
            f"{f(r['epoch_speedup'], x, 'n/a')} & "
            f"{f(r['wall_speedup'], x, 'n/a')} & "
            f"{f(r['step_overhead_pct'], '{:+.1f}', 'n/a')} & "
            f"{K} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="main")
    ap.add_argument("--dataset", default="cifar100")
    args = ap.parse_args()

    res = build(args.campaign, args.dataset)
    persist_dir = get_results_root() / "10_bograd_ablation" / "10_scale_transfer"
    persist_dir.mkdir(parents=True, exist_ok=True)
    for d in (persist_dir, _HERE):
        write_json_atomic(d / "scale_transfer.json", res)
        (d / "scale_transfer.tex").write_text(_latex(res), encoding="utf-8")

    print(f"\n=== Scale transfer ({args.dataset}, campaign={args.campaign}) ===")
    hdr = (f"{'base':<9}{'bl acc':>9}{'bograd':>9}{'d-acc':>8}{'ep-spd':>8}"
           f"{'wall':>7}{'ovh%':>7}{'K':>5}")
    print(hdr); print("-" * len(hdr))
    for r in res["rows"]:
        sp = "{:.2f}x".format(r["epoch_speedup"]) if r["epoch_speedup"] else "n/a"
        wl = "{:.2f}x".format(r["wall_speedup"]) if r["wall_speedup"] else "n/a"
        ov = ("{:+.1f}".format(r["step_overhead_pct"])
              if r["step_overhead_pct"] is not None else "n/a")
        K = str(r["tuned_K"]) if r["tuned_K"] is not None else "--"
        print(f"{r['base']:<9}{r['baseline_acc']:>9.3f}{r['bograd_acc']:>9.3f}"
              f"{r['delta_acc']:>+8.3f}{sp:>8}{wl:>7}{ov:>7}{K:>5}")
    print(f"\nwrote {persist_dir / 'scale_transfer.json'} (+ .tex, + repo-local copies)")


if __name__ == "__main__":
    main()
