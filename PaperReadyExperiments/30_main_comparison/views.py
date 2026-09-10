"""
30 — bakeoff VIEWS (pure presentation; reads the canonical record set, never trains).

Produces, from a bakeoff campaign's per-cell cell_*.json files (or all_rows.json):

  30.00  convergence speed-up per dataset        (grouped bars per base optimiser)
  30.01  training trajectories per dataset       (one row per base optimiser: test
         accuracy and training loss against epoch, the baseline and every method
         arm, mean +/- std band across seeds)
  30.02  fixed-budget accuracy tables            (acc at 10/50/100% of epochs;
         rows = dataset x base, cols = method; mean +/- std; markdown + json)
  30.03  final-accuracy summary bars             (grouped bars per dataset)
  30.09  accuracy-vs-wall-clock Pareto           (scatter per dataset; colour=base,
         marker=method; Pareto frontier overlaid)
  30.10  results table per dataset               (LaTeX, two panels: accuracy and
         loss, then convergence and cost)

Usage:
    python views.py --campaign _core/results/run_<id>            # all views
    python views.py --campaign ... --only 01 10
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.storage import read_json, write_json_atomic   # noqa: E402
from common.plotting import apply_thesis_rcparams, PALETTE, METHOD_STYLE  # noqa: E402
from _summary_utils import TARGET_FRAC                    # noqa: E402
from _curves import BASELINE_STYLE                        # noqa: E402

# GradDrop was run as an arm but is not part of the study. `_group` drops any
# method not listed here, so its cells stay in the record set and out of every view.
METHOD_ORDER = ["baseline", "cosgd", "bograd", "dropout"]

# Order and names used by the thesis figures and tables (30.01, 30.10).
BASE_ORDER = ["sgd", "signsgd", "rmsprop", "adam"]
BASE_NAME = {"sgd": "SGD", "signsgd": "SignSGD", "rmsprop": "RMSprop", "adam": "Adam"}
METHOD_NAME = {"baseline": "Baseline", "cosgd": "COSGD", "bograd": "BOGrad",
               "dropout": "Dropout"}
DATASET_NAME = {"mnist": "MNIST", "emnist_balanced": "EMNIST-Balanced",
                "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
                "covertype": "Covertype", "yahoo_answers": "Yahoo!~Answers"}
# The baseline is drawn in the dashed dark grey of Chapters 4 and 5.
METHOD_COLOUR = {"cosgd": "#1f77b4", "bograd": "#d62728", "dropout": "#2ca02c"}
_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _resolve_campaign(arg: Optional[str]) -> Path:
    root = _HERE / "_core" / "results"
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else (_HERE / arg)
    main = root / "main"
    if main.exists():
        return main
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise SystemExit(f"No bakeoff campaign under {root} (run.py first)")
    return runs[-1]


def _load_rows(campaign: Path) -> List[Dict[str, Any]]:
    # Aggregate the per-cell files first: they never collide across concurrent
    # per-dataset sessions, whereas a campaign-level all_rows.json is overwritten
    # per session. Fall back to all_rows.json only if no cell files exist.
    rows: List[Dict[str, Any]] = []
    for cell in sorted(campaign.rglob("cell_*.json")):
        rows.extend(read_json(cell).get("rows", []))
    if rows:
        return rows
    allp = campaign / "all_rows.json"
    return read_json(allp) if allp.exists() else []


def _group(rows):
    """(dataset) -> (base, method) -> list of seed rows."""
    g: Dict[str, Dict[tuple, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] not in METHOD_ORDER:
            continue
        g[r["dataset"]][(r["base"], r["method"])].append(r)
    return g


def _mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)


def _history(rs, key) -> np.ndarray:
    """Seeds x epochs array of one per-epoch history, NaN where nothing was logged.

    A run resumed from a checkpoint logs only the epochs after the resume point,
    and those are the last epochs of the run, so a short history is padded at the
    front. Truncating every seed to the shortest, as these views once did, drew a
    resumed Yahoo! Answers seed's epochs 5 to 15 as epochs 1 to 11.
    """
    width = max(int(r.get("total_epochs") or len(r.get(key) or [])) for r in rs)
    M = np.full((len(rs), width), np.nan)
    for i, r in enumerate(rs):
        v = np.asarray(r.get(key) or [], dtype=float)[-width:]
        if v.size:
            M[i, width - v.size:] = v
    return M


def _budget_epoch(n_epochs: int, frac: float) -> int:
    """The 1-based epoch that stands for `frac` of an `n_epochs` budget."""
    return max(1, int(round(frac * n_epochs)))


def _plain_log_axis(ax) -> None:
    """Log-scale y axis labelled in plain decimals (0.5, 1, 2), not powers of ten.

    The tick density follows the span of the data: a narrow panel gets steps of
    1, 1.5, 2, 3, 5 and 7 so that it carries at least two labels, and a panel
    spanning more than about a decade and a half keeps only the powers of ten.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ratio = hi / lo
    subs = (1.0,) if ratio > 50 else (1.0, 2.0, 5.0) if ratio > 4 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _speed(cells, base, method, target_frac=TARGET_FRAC) -> Optional[Dict[str, Any]]:
    """Epochs to `target_frac` x the baseline's mean final accuracy, and the speed-up.

    A seed that never reaches the target is charged the budget plus one. A seed
    with unlogged epochs is left out, since it may have reached the target in one
    of them; `complete` counts the seeds that were used.
    """
    base_rows, rows = cells.get((base, "baseline"), []), cells.get((base, method), [])
    if not base_rows or not rows:
        return None
    base_acc = _history(base_rows, "epoch_test_acc")
    target = target_frac * float(np.mean(base_acc[:, -1]))

    def per_seed(M):
        out = []
        for seed in M:
            if np.isnan(seed).any():
                continue
            hit = np.flatnonzero(seed >= target)
            out.append(float(hit[0] + 1) if hit.size else float(len(seed) + 1))
        return out

    acc = _history(rows, "epoch_test_acc")
    b_eps, m_eps = per_seed(base_acc), per_seed(acc)
    if not b_eps or not m_eps:
        return None
    return {"target": target,
            "baseline_epochs": float(np.mean(b_eps)),
            "epochs": float(np.mean(m_eps)), "epochs_std": float(np.std(m_eps)),
            "per_seed": m_eps,
            "reached": sum(e <= acc.shape[1] for e in m_eps),
            "complete": len(m_eps), "n": acc.shape[0],
            "speedup": float(np.mean(b_eps)) / float(np.mean(m_eps))}


# ---- 30.01 trajectories ----------------------------------------------------
def view_trajectories(campaign, grouped):
    """One figure per dataset. Each row is a base optimiser; the left panel is test
    accuracy and the right training loss on a log scale. The baseline is dashed,
    each method arm a solid line, with the mean over seeds and a +/- std band."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10})
    panels = (("epoch_test_acc", "test accuracy"), ("epoch_train_loss", "training loss"))
    for dataset, cells in grouped.items():
        bases = [b for b in BASE_ORDER if (b, "baseline") in cells]
        # sized so a figure, its heading and a short paragraph share one page
        fig, axes = plt.subplots(len(bases), 2, figsize=(7.2, 1.35 * len(bases)),
                                 sharex=True, squeeze=False)
        for row, base in enumerate(bases):
            for col, (key, what) in enumerate(panels):
                ax = axes[row][col]
                for method in METHOD_ORDER:
                    rs = cells.get((base, method))
                    if not rs:
                        continue
                    M = _history(rs, key)
                    x = np.arange(1, M.shape[1] + 1)
                    mean, std = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
                    style = (dict(BASELINE_STYLE, linewidth=1.8) if method == "baseline"
                             else dict(color=METHOD_COLOUR[method], linewidth=1.5))
                    ax.plot(x, mean, label=METHOD_NAME[method], **style)
                    ax.fill_between(x, mean - std, mean + std, color=style["color"],
                                    alpha=0.12, linewidth=0)
                    ax.set_xlim(1, M.shape[1])
                if col == 1:
                    _plain_log_axis(ax)
                ax.set_title(f"{BASE_NAME[base]}: {what}")
                if row == len(bases) - 1:
                    ax.set_xlabel("epoch")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   ncol=len(labels), frameon=False)
        out = campaign / f"30_01_trajectories_{dataset}"
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.with_suffix('.pdf')}")


# ---- 30.02 fixed-budget tables --------------------------------------------
def view_budget_tables(campaign, grouped):
    fractions = {"10pct": 0.1, "50pct": 0.5, "100pct": 1.0}
    out_md = ["# 30.02 Fixed-budget accuracy tables\n"]
    out_json: Dict[str, Any] = {}
    for frac_name, frac in fractions.items():
        out_md.append(f"\n## At {frac_name} of epochs\n")
        for dataset, cells in sorted(grouped.items()):
            bases = sorted({b for b, _ in cells})
            out_md.append(f"\n**{dataset}**\n")
            out_md.append("| base | " + " | ".join(METHOD_ORDER) + " |")
            out_md.append("|" + "---|" * (len(METHOD_ORDER) + 1))
            for base in bases:
                cellvals = []
                for method in METHOD_ORDER:
                    rs = cells.get((base, method), [])
                    if rs:
                        acc = _history(rs, "epoch_test_acc")
                        m, s, n = _mean_std(acc[:, _budget_epoch(acc.shape[1], frac) - 1].tolist())
                    else:
                        m, s, n = float("nan"), float("nan"), 0
                    cellvals.append((method, m, s, n))
                    out_json[f"{frac_name}/{dataset}/{base}/{method}"] = {"mean": m, "std": s, "n": n}
                best = max((v for v in cellvals if v[1] == v[1]), key=lambda v: v[1], default=None)
                cells_str = []
                for method, m, s, n in cellvals:
                    txt = f"{m:.3f}±{s:.3f}" if n else "—"
                    if best and method == best[0] and n:
                        txt = f"**{txt}**"
                    cells_str.append(txt)
                out_md.append(f"| {base} | " + " | ".join(cells_str) + " |")
    (campaign / "30_02_budget_tables.md").write_text("\n".join(out_md), encoding="utf-8")
    write_json_atomic(campaign / "30_02_budget_tables.json", out_json)
    print(f"  wrote {campaign/'30_02_budget_tables.md'}")


# ---- 30.03 final-accuracy bars --------------------------------------------
def view_final_bars(campaign, grouped):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")
    for dataset, cells in grouped.items():
        bases = sorted({b for b, _ in cells})
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.8 / len(METHOD_ORDER)
        x = np.arange(len(bases))
        for mi, method in enumerate(METHOD_ORDER):
            means, stds = [], []
            for base in bases:
                m, s, _ = _mean_std([r.get("final_test_acc") for r in cells.get((base, method), [])])
                means.append(m); stds.append(s)
            ax.bar(x + (mi - (len(METHOD_ORDER) - 1) / 2) * width, means, width, yerr=stds,
                   capsize=2, label=method, linestyle=METHOD_STYLE.get(method))
        ax.set_xticks(x); ax.set_xticklabels(bases)
        ax.set_ylabel("final test accuracy"); ax.set_title(f"30.03  {dataset}: final accuracy")
        ax.legend(fontsize=8, ncol=3)
        out = campaign / f"30_03_final_bars_{dataset}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


# ---- 30.09 accuracy vs wall-clock Pareto ----------------------------------
def view_pareto(campaign, grouped):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")
    markers = {"baseline": "o", "cosgd": "s", "bograd": "^", "dropout": "v"}
    for dataset, cells in grouped.items():
        pts = []
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (base, method), rs in cells.items():
            acc, _, _ = _mean_std([r.get("final_test_acc") for r in rs])
            spt, _, _ = _mean_std([r.get("mean_step_wall_time_s") for r in rs])
            if acc != acc or spt != spt:
                continue
            pts.append((spt, acc))
            ax.scatter(spt, acc, color=PALETTE.get(base), marker=markers.get(method, "o"),
                       s=70, edgecolor="black", linewidth=0.4, label=f"{base}+{method}")
        # Pareto frontier (max acc for min time)
        if pts:
            pts_sorted = sorted(pts)
            frontier, best_acc = [], -1
            for spt, acc in pts_sorted:
                if acc > best_acc:
                    frontier.append((spt, acc)); best_acc = acc
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, "k--", alpha=0.5, label="Pareto frontier")
        ax.set_xscale("log"); ax.set_xlabel("sec / step (log)"); ax.set_ylabel("final test accuracy")
        ax.set_title(f"30.09  {dataset}: accuracy vs wall-clock")
        ax.legend(fontsize=6, ncol=2)
        out = campaign / f"30_09_pareto_{dataset}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


# ---- 30.00 CONVERGENCE SPEED-UP (the headline metric) ---------------------
def view_speedup(campaign, grouped, target_frac=TARGET_FRAC):
    """For each (dataset, base): how many epochs each method needs to reach the
    BASELINE's target accuracy, and the speed-up factor (baseline_epochs /
    method_epochs). This is the primary 'does it train faster' result.

    target = `target_frac` x baseline's final mean accuracy. The default is 0.99,
    not 1.0: see _summary_utils.TARGET_FRAC for why matching the baseline's mean
    final accuracy exactly is an unstable target on a plateaued curve. Also
    reports epoch-1 accuracy (early-progress) and the steps-to-target speed-up
    via mean step time. The statistic itself is `_speed`, shared with 30.10.
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    apply_thesis_rcparams("dense")

    out_md = [f"# 30.00 Convergence speed-up (target = {target_frac:g}x baseline final acc)\n",
              "Speed-up = baseline_epochs_to_target / method_epochs_to_target. "
              ">1 means faster. 'wall speed-up' also folds in per-step cost.\n"]
    out_json: Dict[str, Any] = {}

    for dataset, cells in sorted(grouped.items()):
        bases = sorted({b for b, _ in cells})
        out_md.append(f"\n## {dataset}\n")
        out_md.append("| base | metric | " + " | ".join(METHOD_ORDER) + " |")
        out_md.append("|" + "---|" * (len(METHOD_ORDER) + 2))
        # bar fig: speed-up per (base, method)
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.8 / len(METHOD_ORDER); x = np.arange(len(bases))
        for dataset_method_i, method in enumerate(METHOD_ORDER):
            speedups = []
            for base in bases:
                s = _speed(cells, base, method, target_frac)
                if s is None:
                    speedups.append(np.nan); continue
                sp = s["speedup"]
                speedups.append(sp)
                # wall speed-up: fold in per-step cost
                b_spt = _mean_std([r.get("mean_step_wall_time_s") for r in cells[(base, "baseline")]])[0]
                m_spt = _mean_std([r.get("mean_step_wall_time_s") for r in cells[(base, method)]])[0]
                wall_sp = sp * (b_spt / m_spt) if (m_spt and b_spt and m_spt == m_spt) else None
                out_json[f"{dataset}/{base}/{method}"] = {
                    "epochs_to_target": s["epochs"], "baseline_epochs": s["baseline_epochs"],
                    "epoch_speedup": sp,
                    "wall_speedup": float(wall_sp) if wall_sp else None,
                    "epoch1_acc": float(np.nanmean(
                        _history(cells[(base, method)], "epoch_test_acc")[:, 0])),
                }
            ax.bar(x + (dataset_method_i - (len(METHOD_ORDER) - 1) / 2) * width,
                   [s if s == s else 0 for s in speedups],
                   width, label=method, linestyle=METHOD_STYLE.get(method))
        ax.axhline(1.0, color="black", lw=0.8, ls="--", label="baseline (1.0x)")
        ax.set_xticks(x); ax.set_xticklabels(bases)
        ax.set_ylabel("epoch speed-up vs baseline"); ax.set_title(f"30.00  {dataset}: convergence speed-up")
        ax.legend(fontsize=8, ncol=3)
        out = campaign / f"30_00_speedup_{dataset}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")
        # markdown rows: epoch_speedup + epoch1_acc
        for base in bases:
            for metric_key, label in [("epoch_speedup", "epoch speed-up"), ("epoch1_acc", "epoch-1 acc")]:
                cellstr = []
                for method in METHOD_ORDER:
                    d = out_json.get(f"{dataset}/{base}/{method}", {})
                    v = d.get(metric_key)
                    if v is None:
                        cellstr.append("—")
                    elif metric_key == "epoch_speedup":
                        cellstr.append(f"**{v:.2f}x**" if v > 1.01 else f"{v:.2f}x")
                    else:
                        cellstr.append(f"{v:.3f}")
                out_md.append(f"| {base} | {label} | " + " | ".join(cellstr) + " |")

    (campaign / "30_00_speedup.md").write_text("\n".join(out_md), encoding="utf-8")
    write_json_atomic(campaign / "30_00_speedup.json", out_json)
    print(f"  wrote {campaign/'30_00_speedup.md'}")


# ---- 30.10 results table per dataset --------------------------------------
def _column(M: np.ndarray, idx: int):
    """(mean, std, partial) of one epoch column, over the seeds that logged it."""
    col = M[:, idx]
    col = col[~np.isnan(col)]
    return float(col.mean()), float(col.std()), col.size < M.shape[0]


def _pm(stat, nd: int) -> str:
    """'mean $\\pm$ std', with a dagger when some seed or epoch is missing from it."""
    mean, std, partial = stat
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}" + (r"$^{\dagger}$" if partial else "")


def view_results_tables(campaign, grouped, target_frac=TARGET_FRAC):
    """One LaTeX table per dataset holding every per-cell statistic.

    Panel (a): test accuracy after a tenth and half of the epoch budget (the epochs
    30.02 reads), at the final epoch and the best one, and final training loss.
    Panel (b): the tuned learning rate with K or p; epochs to the speed-up target
    of 30.00, the seeds reaching it and the speed-up; then per-step training
    time, total wall-clock, peak GPU memory and BOGrad's buffer. Accuracy, loss
    and epochs are mean +/- std over seeds. The cost columns are means, and the
    largest relative spread among them over seeds is printed for the prose.

    The columns are defined once in the thesis text, so each caption stays short.
    """
    out_json: Dict[str, Any] = {}
    spread = (0.0, "")
    for dataset, cells in sorted(grouped.items()):
        bases = [b for b in BASE_ORDER if (b, "baseline") in cells]
        first = next(iter(cells.values()))[0]
        n_epochs = int(first["total_epochs"])
        steps_per_epoch = int(first["total_steps"]) // n_epochs
        n_seeds = max(len(rs) for rs in cells.values())
        e_lo, e_mid = _budget_epoch(n_epochs, 0.1), _budget_epoch(n_epochs, 0.5)
        step_max = max(_mean_std([r.get("mean_train_step_s") for r in rs])[0] for rs in cells.values())
        nd_step = 2 if step_max * 1000 < 20 else 1

        rows_a: List[str] = []
        rows_b: List[str] = []
        gaps: List[str] = []
        for bi, base in enumerate(bases):
            methods = [m for m in METHOD_ORDER if (base, m) in cells]
            speeds = {m: _speed(cells, base, m, target_frac) for m in methods}
            top = max(round(s["speedup"], 2) for s in speeds.values())
            if bi:
                rows_a.append(r"\midrule")
                rows_b.append(r"\midrule")
            for mi, method in enumerate(methods):
                rs = cells[(base, method)]
                acc = _history(rs, "epoch_test_acc")
                loss = _history(rs, "epoch_train_loss")
                missing = np.isnan(acc).sum(axis=1)
                for r, k in zip(rs, missing):
                    if k:
                        gaps.append(f"one {METHOD_NAME[method]} seed under {BASE_NAME[base]} "
                                    f"was resumed from a checkpoint and did not log its "
                                    f"first {_WORDS[k] if k < 10 else k} epochs")
                best = np.nanmax(acc, axis=1)
                stats_a = {"acc_lo": _column(acc, e_lo - 1), "acc_mid": _column(acc, e_mid - 1),
                           "acc_final": _column(acc, -1),
                           "acc_best": (float(best.mean()), float(best.std()), bool(missing.any())),
                           "loss_final": _column(loss, -1)}
                label = [BASE_NAME[base] if mi == 0 else "", METHOD_NAME[method]]
                rows_a.append(" & ".join(label + [_pm(stats_a[k], 3) for k in stats_a]) + r" \\")

                s = speeds[method]
                hp = rs[0].get("hp") or {}
                knob = {"bograd": f"{hp.get('K')}",
                        "dropout": f"{hp.get('dropout_p', float('nan')):g}"}.get(method, "--")
                step = _mean_std([r.get("mean_train_step_s") for r in rs])
                total = _mean_std([r.get("total_wall_time_s") for r in rs])
                peak = _mean_std([r.get("peak_mem_mb") for r in rs])
                n_params = _mean_std([r.get("n_params") for r in rs])[0]
                for name, (m_, s_, _n) in (("step", step), ("total", total), ("peak", peak)):
                    if s_ / m_ > spread[0]:
                        spread = (s_ / m_, f"{dataset}/{base}/{method}/{name}")
                buf = hp["K"] * n_params * 4 / 1e6 if method == "bograd" else None
                sp_txt = f"{s['speedup']:.2f}"
                sp_txt = (rf"\textbf{{{sp_txt}}}" if round(s["speedup"], 2) == top else sp_txt) + r"$\times$"
                rows_b.append(" & ".join(label + [
                    f"{hp['lr']:g}", knob,
                    _pm((s["epochs"], s["epochs_std"], s["complete"] < s["n"]), 1),
                    f"{s['reached']}/{s['complete']}", sp_txt,
                    f"{step[0] * 1000:.{nd_step}f}",
                    _pm((total[0] / 60, total[1] / 60, False), 1), f"{peak[0]:.0f}",
                    f"{buf:.1f}" if buf is not None else "--"]) + r" \\")

                out_json[f"{dataset}/{base}/{method}"] = {
                    **{k: {"mean": v[0], "std": v[1], "partial": v[2]} for k, v in stats_a.items()},
                    "epoch_lo": e_lo, "epoch_mid": e_mid, "lr": hp.get("lr"),
                    "K": hp.get("K"), "dropout_p": hp.get("dropout_p"),
                    "target": s["target"], "epochs_to_target": s["epochs"],
                    "epochs_to_target_std": s["epochs_std"], "per_seed_epochs": s["per_seed"],
                    "reached": s["reached"], "complete": s["complete"], "n_seeds": s["n"],
                    "speedup": s["speedup"], "step_ms": step[0] * 1000,
                    "total_min": total[0] / 60, "peak_mem_mb": peak[0], "buffer_mb": buf,
                    "n_params": n_params, "steps_per_epoch": steps_per_epoch,
                }

        steps_txt = f"{steps_per_epoch:,}".replace(",", "{,}")
        caption = (rf"Results on {DATASET_NAME[dataset]}: mean $\pm$ standard deviation "
                   rf"over {_WORDS[n_seeds]} seeds, with the columns of "
                   r"Section~\ref{sec:experiments:tables}. The budget is "
                   rf"${n_epochs}$ epochs of ${steps_txt}$ steps, and a seed that "
                   rf"misses the target is charged ${n_epochs + 1}$ epochs.")
        if gaps:
            caption += r" $^{\dagger}$Over the logged epochs only: " + "; ".join(gaps) + "."
        # 16 rows per panel only fit a float page with the rows set slightly tighter
        lines = [
            r"\begin{table}[p]",
            r"  \centering",
            r"  \footnotesize",
            r"  \linespread{1}\selectfont",
            r"  \renewcommand{\arraystretch}{0.92}",
            r"  \setlength{\tabcolsep}{3pt}",
            r"  \begin{tabular}{llrrrrr}",
            r"    \multicolumn{7}{l}{\small\textbf{(a)} Accuracy and loss} \\",
            r"    \toprule",
            r"    & & \multicolumn{4}{c}{Test accuracy} & Training loss \\",
            r"    \cmidrule(lr){3-6} \cmidrule(lr){7-7}",
            rf"    Base & Method & Epoch {e_lo} & Epoch {e_mid} & Final & Best & Final \\",
            r"    \midrule",
            *("    " + r for r in rows_a),
            r"    \bottomrule",
            r"  \end{tabular}",
            "",
            r"  \vspace{4mm}",
            r"  \begin{tabular}{llrrrrrrrrr}",
            r"    \multicolumn{11}{l}{\small\textbf{(b)} Convergence and cost} \\",
            r"    \toprule",
            r"    & & \multicolumn{2}{c}{Tuned} & \multicolumn{3}{c}{Convergence}"
            r" & \multicolumn{4}{c}{Cost} \\",
            r"    \cmidrule(lr){3-4} \cmidrule(lr){5-7} \cmidrule(lr){8-11}",
            r"    Base & Method & $\eta$ & $K$ / $r$ & Epochs & Reached & Speed-up"
            r" & Step & Total & Peak & Buffer \\",
            r"    & & & & & & & (ms) & (min) & (MB) & (MB) \\",
            r"    \midrule",
            *("    " + r for r in rows_b),
            r"    \bottomrule",
            r"  \end{tabular}",
            rf"  \caption{{{caption}}}",
            rf"  \label{{tab:experiments:results_{dataset}}}",
            r"\end{table}",
        ]
        out = campaign / f"30_10_results_table_{dataset}.tex"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {out}")
    write_json_atomic(campaign / "30_10_results_table.json", out_json)
    print(f"  largest across-seed spread of a cost column: {100 * spread[0]:.1f}% ({spread[1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--only", nargs="+", default=["00", "01", "02", "03", "09", "10"])
    ap.add_argument("--target-frac", type=float, default=TARGET_FRAC,
                    help="speed-up target as fraction of baseline final acc "
                         f"(default {TARGET_FRAC}; 1.0 is unstable on plateaued curves)")
    args = ap.parse_args()

    campaign = _resolve_campaign(args.campaign)
    rows = _load_rows(campaign)
    if not rows:
        raise SystemExit(f"No rows in {campaign}")
    grouped = _group(rows)
    print(f"Views from {campaign} — {len(rows)} rows, {len(grouped)} datasets")

    if "00" in args.only: view_speedup(campaign, grouped, args.target_frac)
    if "01" in args.only: view_trajectories(campaign, grouped)
    if "02" in args.only: view_budget_tables(campaign, grouped)
    if "03" in args.only: view_final_bars(campaign, grouped)
    if "09" in args.only: view_pareto(campaign, grouped)
    if "10" in args.only: view_results_tables(campaign, grouped, args.target_frac)


if __name__ == "__main__":
    main()
