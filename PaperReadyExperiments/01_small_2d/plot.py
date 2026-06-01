"""
Plotter for the small 2D toy experiment.

Reads the latest run (or a specific --run_id) and produces:

  1. trajectory.png — 2D parameter-space plot with the loss landscape contours,
     mixture centres, and the per-seed trajectories overlaid.
  2. metrics_timeseries.png — multi-panel time series of the headline metrics
     (I_inter, I_between_K32, useful descent frac, per-step deficit) with
     mean ± std across seeds.
  3. correlations.png — bar plot of the correlation coefficients between
     geometric summaries and the per-step deficit, per seed and aggregated.

Reads-only; no training happens here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE / "results"


def _resolve_run_dir(run_id: str | None) -> Path:
    if run_id is None:
        candidates = sorted(RESULTS.glob("run_*"))
        if not candidates:
            raise SystemExit(f"No runs found in {RESULTS}")
        return candidates[-1]
    path = RESULTS / f"run_{run_id}"
    if not path.exists():
        raise SystemExit(f"Run dir not found: {path}")
    return path


def _seed_files(run_dir: Path) -> List[int]:
    seeds = []
    for f in run_dir.glob("summary_seed*.json"):
        seeds.append(int(f.stem.replace("summary_seed", "")))
    return sorted(seeds)


def _load_seed(run_dir: Path, seed: int) -> Dict:
    with open(run_dir / f"logs_seed{seed}.json") as f:
        return json.load(f)


def _loss_grid(centres: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Loss = mean over k of 0.5 * (||(x,y) - mu_k||^2)."""
    pts = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)  # [H, W, 2]
    diffs = pts[:, :, None, :] - centres[None, None, :, :]       # [H, W, K, 2]
    loss = 0.5 * (diffs ** 2).sum(axis=-1).mean(axis=-1)          # [H, W]
    return loss


def plot_trajectory(run_dir: Path, seeds: List[int], cfg: Dict) -> None:
    centres = np.array(cfg["centres"], dtype=np.float32)
    R = float(cfg["R"])

    # Determine the trajectory bounds for the plot.
    all_traj = []
    for s in seeds:
        all_traj.append(np.load(run_dir / f"trajectory_seed{s}.npy"))
    pts = np.concatenate(all_traj, axis=0)
    xmin, ymin = pts.min(axis=0) - 0.3
    xmax, ymax = pts.max(axis=0) + 0.3
    # Include centres + origin in bounds
    xmin = min(xmin, centres[:, 0].min() - 0.3, -0.3)
    xmax = max(xmax, centres[:, 0].max() + 0.3, 0.3)
    ymin = min(ymin, centres[:, 1].min() - 0.3, -0.3)
    ymax = max(ymax, centres[:, 1].max() + 0.3, 0.3)

    xs = np.linspace(xmin, xmax, 120)
    ys = np.linspace(ymin, ymax, 120)
    loss = _loss_grid(centres, xs, ys)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    cs = ax.contour(xs, ys, loss, levels=12, colors="0.6", linewidths=0.6, alpha=0.8)
    ax.contourf(xs, ys, loss, levels=20, cmap="viridis", alpha=0.25)
    ax.scatter(centres[:, 0], centres[:, 1], marker="x", s=60, c="black",
               label=f"mixture centres ({len(centres)})")
    ax.scatter([0], [0], marker="*", s=120, c="red", label="centroid (optimum)")

    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(seeds):
        traj = all_traj[i]
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=1.0, color=cmap(i % 10),
                alpha=0.85, label=f"seed {s}")
        ax.scatter([traj[0, 0]], [traj[0, 1]], marker="o", s=30,
                   c=[cmap(i % 10)], edgecolors="black", lw=0.5, zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(f"2D mixture trajectory (n_groups={cfg['n_groups']}, R={R}, "
                 f"batch={cfg['batch_size']}, lr={cfg['lr']})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "trajectory.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {run_dir / 'trajectory.png'}")


def _series_mean_std(seed_logs: List[List[Dict]], key: str):
    """Align series across seeds by step. Returns (steps, mean, std)."""
    # Each log list has its own step grid; here log_every=1 by default so they
    # all share the same step grid in the toy. If they diverge we just take the
    # shortest common length.
    per_seed = []
    common_steps = None
    for logs in seed_logs:
        steps = [l["step"] for l in logs if key in l and l[key] == l[key]]
        vals = [l[key] for l in logs if key in l and l[key] == l[key]]
        per_seed.append((np.array(steps), np.array(vals, dtype=np.float64)))
    if not per_seed:
        return np.array([]), np.array([]), np.array([])
    # Use the intersection of step grids.
    common_steps = set(per_seed[0][0].tolist())
    for s, _ in per_seed[1:]:
        common_steps &= set(s.tolist())
    common_steps = sorted(common_steps)
    if not common_steps:
        return np.array([]), np.array([]), np.array([])
    aligned = []
    for s, v in per_seed:
        idx = np.array([np.where(s == cs)[0][0] for cs in common_steps])
        aligned.append(v[idx])
    M = np.stack(aligned, axis=0)
    return np.array(common_steps), M.mean(axis=0), M.std(axis=0)


def plot_timeseries(run_dir: Path, seeds: List[int]) -> None:
    seed_logs = []
    for s in seeds:
        with open(run_dir / f"logs_seed{s}.json") as f:
            seed_logs.append(json.load(f)["logs"])

    keys = [
        ("I_inter",                 "Inter-batch cancellation index"),
        ("inter_mean_cos",          "Mean per-pair cosine (inter)"),
        ("inter_useful_descent_frac","Useful descent frac of batch grad"),
        ("I_between_K32",           "Between-batch cancellation index (K=32)"),
        ("between_K32_mean_cos",    "Mean pairwise cosine within K=32"),
        ("D_t",                     "Per-step first-order deficit $D_t$"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    axs = axs.flatten()
    for ax, (key, title) in zip(axs, keys):
        steps, mean, std = _series_mean_std(seed_logs, key)
        if steps.size == 0:
            ax.set_title(f"{title}\n(no data)")
            continue
        ax.plot(steps, mean, color="tab:blue", lw=1.5, label="mean")
        ax.fill_between(steps, mean - std, mean + std,
                        color="tab:blue", alpha=0.2, label="±1 std")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Interference metrics over training (mean ± std across seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(run_dir / "metrics_timeseries.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {run_dir / 'metrics_timeseries.png'}")


def plot_correlations(run_dir: Path, seeds: List[int]) -> None:
    summaries = []
    for s in seeds:
        with open(run_dir / f"summary_seed{s}.json") as f:
            summaries.append(json.load(f))
    corr_keys = [
        ("corr_I_inter_vs_Dt",            "$I_{\\rm inter}$ vs $D_t$"),
        ("corr_inter_mean_cos_vs_Dt",     "mean inter cos vs $D_t$"),
        ("corr_inter_max_min_ratio_vs_Dt","max/min ratio vs $D_t$"),
        ("corr_I_between_K32_vs_Dt",      "$I_{\\rm between, 32}$ vs $D_t$"),
        ("corr_between_K32_mean_cos_vs_Dt","K=32 mean cos vs $D_t$"),
    ]
    labels = [c[1] for c in corr_keys]
    means = []
    stds = []
    for key, _ in corr_keys:
        vals = [s.get(key, float("nan")) for s in summaries
                if isinstance(s.get(key), (int, float)) and not math.isnan(s.get(key))]
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(float("nan"))
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color="tab:blue", alpha=0.85, capsize=4)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation: geometric metric vs per-step deficit $D_t$ (mean ± std across seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(run_dir / "correlations.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {run_dir / 'correlations.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None,
                        help="run id (folder name suffix); default: latest")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_id)
    print(f"Plotting from {run_dir}")
    seeds = _seed_files(run_dir)
    if not seeds:
        raise SystemExit(f"No seed summaries in {run_dir}")
    print(f"  found seeds: {seeds}")

    # Read config from first seed's summary
    with open(run_dir / f"summary_seed{seeds[0]}.json") as f:
        cfg = json.load(f)["config"]

    plot_trajectory(run_dir, seeds, cfg)
    plot_timeseries(run_dir, seeds)
    plot_correlations(run_dir, seeds)


if __name__ == "__main__":
    main()
