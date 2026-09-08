"""The 2D validation figure, rebuilt around the per-class pulls.

The original panel (a) plotted only the trajectory: a straight run-in from the
start followed by a noise ball at the optimum. It shows where theta went, not
why the metrics move, and the thing the section actually claims, that the
per-class pulls agree far out and cancel at the centroid, is never drawn.

This draws the pulls themselves at three points along the trajectory, which is
what makes the claim visible rather than asserted.

    python pulls_figure.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, K = 2.0, 10
ANG = np.linspace(0, 2 * np.pi, K, endpoint=False)
CENTRES = R * np.stack([np.cos(ANG), np.sin(ANG)], 1)

# component k contributes 0.5*||theta - mu_k||^2, so its gradient is theta - mu_k
def pulls(theta):
    return np.asarray(theta, float)[None, :] - CENTRES


def stats(theta):
    g = pulls(theta)
    n = np.linalg.norm(g, axis=1)
    I = np.linalg.norm(g.sum(0)) / n.sum()
    gn = g / np.maximum(n, 1e-12)[:, None]
    C = gn @ gn.T
    iu = np.triu_indices(K, 1)
    cos = C[iu]
    return I, float(np.nanmean(cos)), float((cos < 0).mean())


def loss_grid(xs, ys):
    pts = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1)
    d = pts[:, :, None, :] - CENTRES[None, None, :, :]
    return 0.5 * (d ** 2).sum(-1).mean(-1)


def trajectory(lr=0.2, n_steps=500, batch=20, seed=0, start=(3.0, 0.0)):
    rng = np.random.default_rng(seed)
    th = np.array(start, float)
    traj = [th.copy()]
    for _ in range(n_steps):
        idx = rng.integers(0, K, size=batch)
        g = (th[None, :] - CENTRES[idx]).mean(0)
        th = th - lr * g
        traj.append(th.copy())
    return np.array(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        from common.plotting import apply_thesis_rcparams
        apply_thesis_rcparams()
    except Exception:
        pass

    traj = trajectory()
    MARKS = [(3.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    LBL = ["far from the optimum", "midway", "at the optimum"]

    fig = plt.figure(figsize=(12.4, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.32)

    # ---- (a) surface + trajectory, with the three sample points marked
    ax = fig.add_subplot(gs[0, 0])
    lim = 3.35
    xs = ys = np.linspace(-lim, lim, 160)
    Z = loss_grid(xs, ys)
    ax.contour(xs, ys, Z, levels=10, colors="0.7", linewidths=0.6)
    ax.scatter(CENTRES[:, 0], CENTRES[:, 1], marker="x", s=42, c="0.35",
               label="class centres", zorder=3)
    ax.plot(traj[:, 0], traj[:, 1], lw=0.9, color="tab:blue", alpha=0.7,
            label="trajectory", zorder=2)
    for (mx, my), lab in zip(MARKS, "123"):
        ax.scatter([mx], [my], s=60, facecolor="white", edgecolor="black",
                   zorder=6, linewidths=1.1)
        ax.text(mx, my, lab, ha="center", va="center", fontsize=7.5, zorder=7)
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\theta_1$", fontsize=9); ax.set_ylabel(r"$\theta_2$", fontsize=9)
    ax.set_title("(a) loss surface and trajectory", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right")

    # ---- (b,c,d) the pulls themselves at each marked point
    cmap = plt.get_cmap("tab10")
    # A single scale across the three panels: the pull magnitudes differ between
    # them (they are longest far from the centres), and rescaling each panel to
    # its own maximum would hide exactly that.
    scale = 1.0 / max(np.abs(pulls(t)).max() for t in MARKS)
    for j, (theta, lab) in enumerate(zip(MARKS, LBL)):
        ax = fig.add_subplot(gs[0, j + 1])
        g = pulls(theta)
        I, mc, fn = stats(theta)
        for k in range(K):
            v = g[k] * scale
            ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=cmap(k), lw=1.5))
        # the resultant, which is what the optimiser actually steps along
        r = g.sum(0) * scale / K
        if np.linalg.norm(r) > 1e-6:
            ax.annotate("", xy=(r[0], r[1]), xytext=(0, 0), zorder=10,
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4))
        else:
            ax.scatter([0], [0], s=70, facecolor="black", zorder=10)
        ax.axhline(0, color="0.9", lw=0.8, zorder=0)
        ax.axvline(0, color="0.9", lw=0.8, zorder=0)
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({'bcd'[j]}) {lab}", fontsize=9.5)
        # two short lines rather than one long one, so neighbouring panels
        # cannot run their captions into one another
        ax.text(0.5, -0.05,
                f"$I_{{\\mathrm{{inter}}}} = {I:.2f}$\n"
                f"mean cos ${mc:+.2f}$, {100*fn:.0f}% conflicting",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                linespacing=1.6)

    out = Path(args.out) if args.out else (
        _REPO.parent / "dissertation" / "chapters" / "interference" / "figures"
        / "val_2d_pulls.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for theta, lab in zip(MARKS, LBL):
        I, mc, fn = stats(theta)
        print(f"  {str(theta):12s} {lab:22s} I={I:.3f} mean_cos={mc:+.3f} "
              f"conflicting={100*fn:.0f}%")


if __name__ == "__main__":
    main()
