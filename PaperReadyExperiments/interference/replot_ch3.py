"""Redraw the two CIFAR-10 interference figures without the useful-fraction panel.

Chapter 3 now reports two quantities per scale, a cancellation index with the
pairwise angle statistics beneath it, and the per-step deficit. The
useful-versus-wasted decomposition was cut, so the figures must stop showing it.

Reads the stored series (results/interference/chapter3_series.json); nothing is
re-run.

    python replot_ch3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = _REPO.parent / "dissertation" / "chapters" / "interference" / "figures"
SERIES = _REPO / "results" / "interference" / "chapter3_series.json"


def band(ax, d, colour, label=None):
    st = np.asarray(d["steps"], float)
    m = np.asarray(d["mean"], float)
    s = np.asarray(d["std"], float)
    ok = ~np.isnan(m)
    ax.plot(st[ok], m[ok], color=colour, lw=1.4, label=label)
    ax.fill_between(st[ok], (m - s)[ok], (m + s)[ok], color=colour, alpha=0.18, lw=0)


def main():
    try:
        from common.plotting import apply_thesis_rcparams
        apply_thesis_rcparams()
    except Exception:
        pass

    logs = json.loads(SERIES.read_text())["series"]["c10_logs"]

    # ---------- within-batch: index (with the orthogonal null) + cosines ----
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))

    ax = axes[0]
    band(ax, logs["I_inter"], "tab:blue")
    ax.axhline(1 / np.sqrt(10), color="0.4", ls="--", lw=1.1,
               label=r"orthogonal null $1/\sqrt{C}$")
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel(r"$I_{\mathrm{inter}}$", fontsize=9)
    ax.set_title("(a) cancellation index", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)

    ax = axes[1]
    band(ax, logs["inter_mean_cos_pos"], "tab:green", "aligning pairs")
    band(ax, logs["inter_mean_cos"], "tab:blue", "all pairs")
    band(ax, logs["inter_mean_cos_neg"], "tab:red", "conflicting pairs")
    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax2 = ax.twinx()
    fn = logs["inter_frac_neg"]
    ax2.plot(np.asarray(fn["steps"], float), np.asarray(fn["mean"], float),
             color="0.45", ls=":", lw=1.3)
    ax2.set_ylabel("fraction conflicting", fontsize=8, color="0.45")
    ax2.tick_params(labelsize=7.5, colors="0.45")
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("mean pairwise cosine", fontsize=9)
    ax.set_title("(b) where the cancellation comes from", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center right"); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = FIGDIR / "within_batch_cifar10.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)

    # ---------- between-batch: index per K (with the diffusion null) + cosines
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
    colours = {4: "tab:blue", 32: "tab:orange", 128: "tab:green"}

    ax = axes[0]
    for K, c in colours.items():
        band(ax, logs[f"I_between_K{K}"], c, f"$K = {K}$")
        ax.axhline(1 / np.sqrt(K), color=c, ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel(r"$I_{\mathrm{between},K}$", fontsize=9)
    ax.set_title(r"(a) cancellation index, dashed = diffusion null $1/\sqrt{K}$",
                 fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)

    ax = axes[1]
    for K, c in colours.items():
        band(ax, logs[f"between_K{K}_mean_cos"], c, f"$K = {K}$")
    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("within-window mean cosine", fontsize=9)
    ax.set_title("(b) how the updates relate within the window", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = FIGDIR / "between_batch_cifar10.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
