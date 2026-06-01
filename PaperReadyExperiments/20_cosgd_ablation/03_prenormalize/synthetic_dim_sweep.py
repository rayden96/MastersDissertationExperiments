"""
20.03 synthetic — cosine-vs-dimension and L2-change-after-GS (thesis 3.1/3.2).

Pure synthetic measurement (no training). Demonstrates *why* pre-normalisation
matters for COSGD's Gram-Schmidt:

  3.1  average |cos| between random vector pairs vs dimension, for unit-normalised
       vs raw-magnitude vectors. Normalised collapses toward 0 (near-orthogonal
       in high dim); unnormalised stays large because magnitude variance dominates.
  3.2  average L2 change a vector undergoes when Gram-Schmidt-orthogonalised
       against the others, normalised vs unnormalised. Stable+small when
       normalised; large/growing when not.

Writes results.json + a 2x2 figure. Run:
    python synthetic_dim_sweep.py [--trials 50 --max-dim 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent           # 03_prenormalize
_REPO = _HERE.parents[2]                            # repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.optimizers.COSGD import gram_schmidt_normal  # noqa: E402


def mean_abs_cosine(dim, trials, normalise, rng):
    vals = []
    for _ in range(trials):
        a = torch.from_numpy(rng.standard_normal(dim).astype("float32"))
        b = torch.from_numpy(rng.standard_normal(dim).astype("float32"))
        if normalise:
            a = a / a.norm(); b = b / b.norm()
        else:
            # inject magnitude variance so the unnormalised case is non-trivial
            a = a * (rng.uniform(0.2, 5.0)); b = b * (rng.uniform(0.2, 5.0))
        vals.append(abs(float(torch.dot(a, b) / (a.norm() * b.norm()))))
    return float(np.mean(vals))


def mean_l2_change(dim, trials, normalise, rng, n_vectors=5):
    vals = []
    for _ in range(trials):
        V = torch.from_numpy(rng.standard_normal((n_vectors, dim)).astype("float32"))
        if normalise:
            V = V / V.norm(dim=1, keepdim=True)
        else:
            V = V * torch.from_numpy(rng.uniform(0.2, 5.0, size=(n_vectors, 1)).astype("float32"))
        O = gram_schmidt_normal(V)
        vals.append(float((O - V).norm(dim=1).mean()))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max-dim", type=int, default=5000)
    ap.add_argument("--step", type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(2026)

    dims = list(range(2, args.max_dim + 1, args.step))
    out = {"dims": dims, "cos_norm": [], "cos_raw": [], "l2_norm": [], "l2_raw": []}
    for d in dims:
        out["cos_norm"].append(mean_abs_cosine(d, args.trials, True, rng))
        out["cos_raw"].append(mean_abs_cosine(d, args.trials, False, rng))
        out["l2_norm"].append(mean_l2_change(d, args.trials, True, rng))
        out["l2_raw"].append(mean_l2_change(d, args.trials, False, rng))
        print(f"  dim={d:>5}  |cos| norm={out['cos_norm'][-1]:.3f} raw={out['cos_raw'][-1]:.3f}  "
              f"L2 norm={out['l2_norm'][-1]:.3f} raw={out['l2_raw'][-1]:.3f}", flush=True)

    res_dir = _HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "synthetic_dim_sweep.json", "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(dims, out["cos_norm"], "o-", label="normalised")
        ax[0].plot(dims, out["cos_raw"], "s-", label="raw magnitude")
        ax[0].set_title("3.1  mean |cosine| vs dimension"); ax[0].set_xlabel("dimension")
        ax[0].set_ylabel("mean |cos|"); ax[0].legend()
        ax[1].plot(dims, out["l2_norm"], "o-", label="normalised")
        ax[1].plot(dims, out["l2_raw"], "s-", label="raw magnitude")
        ax[1].set_title("3.2  mean L2 change after Gram-Schmidt"); ax[1].set_xlabel("dimension")
        ax[1].set_ylabel("mean ||v - GS(v)||"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(res_dir / "synthetic_dim_sweep.png", dpi=140)
        print(f"wrote {res_dir/'synthetic_dim_sweep.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"wrote {res_dir/'synthetic_dim_sweep.json'}")


if __name__ == "__main__":
    main()
