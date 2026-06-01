"""
Runner for the small 2D toy experiment.

For each seed, trains a Mixture2DProblem with vanilla SGD and an
InterferenceMeter attached. Logs every step (cheap in 2D). Saves:

  - per-seed per-step log JSON
  - per-seed summary JSON
  - per-seed parameter trajectory (every step) as .npy
  - aggregate config + manifest

This script only produces data. Plotting lives in plot.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from collections import deque

import numpy as np

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from interference import InterferenceMeter, summarize_run  # noqa: E402

from problem import Mixture2DProblem  # noqa: E402


def run_one_seed(
    *,
    seed: int,
    n_groups: int,
    R: float,
    batch_size: int,
    lr: float,
    n_steps: int,
    K_values: List[int],
    log_every: int,
    ref_refresh_every: int,
    init_position,
    method: str = "baseline",
    bograd_K: int = 32,
):
    problem = Mixture2DProblem(
        n_groups=n_groups, R=R, batch_size=batch_size, lr=lr,
        init_position=init_position, seed=seed,
    )
    meter = InterferenceMeter(
        problem=problem, lr=lr, K_values=K_values,
        log_every=log_every, ref_refresh_every=ref_refresh_every,
    )
    meter.initialize()

    trajectory = np.empty((n_steps + 1, 2), dtype=np.float32)
    trajectory[0] = problem.flatten_params().numpy()

    losses = np.empty(n_steps, dtype=np.float32)

    # BoGrad needs a rolling buffer of past raw updates.
    bograd_buffer: deque = deque(maxlen=bograd_K) if method == "bograd" else None

    for step in range(n_steps):
        meter.before_step()
        if method == "bograd":
            batch, loss = problem.step_bograd(bograd_buffer, mode="negative")
        else:
            batch, loss = problem.step_sgd()
        meter.after_step(step, batch, loss)
        trajectory[step + 1] = problem.flatten_params().numpy()
        losses[step] = loss

    summary = summarize_run(
        logs=meter.logs,
        calibration_logs=meter.calibration_logs,
        K_values=K_values,
        cum_deficit=meter.cum_deficit,
        cum_deficit_count=meter.cum_deficit_count,
    )
    summary["seed"] = seed
    summary["final_theta"] = problem.flatten_params().tolist()
    summary["final_dist_to_origin"] = float(np.linalg.norm(trajectory[-1]))
    summary["config"] = problem.config()

    return {
        "summary": summary,
        "logs": meter.logs,
        "calibration_logs": meter.calibration_logs,
        "trajectory": trajectory,
        "losses": losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "bograd"], default="baseline",
                        help="baseline = vanilla SGD; bograd = BoGrad-projected SGD")
    parser.add_argument("--bograd_K", type=int, default=32)
    parser.add_argument("--n_groups", type=int, default=10)
    parser.add_argument("--R", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--K_values", type=int, nargs="+", default=[4, 32, 128])
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--ref_refresh_every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--init_x", type=float, default=3.0)
    parser.add_argument("--init_y", type=float, default=0.0)
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _HERE / "results" / f"run_{run_id}_{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "args": {k: v for k, v in vars(args).items()},
        "seeds": list(args.seeds),
        "wall_clock_s_total": 0.0,
    }

    t0 = time.time()
    seed_summaries: List[Dict] = []

    for seed in args.seeds:
        print(f"[seed {seed}] start", flush=True)
        st = time.time()
        result = run_one_seed(
            seed=seed,
            n_groups=args.n_groups, R=args.R,
            batch_size=args.batch_size, lr=args.lr,
            n_steps=args.n_steps,
            K_values=list(args.K_values),
            log_every=args.log_every,
            ref_refresh_every=args.ref_refresh_every,
            init_position=(args.init_x, args.init_y),
            method=args.method,
            bograd_K=args.bograd_K,
        )
        dt = time.time() - st
        print(f"[seed {seed}] done in {dt:.2f}s — "
              f"final dist to origin = {result['summary']['final_dist_to_origin']:.4f}, "
              f"I_inter_mean = {result['summary']['I_inter_mean']:.3f}, "
              f"corr I_inter vs D_t = {result['summary']['corr_I_inter_vs_Dt']:.3f}")

        with open(out_dir / f"logs_seed{seed}.json", "w") as f:
            json.dump(
                {"summary": result["summary"], "logs": result["logs"],
                 "calibration_logs": result["calibration_logs"]},
                f, indent=2, default=str,
            )
        with open(out_dir / f"summary_seed{seed}.json", "w") as f:
            json.dump(result["summary"], f, indent=2, default=str)
        np.save(out_dir / f"trajectory_seed{seed}.npy", result["trajectory"])
        np.save(out_dir / f"losses_seed{seed}.npy", result["losses"])

        seed_summaries.append(result["summary"])

    manifest["wall_clock_s_total"] = time.time() - t0

    # Aggregate across seeds
    def agg(key):
        vals = [s.get(key, float("nan")) for s in seed_summaries
                if isinstance(s.get(key), (int, float))]
        vals = [v for v in vals if v == v]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    headline = {
        "I_inter_mean":               agg("I_inter_mean"),
        "I_between_K32_mean":         agg("I_between_K32_mean"),
        "inter_useful_descent_frac_mean": agg("inter_useful_descent_frac_mean"),
        "cum_deficit":                agg("cum_deficit"),
        "corr_I_inter_vs_Dt":         agg("corr_I_inter_vs_Dt"),
        "corr_inter_mean_cos_vs_Dt":  agg("corr_inter_mean_cos_vs_Dt"),
        "corr_I_between_K32_vs_Dt":   agg("corr_I_between_K32_vs_Dt"),
        "ref_loss_drop":              agg("ref_loss_drop"),
        "final_dist_to_origin":       agg("final_dist_to_origin"),
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(out_dir / "headline.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)

    print("\n=== Headline (mean ± std across seeds) ===")
    for k, v in headline.items():
        print(f"  {k:<36} {v['mean']:+.4f} ± {v['std']:.4f}  (n={v['n']})")
    print(f"\nResults written to {out_dir}")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()
