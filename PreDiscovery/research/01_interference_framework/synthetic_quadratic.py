"""
Synthetic experiment 1 — Controlled-conflict quadratic loss.

Implements the experiment described in framework.md §7.2 (first bullet).

Setting
-------
Quadratic loss
    L(theta) = ½ thetaᵀ H theta - bᵀ theta
with diagonal H of controllable per-coordinate eigenvalues. The optimum is
theta_star = H⁻¹ b. The full-batch gradient is g̃(theta) = Htheta - b.

We simulate mini-batch gradients by adding mean-zero noise plus a *structured
interference* component that alternates sign on a designated subset of
coordinates J (the "conflict" coordinates). For step t:
    g_t(theta) = (Htheta - b) + noise_t + (-1)^t · amp · e_J
where e_J is the indicator vector of J. Across pairs of consecutive steps,
the structured component cancels at expectation but produces a step-by-step
zigzag along J.

What we test
------------
1. Does the framework's geometric metric (cos g_t, g_{t-1}) detect the
   injected interference?  Predict: yes, strongly negative on J-coordinates,
   approximately zero on non-J-coordinates.
2. Does the wasted-work ratio (WW_K) drop under interference?  Predict: yes.
3. Does BoGrad (gradient-stage K=8 negative) suppress the metric signal AND
   recover convergence speed?  Predict: yes for both.

Three configurations are run:
- "no_interference"     : amp = 0, just noise. Reference for null behaviour.
- "with_interference"   : amp > 0, no orthogonalisation. Should show clear signal.
- "with_interference_bograd" : amp > 0 + BoGrad. Should attenuate signal AND speed convergence.

Output
------
research/01_interference_framework/results/synthetic_quadratic/run_<timestamp>/
    config.json
    <variant>/history.json, summary.json, theta_trajectory.json
    summary_table.txt

Interpretation guide
--------------------
The validation succeeds if:
- "no_interference" exhibits framework metrics near baseline values (cos ≈ 0,
  WW reasonable for noise-only).
- "with_interference" shows substantially more negative cos(g_t, g_{t-1}) and
  lower WW_K than the no-interference run.
- "with_interference_bograd" recovers metrics back toward the no-interference
  baseline (cos closer to 0, WW closer to no-interference levels) AND
  converges faster (fewer steps to a target loss).

If those three predictions hold, the framework's metrics are *causally*
sensitive to controlled interference, not just observationally correlated.

Usage
-----
    python research/01_interference_framework/synthetic_quadratic.py
    python research/01_interference_framework/synthetic_quadratic.py --p 200 --steps 800
    python research/01_interference_framework/synthetic_quadratic.py --amplitude 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

# Ensure stdout can handle unicode on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.diagnostics import InterferenceTracker  # noqa: E402
from common.optimizers import BoGrad  # noqa: E402


# ============================================================================
# Quadratic environment
# ============================================================================
@dataclass
class QuadraticEnv:
    """L(theta) = ½ thetaᵀ H theta - bᵀ theta.

    H is diagonal with eigenvalues drawn log-uniformly between
    min_eigenvalue and max_eigenvalue. b is randomly drawn so that the
    optimum theta_star = H⁻¹ b is in a controllable region.

    The "mini-batch" gradient simulator returns
        g_t = (Htheta - b) + xi_t + ι_t
    where:
      - xi_t  ~  N(0, noise_scale² · I)   (iid noise across t)
      - ι_t = (-1)^t · amplitude · e_J  (structured interference on coords J)
    """

    p: int
    H: torch.Tensor
    b: torch.Tensor
    noise_scale: float
    interference_amplitude: float
    interference_indices: torch.Tensor   # 1-D long tensor of coord indices in J
    seed: int

    @classmethod
    def build(
        cls,
        p: int = 100,
        min_eigenvalue: float = 1.0,
        max_eigenvalue: float = 100.0,
        n_interference_coords: int = 20,
        noise_scale: float = 0.05,
        interference_amplitude: float = 0.0,
        seed: int = 2026,
    ) -> "QuadraticEnv":
        rng = torch.Generator().manual_seed(seed)
        # Log-spaced eigenvalues (deterministic — no random component, so no generator needed)
        log_evs = torch.linspace(math.log(min_eigenvalue), math.log(max_eigenvalue), p)
        H_diag = torch.exp(log_evs)
        H = torch.diag(H_diag)
        # b chosen so optimum theta_star has coordinates ~ N(0, 1)
        theta_star_target = torch.randn(p, generator=rng)
        b = H @ theta_star_target
        # Choose J = top-magnitude coords of theta_star for interference (likely
        # high-eigenvalue dims — most penalising of mismatch)
        J = torch.argsort(theta_star_target.abs(), descending=True)[:n_interference_coords]
        return cls(
            p=p, H=H, b=b,
            noise_scale=float(noise_scale),
            interference_amplitude=float(interference_amplitude),
            interference_indices=J,
            seed=int(seed),
        )

    def to(self, device) -> "QuadraticEnv":
        return QuadraticEnv(
            p=self.p,
            H=self.H.to(device),
            b=self.b.to(device),
            noise_scale=self.noise_scale,
            interference_amplitude=self.interference_amplitude,
            interference_indices=self.interference_indices.to(device),
            seed=self.seed,
        )

    def true_loss(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * theta @ (self.H @ theta) - self.b @ theta

    def true_grad(self, theta: torch.Tensor) -> torch.Tensor:
        return self.H @ theta - self.b

    def optimum(self) -> torch.Tensor:
        return torch.linalg.solve(self.H, self.b)

    def minibatch_grad(self, theta: torch.Tensor, t: int, gen: torch.Generator) -> torch.Tensor:
        g = self.true_grad(theta)
        # Noise (iid across t, drawn fresh)
        noise = torch.randn(self.p, generator=gen, device=theta.device) * self.noise_scale
        # Structured interference: zigzag on J coordinates
        sign = 1.0 if (t % 2 == 0) else -1.0
        interference = torch.zeros(self.p, device=theta.device)
        if self.interference_amplitude != 0.0:
            interference[self.interference_indices] = sign * self.interference_amplitude
        return g + noise + interference


# ============================================================================
# Synthetic theta as nn.Module so we can reuse InterferenceTracker
# ============================================================================
class ThetaParam(nn.Module):
    """Wraps a parameter vector theta as an nn.Module with one Parameter.

    We don't run forward() — the loss is computed externally from .theta.
    But the InterferenceTracker iterates self.parameters() to flatten params
    and gradients, so wrapping theta as a Parameter is sufficient.
    """

    def __init__(self, p: int, init_scale: float = 0.5, seed: int = 2026):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.randn(p, generator=rng) * init_scale)

    def forward(self, *args, **kwargs):
        return self.theta


# ============================================================================
# Run one variant
# ============================================================================
@dataclass
class VariantResult:
    name: str
    history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    forgetting: Dict[str, Any]
    wasted_work: Dict[str, Any]
    theta_final: List[float]
    theta_optimum_distance: float
    final_loss: float
    initial_loss: float
    steps_run: int
    wall_clock_s: float


def run_variant(
    name: str,
    env: QuadraticEnv,
    optimizer_factory: Callable[[ThetaParam], torch.optim.Optimizer],
    *,
    steps: int,
    log_every: int,
    seed: int,
    init_seed: int,
    device: torch.device,
) -> VariantResult:
    # Reproducibility — fresh generators each variant for paired comparison
    init_gen = torch.Generator().manual_seed(init_seed)
    grad_gen = torch.Generator(device=device).manual_seed(seed)

    # Init theta (paired across variants when init_seed is shared)
    model = ThetaParam(env.p, init_scale=0.5, seed=init_seed).to(device)
    optimizer = optimizer_factory(model)

    # Tracker (no probe — synthetic, no classes)
    tracker = InterferenceTracker(
        model, criterion=lambda *a, **k: torch.zeros(()),  # unused
        probe_set=None,
        log_every=log_every,
        wasted_work_K=32,
        device=device,
        trajectory_lags=[1, 4, 16],
    )

    initial_loss = float(env.true_loss(model.theta).item())
    theta_optimum = env.optimum().to(device)

    print(f"\n=== {name} ===")
    print(f"  init loss = {initial_loss:.4f}    p={env.p}    "
          f"interference_amp={env.interference_amplitude}    "
          f"|J|={env.interference_indices.numel()}")

    t0 = time.time()
    for t in range(steps):
        # Compute mini-batch gradient
        g = env.minibatch_grad(model.theta.detach(), t, grad_gen)
        # Set p.grad
        optimizer.zero_grad(set_to_none=False)
        if model.theta.grad is None:
            model.theta.grad = torch.zeros_like(model.theta)
        model.theta.grad.copy_(g)

        # Compute the loss for InterferenceTracker (true loss, not the noisy one).
        # We pass it for the loss field; doesn't affect optimisation.
        true_loss_val = float(env.true_loss(model.theta).item())

        tracker.before_step()
        optimizer.step()
        tracker.after_step(loss=true_loss_val)

    final_loss = float(env.true_loss(model.theta).item())
    distance_to_opt = float((model.theta.detach() - theta_optimum).norm().item())
    elapsed = time.time() - t0

    print(f"  final loss = {final_loss:.4f}    dist_opt = {distance_to_opt:.4f}    "
          f"({elapsed:.1f}s)")

    return VariantResult(
        name=name,
        history=tracker.get_history(),
        summary=tracker.summary(),
        forgetting=tracker.forgetting.summary(),
        wasted_work=tracker.wasted_work.summary(),
        theta_final=model.theta.detach().cpu().tolist(),
        theta_optimum_distance=distance_to_opt,
        final_loss=final_loss,
        initial_loss=initial_loss,
        steps_run=steps,
        wall_clock_s=elapsed,
    )


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=100, help="Parameter dimension")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--noise-scale", type=float, default=0.05)
    ap.add_argument("--amplitude", type=float, default=2.0,
                    help="Interference zigzag amplitude on J-coordinates")
    ap.add_argument("--n-interference-coords", type=int, default=20)
    ap.add_argument("--min-eigenvalue", type=float, default=1.0)
    ap.add_argument("--max-eigenvalue", type=float, default=100.0)
    ap.add_argument("--bograd-K", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--env-seed", type=int, default=2026)
    ap.add_argument("--init-seed", type=int, default=2026)
    ap.add_argument("--noise-seed", type=int, default=12345)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")

    # Build TWO environments: one with no interference, one with it.
    env_clean = QuadraticEnv.build(
        p=args.p,
        min_eigenvalue=args.min_eigenvalue,
        max_eigenvalue=args.max_eigenvalue,
        n_interference_coords=args.n_interference_coords,
        noise_scale=args.noise_scale,
        interference_amplitude=0.0,
        seed=args.env_seed,
    ).to(device)

    env_conflict = QuadraticEnv.build(
        p=args.p,
        min_eigenvalue=args.min_eigenvalue,
        max_eigenvalue=args.max_eigenvalue,
        n_interference_coords=args.n_interference_coords,
        noise_scale=args.noise_scale,
        interference_amplitude=args.amplitude,
        seed=args.env_seed,
    ).to(device)

    # Output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "research" / "01_interference_framework" / "results" / "synthetic_quadratic"
    out_dir = out_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id}, fh, indent=2)

    results: Dict[str, VariantResult] = {}

    # Variant 1 — clean baseline (no interference)
    results["no_interference"] = run_variant(
        "SGD on clean env (no interference)",
        env_clean,
        lambda m: torch.optim.SGD([m.theta], lr=args.lr),
        steps=args.steps, log_every=args.log_every,
        seed=args.noise_seed, init_seed=args.init_seed, device=device,
    )

    # Variant 2 — interference-injected, no orthogonalisation
    results["with_interference"] = run_variant(
        "SGD on interference-injected env",
        env_conflict,
        lambda m: torch.optim.SGD([m.theta], lr=args.lr),
        steps=args.steps, log_every=args.log_every,
        seed=args.noise_seed, init_seed=args.init_seed, device=device,
    )

    # Variant 3 — interference + BoGrad (gradient-stage K=8 negative)
    results["with_interference_bograd"] = run_variant(
        f"BoGrad+SGD on interference-injected env (gradient-stage K={args.bograd_K} neg)",
        env_conflict,
        lambda m: BoGrad(
            [m.theta], torch.optim.SGD,
            buffer_size=args.bograd_K, project_stage="gradient",
            projection_mode="negative", orth_method="sequential",
            lr=args.lr,
        ),
        steps=args.steps, log_every=args.log_every,
        seed=args.noise_seed, init_seed=args.init_seed, device=device,
    )

    # Save per-variant outputs
    for name, r in results.items():
        sub = out_dir / name
        sub.mkdir(exist_ok=True)
        with (sub / "history.json").open("w") as fh:
            json.dump(r.history, fh, indent=2)
        with (sub / "summary.json").open("w") as fh:
            json.dump(r.summary, fh, indent=2, default=str)
        with (sub / "result.json").open("w") as fh:
            json.dump({
                "name": r.name,
                "final_loss": r.final_loss,
                "initial_loss": r.initial_loss,
                "theta_optimum_distance": r.theta_optimum_distance,
                "steps_run": r.steps_run,
                "wall_clock_s": r.wall_clock_s,
                "forgetting": r.forgetting,
                "wasted_work": r.wasted_work,
            }, fh, indent=2)
        with (sub / "theta_final.json").open("w") as fh:
            json.dump(r.theta_final, fh)

    # Console summary
    print("\n" + "=" * 92)
    print("SYNTHETIC QUADRATIC — does the framework detect injected interference?")
    print("=" * 92)
    print(f"{'variant':38s} {'final_loss':>11s} {'dist_opt':>10s} "
          f"{'cos(g,g_-1)':>12s} {'WW_K=32':>9s} {'cos(u,u_-1)':>12s}")
    print("-" * 92)
    for name, r in results.items():
        s = r.summary
        cos_g = s.get("cos_g_prev_mean", float("nan"))
        cos_u = s.get("cos_u_prev_mean", float("nan"))
        ww = s.get("ww_wasted_work_ratio_mean", float("nan"))
        print(f"{name:38s} {r.final_loss:>11.4f} {r.theta_optimum_distance:>10.4f} "
              f"{cos_g:>+12.4f} {ww:>9.4f} {cos_u:>+12.4f}")
    print("-" * 92)

    # Decision criterion
    s_clean = results["no_interference"].summary
    s_conflict = results["with_interference"].summary
    s_bograd = results["with_interference_bograd"].summary

    print("\nFRAMEWORK SENSITIVITY CHECK")
    print(f"  cos(g,g_-1):  clean={s_clean.get('cos_g_prev_mean', float('nan')):>+.4f}  "
          f"conflict={s_conflict.get('cos_g_prev_mean', float('nan')):>+.4f}  "
          f"+bograd={s_bograd.get('cos_g_prev_mean', float('nan')):>+.4f}")
    print(f"  WW_K=32:      clean={s_clean.get('ww_wasted_work_ratio_mean', float('nan')):.4f}  "
          f"conflict={s_conflict.get('ww_wasted_work_ratio_mean', float('nan')):.4f}  "
          f"+bograd={s_bograd.get('ww_wasted_work_ratio_mean', float('nan')):.4f}")
    print(f"  dist_opt:          clean={results['no_interference'].theta_optimum_distance:.4f}  "
          f"conflict={results['with_interference'].theta_optimum_distance:.4f}  "
          f"+bograd={results['with_interference_bograd'].theta_optimum_distance:.4f}")

    # Pass/fail summary
    print("\nPREDICTION CHECK:")
    cos_clean = s_clean.get("cos_g_prev_mean", 0.0)
    cos_conflict = s_conflict.get("cos_g_prev_mean", 0.0)
    cos_bograd = s_bograd.get("cos_g_prev_mean", 0.0)
    ww_clean = s_clean.get("ww_wasted_work_ratio_mean", 1.0)
    ww_conflict = s_conflict.get("ww_wasted_work_ratio_mean", 1.0)
    ww_bograd = s_bograd.get("ww_wasted_work_ratio_mean", 1.0)

    p1 = cos_conflict < cos_clean - 0.05  # interference detected geometrically
    p2 = ww_conflict < ww_clean - 0.05   # interference detected by trajectory
    p3 = cos_bograd > cos_conflict + 0.02 or ww_bograd > ww_conflict + 0.02  # bograd attenuates
    p4 = (results["with_interference_bograd"].theta_optimum_distance
          < results["with_interference"].theta_optimum_distance)

    def mark(b: bool) -> str:
        return "PASS" if b else "FAIL"

    print(f"  P1 (cos detects interference):     {mark(p1)}  "
          f"({cos_clean:+.4f} → {cos_conflict:+.4f}, want decrease)")
    print(f"  P2 (WW_K detects interference):    {mark(p2)}  "
          f"({ww_clean:.4f} → {ww_conflict:.4f}, want decrease)")
    print(f"  P3 (BoGrad attenuates signal):     {mark(p3)}  "
          f"(cos: {cos_conflict:+.4f} → {cos_bograd:+.4f}, "
          f"WW: {ww_conflict:.4f} → {ww_bograd:.4f})")
    print(f"  P4 (BoGrad converges closer to theta_star): {mark(p4)}  "
          f"({results['with_interference'].theta_optimum_distance:.4f} → "
          f"{results['with_interference_bograd'].theta_optimum_distance:.4f})")

    overall = all([p1, p2, p3, p4])
    print(f"\nOVERALL: {mark(overall)} — "
          f"{'framework metrics are causally sensitive to controlled interference.' if overall else 'one or more predictions failed; iterate.'}")
    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
