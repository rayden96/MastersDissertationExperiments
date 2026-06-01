"""
Synthetic experiment 1c — PERPENDICULAR-bias quadratic loss.

The (W') validation experiment. The previous biased-quadratic test (F10)
showed BoGrad cannot recover from drift bias *aligned* with the descent
direction, because removing the buffered direction also strips descent.
Hypothesis (W'), per framework.md §9 question 10:

> orthogonalisation helps only when the interference is meaningfully
> separable from the descent direction.

This experiment constructs a setting where bias is *guaranteed* perpendicular
to descent and tests whether BoGrad recovers there.

Construction
------------
Loss: L(theta) = ½ thetaᵀ H theta - bᵀ theta with diagonal H.

The parameter space is split into two subspaces:
  - ACTIVE (n_active dims):   H_diag = log-spaced eigenvalues in [1, 100],
                               b = randomly chosen so theta_star[active] is non-zero.
                               Descent IS active here.
  - INACTIVE (p - n_active):  H_diag = epsilon (tiny, e.g. 1e-3),
                               b = 0, so theta_star[inactive] = 0.
                               Descent has near-zero component here.

Bias is injected ONLY on the inactive dims:
  bias_t = amp * e_INACTIVE   (drift: constant pull on inactive coordinates)

Because the descent gradient has near-zero component on the inactive dims,
the bias is geometrically perpendicular (within tolerance epsilon) to the
descent direction.

Predictions
-----------
- P1: clean baseline converges (theta -> theta_star, both active and inactive
  dims close to optimum).
- P2: biased baseline does NOT converge on the inactive dims (bias accumulates
  there, drives theta[inactive] away from 0). On the active dims, descent
  still works, so dist_opt_active is small but dist_opt_inactive is large.
- P3 (W' validation): BoGrad FULL or POSITIVE mode recovers — these modes
  subtract the buffered direction (which is the bias), and since the bias
  is perpendicular to descent, removing it doesn't strip descent. dist_opt
  should drop substantially.
- P4: BoGrad NEGATIVE mode does NOT recover — the bias makes consecutive
  gradients positively aligned (they share the bias direction), and
  negative-mode skips when cos > 0. So negative-mode is essentially a
  no-op here.

If P3 passes (BoGrad full/positive recovers), Hypothesis (W') is validated:
orthogonalisation works when interference is separable from descent.

Output
------
research/01_interference_framework/results/synthetic_quadratic_perpendicular/run_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure stdout can handle unicode on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.diagnostics import InterferenceTracker  # noqa: E402
from common.optimizers import BoGrad  # noqa: E402


# ============================================================================
# Perpendicular-bias quadratic environment
# ============================================================================
@dataclass
class PerpendicularQuadraticEnv:
    """Quadratic loss with sparse theta_star and bias on inactive dimensions.

    Active dims have full descent dynamics. Inactive dims have near-zero
    eigenvalue and b=0, so descent has near-zero component on them. Bias on
    inactive dims is therefore (approximately) perpendicular to descent.
    """

    p: int
    n_active: int
    H_diag: torch.Tensor   # 1-D diagonal
    b: torch.Tensor
    active_indices: torch.Tensor   # long tensor
    inactive_indices: torch.Tensor   # long tensor
    noise_scale: float
    bias_amplitude: float
    seed: int

    @classmethod
    def build(
        cls,
        p: int = 100,
        n_active: int = 50,
        min_eigenvalue: float = 1.0,
        max_eigenvalue: float = 100.0,
        inactive_eigenvalue: float = 1e-3,
        noise_scale: float = 0.05,
        bias_amplitude: float = 0.0,
        seed: int = 2026,
    ) -> "PerpendicularQuadraticEnv":
        if n_active >= p:
            raise ValueError("n_active must be < p")
        rng = torch.Generator().manual_seed(seed)

        # Build diagonal H: active dims log-spaced in [min, max]; inactive tiny.
        active_evs = torch.exp(torch.linspace(math.log(min_eigenvalue), math.log(max_eigenvalue), n_active))
        H_diag = torch.full((p,), float(inactive_eigenvalue))
        # Randomly assign which p indices are active.
        perm = torch.randperm(p, generator=rng)
        active_indices = perm[:n_active].sort().values
        inactive_indices = perm[n_active:].sort().values
        for i, idx in enumerate(active_indices.tolist()):
            H_diag[idx] = active_evs[i]

        # theta_star_target: nonzero on active dims, zero on inactive.
        theta_star = torch.zeros(p)
        theta_star[active_indices] = torch.randn(n_active, generator=rng)
        # b = H @ theta_star — diagonal H, so b = H_diag * theta_star elementwise.
        b = H_diag * theta_star

        return cls(
            p=p, n_active=n_active, H_diag=H_diag, b=b,
            active_indices=active_indices,
            inactive_indices=inactive_indices,
            noise_scale=float(noise_scale),
            bias_amplitude=float(bias_amplitude),
            seed=int(seed),
        )

    def to(self, device) -> "PerpendicularQuadraticEnv":
        return PerpendicularQuadraticEnv(
            p=self.p, n_active=self.n_active,
            H_diag=self.H_diag.to(device), b=self.b.to(device),
            active_indices=self.active_indices.to(device),
            inactive_indices=self.inactive_indices.to(device),
            noise_scale=self.noise_scale,
            bias_amplitude=self.bias_amplitude,
            seed=self.seed,
        )

    def true_loss(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * (theta * theta * self.H_diag).sum() - (self.b * theta).sum()

    def true_grad(self, theta: torch.Tensor) -> torch.Tensor:
        # Diagonal H ⇒ Hθ = H_diag * θ.
        return self.H_diag * theta - self.b

    def optimum(self) -> torch.Tensor:
        # Diagonal H, so theta_star = b / H_diag elementwise.
        return self.b / self.H_diag

    def dist_to_optimum(self, theta: torch.Tensor) -> Dict[str, float]:
        opt = self.optimum().to(theta.device)
        diff = theta - opt
        return {
            "dist_total": float(diff.norm().item()),
            "dist_active": float(diff[self.active_indices].norm().item()),
            "dist_inactive": float(diff[self.inactive_indices].norm().item()),
        }

    def minibatch_grad(self, theta: torch.Tensor, t: int, gen: torch.Generator) -> torch.Tensor:
        g = self.true_grad(theta)
        noise = torch.randn(self.p, generator=gen, device=theta.device) * self.noise_scale
        bias = torch.zeros(self.p, device=theta.device)
        if self.bias_amplitude != 0.0:
            bias[self.inactive_indices] = self.bias_amplitude
        return g + noise + bias


# ============================================================================
# Theta wrapper
# ============================================================================
class ThetaParam(nn.Module):
    def __init__(self, p: int, init_scale: float = 0.5, seed: int = 2026):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.randn(p, generator=rng) * init_scale)

    def forward(self, *a, **k):
        return self.theta


# ============================================================================
# Variant runner
# ============================================================================
@dataclass
class VariantResult:
    name: str
    final_loss: float
    initial_loss: float
    dist_total: float
    dist_active: float
    dist_inactive: float
    history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    wall_clock_s: float


def run_variant(
    name: str,
    env: PerpendicularQuadraticEnv,
    optimizer_factory: Callable[[ThetaParam], torch.optim.Optimizer],
    *,
    steps: int,
    log_every: int,
    seed: int,
    init_seed: int,
    device: torch.device,
    pairwise_K: int = 32,
) -> VariantResult:
    grad_gen = torch.Generator(device=device).manual_seed(seed)
    model = ThetaParam(env.p, init_scale=0.5, seed=init_seed).to(device)
    optimizer = optimizer_factory(model)
    tracker = InterferenceTracker(
        model, criterion=lambda *a, **k: torch.zeros(()),
        probe_set=None,
        log_every=log_every, wasted_work_K=32,
        device=device, trajectory_lags=[1, 4, 16],
        pairwise_K=pairwise_K,
    )

    initial_loss = float(env.true_loss(model.theta).item())
    print(f"\n=== {name} ===  init_loss={initial_loss:.4f}")
    t0 = time.time()
    for t in range(steps):
        g = env.minibatch_grad(model.theta.detach(), t, grad_gen)
        optimizer.zero_grad(set_to_none=False)
        if model.theta.grad is None:
            model.theta.grad = torch.zeros_like(model.theta)
        model.theta.grad.copy_(g)
        true_loss_val = float(env.true_loss(model.theta).item())
        tracker.before_step()
        optimizer.step()
        tracker.after_step(loss=true_loss_val)

    final_loss = float(env.true_loss(model.theta).item())
    dist = env.dist_to_optimum(model.theta.detach())
    elapsed = time.time() - t0
    print(f"  final_loss={final_loss:.4f}  dist_total={dist['dist_total']:.4f}  "
          f"dist_active={dist['dist_active']:.4f}  dist_inactive={dist['dist_inactive']:.4f}  "
          f"({elapsed:.1f}s)")

    return VariantResult(
        name=name,
        final_loss=final_loss,
        initial_loss=initial_loss,
        dist_total=dist["dist_total"],
        dist_active=dist["dist_active"],
        dist_inactive=dist["dist_inactive"],
        history=tracker.get_history(),
        summary=tracker.summary(),
        wall_clock_s=elapsed,
    )


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--n-active", type=int, default=50,
                    help="Number of active (descent) dimensions; the rest are inactive (perpendicular)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--noise-scale", type=float, default=0.05)
    ap.add_argument("--bias-amplitude", type=float, default=0.5)
    ap.add_argument("--inactive-eigenvalue", type=float, default=1e-3,
                    help="Eigenvalue of the inactive subspace (must be small)")
    ap.add_argument("--bograd-K", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--pairwise-K", type=int, default=32)
    ap.add_argument("--env-seed", type=int, default=2026)
    ap.add_argument("--init-seed", type=int, default=2026)
    ap.add_argument("--noise-seed", type=int, default=12345)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"p={args.p} n_active={args.n_active} bias_amp={args.bias_amplitude}")
    print(f"  ⇒ {args.p - args.n_active} inactive (perpendicular-bias) dims")

    env_clean = PerpendicularQuadraticEnv.build(
        p=args.p, n_active=args.n_active,
        inactive_eigenvalue=args.inactive_eigenvalue,
        noise_scale=args.noise_scale, bias_amplitude=0.0,
        seed=args.env_seed,
    ).to(device)
    env_biased = PerpendicularQuadraticEnv.build(
        p=args.p, n_active=args.n_active,
        inactive_eigenvalue=args.inactive_eigenvalue,
        noise_scale=args.noise_scale, bias_amplitude=args.bias_amplitude,
        seed=args.env_seed,
    ).to(device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "research" / "01_interference_framework" / "results" / "synthetic_quadratic_perpendicular" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id}, fh, indent=2)

    results: Dict[str, VariantResult] = {}

    def sgd_factory(m):
        return torch.optim.SGD([m.theta], lr=args.lr)

    def make_bograd_factory(mode: str):
        def factory(m):
            return BoGrad(
                [m.theta], torch.optim.SGD,
                buffer_size=args.bograd_K, project_stage="gradient",
                projection_mode=mode, orth_method="sequential",
                lr=args.lr,
            )
        return factory

    runs = [
        ("clean_baseline", env_clean, sgd_factory),
        ("biased_baseline", env_biased, sgd_factory),
        ("biased_bograd_full", env_biased, make_bograd_factory("full")),
        ("biased_bograd_neg", env_biased, make_bograd_factory("negative")),
        ("biased_bograd_pos", env_biased, make_bograd_factory("positive")),
    ]
    for key, env, factory in runs:
        results[key] = run_variant(
            key, env, factory, steps=args.steps, log_every=args.log_every,
            seed=args.noise_seed, init_seed=args.init_seed, device=device,
            pairwise_K=args.pairwise_K,
        )

    for key, r in results.items():
        sub = out_dir / key
        sub.mkdir(exist_ok=True)
        with (sub / "history.json").open("w") as fh:
            json.dump(r.history, fh, indent=2)
        with (sub / "summary.json").open("w") as fh:
            json.dump(r.summary, fh, indent=2, default=str)
        with (sub / "result.json").open("w") as fh:
            json.dump({
                "name": r.name, "final_loss": r.final_loss,
                "initial_loss": r.initial_loss,
                "dist_total": r.dist_total,
                "dist_active": r.dist_active,
                "dist_inactive": r.dist_inactive,
                "wall_clock_s": r.wall_clock_s,
            }, fh, indent=2)

    # Console summary
    print("\n" + "=" * 130)
    print(f"PERPENDICULAR QUADRATIC (n_active={args.n_active}/{args.p}, bias_amp={args.bias_amplitude})")
    print(f"  Bias on the {args.p - args.n_active} INACTIVE dims (~ orthogonal to descent)")
    print(f"  This is the (W') validation: does BoGrad recover when bias is separable from descent?")
    print("=" * 130)
    header = (f"{'variant':28s} {'final_loss':>11s} {'dist_total':>11s} {'dist_active':>12s} {'dist_inactive':>14s} "
              f"{'%pos':>6s} {'%neg':>6s} {'⟨cos+⟩':>8s} {'⟨cos-⟩':>8s}")
    print(header)
    print("-" * 130)
    for key, r in results.items():
        s = r.summary
        pw = s.get("pairwise_summary", {}) or {}
        gp = pw.get("grad_frac_positive_mean", float("nan"))
        gn = pw.get("grad_frac_negative_mean", float("nan"))
        gp_cos = pw.get("grad_mean_positive_cos_mean", float("nan"))
        gn_cos = pw.get("grad_mean_negative_cos_mean", float("nan"))
        print(f"{key:28s} {r.final_loss:>11.4f} {r.dist_total:>11.4f} "
              f"{r.dist_active:>12.4f} {r.dist_inactive:>14.4f} "
              f"{gp:>6.3f} {gn:>6.3f} {gp_cos:>+8.3f} {gn_cos:>+8.3f}")

    # Predictions
    print("\nPREDICTION CHECK:")
    clean = results["clean_baseline"]
    biased = results["biased_baseline"]
    full = results["biased_bograd_full"]
    neg = results["biased_bograd_neg"]
    pos = results["biased_bograd_pos"]

    p1 = clean.dist_total < 0.5
    print(f"  P1 (clean converges):                       {'PASS' if p1 else 'FAIL'}  "
          f"(dist_total={clean.dist_total:.3f})")

    p2 = biased.dist_inactive > clean.dist_inactive * 5
    print(f"  P2 (bias drives off-optimum on inactive):   {'PASS' if p2 else 'FAIL'}  "
          f"(dist_inactive: clean={clean.dist_inactive:.3f}, biased={biased.dist_inactive:.3f})")

    p3 = full.dist_total < biased.dist_total * 0.5 or pos.dist_total < biased.dist_total * 0.5
    print(f"  P3 (BoGrad full or pos recovers — W'):      {'PASS' if p3 else 'FAIL'}  "
          f"(biased={biased.dist_total:.3f}, full={full.dist_total:.3f}, pos={pos.dist_total:.3f})")

    p4 = neg.dist_total > biased.dist_total * 0.7
    print(f"  P4 (BoGrad neg does NOT recover):           {'PASS' if p4 else 'FAIL'}  "
          f"(biased={biased.dist_total:.3f}, neg={neg.dist_total:.3f})")

    overall = all([p1, p2, p3, p4])
    if overall:
        msg = "Hypothesis (W') VALIDATED — orthogonalisation recovers when interference is separable from descent."
    else:
        msg = "one or more predictions failed; see numbers above."
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'} — {msg}")
    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
