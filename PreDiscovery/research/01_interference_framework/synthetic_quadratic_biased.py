"""
Synthetic experiment 1b — BIASED-interference quadratic loss.

This is the follow-up to synthetic_quadratic.py (Finding F5), which showed
that zero-mean alternating interference is detected by the framework
metrics but does NOT slow training (because SGD's averaging absorbs it),
and BoGrad applied to it actively HURTS by stripping useful signal.

Setting
-------
Same quadratic loss as synthetic_quadratic.py:
    L(theta) = ½ thetaᵀ H theta - bᵀ theta
but with **biased** interference instead of alternating. For step t:
    g_t(theta) = (Htheta - b) + noise_t + bias_t
where ``bias_t`` is a structured perturbation that does NOT cancel across
consecutive steps:

  - "drift"     : bias_t = amp * e_J          (constant push along J)
  - "rotating"  : bias_t = amp * R_t e_J      (rotates within J subspace
                                                 over a slow period — non-cancelling
                                                 within K-window).
  - "decaying"  : bias_t = amp * (0.99)^t e_J (initial bias that fades —
                                                 simulates an early-training
                                                 distortion).

The key difference from synthetic_quadratic.py: the bias is NOT zero-mean,
so SGD cannot average it out. It actually shifts the optimum (drift case)
or keeps applying the same direction (rotating, decaying). Hypothesis (W)
predicts this kind of interference DOES slow training, and orthogonalisation
should genuinely help.

Configurations
--------------
Three injection modes (drift / rotating / decaying) × four optimisers:
- baseline (SGD)
- BoGrad full (subtract all overlap)
- BoGrad negative (subtract destructive overlap only)
- BoGrad positive (subtract redundant overlap only)

Predictions
-----------
- P1: clean run converges, dist_opt small.
- P2: biased run converges WORSE than clean (real slowdown — F5's null
  result fixed in this version).
- P3: BoGrad full and BoGrad negative recover convergence on biased run
  (closer to clean).
- P4: BoGrad positive does NOT help (it removes the wrong thing for this
  kind of interference; the bias is a "destructive" pull, removing only
  destructive components is the right intervention).

Output
------
research/01_interference_framework/results/synthetic_quadratic_biased/run_<timestamp>/
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
# Quadratic environment with biased interference
# ============================================================================
@dataclass
class BiasedQuadraticEnv:
    """L(theta) = ½ thetaᵀ H theta - bᵀ theta with biased mini-batch noise.

    Three bias modes for the structured-interference component:
      - "drift": bias_t = amp * e_J  (constant pull off the true gradient)
      - "rotating": bias_t = amp * cos(t * 2π / period) * e_J +
                            amp * sin(t * 2π / period) * e_J2
                    (slow rotation in a 2-coord subspace — adjacent steps
                    share a strong common bias direction, but the direction
                    drifts; non-cancelling within a K-step window for K << period)
      - "decaying": bias_t = amp * (decay)^t * e_J  (initial bias that fades)
    """

    p: int
    H: torch.Tensor
    b: torch.Tensor
    noise_scale: float
    bias_amplitude: float
    bias_mode: str
    interference_indices: torch.Tensor
    bias_period: int
    bias_decay: float
    seed: int

    @classmethod
    def build(
        cls,
        p: int = 100,
        min_eigenvalue: float = 1.0,
        max_eigenvalue: float = 100.0,
        n_interference_coords: int = 20,
        noise_scale: float = 0.05,
        bias_amplitude: float = 0.0,
        bias_mode: str = "drift",
        bias_period: int = 200,
        bias_decay: float = 0.99,
        seed: int = 2026,
    ) -> "BiasedQuadraticEnv":
        if bias_mode not in ("drift", "rotating", "decaying"):
            raise ValueError(f"bias_mode must be drift/rotating/decaying, got {bias_mode}")
        rng = torch.Generator().manual_seed(seed)
        log_evs = torch.linspace(math.log(min_eigenvalue), math.log(max_eigenvalue), p)
        H_diag = torch.exp(log_evs)
        H = torch.diag(H_diag)
        theta_star_target = torch.randn(p, generator=rng)
        b = H @ theta_star_target
        J = torch.argsort(theta_star_target.abs(), descending=True)[:n_interference_coords]
        return cls(
            p=p, H=H, b=b,
            noise_scale=float(noise_scale),
            bias_amplitude=float(bias_amplitude),
            bias_mode=bias_mode,
            interference_indices=J,
            bias_period=int(bias_period),
            bias_decay=float(bias_decay),
            seed=int(seed),
        )

    def to(self, device) -> "BiasedQuadraticEnv":
        return BiasedQuadraticEnv(
            p=self.p, H=self.H.to(device), b=self.b.to(device),
            noise_scale=self.noise_scale, bias_amplitude=self.bias_amplitude,
            bias_mode=self.bias_mode, interference_indices=self.interference_indices.to(device),
            bias_period=self.bias_period, bias_decay=self.bias_decay, seed=self.seed,
        )

    def true_loss(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * theta @ (self.H @ theta) - self.b @ theta

    def true_grad(self, theta: torch.Tensor) -> torch.Tensor:
        return self.H @ theta - self.b

    def optimum(self) -> torch.Tensor:
        return torch.linalg.solve(self.H, self.b)

    def minibatch_grad(self, theta: torch.Tensor, t: int, gen: torch.Generator) -> torch.Tensor:
        g = self.true_grad(theta)
        # iid noise
        noise = torch.randn(self.p, generator=gen, device=theta.device) * self.noise_scale
        # Structured biased interference (NOT zero-mean across consecutive steps)
        bias = torch.zeros(self.p, device=theta.device)
        if self.bias_amplitude != 0.0:
            J = self.interference_indices
            if self.bias_mode == "drift":
                # Constant pull on J coordinates
                bias[J] = self.bias_amplitude
            elif self.bias_mode == "rotating":
                # Slow rotation within the J subspace.
                # Split J into two halves; rotate within them.
                half = max(len(J) // 2, 1)
                J1 = J[:half]
                J2 = J[half:2 * half] if 2 * half <= len(J) else J1
                phase = 2 * math.pi * t / self.bias_period
                bias[J1] = self.bias_amplitude * math.cos(phase)
                bias[J2] = self.bias_amplitude * math.sin(phase)
            elif self.bias_mode == "decaying":
                bias[J] = self.bias_amplitude * (self.bias_decay ** t)
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
    dist_opt: float
    history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    wall_clock_s: float


def run_variant(
    name: str,
    env: BiasedQuadraticEnv,
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
    theta_star = env.optimum().to(device)
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
    dist = float((model.theta.detach() - theta_star).norm().item())
    elapsed = time.time() - t0
    print(f"  final_loss={final_loss:.4f}  dist_opt={dist:.4f}  ({elapsed:.1f}s)")

    return VariantResult(
        name=name,
        final_loss=final_loss,
        initial_loss=initial_loss,
        dist_opt=dist,
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
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--noise-scale", type=float, default=0.05)
    ap.add_argument("--bias-amplitude", type=float, default=0.5,
                    help="Persistent bias magnitude. Smaller than synth_quadratic's "
                         "amp because biased interference is more impactful.")
    ap.add_argument("--bias-mode", type=str, default="drift",
                    choices=["drift", "rotating", "decaying"],
                    help="Type of biased interference to inject.")
    ap.add_argument("--bias-period", type=int, default=200)
    ap.add_argument("--bias-decay", type=float, default=0.99)
    ap.add_argument("--n-interference-coords", type=int, default=20)
    ap.add_argument("--bograd-K", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--pairwise-K", type=int, default=32)
    ap.add_argument("--env-seed", type=int, default=2026)
    ap.add_argument("--init-seed", type=int, default=2026)
    ap.add_argument("--noise-seed", type=int, default=12345)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Bias: mode={args.bias_mode}  amp={args.bias_amplitude}")

    env_clean = BiasedQuadraticEnv.build(
        p=args.p, n_interference_coords=args.n_interference_coords,
        noise_scale=args.noise_scale, bias_amplitude=0.0, bias_mode=args.bias_mode,
        seed=args.env_seed,
    ).to(device)
    env_biased = BiasedQuadraticEnv.build(
        p=args.p, n_interference_coords=args.n_interference_coords,
        noise_scale=args.noise_scale, bias_amplitude=args.bias_amplitude,
        bias_mode=args.bias_mode, bias_period=args.bias_period, bias_decay=args.bias_decay,
        seed=args.env_seed,
    ).to(device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "research" / "01_interference_framework" / "results" / "synthetic_quadratic_biased" / f"run_{run_id}"
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

    # Save outputs
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
                "initial_loss": r.initial_loss, "dist_opt": r.dist_opt,
                "wall_clock_s": r.wall_clock_s,
            }, fh, indent=2)

    # Console summary
    print("\n" + "=" * 110)
    print(f"BIASED QUADRATIC ({args.bias_mode}, amp={args.bias_amplitude}) — does (W) hold?")
    print("=" * 110)
    header = (f"{'variant':28s} {'final_loss':>11s} {'dist_opt':>10s} "
              f"{'cos(g,prev)':>12s} {'WW_K':>7s} {'%pos':>6s} {'%neg':>6s} {'⟨cos+⟩':>8s} {'⟨cos-⟩':>8s}")
    print(header)
    print("-" * 110)
    for key, r in results.items():
        s = r.summary
        cos_prev = s.get("cos_g_prev_mean", float("nan"))
        ww = s.get("ww_wasted_work_ratio_mean", float("nan"))
        pw = s.get("pairwise_summary", {}) or {}
        gp = pw.get("grad_frac_positive_mean", float("nan"))
        gn = pw.get("grad_frac_negative_mean", float("nan"))
        gp_cos = pw.get("grad_mean_positive_cos_mean", float("nan"))
        gn_cos = pw.get("grad_mean_negative_cos_mean", float("nan"))
        print(f"{key:28s} {r.final_loss:>11.4f} {r.dist_opt:>10.4f} "
              f"{cos_prev:>+12.4f} {ww:>7.3f} {gp:>6.3f} {gn:>6.3f} "
              f"{gp_cos:>+8.3f} {gn_cos:>+8.3f}")

    # Predictions
    print("\nPREDICTION CHECK:")
    clean = results["clean_baseline"].dist_opt
    biased = results["biased_baseline"].dist_opt
    full = results["biased_bograd_full"].dist_opt
    neg = results["biased_bograd_neg"].dist_opt
    pos = results["biased_bograd_pos"].dist_opt

    p1 = clean < biased * 0.8 or clean < 0.2
    print(f"  P1 (clean converges):              {'PASS' if p1 else 'FAIL'}  "
          f"(dist_opt clean={clean:.3f})")
    p2 = biased > clean * 1.5
    print(f"  P2 (bias slows convergence):       {'PASS' if p2 else 'FAIL'}  "
          f"(dist_opt: clean={clean:.3f} vs biased={biased:.3f})")
    p3 = full < biased and neg < biased
    print(f"  P3 (BoGrad full+neg recover):      {'PASS' if p3 else 'FAIL'}  "
          f"(biased={biased:.3f}, full={full:.3f}, neg={neg:.3f})")
    p4 = pos >= neg
    print(f"  P4 (BoGrad positive ≮ negative):   {'PASS' if p4 else 'FAIL'}  "
          f"(neg={neg:.3f}, pos={pos:.3f})")

    overall = all([p1, p2, p3, p4])
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'} — "
          f"{'Hypothesis (W) supported on biased interference; mode comparison informative.' if overall else 'one or more predictions failed; iterate.'}")
    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
