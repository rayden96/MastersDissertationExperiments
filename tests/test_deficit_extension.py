"""
Verify the preconditioned-deficit extension to InterferenceMeter.

    python tests/test_deficit_extension.py

Checks:
  1. Regression: the SGD-yardstick D_t equals the canonical positive=hurt
     formula <g,u> + lr*||g||^2 exactly, on random (g, u).
  2. Hook off (default): cum_deficit_precond stays 0 / count 0; logs carry NaN
     for D_t_precond -> original behaviour preserved.
  3. Hook on with an SGD ideal (u_ideal = -lr*g): D_t_precond == D_t exactly,
     confirming the general form reduces to the yardstick for plain SGD.
  4. Hook on with an Adam-like elementwise-scaled ideal: D_t_precond differs
     from the SGD-yardstick D_t (it isolates preconditioning), and accumulates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

_REPO = Path(__file__).resolve().parent.parent
_PRE = _REPO / "PaperReadyExperiments"
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from interference.problem import InterferenceProblem  # noqa: E402
from interference.meter import InterferenceMeter  # noqa: E402


class _ToyProblem(InterferenceProblem):
    """Minimal problem driving the meter with scripted params/grads.

    `ideal_mode`: None -> no hook; 'sgd' -> -lr*g; 'adam' -> elementwise-scaled.
    """

    def __init__(self, dim=12, lr=0.1, ideal_mode=None):
        self.dim = dim
        self.lr = lr
        self.ideal_mode = ideal_mode
        self._theta = torch.zeros(dim)
        g = torch.randn(dim)
        self._ref = g / g.norm()
        self._scale = (torch.rand(dim) + 0.5)  # Adam-like per-coord precond

    def flatten_params(self):
        return self._theta.clone()

    @property
    def device(self):
        return torch.device("cpu")

    def compute_reference(self) -> Tuple[torch.Tensor, float]:
        return self._ref.clone(), 1.0

    def per_subgroup_grads(self, batch) -> Dict[int, torch.Tensor]:
        return {}

    def preconditioned_ideal(self, ref_grad):
        if self.ideal_mode is None:
            return None
        if self.ideal_mode == "sgd":
            return -self.lr * ref_grad
        if self.ideal_mode == "adam":
            return -self.lr * ref_grad * self._scale
        raise ValueError(self.ideal_mode)

    # test helper: apply a chosen update and notify the meter
    def apply_update(self, u):
        self._theta = self._theta + u


def _expected_D(lr, g, u):
    # Canonical deficit (positive = hurt): D_t = <g, u> - <g, u_ideal>,
    # u_ideal = -lr*g  =>  <g, u> + lr*||g||^2.
    return torch.dot(g, u).item() + lr * torch.dot(g, g).item()


def test_regression_and_hook():
    torch.manual_seed(0)
    lr = 0.1

    # --- hook OFF: regression vs original formula ---
    prob = _ToyProblem(lr=lr, ideal_mode=None)
    meter = InterferenceMeter(prob, lr=lr, log_every=1, ref_refresh_every=10**9)
    meter.initialize()
    g = meter.ref_grad.clone()
    max_diff = 0.0
    for step in range(20):
        u = torch.randn(prob.dim) * 0.05
        meter.before_step()
        prob.apply_update(u)
        meter.after_step(step, batch=None, applied_loss=0.0)
        log = meter.logs[-1]
        ref_D = _expected_D(lr, g, u)
        max_diff = max(max_diff, abs(log["D_t"] - ref_D))
        # hook off -> precond is NaN and uncounted
        assert log["D_t_precond"] != log["D_t_precond"], "precond should be NaN when hook off"
    assert max_diff < 1e-5, f"D_t regression diff {max_diff}"
    assert meter.cum_deficit_precond == 0.0 and meter.cum_deficit_precond_count == 0
    print(f"  [1] D_t == original formula (max diff {max_diff:.2e})  [2] hook-off preserves behaviour  OK")

    # --- hook ON, SGD ideal: D_t_precond == D_t ---
    prob = _ToyProblem(lr=lr, ideal_mode="sgd")
    meter = InterferenceMeter(prob, lr=lr, log_every=1, ref_refresh_every=10**9)
    meter.initialize()
    for step in range(20):
        u = torch.randn(prob.dim) * 0.05
        meter.before_step(); prob.apply_update(u); meter.after_step(step, None, 0.0)
        log = meter.logs[-1]
        assert abs(log["D_t_precond"] - log["D_t"]) < 1e-6, "SGD ideal must equal yardstick"
    assert meter.cum_deficit_precond_count == 20
    print("  [3] SGD ideal: D_t_precond == D_t  OK")

    # --- hook ON, Adam-like ideal: differs, accumulates ---
    prob = _ToyProblem(lr=lr, ideal_mode="adam")
    meter = InterferenceMeter(prob, lr=lr, log_every=1, ref_refresh_every=10**9)
    meter.initialize()
    any_diff = False
    for step in range(20):
        u = torch.randn(prob.dim) * 0.05
        meter.before_step(); prob.apply_update(u); meter.after_step(step, None, 0.0)
        log = meter.logs[-1]
        if abs(log["D_t_precond"] - log["D_t"]) > 1e-4:
            any_diff = True
    assert any_diff, "Adam-like ideal should differ from SGD yardstick"
    assert meter.cum_deficit_precond_count == 20 and meter.cum_deficit_precond != 0.0
    print("  [4] Adam-like ideal: D_t_precond isolates preconditioning (differs, accumulates)  OK")


if __name__ == "__main__":
    print("Preconditioned-deficit extension verification:")
    test_regression_and_hook()
    print("\nAll deficit-extension assertions passed.")
