"""
MoGrad — Momentum-orthogonalised gradient optimizer.

Maintains a momentum-style EMA of the current gradient as a reference
vector, then applies vanilla SGD with a *projected* gradient. Importantly:
the EMA is NOT added to the update (unlike standard momentum); it is used
ONLY as a direction to orthogonalise against.

  m_t  = β · m_{t-1} + (1 - β) · g_t           (EMA reference, not in update)
  g̃_t  = g_t  -  α · ⟨g_t, m̂_t⟩ · m̂_t          (project against m_t direction)
  θ_{t+1} = θ_t - lr · g̃_t                       (vanilla SGD on projected grad)

α is 0/1 controlled by `projection_mode`:
  - "full"     : α = 1 always
  - "negative" : α = 1 when ⟨g_t, m̂_t⟩ < 0 (remove destructive overlap)
  - "positive" : α = 1 when ⟨g_t, m̂_t⟩ > 0 (remove redundant overlap)

Projection only fires after `start_step`. Before that, the EMA is
accumulated but not used. This handles the "wait for momentum to
stabilise" requirement.

The projection is per-parameter-tensor (each tensor has its own m_t).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


_VALID_MODES = ("full", "negative", "positive")


class MoGrad(Optimizer):
    """Vanilla SGD with the gradient projected against a momentum reference.

    Parameters
    ----------
    params : iterable
        Parameters to optimise.
    lr : float, default 0.05
    momentum_beta : float, default 0.9
        EMA coefficient for the reference momentum vector.
    start_step : int, default 100
        Step (counting from 1) at which projection begins. Before this,
        the optimiser is plain SGD; the reference is still accumulated.
    projection_mode : {"full", "negative", "positive"}, default "negative"
        Which sign of overlap to remove (see module docstring).
    weight_decay : float, default 0.0
    eps : float, default 1e-12

    Notes
    -----
    Step magnitudes here are bounded by ‖g_t‖ — there is no momentum
    accumulation in the update direction. So the optimiser behaves
    closer to vanilla SGD than to SGD-with-momentum, with a structural
    "subtract reference overlap" modification of the gradient.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 0.05,
        momentum_beta: float = 0.9,
        start_step: int = 100,
        projection_mode: str = "negative",
        weight_decay: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        if projection_mode not in _VALID_MODES:
            raise ValueError(f"projection_mode must be one of {_VALID_MODES}")
        if start_step < 0:
            raise ValueError("start_step must be >= 0")

        defaults = dict(
            lr=float(lr),
            momentum_beta=float(momentum_beta),
            start_step=int(start_step),
            projection_mode=projection_mode,
            weight_decay=float(weight_decay),
            eps=float(eps),
        )
        super().__init__(params, defaults)
        self.global_step = 0

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                # Reference momentum vector — same shape as p.
                self.state[p]["mo_ref"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

        # Diagnostics — track when projection actually fires.
        self._projections_applied = 0
        self._steps_after_warmup = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum_beta"]
            start = group["start_step"]
            mode = group["projection_mode"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # 1. Update EMA reference (always — even during warmup).
                ref = self.state[p]["mo_ref"]
                ref.mul_(beta).add_(grad, alpha=1.0 - beta)

                # 2. Decide whether to project this step.
                projected_grad = grad
                if self.global_step > start:
                    self._steps_after_warmup += 1
                    g_flat = grad.reshape(-1)
                    ref_flat = ref.reshape(-1)
                    ref_norm = ref_flat.norm()
                    if ref_norm.item() > eps:
                        ref_unit = ref_flat / (ref_norm + eps)
                        dot_val = torch.dot(g_flat, ref_unit)
                        dot_val_item = dot_val.item()
                        fire = (
                            mode == "full"
                            or (mode == "negative" and dot_val_item < 0)
                            or (mode == "positive" and dot_val_item > 0)
                        )
                        if fire:
                            self._projections_applied += 1
                            new_g_flat = g_flat - dot_val * ref_unit
                            projected_grad = new_g_flat.reshape_as(grad)

                # 3. Apply vanilla SGD with the (possibly projected) grad.
                p.add_(projected_grad, alpha=-lr)

        return loss

    def projection_rate(self) -> float:
        """Fraction of post-warmup steps where projection actually fired."""
        if self._steps_after_warmup == 0:
            return 0.0
        return self._projections_applied / self._steps_after_warmup


__all__ = ["MoGrad"]
