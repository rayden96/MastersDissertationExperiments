"""
SignSGD / Signum — sign-of-gradient optimiser.

A clean, paper-ready port of the SignSGD variant from
`PreDiscovery/discoveryPhase2/enhanced_variants.py` (archive). Lives in
`common/` so paper-ready experiments can import it as a first-class base
optimiser and so it can be wrapped by BoGrad / COSGD / GradDrop (all of which
take a `base_optimizer_cls`).

Update rule
-----------
- Vanilla SignSGD (`momentum == 0`):           θ ← θ − lr · sign(g)
- Signum       (`momentum  > 0`, Bernstein et al. 2018):
      buf ← μ · buf + g
      θ   ← θ − lr · sign(buf)

`weight_decay` (L2) is folded into the gradient before the sign is taken, as
in `torch.optim.SGD`.

The optimiser only reads `p.grad` and writes `p.data`, so it composes cleanly
as a base optimiser inside the BoGrad gradient/update-stage wrappers.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SignSGD(Optimizer):
    """SignSGD with optional Signum momentum.

    Parameters
    ----------
    params : iterable of torch.Tensor
        Parameters to optimise.
    lr : float, default 1e-3
        Learning rate.
    momentum : float, default 0.0
        Signum momentum coefficient μ. 0 → vanilla SignSGD.
    weight_decay : float, default 0.0
        L2 penalty, folded into the gradient before the sign.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SignSGD does not support sparse gradients")

                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum != 0.0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(grad).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(grad)
                    direction = buf
                else:
                    direction = grad

                p.add_(torch.sign(direction), alpha=-lr)

        return loss


__all__ = ["SignSGD"]
