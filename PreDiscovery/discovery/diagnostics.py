"""
Diagnostic harness for the discovery ablation.

Wraps the train step to capture, every N steps, the metrics from the geometry
critique:

  g_norm        — ‖g_t‖
  u_norm        — ‖θ_t − θ_{t-1}‖   (the actually-applied parameter delta)
  cos_g_prev    — cos(g_t, g_{t-1})  — gradient-space alignment
  cos_u_prev    — cos(u_t, u_{t-1})  — UPDATE-space alignment (the one that
                   actually maps to interference in parameter space)
  cos_u_g       — cos(u_t, -g_t)     — descent-direction quality
                   (positive ⇒ update is in a descent direction; ≈ 1 means
                   close to negative gradient direction)

All quantities are aggregated over the *full flattened parameter vector* — i.e.
all parameter tensors concatenated. This is the right scope for "is this
optimiser making globally-consistent steps", which is what the prompt cared about.

Usage
-----

    harness = DiagnosticHarness(model, log_every=50)
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        harness.before_step()           # snapshot grads + params
        optimizer.step()
        harness.after_step(loss=loss.item())  # compute deltas, log if log_every

    history = harness.history  # list of dicts
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn


def _flat(params: List[Tensor]) -> Tensor:
    return torch.cat([p.reshape(-1) for p in params])


def _safe_cos(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    na = a.norm()
    nb = b.norm()
    if na.item() <= eps or nb.item() <= eps:
        return float("nan")
    return float((a @ b / (na * nb)).item())


class DiagnosticHarness:
    def __init__(self, model: nn.Module, log_every: int = 50):
        self.model = model
        self.log_every = int(log_every)
        self.history: List[Dict[str, float]] = []

        self._step = 0
        self._params_before: Optional[Tensor] = None
        self._grad_before: Optional[Tensor] = None
        self._prev_grad: Optional[Tensor] = None
        self._prev_update: Optional[Tensor] = None

    def reset(self) -> None:
        self.history.clear()
        self._step = 0
        self._params_before = None
        self._grad_before = None
        self._prev_grad = None
        self._prev_update = None

    def _gather_params(self) -> List[Tensor]:
        return [p.detach() for p in self.model.parameters() if p.requires_grad]

    def _gather_grads(self) -> List[Tensor]:
        out = []
        for p in self.model.parameters():
            if p.grad is None or not p.requires_grad:
                continue
            out.append(p.grad.detach())
        return out

    def before_step(self) -> None:
        """Call AFTER loss.backward() and BEFORE optimizer.step()."""
        if self._step % self.log_every == 0:
            self._params_before = _flat(self._gather_params()).clone()
            grads = self._gather_grads()
            self._grad_before = _flat(grads).clone() if grads else None

    def after_step(self, loss: Optional[float] = None) -> None:
        """Call AFTER optimizer.step()."""
        if self._step % self.log_every == 0:
            self._record(loss)
        self._step += 1

    def _record(self, loss: Optional[float]) -> None:
        params_after = _flat(self._gather_params())
        if self._params_before is None or self._grad_before is None:
            self._step += 0  # no-op; just bail
            return

        update = params_after - self._params_before  # actually-applied delta
        grad = self._grad_before

        row: Dict[str, float] = {
            "step": float(self._step),
            "loss": float(loss) if loss is not None else float("nan"),
            "g_norm": float(grad.norm().item()),
            "u_norm": float(update.norm().item()),
        }

        # cos(u_t, -g_t)  — descent if positive (update opposes gradient).
        if grad.numel() == update.numel():
            row["cos_u_neg_g"] = _safe_cos(update, -grad)

        if self._prev_grad is not None and self._prev_grad.numel() == grad.numel():
            row["cos_g_prev"] = _safe_cos(grad, self._prev_grad)
        if self._prev_update is not None and self._prev_update.numel() == update.numel():
            row["cos_u_prev"] = _safe_cos(update, self._prev_update)

        self.history.append(row)
        self._prev_grad = grad.clone()
        self._prev_update = update.clone()

    # Convenience for summary stats over a window
    def aggregate_window(self, last_k: Optional[int] = None) -> Dict[str, float]:
        rows = self.history if last_k is None else self.history[-last_k:]
        if not rows:
            return {}
        out: Dict[str, float] = {}
        keys = set()
        for r in rows:
            keys.update(r.keys())
        for k in keys:
            if k == "step":
                continue
            vals = [r[k] for r in rows if k in r and isinstance(r[k], float) and math.isfinite(r[k])]
            if vals:
                out[f"{k}_mean"] = sum(vals) / len(vals)
        return out


__all__ = ["DiagnosticHarness"]
