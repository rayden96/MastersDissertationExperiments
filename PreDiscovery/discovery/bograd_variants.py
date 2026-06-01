"""
Adam variants with BoGrad-style projection inserted at different points in the
update pipeline.

The standard Adam update is:

    g_t        = ∇L(θ_{t-1})
    m_t        = β₁ · m_{t-1} + (1 - β₁) · g_t
    v_t        = β₂ · v_{t-1} + (1 - β₂) · g_t ⊙ g_t
    m̂_t       = m_t / (1 - β₁^t)
    v̂_t       = v_t / (1 - β₂^t)
    u_t        = m̂_t / (√v̂_t + ε)
    θ_t        = θ_{t-1} - lr · u_t

BoGrad-style projection can be inserted at four points, giving substantively
different algorithms (not just different framings of the same one). We
implement all four in a single class, switchable via `projection_point`:

  "off"            — pure Adam, no projection.
  "grad"           — project g_t against past g's BEFORE updating m_t, v_t.
                     This is the original BOSGD recipe; for Adam it's the
                     incoherent variant — included for direct comparison.
  "momentum"       — let m_t accumulate naturally; project m_t against past
                     m_t's; use proj(m_t) to compute the step but DO NOT
                     overwrite the EMA state. Avoids compounding projection
                     into the EMA itself.
  "natural"        — project g_t in the v_t-weighted inner product
                     ⟨a,b⟩_v = Σ a_j b_j / (√v_j + ε). This is the projection
                     in Adam's "natural" metric. Uses current v_t (acknowledged
                     staleness — buffer entries were appended under earlier v).
  "update"         — let Adam compute u_t fully; project u_t against past
                     applied updates.

All four respect a common `projection_mode`:
  "full"     — subtract the full projection component along each buffered
               direction.
  "negative" — subtract only when the dot product is negative (PCGrad-style
               asymmetric projection — preserves coherent descent, removes
               only destructive interference).

Buffers are per-parameter-tensor (matches the rest of BoGrad's per-tensor
geometry). Stored as lists of normalised 1-D tensors.

Note: under "momentum" mode, the EMA buffer m_t is NOT overwritten with
proj(m_t). This is a deliberate choice — overwriting compounds projection
into Adam's state across steps, which makes the algorithm increasingly
aggressive over time and was empirically worse in early tests.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


_VALID_POINTS = ("off", "grad", "momentum", "natural", "update")
_VALID_MODES = ("full", "negative")


class AdamWithBoGrad(Optimizer):
    """Adam with a configurable BoGrad-style projection inserted at one point.

    Re-implements Adam from scratch (rather than wrapping torch.optim.Adam) so
    the projection can be inserted before m/v updates, between m and v scaling,
    or after the full update is computed. The "off" mode reproduces standard
    Adam exactly.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        # BoGrad knobs
        buffer_size: int = 8,
        projection_point: str = "off",
        projection_mode: str = "negative",
        store_normalised: bool = True,
        projection_dtype: torch.dtype = torch.float32,
        projection_eps: float = 1e-12,
    ) -> None:
        if projection_point not in _VALID_POINTS:
            raise ValueError(f"projection_point must be one of {_VALID_POINTS}")
        if projection_mode not in _VALID_MODES:
            raise ValueError(f"projection_mode must be one of {_VALID_MODES}")
        if buffer_size < 0:
            raise ValueError("buffer_size must be >= 0")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.buffer_size = int(buffer_size)
        self.projection_point = projection_point
        self.projection_mode = projection_mode
        self.store_normalised = bool(store_normalised)
        self.projection_dtype = projection_dtype
        self.projection_eps = float(projection_eps)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    # Skip sparse — same convention as the rest of BoGrad.
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["buffer"] = []  # list of 1D normalised tensors
                state["step"] += 1
                step_t = state["step"]

                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)  # decoupled-style: g <- g + wd * p

                # ---------------- "grad" projection (Euclidean, on raw g) ----------------
                if self.projection_point == "grad" and self.buffer_size > 0:
                    g_flat = grad.view(-1).to(dtype=self.projection_dtype)
                    g_proj = self._project_euclidean(g_flat, state["buffer"])
                    self._append(state["buffer"], g_flat)
                    grad = g_proj.view_as(grad).to(dtype=grad.dtype)

                # ---------------- "natural" projection (v-weighted, on raw g) ----------------
                if self.projection_point == "natural" and self.buffer_size > 0:
                    g_flat = grad.view(-1).to(dtype=self.projection_dtype)
                    v_flat = state["exp_avg_sq"].view(-1).to(dtype=self.projection_dtype)
                    g_proj = self._project_natural(g_flat, state["buffer"], v_flat, eps=eps)
                    self._append(state["buffer"], g_flat)
                    grad = g_proj.view_as(grad).to(dtype=grad.dtype)

                # ---------------- Adam state update ----------------
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** step_t
                bias_corr2 = 1.0 - beta2 ** step_t

                # ---------------- "momentum" projection (Euclidean, on m_t) ----------------
                if self.projection_point == "momentum" and self.buffer_size > 0:
                    m_flat = exp_avg.view(-1).to(dtype=self.projection_dtype)
                    m_proj = self._project_euclidean(m_flat, state["buffer"])
                    self._append(state["buffer"], m_flat)
                    # Use projected m for this step, but DO NOT overwrite the EMA state.
                    m_for_step = m_proj.view_as(exp_avg).to(dtype=exp_avg.dtype)
                else:
                    m_for_step = exp_avg

                m_hat = m_for_step / bias_corr1
                v_hat = exp_avg_sq / bias_corr2
                update = m_hat / (v_hat.sqrt().add_(eps))

                # ---------------- "update" projection (Euclidean, on final u) ----------------
                if self.projection_point == "update" and self.buffer_size > 0:
                    u_flat = update.view(-1).to(dtype=self.projection_dtype)
                    u_proj = self._project_euclidean(u_flat, state["buffer"])
                    self._append(state["buffer"], u_flat)
                    update = u_proj.view_as(update).to(dtype=update.dtype)

                p.add_(update, alpha=-lr)

        return loss

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _project_euclidean(self, x_flat: Tensor, buffer: List[Tensor]) -> Tensor:
        """Sequential Gram-Schmidt subtraction (the IJCNN-working method)."""
        if not buffer:
            return x_flat
        out = x_flat.clone()
        for b in buffer:
            b_vec = b.to(device=x_flat.device, dtype=self.projection_dtype)
            denom = 1.0 if self.store_normalised else float(torch.dot(b_vec, b_vec).item())
            if denom <= self.projection_eps:
                continue
            dot_val = torch.dot(out, b_vec)
            if self.projection_mode == "negative" and dot_val.item() >= 0:
                continue
            out = out - (dot_val / denom) * b_vec
        return out

    def _project_natural(
        self, x_flat: Tensor, buffer: List[Tensor], v_flat: Tensor, eps: float,
    ) -> Tensor:
        """Sequential GS in v-weighted inner product ⟨a,b⟩_v = Σ a_j b_j / (√v_j + eps).

        Buffer entries were stored under PAST v values; we use CURRENT v_t for
        the inner product. The "natural" geometry therefore drifts. Acceptable
        for ablation; an exact treatment would re-scale buffer entries.
        """
        if not buffer:
            return x_flat
        D = v_flat.sqrt().add_(eps)  # per-coordinate scale
        out = x_flat.clone()
        for b in buffer:
            b_vec = b.to(device=x_flat.device, dtype=self.projection_dtype)
            # ⟨out, b⟩_v = Σ out_j b_j / D_j
            num = torch.sum(out * b_vec / D)
            denom = torch.sum(b_vec * b_vec / D)
            if denom.item() <= self.projection_eps:
                continue
            if self.projection_mode == "negative" and num.item() >= 0:
                continue
            out = out - (num / denom) * b_vec
        return out

    def _append(self, buffer: List[Tensor], x_flat: Tensor) -> None:
        x = x_flat.detach()
        x_norm = x.norm()
        if not torch.isfinite(x_norm) or x_norm.item() <= self.projection_eps:
            return
        if self.store_normalised:
            x = x / (x_norm + self.projection_eps)
        buffer.append(x.to(dtype=self.projection_dtype))
        while len(buffer) > self.buffer_size:
            buffer.pop(0)


__all__ = ["AdamWithBoGrad"]
