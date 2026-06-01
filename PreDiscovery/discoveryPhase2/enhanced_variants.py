"""
Phase-2 BoGrad enhancements. Custom optimisers plus thin wrappers.

  ComplementMomentumSGD        — SGD with momentum + perpendicular boost
  ComplementAdam                — Adam with perpendicular preconditioned boost
                                  (now with warmup_steps to avoid early v ≈ 0
                                  divergence)
  ComplementRMSprop             — RMSprop with momentum + perp boost on the
                                  preconditioned gradient
  ComplementSignSGD             — SignSGD with momentum + perp boost before sign()
  SignSGD                       — plain SignSGD (with optional momentum) baseline
  AdaptiveTriggerBoGrad         — wraps any base; projects only when
                                  cos(g_t, EMA(g)) < threshold
  GradientDifferenceBoGrad      — wraps any base; buffer holds (g_t - g_{t-1})
  MultiScaleBoGrad              — wraps any base; two buffers (short + long)

(Variant 1 from the README — long-buffer trajectory projection — is just the
existing common.optimizers.BoGrad with project_stage="update", buffer_size=32,
projection_mode="negative". No new code needed; configured in the runner.)

All wrappers are per-parameter-tensor (matches the rest of BoGrad). Buffer
entries are stored as 1-D normalised tensors. Projection is sequential
Gram-Schmidt subtraction with optional asymmetric ("negative") gating —
phase 1 confirmed sequential is the right algorithm.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


_EPS = 1e-12


# ----------------------------------------------------------------------------
# Shared projection primitive
# ----------------------------------------------------------------------------
def _project_sequential(
    x_flat: Tensor,
    buffer: List[Tensor],
    *,
    projection_mode: str = "negative",
    store_normalised: bool = True,
    eps: float = _EPS,
) -> Tensor:
    """Sequential Gram-Schmidt subtraction (the IJCNN-working algorithm)."""
    if not buffer:
        return x_flat
    out = x_flat.clone()
    for b in buffer:
        b_vec = b.to(device=x_flat.device, dtype=x_flat.dtype)
        denom = 1.0 if store_normalised else float(torch.dot(b_vec, b_vec).item())
        if denom <= eps:
            continue
        dot_val = torch.dot(out, b_vec)
        if projection_mode == "negative" and dot_val.item() >= 0:
            continue
        out = out - (dot_val / denom) * b_vec
    return out


def _append_buffer(
    buffer: List[Tensor],
    x_flat: Tensor,
    max_size: int,
    *,
    store_normalised: bool = True,
    dtype: torch.dtype = torch.float32,
    eps: float = _EPS,
) -> None:
    x = x_flat.detach()
    x_norm = x.norm()
    if not torch.isfinite(x_norm) or x_norm.item() <= eps:
        return
    if store_normalised:
        x = x / (x_norm + eps)
    buffer.append(x.to(dtype=dtype))
    while len(buffer) > max_size:
        buffer.pop(0)


# ============================================================================
# Variant 2a — Complement-aware momentum (SGD)
# ============================================================================
class ComplementMomentumSGD(Optimizer):
    """SGD with momentum AND an explicit perpendicular-to-momentum boost.

    Standard SGD with momentum:
        v_t = mu * v_{t-1} + g_t
        θ ← θ - lr * v_t

    Complement-aware variant:
        v_t = mu * v_{t-1} + g_t
        compute g_perp = g_t - α * v̂_{t-1}     (α = ⟨g_t, v̂_{t-1}⟩)
        θ ← θ - lr * (v_t + γ * g_perp)

    where v̂_{t-1} is the unit vector of the *previous* velocity. The
    perpendicular-to-previous-momentum component of g_t is added on top of
    the standard momentum step, scaled by perp_weight γ.

    Rationale: momentum's EMA dilutes new information that's orthogonal to
    its current direction. Explicitly amplifying that orthogonal component
    targets exactly the dimension momentum is structurally blind to.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.05,
        momentum: float = 0.9,
        perp_weight: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, perp_weight=perp_weight, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            gamma = group["perp_weight"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                v = state["velocity"]

                # Snapshot v_{t-1} BEFORE the update.
                v_prev_flat = v.view(-1).clone()
                v_prev_norm = v_prev_flat.norm()

                # Update velocity: v_t = mu * v_{t-1} + g_t
                v.mul_(mu).add_(grad)

                # Compute g_perp w.r.t. v_{t-1}.
                g_flat = grad.view(-1)
                if v_prev_norm.item() > _EPS:
                    v_hat = v_prev_flat / v_prev_norm
                    alpha = torch.dot(g_flat, v_hat)
                    g_perp_flat = g_flat - alpha * v_hat
                    g_perp = g_perp_flat.view_as(grad)
                else:
                    # First step (or v_prev is zero): no momentum to complement;
                    # set g_perp = 0 to avoid double-counting g_t.
                    g_perp = torch.zeros_like(grad)

                # Step: -lr * (v_t + gamma * g_perp)
                step_dir = v + gamma * g_perp
                p.add_(step_dir, alpha=-lr)

        return loss


# ============================================================================
# Variant 2b — Complement-aware Adam
# ============================================================================
class ComplementAdam(Optimizer):
    """Adam with a perpendicular-to-momentum boost added to the effective gradient.

    Standard Adam:
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        u_t = m̂_t / (√v̂_t + ε)
        θ ← θ - lr · u_t

    Complement variant (Fix A — decompose in RAW gradient space):
        compute m_t, v_t, m̂_t, v̂_t as usual
        m̂_unit = m̂_t / ‖m̂_t‖                              (unit vector in raw-grad space)
        α       = ⟨g_t, m̂_unit⟩
        g_perp  = g_t - α · m̂_unit                          (RAW-space decomposition)
        step    = (m̂_t + γ · g_perp) / (√v̂_t + ε)           (combine then precondition once)
        θ ← θ - lr · step

    Rationale for the fix
    ---------------------
    A previous version decomposed `g` against `u = m̂/√v̂` (the preconditioned
    step direction). Because `√v̂` rescales coordinates differently, `u` lives
    in a geometrically different space from `g`, and projecting `g` onto a
    `û` that is dominated by small-`v̂` coordinates produced a `g_perp` whose
    re-preconditioning blew up exactly those coordinates. Training diverged
    even at small γ.

    Fix A keeps the decomposition entirely in raw-gradient space — the same
    space `m̂` lives in — so the projection has a consistent metric. The
    combined `(m̂ + γ·g_perp)` is then preconditioned once at the end through
    Adam's standard `√v̂` denominator. Geometrically equivalent to "add the
    perpendicular complement to the smoothed gradient before Adam's scaling."

    With this formulation, the warmup_steps safeguard is no longer required
    (default reduced to 0). At step 1, `m̂ ≈ g`, so `g_perp ≈ 0` automatically.

    All operations are per-parameter-tensor.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        perp_weight: float = 1.0,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        """warmup_steps: optional safeguard. The Fix-A formulation is stable
        from step 1, so default is 0. Kept as a knob for ablation studies."""
        defaults = dict(lr=lr, betas=betas, eps=eps, perp_weight=perp_weight,
                        weight_decay=weight_decay, warmup_steps=int(warmup_steps))
        super().__init__(params, defaults)

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
            gamma = group["perp_weight"]
            wd = group["weight_decay"]
            warmup = group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2
                denom = v_hat.sqrt().add_(eps)

                if t <= warmup or gamma == 0.0:
                    # Pure Adam during warmup or when complement is disabled.
                    p.addcdiv_(m_hat, denom, value=-lr)
                    continue

                # Fix A: decompose g_t against m̂_t in RAW gradient space.
                g_flat = grad.view(-1)
                m_hat_flat = m_hat.view(-1)
                m_hat_norm = m_hat_flat.norm()

                if m_hat_norm.item() > _EPS:
                    m_hat_unit = m_hat_flat / m_hat_norm
                    alpha = torch.dot(g_flat, m_hat_unit)
                    g_perp_flat = g_flat - alpha * m_hat_unit
                    g_perp = g_perp_flat.view_as(grad)
                else:
                    g_perp = torch.zeros_like(grad)

                # Combine in raw grad space, precondition once through √v̂.
                # step = (m̂ + γ · g_perp) / (√v̂ + ε)
                combined = m_hat + gamma * g_perp
                p.addcdiv_(combined, denom, value=-lr)

        return loss


# ============================================================================
# Variant 3 — Adaptive triggering
# ============================================================================
class AdaptiveTriggerBoGrad(Optimizer):
    """Wrap a base optimiser. Apply BoGrad-style projection ONLY when the
    current gradient meaningfully fights an EMA proxy of recent gradient
    direction.

    Trigger condition (per parameter tensor):
        cos(g_t, ema(g)_t) < trigger_threshold   (typically -0.3)

    When the trigger fires, project g_t against the buffer of recent g's
    (asymmetric / "negative" mode by default) and forward the projected grad
    to the base optimiser. Otherwise the base optimiser sees raw g_t.

    The EMA proxy is maintained internally so this works regardless of
    whether the base optimiser has its own momentum.
    """

    def __init__(
        self,
        params,
        base_optimizer_cls: Type[Optimizer],
        *,
        buffer_size: int = 8,
        projection_mode: str = "negative",
        trigger_threshold: float = -0.3,
        ema_beta: float = 0.9,
        store_normalised: bool = True,
        projection_dtype: torch.dtype = torch.float32,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(params, defaults={})
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)

        self.buffer_size = int(buffer_size)
        self.projection_mode = projection_mode
        self.trigger_threshold = float(trigger_threshold)
        self.ema_beta = float(ema_beta)
        self.store_normalised = bool(store_normalised)
        self.projection_dtype = projection_dtype

        self.trigger_count = 0
        self.step_count = 0

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                self.state[p]["buffer"] = []
                self.state[p]["ema"] = None  # 1-D tensor, lazily inited

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.is_sparse:
                    continue

                state = self.state[p]
                g_flat = p.grad.detach().view(-1).to(dtype=self.projection_dtype)

                ema = state["ema"]
                if ema is None:
                    ema = g_flat.clone()
                else:
                    ema.mul_(self.ema_beta).add_(g_flat, alpha=1.0 - self.ema_beta)
                state["ema"] = ema

                self.step_count += 1
                projection_applied = False

                if state["buffer"]:
                    g_norm = g_flat.norm()
                    ema_norm = ema.norm()
                    if g_norm.item() > _EPS and ema_norm.item() > _EPS:
                        cos = (torch.dot(g_flat, ema) / (g_norm * ema_norm)).item()
                        if cos < self.trigger_threshold:
                            projected = _project_sequential(
                                g_flat, state["buffer"],
                                projection_mode=self.projection_mode,
                                store_normalised=self.store_normalised,
                            )
                            p.grad.copy_(projected.view_as(p.grad).to(dtype=p.grad.dtype))
                            self.trigger_count += 1
                            projection_applied = True

                _append_buffer(
                    state["buffer"], g_flat, self.buffer_size,
                    store_normalised=self.store_normalised,
                    dtype=self.projection_dtype,
                )

        self.base_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def trigger_rate(self) -> float:
        return self.trigger_count / max(self.step_count, 1)


# ============================================================================
# Variant 4 — Gradient-difference buffer
# ============================================================================
class GradientDifferenceBoGrad(Optimizer):
    """Wrap a base optimiser. Buffer holds (g_t - g_{t-1}) — gradient *changes*
    rather than gradients themselves. Project current g_t against directions
    of recent change before passing to the base optimiser.

    Rationale: if successive gradients change rapidly along some direction,
    that direction is high-curvature; the loss is "ringing" along it. Removing
    the component of g_t along recent change directions damps the ringing
    without touching coherent descent. A poor man's curvature-aware step.

    Asymmetric "negative" mode is recommended (and default) — only remove
    when current gradient is anti-aligned with recent change, which is the
    signature of an actively destructive ring.
    """

    def __init__(
        self,
        params,
        base_optimizer_cls: Type[Optimizer],
        *,
        buffer_size: int = 8,
        projection_mode: str = "negative",
        store_normalised: bool = True,
        projection_dtype: torch.dtype = torch.float32,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(params, defaults={})
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)

        self.buffer_size = int(buffer_size)
        self.projection_mode = projection_mode
        self.store_normalised = bool(store_normalised)
        self.projection_dtype = projection_dtype

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                self.state[p]["buffer"] = []        # buffer of (g_t - g_{t-1})
                self.state[p]["prev_grad"] = None    # 1-D tensor

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.is_sparse:
                    continue
                state = self.state[p]
                g_flat = p.grad.detach().view(-1).to(dtype=self.projection_dtype)

                # Project against accumulated gradient-difference buffer.
                if state["buffer"]:
                    projected = _project_sequential(
                        g_flat, state["buffer"],
                        projection_mode=self.projection_mode,
                        store_normalised=self.store_normalised,
                    )
                    p.grad.copy_(projected.view_as(p.grad).to(dtype=p.grad.dtype))

                # Update buffer with the (raw, pre-projection) gradient difference.
                prev = state["prev_grad"]
                if prev is not None and prev.numel() == g_flat.numel():
                    diff = g_flat - prev
                    _append_buffer(
                        state["buffer"], diff, self.buffer_size,
                        store_normalised=self.store_normalised,
                        dtype=self.projection_dtype,
                    )
                state["prev_grad"] = g_flat.clone()

        self.base_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


# ============================================================================
# Variant 5 — Multi-scale buffer
# ============================================================================
class MultiScaleBoGrad(Optimizer):
    """Wrap a base optimiser. Two buffers per parameter:
      - short window (K_short=4) for immediate mini-batch interference,
      - long  window (K_long=32) for landscape-level oscillation.

    Both apply asymmetric projection. Short-buffer projection runs first
    (cleans the immediate noise), then long-buffer projection runs on the
    result (handles the longer-horizon signal).

    Insertion: every step appends the *raw* (pre-projection) gradient to
    BOTH buffers, so each buffer maintains an independent history.
    """

    def __init__(
        self,
        params,
        base_optimizer_cls: Type[Optimizer],
        *,
        short_buffer_size: int = 4,
        long_buffer_size: int = 32,
        projection_mode: str = "negative",
        store_normalised: bool = True,
        projection_dtype: torch.dtype = torch.float32,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(params, defaults={})
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)

        self.short_buffer_size = int(short_buffer_size)
        self.long_buffer_size = int(long_buffer_size)
        self.projection_mode = projection_mode
        self.store_normalised = bool(store_normalised)
        self.projection_dtype = projection_dtype

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                self.state[p]["short"] = []
                self.state[p]["long"] = []

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.is_sparse:
                    continue
                state = self.state[p]
                g_flat = p.grad.detach().view(-1).to(dtype=self.projection_dtype)
                raw = g_flat.clone()

                # Short-buffer pass.
                if state["short"]:
                    g_flat = _project_sequential(
                        g_flat, state["short"],
                        projection_mode=self.projection_mode,
                        store_normalised=self.store_normalised,
                    )

                # Long-buffer pass.
                if state["long"]:
                    g_flat = _project_sequential(
                        g_flat, state["long"],
                        projection_mode=self.projection_mode,
                        store_normalised=self.store_normalised,
                    )

                p.grad.copy_(g_flat.view_as(p.grad).to(dtype=p.grad.dtype))

                # Append raw (pre-projection) g to both buffers.
                _append_buffer(
                    state["short"], raw, self.short_buffer_size,
                    store_normalised=self.store_normalised,
                    dtype=self.projection_dtype,
                )
                _append_buffer(
                    state["long"], raw, self.long_buffer_size,
                    store_normalised=self.store_normalised,
                    dtype=self.projection_dtype,
                )

        self.base_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


# ============================================================================
# Plain SignSGD (with optional momentum) — baseline for the SignSGD family
# ============================================================================
class SignSGD(Optimizer):
    """SignSGD with optional momentum. Step is -lr * sign(m_t) (or sign(g) if no momentum)."""

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
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
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                if mu > 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    m = state["momentum_buffer"]
                    m.mul_(mu).add_(grad)
                    p.add_(m.sign(), alpha=-lr)
                else:
                    p.add_(grad.sign(), alpha=-lr)

        return loss


# ============================================================================
# Variant 2c — Complement-aware RMSprop
# ============================================================================
class ComplementRMSprop(Optimizer):
    """RMSprop with momentum + perpendicular boost on the preconditioned gradient.

    Standard RMSprop with momentum (PyTorch convention):
        v_t = α · v_{t-1} + (1 - α) · g_t²
        p_t = g_t / (√v_t + ε)         (preconditioned gradient)
        m_t = μ · m_{t-1} + p_t
        step = -lr · m_t

    Complement variant:
        v_t = α · v_{t-1} + (1 - α) · g_t²
        p_t = g_t / (√v_t + ε)
        decompose p_t w.r.t. m_{t-1}:
            β = ⟨p_t, m̂_{t-1}⟩
            p_perp = p_t - β · m̂_{t-1}
        m_t = μ · m_{t-1} + p_t
        step = -lr · (m_t + γ · p_perp)

    Decomposition is in *preconditioned-gradient space* because that's the
    space the momentum buffer m lives in. The perpendicular boost adds
    information from p_t that wasn't already in the running m direction.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.9,
        perp_weight: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        if momentum <= 0:
            raise ValueError("ComplementRMSprop requires momentum > 0 (need m_{t-1} to complement against)")
        defaults = dict(lr=lr, alpha=alpha, eps=eps, momentum=momentum,
                        perp_weight=perp_weight, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            a = group["alpha"]
            eps = group["eps"]
            mu = group["momentum"]
            gamma = group["perp_weight"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                v = state["square_avg"]
                m = state["momentum_buffer"]

                # v_t = α v_{t-1} + (1 - α) g²
                v.mul_(a).addcmul_(grad, grad, value=1.0 - a)
                precond = grad / (v.sqrt() + eps)

                # Snapshot m_{t-1} BEFORE updating m.
                m_prev_flat = m.view(-1).clone()
                m_prev_norm = m_prev_flat.norm()

                # Perpendicular component of precond w.r.t. m_{t-1}.
                if m_prev_norm.item() > _EPS:
                    m_hat = m_prev_flat / m_prev_norm
                    p_flat = precond.view(-1)
                    beta = torch.dot(p_flat, m_hat)
                    p_perp_flat = p_flat - beta * m_hat
                    p_perp = p_perp_flat.view_as(precond)
                else:
                    p_perp = torch.zeros_like(precond)

                # Update momentum buffer with raw preconditioned gradient.
                m.mul_(mu).add_(precond)

                step_dir = m + gamma * p_perp
                p.add_(step_dir, alpha=-lr)

        return loss


# ============================================================================
# Variant 2d — Complement-aware SignSGD
# ============================================================================
class ComplementSignSGD(Optimizer):
    """SignSGD with momentum + perpendicular boost on the pre-sign direction.

    Standard SignSGD with momentum:
        m_t = β · m_{t-1} + g_t
        step = -lr · sign(m_t)

    Complement variant:
        decompose g_t w.r.t. m_{t-1}:
            α = ⟨g_t, m̂_{t-1}⟩
            g_perp = g_t - α · m̂_{t-1}
        m_t = β · m_{t-1} + g_t
        step = -lr · sign(m_t + γ · g_perp)

    The perpendicular boost shifts the sign pattern toward coordinates where
    the new orthogonal information matters. Magnitude per coordinate stays
    exactly lr (preserving SignSGD's defining property).
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        momentum: float = 0.9,
        perp_weight: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        if momentum <= 0:
            raise ValueError("ComplementSignSGD requires momentum > 0 (need m_{t-1} to complement against)")
        defaults = dict(lr=lr, momentum=momentum, perp_weight=perp_weight, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            gamma = group["perp_weight"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m = state["momentum_buffer"]

                # Snapshot m_{t-1}.
                m_prev_flat = m.view(-1).clone()
                m_prev_norm = m_prev_flat.norm()

                # Perpendicular component of g w.r.t. m_{t-1}.
                if m_prev_norm.item() > _EPS:
                    m_hat = m_prev_flat / m_prev_norm
                    g_flat = grad.view(-1)
                    alpha = torch.dot(g_flat, m_hat)
                    g_perp_flat = g_flat - alpha * m_hat
                    g_perp = g_perp_flat.view_as(grad)
                else:
                    g_perp = torch.zeros_like(grad)

                # Update m_t.
                m.mul_(mu).add_(grad)

                # step = -lr · sign(m_t + γ · g_perp)
                p.add_((m + gamma * g_perp).sign(), alpha=-lr)

        return loss


# ============================================================================
# In-pipeline RMSprop — project the PRECONDITIONED gradient before momentum
# ============================================================================
class InPipelineRMSprop(Optimizer):
    """RMSprop with BoGrad-style projection of the preconditioned gradient,
    inserted between preconditioning and the momentum buffer update.

    Standard RMSprop with momentum:
        v_t  = α · v_{t-1} + (1 - α) · g_t²
        p_t  = g_t / (√v_t + ε)             (preconditioned gradient)
        m_t  = μ · m_{t-1} + p_t
        step = -lr · m_t

    In-pipeline variant:
        v_t  = α · v_{t-1} + (1 - α) · g_t²
        p_t  = g_t / (√v_t + ε)
        p̃_t  = project(p_t, buffer of past p_t's)        (asymmetric)
        m_t  = μ · m_{t-1} + p̃_t                          (momentum eats projected)
        step = -lr · m_t
        buffer.append(p_t)                                 (raw, not projected)

    Why: the preconditioned gradient is the "elementary direction" that
    RMSprop's momentum accumulates. Projecting at this stage removes
    short-horizon redundancy BEFORE momentum smooths it in — the momentum
    buffer never sees the redundant components. Symmetric to projecting Adam's
    m_t between EMA and preconditioning.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        buffer_size: int = 8,
        projection_mode: str = "negative",
        store_normalised: bool = True,
        projection_dtype: torch.dtype = torch.float32,
    ) -> None:
        if momentum <= 0:
            raise ValueError("InPipelineRMSprop requires momentum > 0")
        defaults = dict(lr=lr, alpha=alpha, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.buffer_size = int(buffer_size)
        self.projection_mode = projection_mode
        self.store_normalised = bool(store_normalised)
        self.projection_dtype = projection_dtype

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                self.state[p]["buffer"] = []

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            a = group["alpha"]
            eps = group["eps"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "square_avg" not in state:
                    state["square_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                v = state["square_avg"]
                m = state["momentum_buffer"]
                buf = state["buffer"]

                # v_t update.
                v.mul_(a).addcmul_(grad, grad, value=1.0 - a)
                precond = grad / (v.sqrt() + eps)

                # Project preconditioned gradient against buffer (if applicable).
                if self.buffer_size > 0 and buf:
                    p_flat = precond.view(-1).to(dtype=self.projection_dtype)
                    p_proj_flat = _project_sequential(
                        p_flat, buf,
                        projection_mode=self.projection_mode,
                        store_normalised=self.store_normalised,
                    )
                    p_for_momentum = p_proj_flat.view_as(precond).to(dtype=precond.dtype)
                else:
                    p_for_momentum = precond

                # Momentum buffer accumulates the projected preconditioned grad.
                m.mul_(mu).add_(p_for_momentum)

                # Apply step.
                p.add_(m, alpha=-lr)

                # Buffer the RAW preconditioned grad (not projected).
                if self.buffer_size > 0:
                    raw_flat = precond.view(-1).to(dtype=self.projection_dtype)
                    _append_buffer(
                        buf, raw_flat, self.buffer_size,
                        store_normalised=self.store_normalised,
                        dtype=self.projection_dtype,
                    )

        return loss


# ============================================================================
# In-pipeline SignSGD — project m_t BEFORE the sign() nonlinearity
# ============================================================================
class InPipelineSignSGD(Optimizer):
    """SignSGD with BoGrad-style projection of m_t inserted before sign().

    Standard SignSGD with momentum:
        m_t  = μ · m_{t-1} + g_t
        step = -lr · sign(m_t)

    In-pipeline variant:
        m_t  = μ · m_{t-1} + g_t
        m̃_t  = project(m_t, buffer of past m_t's)    (asymmetric)
        step = -lr · sign(m̃_t)
        buffer.append(m_t)                            (raw, not projected)

    Why: the pre-sign m_t is a real-valued vector with rich geometry; sign()
    discards magnitude. Projecting in m_t's space (where Gram-Schmidt makes
    sense) before sign() lets us influence WHICH coordinates flip without
    losing the per-coord ±lr step magnitude property.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        buffer_size: int = 8,
        projection_mode: str = "negative",
        store_normalised: bool = True,
        projection_dtype: torch.dtype = torch.float32,
    ) -> None:
        if momentum <= 0:
            raise ValueError("InPipelineSignSGD requires momentum > 0")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.buffer_size = int(buffer_size)
        self.projection_mode = projection_mode
        self.store_normalised = bool(store_normalised)
        self.projection_dtype = projection_dtype

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                self.state[p]["buffer"] = []

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m = state["momentum_buffer"]
                buf = state["buffer"]

                # m_t = μ·m + g.
                m.mul_(mu).add_(grad)

                # Project m_t against buffer (if applicable).
                if self.buffer_size > 0 and buf:
                    m_flat = m.view(-1).to(dtype=self.projection_dtype)
                    m_proj_flat = _project_sequential(
                        m_flat, buf,
                        projection_mode=self.projection_mode,
                        store_normalised=self.store_normalised,
                    )
                    m_for_sign = m_proj_flat.view_as(m).to(dtype=m.dtype)
                else:
                    m_for_sign = m

                # Apply sign-step using the projected m.
                p.add_(m_for_sign.sign(), alpha=-lr)

                # Buffer the RAW m_t (not projected).
                if self.buffer_size > 0:
                    raw_flat = m.view(-1).to(dtype=self.projection_dtype)
                    _append_buffer(
                        buf, raw_flat, self.buffer_size,
                        store_normalised=self.store_normalised,
                        dtype=self.projection_dtype,
                    )

        return loss


__all__ = [
    "ComplementMomentumSGD",
    "ComplementAdam",
    "ComplementRMSprop",
    "ComplementSignSGD",
    "SignSGD",
    "AdaptiveTriggerBoGrad",
    "GradientDifferenceBoGrad",
    "MultiScaleBoGrad",
    "InPipelineRMSprop",
    "InPipelineSignSGD",
]
