"""
BoGrad: Batch Orthogonalised Gradient wrapper for PyTorch optimisers.

Production version (used by `testing/` and `experiments/`). The legacy /
reference implementation lives in BoGradScrutinized.py.

The core projection algorithm is sequential Gram-Schmidt subtraction (the
IJCNN-era recipe). This file adds, on top of the scrutinised reference:
  - `preserve_magnitude` flag — rescale projected vector so ‖g̃‖ = ‖g‖
    (cleanly separates direction effect from magnitude effect; see testing/01)
  - `random_projection` flag — replace buffer with random unit vectors at
    projection time (control for Test 1 — distinguishes BoGrad's specific
    projection direction from "any projection that shrinks magnitude")
  - `max_rescale` — clip on the rescale factor when preserve_magnitude is on,
    to handle the degenerate case where ‖g̃‖ → 0.

Important empirical finding
---------------------------
The original BOSGD used sequential subtraction against a non-orthogonalised
buffer (i.e. not a true orthogonal projection onto span(B)^⊥). An earlier
revision of this file replaced that with a QR-based true orthogonal projection.
Training results were consistently worse under QR.

Reason: K recent gradients are strongly correlated, so span(B) covers most of
the useful descent direction. Truly projecting orthogonal to it strips out too
much signal. The original sequential method removes only "some" of each
component at a time and is empirically the right amount of correction.

Takeaway:
- Default orth_method is "sequential" (matches the IJCNN paper's working
  algorithm). Use this unless you have a reason not to.
- orth_method="qr" is retained for Experiment 4.3 (method comparison),
  NOT as a production default.
- projection_mode="negative" is softer still (removes only destructive
  components) and is what the IJCNN CIFAR results used. If you're looking for
  "beat the baseline" settings, start there.

What changed vs. the original BOSGD
-----------------------------------

1. Update-stage projection (project_stage="update"). Measure the actual applied
   parameter delta after the base optimiser's step, project *that* against past
   deltas, and rewrite the parameter. Principled for preconditioned optimisers
   (Adam/AdamW) and for SGD+momentum where the momentum buffer already smooths
   raw gradients.

2. orth_method option. "sequential" (default, original BOSGD behaviour),
   "qr" (true orthogonal projection, provided for Exp 4.3 comparison),
   "householder" (reserved — not yet implemented; see TODO).

3. projection_scope option. "per_tensor" (default, original behaviour) or
   "global" (one flat buffer for all parameters together).

4. Logging hooks. Per-step diagnostics (g_norm, g_tilde_norm, g_ratio,
   cos_prev_direction, fraction_removed) accumulated internally and exposed via
   get_last_step_stats(). Experiments 4.1, 4.2, 4.4, 5.8 can log without
   monkey-patching.

5. Memory options. buffer_dtype can be fp16/bf16 on large models.
   min_projection_dim optionally skips tiny tensors.

6. Serialisation. state_dict()/load_state_dict() preserve both BoGrad and base
   optimiser state explicitly; buffer stored as list (round-trips cleanly).

7. Clean separation of collect_stats cost (stats computations gated behind the
   flag so "off" is truly zero-cost).

Known limitations
-----------------
- In update-stage mode with a momentum-based base optimiser, the base's
  internal state (m_t, v_t, velocity buffer) updates based on gradients at
  parameters we partially roll back. This is benign in practice because the
  next gradient is evaluated at the actual (BoGrad-modified) location, but
  very long runs may drift.
- Sparse gradients are skipped (one warning). Embedding-heavy models
  (DBpedia) will have BoGrad only act on the non-sparse parts.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class BoGrad(Optimizer):
    """Batch Orthogonalised Gradient wrapper.

    Parameters
    ----------
    params : iterable of torch.Tensor
        Parameters to optimise.
    base_optimizer_cls : Type[Optimizer]
        Base optimiser class (e.g. torch.optim.SGD, torch.optim.Adam).
    buffer_size : int, default 8
        Maximum number of past directions (K) to maintain.
    project_stage : {"gradient", "update"}, default "gradient"
        - "gradient": project p.grad before base_optimizer.step(). The original
          BOSGD behaviour. Recommended for plain SGD, SignSGD.
        - "update": run base_optimizer.step(), project the applied parameter
          delta, rewrite the parameter. Recommended for Adam/AdamW and for
          SGD+momentum where the gradient-stage projection fights the momentum
          buffer.
    projection_mode : {"full", "negative", "positive"}, default "full"
        - "full": subtract the full projection component for each buffered
          direction (removes both destructive and redundant overlap).
        - "negative": subtract only when the dot product is negative
          (destructive interference). Softer; this is what IJCNN CIFAR-10
          results used — recommended if you want to reproduce the paper gaps.
        - "positive": subtract only when the dot product is positive
          (redundant / "stale-direction" overlap). The complementary mode
          to "negative" — removes the part of the current step that
          re-applies what recent steps already did, while preserving any
          destructive components. Useful for ablating which kind of
          interference (destructive vs redundant) actually slows training.
    orth_method : {"sequential", "qr"}, default "sequential"
        - "sequential": iteratively subtract ⟨x, b_i⟩ b_i for each b_i in the
          buffer. NOT a true orthogonal projection onto span(B)^⊥ (since the
          buffer isn't mutually orthogonalised), but empirically this is the
          *right* amount of correction — IJCNN's working algorithm. USE THIS.
        - "qr": stack the buffer, QR-decompose, project onto span(B)^⊥.
          Mathematically cleaner; empirically more aggressive and tends to
          hurt training. Provided for Experiment 4.3 comparisons only.
    projection_scope : {"per_tensor", "global"}, default "per_tensor"
        Where the buffer lives and what it spans.
    store_normalised : bool, default True
        Store unit-norm vectors in the buffer.
    projection_dtype : torch.dtype, default torch.float32
        Dtype used for projection arithmetic.
    buffer_dtype : torch.dtype or None, default None
        Dtype for buffered vectors. None → match projection_dtype.
    min_projection_dim : int, default 0
        Per-tensor projection is skipped on tensors with numel() < this. Only
        relevant for orth_method="qr" on small biases/BN params where the
        QR-based projection can collapse a tiny tensor into a zero subspace.
        For "sequential" (default) it's safe to leave at 0.
    eps : float, default 1e-12
        Numerical floor.
    collect_stats : bool, default True
        Compute per-step diagnostics for get_last_step_stats(). Set False for
        max throughput (skips norm/cosine computations entirely).
    warn_on_sparse : bool, default True
        Emit a single warning if a sparse gradient is encountered.
    preserve_magnitude : bool, default False
        After projection, rescale `g̃` so that ‖g̃‖ = ‖g‖. Decouples direction
        change from magnitude change. Off by default; enable for testing/02.
    max_rescale : float or None, default None
        When preserve_magnitude is on, clip the rescale factor at this value.
        Protects against ‖g̃‖ → 0 cases where rescaling would explode.
    random_projection : bool, default False
        At projection time, replace the buffer with `buffer_size` random
        unit-norm vectors. Control for testing/01 — distinguishes BoGrad's
        specific projection direction from "any magnitude-reducing perturbation".
    projection_strength : float, default 1.0
        Partial-projection coefficient α ∈ [0, 1]. The returned vector is
        `x + α·(proj − x)` = `(1−α)·x + α·proj`, so α=1 is the full projection
        (default, original behaviour) and α=0 is the unmodified input (baseline).
        Lets the ablation sweep the baseline↔projection continuum with a single
        knob rather than only the discrete modes. Applied after the orth step
        and before `preserve_magnitude`.
    **base_optimizer_kwargs
        Forwarded to base_optimizer_cls.
    """

    _VALID_STAGES = ("gradient", "update")
    _VALID_MODES = ("full", "negative", "positive")
    _VALID_METHODS = ("sequential", "qr")
    _VALID_SCOPES = ("per_tensor", "global")

    def __init__(
        self,
        params: Iterable[Tensor],
        base_optimizer_cls: Type[Optimizer],
        *,
        buffer_size: int = 8,
        project_stage: str = "gradient",
        projection_mode: str = "full",
        orth_method: str = "sequential",
        projection_scope: str = "per_tensor",
        store_normalised: bool = True,
        projection_dtype: torch.dtype = torch.float32,
        buffer_dtype: Optional[torch.dtype] = None,
        min_projection_dim: int = 0,
        eps: float = 1e-12,
        collect_stats: bool = True,
        warn_on_sparse: bool = True,
        preserve_magnitude: bool = False,
        max_rescale: Optional[float] = None,
        random_projection: bool = False,
        projection_strength: float = 1.0,
        **base_optimizer_kwargs: Any,
    ) -> None:
        if buffer_size < 0:
            raise ValueError("buffer_size must be >= 0")
        if not (0.0 <= projection_strength <= 1.0):
            raise ValueError("projection_strength must be in [0, 1]")
        if project_stage not in self._VALID_STAGES:
            raise ValueError(f"project_stage must be one of {self._VALID_STAGES}")
        if projection_mode not in self._VALID_MODES:
            raise ValueError(f"projection_mode must be one of {self._VALID_MODES}")
        if orth_method not in self._VALID_METHODS:
            raise ValueError(f"orth_method must be one of {self._VALID_METHODS}")
        if projection_scope not in self._VALID_SCOPES:
            raise ValueError(f"projection_scope must be one of {self._VALID_SCOPES}")

        super().__init__(params, defaults={})
        self.base_optimizer: Optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_kwargs)

        self.buffer_size = int(buffer_size)
        self.project_stage = project_stage
        self.projection_mode = projection_mode
        self.orth_method = orth_method
        self.projection_scope = projection_scope
        self.store_normalised = bool(store_normalised)
        self.projection_dtype = projection_dtype
        self.buffer_dtype = buffer_dtype if buffer_dtype is not None else projection_dtype
        self.min_projection_dim = int(min_projection_dim)
        self.eps = float(eps)
        self.collect_stats = bool(collect_stats)
        self.warn_on_sparse = bool(warn_on_sparse)
        self.preserve_magnitude = bool(preserve_magnitude)
        self.max_rescale = float(max_rescale) if max_rescale is not None else None
        self.random_projection = bool(random_projection)
        self.projection_strength = float(projection_strength)

        self._sparse_warned = False

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                self.state[p]["buffer"] = []            # List[Tensor]
                self.state[p]["prev_projected"] = None  # 1D tensor or None

        self._global_buffer: List[Tensor] = []
        self._global_prev_projected: Optional[Tensor] = None

        self._last_step_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Optimizer API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.buffer_size == 0:
            self.base_optimizer.step()
            self._last_step_stats = {}
            return loss

        if self.project_stage == "gradient":
            self._gradient_stage()
            self.base_optimizer.step()
        else:
            self._update_stage()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------
    def _gradient_stage(self) -> None:
        if self.projection_scope == "per_tensor":
            rows: List[Dict[str, float]] = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        self._warn_sparse()
                        continue
                    if p.numel() < self.min_projection_dim:
                        continue

                    g_flat = p.grad.detach().view(-1).to(dtype=self.projection_dtype)
                    buf = self.state[p]["buffer"]

                    projected, stats = self._project(g_flat, buf)
                    if self.collect_stats:
                        stats.update(self._update_cosine_stats(p, projected))
                        rows.append(stats)

                    p.grad.copy_(projected.view_as(p.grad).to(dtype=p.grad.dtype))
                    self._append(buf, g_flat)

            self._aggregate_stats(rows)
        else:
            grads, params = self._collect_grads_for_global_projection()
            if not grads:
                self._last_step_stats = {}
                return
            g_flat = torch.cat(grads)
            projected, stats = self._project(g_flat, self._global_buffer)
            if self.collect_stats:
                stats.update(self._update_global_cosine_stats(projected))
            self._scatter_back_grads(projected, params)
            self._append(self._global_buffer, g_flat)
            self._aggregate_stats([stats] if self.collect_stats else [])

    def _update_stage(self) -> None:
        params_before: Dict[int, Tensor] = {}
        param_list: List[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    self._warn_sparse()
                    continue
                params_before[id(p)] = p.data.detach().clone()
                param_list.append(p)

        self.base_optimizer.step()

        if self.projection_scope == "per_tensor":
            rows: List[Dict[str, float]] = []
            for p in param_list:
                if p.numel() < self.min_projection_dim:
                    continue
                before = params_before[id(p)]
                delta = (p.data - before).view(-1).to(dtype=self.projection_dtype)
                buf = self.state[p]["buffer"]

                projected, stats = self._project(delta, buf)
                if self.collect_stats:
                    stats.update(self._update_cosine_stats(p, projected))
                    rows.append(stats)

                p.data.copy_((before + projected.view_as(p.data).to(dtype=p.dtype)))
                self._append(buf, delta)

            self._aggregate_stats(rows)
        else:
            deltas: List[Tensor] = []
            shapes: List[torch.Size] = []
            for p in param_list:
                before = params_before[id(p)]
                deltas.append((p.data - before).view(-1).to(dtype=self.projection_dtype))
                shapes.append(p.data.shape)
            if not deltas:
                self._last_step_stats = {}
                return
            delta_flat = torch.cat(deltas)

            projected, stats = self._project(delta_flat, self._global_buffer)
            if self.collect_stats:
                stats.update(self._update_global_cosine_stats(projected))
            self._append(self._global_buffer, delta_flat)

            offset = 0
            for p, shape in zip(param_list, shapes):
                n = p.numel()
                slice_proj = projected[offset:offset + n].view(shape).to(dtype=p.dtype)
                p.data.copy_(params_before[id(p)] + slice_proj)
                offset += n

            self._aggregate_stats([stats] if self.collect_stats else [])

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------
    def _project(self, x_flat: Tensor, buffer: List[Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Project x_flat against the buffer. Returns (projected, stats)."""
        if self.collect_stats:
            x_norm_val = float(x_flat.norm().item())
            stats: Dict[str, float] = {
                "g_norm": x_norm_val,
                "g_tilde_norm": x_norm_val,
                "g_ratio": 1.0,
                "fraction_removed": 0.0,
                "rescale_factor": 1.0,
            }
        else:
            stats = {}
            x_norm_val = None  # type: ignore[assignment]

        # Determine which buffer to use for projection.
        # `random_projection=True` substitutes random unit vectors as a control —
        # tests whether BoGrad's *specific* directions matter, or whether any
        # comparable projection works.
        if self.random_projection and self.buffer_size > 0:
            buffer_for_proj = self._make_random_buffer(x_flat)
        else:
            buffer_for_proj = buffer

        if not buffer_for_proj:
            return x_flat, stats
        if self.collect_stats and x_norm_val is not None and x_norm_val <= self.eps:
            return x_flat, stats

        if self.orth_method == "sequential":
            projected = self._project_sequential(x_flat, buffer_for_proj)
        elif self.orth_method == "qr":
            projected = self._project_qr(x_flat, buffer_for_proj)
        else:
            raise RuntimeError(f"Unknown orth_method {self.orth_method}")

        # Optional partial projection — interpolate baseline (alpha=0) <-> full
        # projection (alpha=1). alpha=1 is a no-op (original behaviour).
        if self.projection_strength != 1.0:
            projected = x_flat + self.projection_strength * (projected - x_flat)

        # Optional magnitude preservation — rescale projected to match input norm.
        if self.preserve_magnitude:
            projected = self._rescale_to_input_norm(projected, x_flat, stats)

        if self.collect_stats:
            tilde_norm = float(projected.norm().item())
            denom = (x_norm_val or 0.0) + self.eps
            stats["g_tilde_norm"] = tilde_norm
            stats["g_ratio"] = tilde_norm / denom
            stats["fraction_removed"] = 1.0 - stats["g_ratio"]
        return projected, stats

    def _make_random_buffer(self, x_flat: Tensor) -> List[Tensor]:
        """Generate a buffer of `buffer_size` random unit vectors matching x_flat's shape.

        Used by random_projection mode as a control. Each call produces fresh
        random vectors so consecutive steps are not correlated (matches the
        spirit of "what if the buffer carried no temporal information at all").
        """
        out: List[Tensor] = []
        for _ in range(self.buffer_size):
            v = torch.randn_like(x_flat, dtype=self.projection_dtype)
            v_norm = v.norm()
            if v_norm.item() <= self.eps:
                continue
            out.append(v / (v_norm + self.eps))
        return out

    def _rescale_to_input_norm(
        self, projected: Tensor, x_flat: Tensor, stats: Dict[str, float],
    ) -> Tensor:
        """Rescale projected vector so its norm matches x_flat's norm.

        If `max_rescale` is set, clip the rescale factor. This protects against
        the degenerate case where ‖g̃‖ → 0 (e.g. the buffer fully spans g),
        which would otherwise produce an unbounded factor.
        """
        x_norm = x_flat.norm()
        proj_norm = projected.norm()
        # GPU-resident decision — avoid syncing.
        safe_proj_norm = torch.clamp(proj_norm, min=self.eps)
        scale = x_norm / safe_proj_norm
        if self.max_rescale is not None:
            scale = torch.clamp(scale, max=self.max_rescale)
        # If projected was effectively zero, scale is large but x_flat ⋅ scale
        # would point in an arbitrary direction; in that case fall back to x_flat.
        if proj_norm.item() <= self.eps:
            if self.collect_stats:
                stats["rescale_factor"] = 1.0
            return x_flat
        if self.collect_stats:
            stats["rescale_factor"] = float(scale.item())
        return projected * scale

    def _project_sequential(self, x_flat: Tensor, buffer: List[Tensor]) -> Tensor:
        """Original BOSGD algorithm: iterative component removal.

        For each b in buffer (FIFO order):
            coeff = <x, b> / <b, b>        (or /1 if buffer stores unit vectors)
            if projection_mode == "negative" and <x, b> >= 0: skip
            x <- x - coeff * b

        NOT a true orthogonal projection onto span(buffer)^⊥ (the buffer isn't
        orthogonalised against itself), but that's the point — empirically this
        "soft" projection preserves more of the descent signal than a true
        orthogonal projection would, and it's what the IJCNN result relied on.

        Note on the `.item()` sign-gate: it forces a GPU→CPU sync per buffered
        direction, which looks wasteful. We benchmarked replacing it with an
        on-GPU multiplicative mask (no sync) and it was ~30% SLOWER on large
        tensors — because in "negative"/"positive" mode the `continue` skips the
        (expensive) AXPY subtraction roughly half the time, whereas a mask must
        run every subtraction unconditionally. The sync is cheaper than the
        redundant arithmetic. So the explicit-skip form below is kept on purpose.
        """
        projected = x_flat.clone()
        for b in buffer:
            b_vec = b.to(device=x_flat.device, dtype=self.projection_dtype)
            if self.store_normalised:
                denom = 1.0
            else:
                bb = float(torch.dot(b_vec, b_vec).item())
                if bb <= self.eps:
                    continue
                denom = bb
            dot_val = torch.dot(projected, b_vec)
            if self.projection_mode == "negative" and dot_val.item() >= 0:
                continue
            if self.projection_mode == "positive" and dot_val.item() <= 0:
                continue
            projected = projected - (dot_val / denom) * b_vec
        return projected

    def _project_qr(self, x_flat: Tensor, buffer: List[Tensor]) -> Tensor:
        """True orthogonal projection onto span(buffer)^⊥ via QR.

        Provided for Experiment 4.3 method comparisons. Empirically more
        aggressive than sequential GS and not recommended as a default.
        """
        B = torch.stack(
            [b.to(device=x_flat.device, dtype=self.projection_dtype) for b in buffer],
            dim=0,
        )
        if self.projection_mode in ("negative", "positive"):
            # QR doesn't naturally support per-component sign-gating — fall
            # back to sequential behaviour under "negative" / "positive".
            # Users who want true orthogonal projection should use "full".
            return self._project_sequential(x_flat, buffer)

        try:
            Q, _ = torch.linalg.qr(B.T, mode="reduced")
        except RuntimeError:
            return x_flat
        coeffs = Q.T @ x_flat
        return x_flat - Q @ coeffs

    def _append(self, buffer: List[Tensor], x_flat: Tensor) -> None:
        x = x_flat.detach()
        x_norm = x.norm()
        if not torch.isfinite(x_norm) or x_norm.item() <= self.eps:
            return
        if self.store_normalised:
            x = x / (x_norm + self.eps)
        x = x.to(dtype=self.buffer_dtype)
        buffer.append(x)
        while len(buffer) > self.buffer_size:
            buffer.pop(0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _update_cosine_stats(self, param: Tensor, projected: Tensor) -> Dict[str, float]:
        prev = self.state[param].get("prev_projected")
        out: Dict[str, float] = {}
        p_norm = projected.norm()
        if prev is not None and p_norm.item() > self.eps:
            cos = torch.dot(projected, prev) / (p_norm * prev.norm() + self.eps)
            out["cos_prev_direction"] = float(cos.item())
        self.state[param]["prev_projected"] = projected.detach().clone()
        return out

    def _update_global_cosine_stats(self, projected: Tensor) -> Dict[str, float]:
        out: Dict[str, float] = {}
        p_norm = projected.norm()
        if self._global_prev_projected is not None and p_norm.item() > self.eps:
            prev = self._global_prev_projected
            cos = torch.dot(projected, prev) / (p_norm * prev.norm() + self.eps)
            out["cos_prev_direction"] = float(cos.item())
        self._global_prev_projected = projected.detach().clone()
        return out

    def _aggregate_stats(self, rows: List[Dict[str, float]]) -> None:
        if not self.collect_stats or not rows:
            self._last_step_stats = {}
            return
        keys: set = set()
        for r in rows:
            keys.update(r.keys())
        agg: Dict[str, float] = {}
        for k in keys:
            vals = [r[k] for r in rows if k in r and math.isfinite(r[k])]
            if vals:
                agg[f"{k}_mean"] = sum(vals) / len(vals)
        agg["num_tensors"] = float(len(rows))
        self._last_step_stats = agg

    def get_last_step_stats(self) -> Dict[str, float]:
        """Return aggregated diagnostics from the most recent step()."""
        return dict(self._last_step_stats)

    # ------------------------------------------------------------------
    # Global-scope helpers
    # ------------------------------------------------------------------
    def _collect_grads_for_global_projection(self) -> Tuple[List[Tensor], List[Tensor]]:
        grads: List[Tensor] = []
        params: List[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    self._warn_sparse()
                    continue
                grads.append(p.grad.detach().view(-1).to(dtype=self.projection_dtype))
                params.append(p)
        return grads, params

    def _scatter_back_grads(self, projected: Tensor, params: List[Tensor]) -> None:
        offset = 0
        for p in params:
            n = p.numel()
            p.grad.copy_(
                projected[offset:offset + n].view_as(p.grad).to(dtype=p.grad.dtype)
            )
            offset += n

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _warn_sparse(self) -> None:
        if self.warn_on_sparse and not self._sparse_warned:
            warnings.warn(
                "BoGrad: encountered sparse gradient; skipping projection for that parameter.",
                RuntimeWarning,
            )
            self._sparse_warned = True

    def reset_buffers(self) -> None:
        """Clear all per-parameter (and global) buffers and cosine memory."""
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["buffer"] = []
                self.state[p]["prev_projected"] = None
        self._global_buffer = []
        self._global_prev_projected = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        config = {
            "buffer_size": self.buffer_size,
            "project_stage": self.project_stage,
            "projection_mode": self.projection_mode,
            "orth_method": self.orth_method,
            "projection_scope": self.projection_scope,
            "store_normalised": self.store_normalised,
            "projection_dtype": str(self.projection_dtype),
            "buffer_dtype": str(self.buffer_dtype),
            "min_projection_dim": self.min_projection_dim,
            "eps": self.eps,
            "collect_stats": self.collect_stats,
            "projection_strength": self.projection_strength,
        }
        global_state = {
            "buffer": [b.detach().cpu() for b in self._global_buffer],
            "prev_projected": (
                self._global_prev_projected.detach().cpu()
                if self._global_prev_projected is not None
                else None
            ),
        }
        return {
            "config": config,
            "bograd_optimizer": super().state_dict(),
            "base_optimizer": self.base_optimizer.state_dict(),
            "global_state": global_state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore[override]
        if "bograd_optimizer" in state_dict and "base_optimizer" in state_dict:
            cfg = state_dict.get("config", {})
            for attr in (
                "buffer_size", "project_stage", "projection_mode", "orth_method",
                "projection_scope", "store_normalised", "min_projection_dim",
                "eps", "collect_stats", "projection_strength",
            ):
                if attr in cfg:
                    setattr(self, attr, cfg[attr])

            super().load_state_dict(state_dict["bograd_optimizer"])
            self.base_optimizer.load_state_dict(state_dict["base_optimizer"])

            gs = state_dict.get("global_state", {})
            self._global_buffer = [b.clone() for b in gs.get("buffer", [])]
            prev = gs.get("prev_projected", None)
            self._global_prev_projected = prev.clone() if prev is not None else None
            return

        try:
            super().load_state_dict(state_dict)
        except Exception:
            self.base_optimizer.load_state_dict(state_dict)


__all__ = ["BoGrad"]
