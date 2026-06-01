"""
WindowedMeanBoGrad — orthogonalise the applied parameter delta against the
arithmetic mean of the last K applied deltas (a single reference direction),
rather than against the K-dim buffer subspace as BoGrad does.

Algorithm (per parameter tensor, update-stage):

    snapshot params before base step
    base_optimizer.step()                       # raw step
    delta = params_after - params_before        # applied update
    if buffer non-empty:
        mean = (1/|buf|) * sum(buf)             # rolling arithmetic mean
        if ||mean|| > eps:
            dot = <delta, mean>
            if mode == "negative" and dot >= 0:  skip
            elif mode == "positive" and dot <= 0: skip
            else:
                delta <- delta - (dot / ||mean||^2) * mean
    params_after <- params_before + delta
    buffer.append(delta_pre_projection)         # FIFO, max len = K

Same shape as BoGrad's update stage, but projection target is the *mean
direction* of the buffer rather than its full subspace. Memory is the same
(O(K)) since we keep the buffer. Compute is K× cheaper per step (one dot
product against the mean instead of K dot products against each entry).

Intended as a falsifiable rank-1 reduction of BoGrad to test the hypothesis:
"BoGrad's value comes from variance across past steps, not from any single
average direction." If WindowedMeanBoGrad performs nearly as well as BoGrad,
the K-dim subspace is overkill and the mean direction suffices. If it's
strictly worse, BoGrad's strength comes from per-step conflict detection.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable, List, Optional, Type

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


_VALID_MODES = ("full", "negative", "positive")


class WindowedMeanBoGrad(Optimizer):
    """Wrap any base PyTorch optimiser. After the base step, project the
    applied parameter delta against the arithmetic mean of the last K applied
    deltas (rank-1 reference)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        base_optimizer_cls: Type[Optimizer],
        *,
        buffer_size: int = 32,
        projection_mode: str = "negative",
        projection_dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
        **base_optimizer_kwargs: Any,
    ) -> None:
        if buffer_size < 0:
            raise ValueError("buffer_size must be >= 0")
        if projection_mode not in _VALID_MODES:
            raise ValueError(f"projection_mode must be one of {_VALID_MODES}")

        super().__init__(params, defaults={})
        self.base_optimizer: Optimizer = base_optimizer_cls(
            self.param_groups, **base_optimizer_kwargs
        )

        self.buffer_size = int(buffer_size)
        self.projection_mode = projection_mode
        self.projection_dtype = projection_dtype
        self.eps = float(eps)

        for group in self.param_groups:
            for p in group["params"]:
                self.state.setdefault(p, {})
                # FIFO ring buffer of recent flat deltas, max len K.
                self.state[p]["buffer"] = deque(maxlen=max(self.buffer_size, 1))

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.buffer_size == 0:
            self.base_optimizer.step()
            return loss

        # Snapshot params and collect the param list (skip sparse / no-grad).
        params_before = {}
        param_list: List[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    continue
                params_before[id(p)] = p.data.detach().clone()
                param_list.append(p)

        # Run the base optimiser's step (uses the raw gradient).
        self.base_optimizer.step()

        # For each tracked parameter, project the applied delta against the
        # rolling-window mean of past deltas, then rewrite the parameter.
        for p in param_list:
            before = params_before[id(p)]
            raw_delta = (p.data - before).view(-1).to(dtype=self.projection_dtype)
            delta = raw_delta.clone()
            buf: deque = self.state[p]["buffer"]

            if len(buf) > 0:
                mean = torch.stack(list(buf)).mean(dim=0)
                mean_norm_sq = float(torch.dot(mean, mean).item())
                if mean_norm_sq > self.eps:
                    dot_val = float(torch.dot(delta, mean).item())
                    skip = (
                        (self.projection_mode == "negative" and dot_val >= 0)
                        or (self.projection_mode == "positive" and dot_val <= 0)
                    )
                    if not skip:
                        coeff = dot_val / mean_norm_sq
                        delta = delta - coeff * mean

            # Append RAW delta (pre-projection) — matches BoGrad convention.
            buf.append(raw_delta.clone())

            # Rewrite parameter with the (possibly-projected) delta.
            p.data.copy_(before + delta.view_as(p.data).to(dtype=p.dtype))

        return loss

    def reset_buffers(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "buffer" in self.state[p]:
                    self.state[p]["buffer"].clear()
