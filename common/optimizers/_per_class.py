"""
PerClassGradientOptimizer — shared base for optimisers that act on the set of
**per-class subgradients** within a mini-batch (the §03 inter-batch objects).

COSGD and GradDrop are the same machinery with a different *fold* step:

  per-class subgradients {g_c}  ->  _combine({g_c})  ->  one vector  ->  p.grad
                                                                       -> base_optimizer.step()

  - COSGD's _combine: Gram-Schmidt orthogonalise then sum/mean/freq-weight.
  - GradDrop's _combine: sign-purity masking then sum.

This base owns everything they share — the parameter layout, gradient
extraction, the three per-class forward/backward strategies (single_forward,
multi_forward, multi_forward_with_BN), the optional timer, and the
write-grad + base_optimizer.step() plumbing. Subclasses implement only
`_combine(class_grads, counts) -> combined_vector`.

Keeping the extraction in one place means COSGD and GradDrop are guaranteed to
see *identical* per-class subgradients, so any difference between them is
attributable to the fold rule alone — which is exactly the inter-batch
mechanism contrast the dissertation draws.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Type

import torch
from torch.optim.optimizer import Optimizer


class PerClassTimer:
    """Lightweight timing. No-op (zero overhead) unless `enabled`."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.times = defaultdict(float)
        self.call_counts = defaultdict(int)

    @contextmanager
    def time_context(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.times[name] += elapsed
            self.call_counts[name] += 1

    def get_stats(self):
        total = sum(self.times.values())
        stats = {}
        for op, t in self.times.items():
            stats[op] = {
                "total_time": t,
                "percentage": (t / total * 100) if total > 0 else 0.0,
                "call_count": self.call_counts[op],
                "avg_time": t / self.call_counts[op] if self.call_counts[op] else 0.0,
            }
        return stats, total

    def print_detailed_stats(self):
        stats, total = self.get_stats()
        if total == 0:
            print("No timing data (collect_timing=False?).")
            return
        print("\n" + "=" * 78)
        print(f"{'Operation':<30} {'Time (s)':<12} {'%':<8} {'Calls':<8} {'Avg (ms)':<10}")
        print("-" * 78)
        for op, d in sorted(stats.items(), key=lambda x: x[1]["percentage"], reverse=True):
            print(f"{op:<30} {d['total_time']:<12.6f} {d['percentage']:<8.1f} "
                  f"{d['call_count']:<8} {d['avg_time'] * 1000:<10.3f}")
        print("-" * 78)
        print(f"{'TOTAL':<30} {total:.6f}s   100.0%\n")


class PerClassGradientOptimizer(Optimizer):
    """Base wrapper for per-class-subgradient methods (COSGD, GradDrop).

    Parameters
    ----------
    params : iterable of torch.Tensor
    base_optimizer_cls : Type[Optimizer], default torch.optim.SGD
        Applied to the combined vector. Composes with SGD, SignSGD, RMSprop, Adam.
    lr : float, default 1e-3
        Forwarded to the base optimiser (so `Opt(..., lr=...)` works).
    model, criterion : required — the wrapper does its own per-class fwd/bwd.
    step_method : {'single_forward','multi_forward','multi_forward_with_BN'}
    collect_timing : bool, default False
    **base_optimizer_kwargs : forwarded to base_optimizer_cls (momentum, betas, …).
    """

    _VALID_STEP = ("single_forward", "multi_forward", "multi_forward_with_BN")

    def __init__(
        self,
        params,
        *,
        base_optimizer_cls: Type[Optimizer] = torch.optim.SGD,
        lr: float = 1e-3,
        model=None,
        criterion=None,
        step_method: str = "single_forward",
        collect_timing: bool = False,
        eps: float = 1e-12,
        **base_optimizer_kwargs,
    ):
        if model is None or criterion is None:
            raise ValueError("Model and criterion must be provided.")
        if step_method not in self._VALID_STEP:
            raise ValueError(f"step_method must be one of {self._VALID_STEP}")

        super().__init__(params, defaults={})
        self.base_optimizer: Optimizer = base_optimizer_cls(
            self.param_groups, lr=lr, **base_optimizer_kwargs
        )

        self.model = model
        self.criterion = criterion
        self.step_method = step_method
        self.eps = float(eps)
        self.timer = PerClassTimer(enabled=bool(collect_timing))

        self.device = next(model.parameters()).device
        self._param_info, self._total_params = self._compute_param_info()

    # ------------------------------------------------------------------
    # Parameter layout
    # ------------------------------------------------------------------
    def _compute_param_info(self):
        info, total = [], 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.numel() > 0:
                    info.append({"param": p, "shape": p.shape,
                                 "start": total, "end": total + p.numel()})
                    total += p.numel()
        return info, total

    def _extract_flat_grad(self):
        if self._total_params == 0:
            return torch.tensor([], device=self.device)
        out = torch.zeros(self._total_params, device=self.device, dtype=torch.float32)
        for info in self._param_info:
            g = info["param"].grad
            if g is not None:
                out[info["start"]:info["end"]] = g.view(-1)
        return out

    def _write_grad(self, combined):
        if combined.numel() == 0:
            return
        for info in self._param_info:
            sl = combined[info["start"]:info["end"]].view(info["shape"])
            p = info["param"]
            if p.grad is None:
                p.grad = sl.detach().clone()
            else:
                p.grad.copy_(sl)

    # ------------------------------------------------------------------
    # Optimizer API
    # ------------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, data, labels, unique_labels, closure=None):
        with self.timer.time_context("total_step"):
            with self.timer.time_context("data_preparation"):
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                unique_labels = unique_labels.to(self.device, non_blocking=True)
                valid = unique_labels[torch.isin(unique_labels, labels)]
                if len(valid) == 0:
                    return 0.0

            if self.step_method == "single_forward":
                class_grads, running_loss, counts = self._grads_single_forward(data, labels, valid)
            elif self.step_method == "multi_forward":
                class_grads, running_loss, counts = self._grads_multi_forward(data, labels, valid)
            else:
                class_grads, running_loss, counts = self._grads_multi_forward_bn(data, labels, valid)

            with self.timer.time_context("combine"):
                combined = self._combine(class_grads, counts)
            self._write_grad(combined)
            with self.timer.time_context("base_step"):
                self.base_optimizer.step()
        return running_loss / max(len(valid), 1)

    # ------------------------------------------------------------------
    # Per-class gradient strategies
    # ------------------------------------------------------------------
    def _grads_single_forward(self, data, labels, valid):
        with self.timer.time_context("forward_pass"):
            outputs = self.model(data)
        return self._per_class_from_outputs(outputs, labels, valid, retain=True)

    def _grads_multi_forward(self, data, labels, valid):
        n = len(valid)
        class_grads = torch.zeros(n, self._total_params, device=self.device)
        counts = torch.zeros(n, device=self.device)
        running_loss = 0.0
        for i, label in enumerate(valid):
            with self.timer.time_context("data_indexing"):
                mask = (labels == label)
                data_c, labels_c = data[mask], labels[mask]
                counts[i] = mask.sum()
            with self.timer.time_context("forward_pass"):
                outputs = self.model(data_c)
            with self.timer.time_context("loss_computation"):
                loss = self.criterion(outputs, labels_c)
                running_loss += loss.item()
            with self.timer.time_context("backward_pass"):
                self.zero_grad()
                loss.backward()
            with self.timer.time_context("gradient_extraction"):
                class_grads[i] = self._extract_flat_grad()
        return class_grads, running_loss, counts

    def _grads_multi_forward_bn(self, data, labels, valid):
        original_training = self.model.training
        with self.timer.time_context("bn_stats_update"):
            self.model.train()
            with torch.no_grad():
                _ = self.model(data)
            self.model.eval()
        try:
            cg, rl, counts = self._grads_multi_forward(data, labels, valid)
        finally:
            with self.timer.time_context("mode_restoration"):
                self.model.train(original_training)
        return cg, rl, counts

    def _per_class_from_outputs(self, outputs, labels, valid, retain):
        n = len(valid)
        class_grads = torch.zeros(n, self._total_params, device=self.device)
        counts = torch.zeros(n, device=self.device)
        running_loss = 0.0
        for i, label in enumerate(valid):
            with self.timer.time_context("data_indexing"):
                mask = (labels == label)
                outputs_c, labels_c = outputs[mask], labels[mask]
                counts[i] = mask.sum()
            with self.timer.time_context("loss_computation"):
                loss = self.criterion(outputs_c, labels_c)
                running_loss += loss.item()
            with self.timer.time_context("backward_pass"):
                self.zero_grad()
                loss.backward(retain_graph=(retain and i < n - 1))
            with self.timer.time_context("gradient_extraction"):
                class_grads[i] = self._extract_flat_grad()
        return class_grads, running_loss, counts

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _combine(self, class_grads, counts):  # pragma: no cover - abstract
        """Fold [n_classes, n_params] subgradients into one [n_params] vector."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Timing accessors
    # ------------------------------------------------------------------
    def reset_time_stats(self):
        self.timer.reset()

    def get_time_stats(self):
        return self.timer.get_stats()

    def print_time_stats(self):
        self.timer.print_detailed_stats()


__all__ = ["PerClassGradientOptimizer", "PerClassTimer"]
