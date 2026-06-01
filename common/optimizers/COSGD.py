# COSGD.py
"""
COSGD — Class-Orthogonalised Gradient Descent.

Computes per-class subgradients on the current batch, **orthogonalises** them
against each other (Gram-Schmidt), combines them into a single vector, writes
that into each parameter's `.grad`, then lets a base optimiser apply the step.

The shared per-class machinery (parameter layout, gradient extraction, the
three forward/backward strategies, timing, write-grad + base step) lives in
`_per_class.PerClassGradientOptimizer`. COSGD only implements the fold step
`_combine`: Gram-Schmidt, then sum / mean / freq-weight. (GradDrop is the
sibling that swaps GS for sign-purity masking — same extraction, different fold.)

Composes with SGD, SignSGD, RMSprop, Adam via `base_optimizer_cls`.

Design axes (driven by PaperReadyExperiments/20_cosgd_ablation):
  - orthogonalization_method : {gram_schmidt_normal, gram_schmidt_negative,
                                modified_gs_normal, modified_gs_negative}
  - step_method              : {single_forward, multi_forward, multi_forward_with_BN}
  - class_order              : {fixed, desc, asc, random}  (GS is order-dependent)
  - prenormalize             : unit-normalise per-class grads before GS (thesis 3.1/3.2)
  - combine                  : {sum, mean, freq}  (freq = §02 class-frequency weighting)

Back-compat: `COSGD(params, lr=..., model=..., criterion=...,
orthogonalization_method=..., step_method=...)` with defaults
(base=SGD, combine="sum", class_order="fixed", prenormalize=False) reproduces
the previous SGD-only behaviour exactly. Step signature unchanged:
`step(data, labels, unique_labels)`.
"""

from typing import Type

import torch
from torch.optim.optimizer import Optimizer

from common.optimizers._per_class import PerClassGradientOptimizer


###############################
# Orthogonalization Utilities #
###############################

def gram_schmidt_normal(vectors):
    """Standard Gram-Schmidt. vectors: [num_vectors, dim]."""
    num_vectors, _ = vectors.shape
    ortho = torch.zeros_like(vectors)
    for i in range(num_vectors):
        v = vectors[i].clone()
        if i > 0:
            prev = ortho[:i]
            dots = torch.mv(prev, v)
            norms_sq = torch.clamp(torch.sum(prev ** 2, dim=1), min=1e-12)
            v = v - torch.sum((dots / norms_sq).unsqueeze(1) * prev, dim=0)
        ortho[i] = v
    return ortho


def gram_schmidt_negative(vectors):
    """Gram-Schmidt removing only negative (destructive) projections."""
    n = vectors.shape[0]
    ortho = torch.zeros_like(vectors)
    for i in range(n):
        v = vectors[i].clone()
        for j in range(i):
            u = ortho[j]
            dot = torch.dot(v, u)
            if dot < 0:
                v = v - (dot / torch.clamp(torch.dot(u, u), min=1e-12)) * u
        ortho[i] = v
    return ortho


def modified_gram_schmidt_normal(vectors):
    """Modified Gram-Schmidt (more numerically stable)."""
    n = vectors.shape[0]
    ortho = torch.zeros_like(vectors)
    for i in range(n):
        v = vectors[i].clone()
        for j in range(i):
            u = ortho[j]
            u_norm = torch.norm(u)
            if u_norm > 1e-12:
                un = u / u_norm
                v = v - torch.dot(v, un) * un
        ortho[i] = v
    return ortho


def modified_gram_schmidt_negative(vectors):
    """Modified Gram-Schmidt removing only negative projections."""
    n = vectors.shape[0]
    ortho = torch.zeros_like(vectors)
    for i in range(n):
        v = vectors[i].clone()
        for j in range(i):
            u = ortho[j]
            u_norm = torch.norm(u)
            if u_norm > 1e-12:
                un = u / u_norm
                dot = torch.dot(v, un)
                if dot < 0:
                    v = v - dot * un
        ortho[i] = v
    return ortho


ORTHOGONALIZATION_METHODS = {
    "gram_schmidt_normal": gram_schmidt_normal,
    "gram_schmidt_negative": gram_schmidt_negative,
    "modified_gs_normal": modified_gram_schmidt_normal,
    "modified_gs_negative": modified_gram_schmidt_negative,
}


def orthogonalize_gradients(gradient_list, method):
    """Orthogonalise a [n, dim] stack (or list) and return the summed vector.

    Retained for back-compat / external callers. COSGD itself calls the
    per-method function directly so it can apply its own combine rule.
    """
    if isinstance(gradient_list, list):
        if not gradient_list:
            raise ValueError("Empty gradient list provided")
        gradients = torch.stack(gradient_list, dim=0)
    else:
        gradients = gradient_list
    fn = ORTHOGONALIZATION_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown orthogonalization method: {method}")
    return torch.sum(fn(gradients), dim=0)


###############################
# COSGD                       #
###############################

class COSGD(PerClassGradientOptimizer):
    _VALID_ORDER = ("fixed", "desc", "asc", "random")
    _VALID_COMBINE = ("sum", "mean", "freq")

    def __init__(
        self,
        params,
        *,
        base_optimizer_cls: Type[Optimizer] = torch.optim.SGD,
        lr: float = 1e-3,
        model=None,
        criterion=None,
        orthogonalization_method: str = "gram_schmidt_normal",
        step_method: str = "single_forward",
        class_order: str = "fixed",
        prenormalize: bool = False,
        combine: str = "sum",
        collect_timing: bool = False,
        eps: float = 1e-12,
        **base_optimizer_kwargs,
    ):
        if orthogonalization_method not in ORTHOGONALIZATION_METHODS:
            raise ValueError(f"Unknown orthogonalization method: {orthogonalization_method}")
        if class_order not in self._VALID_ORDER:
            raise ValueError(f"class_order must be one of {self._VALID_ORDER}")
        if combine not in self._VALID_COMBINE:
            raise ValueError(f"combine must be one of {self._VALID_COMBINE}")

        super().__init__(
            params, base_optimizer_cls=base_optimizer_cls, lr=lr,
            model=model, criterion=criterion, step_method=step_method,
            collect_timing=collect_timing, eps=eps, **base_optimizer_kwargs,
        )
        self.orthogonalization_method = orthogonalization_method
        self.class_order = class_order
        self.prenormalize = bool(prenormalize)
        self.combine = combine

    def _row_order(self, class_grads):
        if self.class_order == "fixed":
            return None
        if self.class_order == "random":
            return torch.randperm(class_grads.shape[0], device=class_grads.device)
        norms = class_grads.norm(dim=1)
        return torch.argsort(norms, descending=(self.class_order == "desc"))

    def _combine(self, class_grads, counts):
        with self.timer.time_context("ordering"):
            order = self._row_order(class_grads)
            if order is not None:
                class_grads = class_grads[order]
                counts = counts[order]
        if self.prenormalize:
            class_grads = class_grads / class_grads.norm(dim=1, keepdim=True).clamp_min(self.eps)
        with self.timer.time_context("orthogonalization"):
            ortho = ORTHOGONALIZATION_METHODS[self.orthogonalization_method](class_grads)
        if self.combine == "sum":
            return ortho.sum(dim=0)
        if self.combine == "mean":
            return ortho.mean(dim=0)
        w = (counts / counts.sum().clamp_min(1.0)).unsqueeze(1)  # freq
        return (w * ortho).sum(dim=0)


__all__ = ["COSGD", "orthogonalize_gradients", "ORTHOGONALIZATION_METHODS"]
