# GradDrop.py
"""
GradDrop — Gradient Sign Dropout, adapted to single-task per-class subgradients.

Published as a multi-task method (Chen et al., "Just Pick a Sign: Optimizing
Deep Multitask Models with Gradient Sign Dropout", NeurIPS 2020), where the
objects masked are per-task gradients. Here we apply the *same* sign-purity
masking rule to the per-class subgradients within a single-task mini-batch, so
it is a direct inter-batch (§03) sibling/competitor to COSGD: identical per-class
extraction, a different fold step.

Sign-purity fold
----------------
Given per-class subgradients {g_c} stacked as G ∈ R^{C×P}, for each coordinate j:

    P_j = 1/2 * ( 1 + ( Σ_c G[c,j] ) / ( Σ_c |G[c,j]| + eps ) )    ∈ [0, 1]

P_j is the "positive sign purity": 1 if every class agrees on a positive sign at
coordinate j, 0 if all agree negative, 0.5 if perfectly conflicted. Draw a single
mask U_j ~ Uniform(0,1) per coordinate and keep, for each class c, only the
gradient components whose sign matches the sampled majority:

    keep positive components where U_j < P_j ;  keep negative components otherwise.

Equivalently (the standard formulation): for class c, coordinate j,
    mask = [ (G[c,j] > 0) & (U_j < P_j) ] | [ (G[c,j] < 0) & (U_j >= P_j) ]
and the combined update is Σ_c mask ⊙ G[c].

Sign-consistent coordinates (P_j ≈ 0 or 1) survive deterministically; conflicted
coordinates are stochastically resolved to one sign, never summed to a damped
mix. This targets exactly the within-batch per-class cancellation the §03 metric
I_inter measures.

Composes with SGD, SignSGD, RMSprop, Adam via `base_optimizer_cls`. Memory is
O(C·P) like COSGD for the dense stack; the combine itself is O(C·P) elementwise
(no O(C²) Gram-Schmidt), which is a useful contrast in the scalability study.
"""

from typing import Optional, Type

import torch
from torch.optim.optimizer import Optimizer

from common.optimizers._per_class import PerClassGradientOptimizer


class GradDrop(PerClassGradientOptimizer):
    """Sign-purity Gradient Dropout over per-class subgradients.

    Parameters
    ----------
    leak : float, default 0.0
        Optional leak ∈ [0, 1]: fraction of the dropped-sign components retained
        (leak=0 is standard GradDrop; leak=1 recovers the plain per-class sum).
        Exposed as an ablation knob; default reproduces the published method.
    generator : torch.Generator, optional
        For reproducible masking. If None, uses the global RNG (so a seeded run
        is still deterministic).
    (plus all PerClassGradientOptimizer args: base_optimizer_cls, lr, model,
     criterion, step_method, collect_timing, **base_optimizer_kwargs)
    """

    def __init__(
        self,
        params,
        *,
        base_optimizer_cls: Type[Optimizer] = torch.optim.SGD,
        lr: float = 1e-3,
        model=None,
        criterion=None,
        step_method: str = "single_forward",
        leak: float = 0.0,
        generator: Optional[torch.Generator] = None,
        collect_timing: bool = False,
        eps: float = 1e-12,
        **base_optimizer_kwargs,
    ):
        if not (0.0 <= leak <= 1.0):
            raise ValueError("leak must be in [0, 1]")
        super().__init__(
            params, base_optimizer_cls=base_optimizer_cls, lr=lr,
            model=model, criterion=criterion, step_method=step_method,
            collect_timing=collect_timing, eps=eps, **base_optimizer_kwargs,
        )
        self.leak = float(leak)
        self.generator = generator

    def _combine(self, class_grads, counts):
        # class_grads: [C, P]
        with self.timer.time_context("sign_purity"):
            col_sum = class_grads.sum(dim=0)                       # [P]
            col_abs = class_grads.abs().sum(dim=0).clamp_min(self.eps)  # [P]
            purity = 0.5 * (1.0 + col_sum / col_abs)              # P_j in [0,1]
            u = torch.rand(
                purity.shape, device=class_grads.device,
                dtype=class_grads.dtype, generator=self.generator,
            )
            keep_pos = (u < purity)                               # [P] bool
        with self.timer.time_context("mask_apply"):
            pos = class_grads > 0
            neg = class_grads < 0
            mask = (pos & keep_pos.unsqueeze(0)) | (neg & (~keep_pos).unsqueeze(0))
            mask = mask.to(class_grads.dtype)
            if self.leak > 0.0:
                mask = mask + self.leak * (1.0 - mask)
            combined = (mask * class_grads).sum(dim=0)
        return combined


__all__ = ["GradDrop"]
