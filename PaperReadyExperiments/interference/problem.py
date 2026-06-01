"""
InterferenceProblem — adapter interface that decouples the metrics from the
specific training problem.

Each experiment provides an implementation that knows how to:

- flatten / unflatten parameters (so the metrics can work with 1D tensors);
- compute the reference (full-batch or large-reference) gradient at the
  current parameters, plus the reference loss;
- compute per-subgroup subgradients on a given batch (per-class for image
  classification; per-mixture-component for a 2D toy; etc.);
- compute the applied update u_t for a single training step (so the
  cancellation indices and trajectory metrics have access to it).

Keep the interface deliberately narrow. Specific problems can expose more on
their own objects, but the metrics only depend on what's here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple

import torch


class InterferenceProblem(ABC):
    """Adapter interface bridging the metrics module to a specific training
    problem (2D toy, CIFAR-10 CNN, CIFAR-100 ResNet, etc.).

    Implementations must keep `flatten_params()` and `flatten_grad()` consistent
    — same ordering, same dtype, same device — so the metrics can compare
    parameter snapshots, gradients, and applied updates as 1D vectors.
    """

    # ----- parameters --------------------------------------------------
    @abstractmethod
    def flatten_params(self) -> torch.Tensor:
        """Return the current trainable parameters as a flat 1D tensor (clone)."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the problem lives on."""

    # ----- reference gradient ------------------------------------------
    @abstractmethod
    def compute_reference(self) -> Tuple[torch.Tensor, float]:
        """Compute the reference gradient at the current parameters.

        Returns:
            (g_tilde, loss): a flat 1D tensor (same shape as flatten_params())
            holding the population (or large-reference) gradient, and the
            reference-set loss as a Python float.

        The implementation may use a fixed reference loader, the full training
        set, or a held-out subset — whatever is most natural for the problem.
        """

    # ----- per-subgroup subgradients on the current batch --------------
    @abstractmethod
    def per_subgroup_grads(self, batch) -> Dict[int, torch.Tensor]:
        """Per-subgroup subgradients on `batch`, as flat tensors.

        The "subgroups" depend on the problem:
        - image classification: one subgroup per class present in the batch;
        - 2D mixture-of-quadratics: one subgroup per mixture component
          represented in the batch.

        Returns a dict {subgroup_id: flat_grad_tensor}. Subgroups with zero
        samples in the batch are omitted.
        """

    # ----- preconditioned counterfactual ideal (optional) --------------
    def preconditioned_ideal(self, ref_grad: torch.Tensor):
        """Optional: the update the *training optimiser itself* would apply to
        the reference (full-batch) gradient at the current parameters.

        Returns a flat tensor `u_ideal` (same shape as flatten_params()), or
        `None` if the problem does not implement it. When provided, the meter
        records a per-optimiser deficit

            D_t_precond = <g_tilde, u_ideal> - <g_tilde, u_t>

        which, unlike the SGD-yardstick `D_t`, isolates mini-batch/trajectory
        cancellation from the optimiser's preconditioning (RMSprop/Adam). The
        SGD yardstick (u_ideal = -lr * g_tilde) is always recorded too as a
        cross-method common scale; for plain SGD the two coincide.

        Implementations compute this via a state-preserving "shadow step": load
        g_tilde into .grad on a clone of the optimiser state, step, and measure
        the applied delta — without perturbing the real training optimiser.
        Default returns None (preserves the original SGD-only behaviour exactly).
        """
        return None
