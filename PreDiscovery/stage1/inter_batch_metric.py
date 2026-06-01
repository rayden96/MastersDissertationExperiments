"""
Inter-batch interference metric — pairwise gradient conflict across classes.

For a single batch (or balanced probe), compute per-class gradients
$g_c = \\nabla_\\theta L_c(\\theta)$ where $L_c$ is the loss on class c
examples only. Then compute pairwise cos$(g_c, g_{c'})$ across all class
pairs. Aggregates:

  - n_pairs            number of class pairs (C(C-1)/2 for C classes)
  - frac_positive      cos > 0 fraction
  - frac_negative      cos < 0 fraction (the "destructive intra-batch
                       interference" signal — the one COSGD targets)
  - mean_cos
  - mean_abs_cos
  - mean_positive_cos / mean_negative_cos
  - min_cos / max_cos

This is conceptually distinct from the *between-batch* pairwise alignment
(handled by `PairwiseAlignmentTracker` in `common/diagnostics`):

  - between-batch pairwise: cos(g_t, g_{t-k}) — same batch's gradient
                            compared to past batches' gradients.
  - inter-batch pairwise:   cos(g_c, g_{c'}) — within one batch, gradients
                            of different classes inside that batch.

The two metrics target the two types of interference in the framework.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from common.diagnostics.per_class_probe import ClassProbeSet


def _flatten_grads(model: nn.Module) -> Optional[torch.Tensor]:
    pieces = []
    for p in model.parameters():
        if p.grad is None or not p.requires_grad:
            continue
        pieces.append(p.grad.detach().reshape(-1))
    if not pieces:
        return None
    return torch.cat(pieces)


@torch.enable_grad()
def measure_inter_batch_interference(
    model: nn.Module,
    criterion: nn.Module,
    probe_set: ClassProbeSet,
    *,
    min_examples_per_class: int = 4,
) -> Dict[str, float]:
    """Compute pairwise per-class gradient cosines via the probe set.

    Procedure:
      1. For each class c with >= min_examples_per_class examples in the probe:
         - Zero gradients
         - Forward + backward on class-c probe examples only
         - Capture the resulting flat gradient vector g_c
      2. Compute pairwise cos(g_c, g_{c'}) for all class pairs
      3. Aggregate distribution statistics
      4. Restore: zero gradients (so the next training step is unaffected)

    Cost: ~C extra forward+backward passes over the probe set per call.
    For CIFAR-10 with 64 examples per class, that's 10 fwd+bwd on tiny
    inputs — cheap relative to a real training step on a real batch.

    Returns
    -------
    dict
        Aggregated stats. Empty dict if fewer than 2 classes have enough
        examples to form pairs.
    """
    was_training = model.training
    model.train()  # for batchnorm consistency with normal training

    grads_per_class: Dict[int, torch.Tensor] = {}

    try:
        for c in range(probe_set.num_classes):
            start = probe_set.class_offsets[c]
            end = probe_set.class_offsets[c + 1]
            n_class = end - start
            if n_class < min_examples_per_class:
                continue

            x_c = probe_set.x[start:end]
            y_c = probe_set.y[start:end]

            # Zero existing gradients
            model.zero_grad(set_to_none=False)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Forward + backward on this class's examples only
            logits = model(x_c)
            loss = criterion(logits, y_c)
            loss.backward()

            # Capture gradient
            g = _flatten_grads(model)
            if g is not None:
                grads_per_class[c] = g.clone()
    finally:
        # Always restore — zero grads + restore training mode
        model.zero_grad(set_to_none=False)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if not was_training:
            model.eval()

    classes = sorted(grads_per_class.keys())
    if len(classes) < 2:
        return {"n_pairs": 0.0}

    # Pairwise cosines
    cos_values: List[float] = []
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1:]:
            g1 = grads_per_class[c1]
            g2 = grads_per_class[c2]
            n1 = g1.norm()
            n2 = g2.norm()
            if n1.item() < 1e-12 or n2.item() < 1e-12:
                continue
            cos = float((g1 @ g2 / (n1 * n2)).item())
            cos_values.append(cos)

    if not cos_values:
        return {"n_pairs": 0.0}

    n = len(cos_values)
    positive = [v for v in cos_values if v > 0]
    negative = [v for v in cos_values if v < 0]

    # Magnitude of mean per-class gradient (average ‖g_c‖ across c)
    norm_means = [float(grads_per_class[c].norm().item()) for c in classes]
    return {
        "n_pairs": float(n),
        "n_classes_measured": float(len(classes)),
        "frac_positive": len(positive) / n,
        "frac_negative": len(negative) / n,
        "mean_cos": sum(cos_values) / n,
        "mean_abs_cos": sum(abs(v) for v in cos_values) / n,
        "mean_positive_cos": sum(positive) / len(positive) if positive else 0.0,
        "mean_negative_cos": sum(negative) / len(negative) if negative else 0.0,
        "min_cos": min(cos_values),
        "max_cos": max(cos_values),
        "mean_class_grad_norm": sum(norm_means) / len(norm_means),
    }


__all__ = ["measure_inter_batch_interference"]
