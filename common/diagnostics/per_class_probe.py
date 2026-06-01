"""
Balanced per-class probe set for fast in-training evaluation.

The probe is a small, balanced subset of the test (or validation) set,
held in memory on the target device so each evaluation is one forward pass.
Used by InterferenceTracker to monitor per-class loss/accuracy at high
cadence during training, which is the basis for the forgetting-event metric
defined in research/01_interference_framework/framework.md.

Typical usage:

    from common.diagnostics import ClassProbeSet
    probe = ClassProbeSet(test_set, num_classes=10, n_per_class=64, device=device)

    # Cheap per-class evaluation any time during training:
    per_class = probe.evaluate(model, criterion)
    # → {0: {'loss': 1.23, 'acc': 0.78, 'n': 64}, 1: {...}, ...}
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset


class ClassProbeSet:
    """A small, balanced, pre-loaded probe set for per-class evaluation.

    Build once at the start of a run; evaluate repeatedly. The probe tensors
    live on `device`, so each call to ``evaluate`` is one forward pass.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Source dataset (typically the test set). Items must yield (x, y) where
        y is an int-castable class label.
    num_classes : int
        Total number of classes.
    n_per_class : int, default 64
        Examples sampled per class. Total probe size = num_classes * n_per_class.
    seed : int, default 2026
        Sampling seed for reproducibility.
    device : torch.device or str, default 'cpu'
        Where to hold the probe tensors.
    max_scan : int, optional
        Upper bound on dataset indices to scan when sampling. Useful if the
        dataset is huge but classes are dense at the start (early-stop after
        finding enough of each class). Default: full dataset length.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_classes: int,
        n_per_class: int = 64,
        seed: int = 2026,
        device="cpu",
        max_scan: int = None,
    ):
        self.num_classes = int(num_classes)
        self.n_per_class = int(n_per_class)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # Deterministic walk: shuffle indices once, take first n_per_class per class.
        n_total = len(dataset)
        scan_limit = min(max_scan, n_total) if max_scan is not None else n_total
        rng = torch.Generator()
        rng.manual_seed(int(seed))
        order = torch.randperm(n_total, generator=rng).tolist()[:scan_limit]

        per_class_indices: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        for idx in order:
            _, label = dataset[idx]
            label = int(label)
            if label in per_class_indices and len(per_class_indices[label]) < self.n_per_class:
                per_class_indices[label].append(idx)
            if all(len(v) >= self.n_per_class for v in per_class_indices.values()):
                break

        # Stack contiguously per class so we can slice per class quickly.
        x_list = []
        y_list = []
        class_offsets = [0]
        for c in range(self.num_classes):
            indices = per_class_indices[c][: self.n_per_class]
            for idx in indices:
                x, y = dataset[idx]
                x_list.append(x)
                y_list.append(int(y))
            class_offsets.append(class_offsets[-1] + len(indices))

        if not x_list:
            raise ValueError("ClassProbeSet found no examples — check dataset / num_classes.")

        self.x = torch.stack(x_list).to(self.device)
        self.y = torch.tensor(y_list, dtype=torch.long, device=self.device)
        self.class_offsets = class_offsets  # length num_classes + 1

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        loss_cap: float = 50.0,
    ) -> Dict[int, Dict[str, float]]:
        """Run the model over the full probe set, return per-class loss/acc.

        Parameters
        ----------
        model : nn.Module
        criterion : nn.Module, optional
            Loss function. Default: ``nn.functional.cross_entropy`` reduction='none'.
            (Ignored unless you subclass and want a different per-example loss.)
        loss_cap : float, default 50.0
            Per-example loss values are clamped to this maximum. Cross-entropy
            can otherwise reach astronomical values when the model is forced
            into a near-one-hot prediction on the wrong class (e.g. mid
            continual-learning forgetting), which corrupts the forgetting
            magnitude metric. Default 50 corresponds to predicting probability
            ~2e-22 for the correct class — well past any meaningful gradient
            signal. Set to ``float('inf')`` to disable.

        Returns
        -------
        dict
            ``{class_id: {'loss': float, 'acc': float, 'n': int}}``
        """
        was_training = model.training
        model.eval()
        try:
            logits = model(self.x)
            losses = nn.functional.cross_entropy(logits, self.y, reduction="none")
            if loss_cap != float("inf"):
                losses = torch.clamp(losses, max=loss_cap)
            preds = logits.argmax(dim=1)

            out: Dict[int, Dict[str, float]] = {}
            for c in range(self.num_classes):
                start = self.class_offsets[c]
                end = self.class_offsets[c + 1]
                if start == end:
                    out[c] = {"loss": float("nan"), "acc": float("nan"), "n": 0}
                    continue
                cls_losses = losses[start:end]
                cls_preds = preds[start:end]
                cls_y = self.y[start:end]
                out[c] = {
                    "loss": float(cls_losses.mean().item()),
                    "acc": float((cls_preds == cls_y).float().mean().item()),
                    "n": int(end - start),
                }
        finally:
            if was_training:
                model.train()

        return out

    def total_size(self) -> int:
        return self.x.size(0)


__all__ = ["ClassProbeSet"]
