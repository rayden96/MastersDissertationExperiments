"""
TorchClassificationProblem — `InterferenceProblem` adapter for any PyTorch
image-classification model with cross-entropy loss.

Used by `02_medium_cifar10/` and `03_large_cifar100/`. Handles:

- flattening parameters and gradients consistently;
- computing the reference gradient on a fixed reference DataLoader;
- computing per-class subgradients on the current batch (one masked
  forward+backward per class present);
- snapshotting + restoring BatchNorm running statistics around diagnostic
  forwards so the metric computation does not pollute training-time BN state.

The training loop owns the optimizer; this object only provides the
read/write surface that the metrics need.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from interference.problem import InterferenceProblem


def _flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([
        p.detach().view(-1) for p in model.parameters() if p.requires_grad
    ])


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            chunks.append(torch.zeros_like(p).view(-1))
        else:
            chunks.append(p.grad.detach().view(-1))
    return torch.cat(chunks)


def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


@contextmanager
def _bn_state_preserved(model: nn.Module):
    """Snapshot BatchNorm running stats on entry; restore on exit.
    Lets us run diagnostic forwards (per-class subgrads, ref grad) without
    corrupting the training-time BN running statistics.
    """
    snapshots: List[Tuple[nn.modules.batchnorm._BatchNorm, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            snapshots.append((
                m,
                m.running_mean.detach().clone() if m.running_mean is not None else None,
                m.running_var.detach().clone() if m.running_var is not None else None,
                m.num_batches_tracked.detach().clone() if m.num_batches_tracked is not None else None,
            ))
    try:
        yield
    finally:
        for m, rm, rv, nbt in snapshots:
            if rm is not None and m.running_mean is not None:
                m.running_mean.copy_(rm)
            if rv is not None and m.running_var is not None:
                m.running_var.copy_(rv)
            if nbt is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(nbt)


class TorchClassificationProblem(InterferenceProblem):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        ref_loader: DataLoader,
        device: torch.device,
        precond_optimizer=None,
    ):
        self.model = model
        self.criterion = criterion
        self.ref_loader = ref_loader
        self._device = device
        # Optional base optimiser used to compute the per-optimiser
        # counterfactual ideal in `preconditioned_ideal`. Set by the Trainer to
        # the *base* optimiser (e.g. optimizer.base_optimizer for wrapped
        # methods), so the ideal reflects preconditioning only — not the
        # projection/orthogonalisation the wrapper adds.
        self._precond_optimizer = precond_optimizer

    @property
    def device(self) -> torch.device:
        return self._device

    def set_precond_optimizer(self, optimizer) -> None:
        self._precond_optimizer = optimizer

    def flatten_params(self) -> torch.Tensor:
        return _flatten_params(self.model)

    # -----------------------------------------------------------------
    def preconditioned_ideal(self, ref_grad: torch.Tensor):
        """The update the base optimiser would apply to the reference gradient
        at the current parameters, via a state-preserving shadow step.

        Snapshots params + optimiser state, loads `ref_grad` into `.grad`, steps,
        measures the applied delta, then restores everything. Reflects the
        *current* preconditioner state (Adam m/v, RMSprop avg) so the ideal is
        the optimiser's genuine response to the clean gradient at this point in
        training. Returns None if no optimiser was supplied.
        """
        opt = self._precond_optimizer
        if opt is None:
            return None

        import copy

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        params_before = [p.detach().clone() for p in trainable]
        grads_before = [None if p.grad is None else p.grad.detach().clone() for p in trainable]
        try:
            state_before = copy.deepcopy(opt.state_dict())
        except Exception:
            state_before = None

        # Load ref_grad slices into .grad.
        offset = 0
        ref_grad = ref_grad.to(self._device)
        for p in trainable:
            n = p.numel()
            p.grad = ref_grad[offset:offset + n].view_as(p).detach().clone()
            offset += n

        with torch.no_grad():
            opt.step()
            u_ideal = torch.cat([
                (p.detach() - b.detach()).view(-1) for p, b in zip(trainable, params_before)
            ])

            # Restore params, grads, optimiser state.
            for p, b in zip(trainable, params_before):
                p.copy_(b)
        for p, gb in zip(trainable, grads_before):
            p.grad = gb
        if state_before is not None:
            opt.load_state_dict(state_before)

        return u_ideal

    # -----------------------------------------------------------------
    def compute_reference(self) -> Tuple[torch.Tensor, float]:
        """Gradient and loss of the mean cross-entropy over `ref_loader`.

        Uses backward on (loss * batch_size) per ref batch so the accumulated
        gradient is the sum-grad; divided by n_total at the end to get the
        mean-grad. Wraps the whole pass in `_bn_state_preserved` so BN
        running stats are not corrupted by this diagnostic call.
        """
        was_training = self.model.training
        self.model.train()
        with _bn_state_preserved(self.model):
            _zero_grads(self.model)
            n_total = 0
            loss_sum = 0.0
            for x, y in self.ref_loader:
                x = x.to(self._device, non_blocking=True)
                y = y.to(self._device, non_blocking=True)
                out = self.model(x)
                loss = self.criterion(out, y)
                (loss * x.size(0)).backward()
                n_total += x.size(0)
                loss_sum += loss.item() * x.size(0)
            grad = _flatten_grads(self.model) / max(n_total, 1)
            _zero_grads(self.model)
        if not was_training:
            self.model.eval()
        return grad, loss_sum / max(n_total, 1)

    # -----------------------------------------------------------------
    def per_subgroup_grads(self, batch) -> Dict[int, torch.Tensor]:
        """`batch` is (x, y). Returns dict {class_label: flat_grad}."""
        x, y = batch
        x = x.to(self._device, non_blocking=True)
        y = y.to(self._device, non_blocking=True)

        was_training = self.model.training
        self.model.train()
        out: Dict[int, torch.Tensor] = {}
        with _bn_state_preserved(self.model):
            for c in torch.unique(y).cpu().tolist():
                mask = (y == c)
                if int(mask.sum().item()) == 0:
                    continue
                xc, yc = x[mask], y[mask]
                _zero_grads(self.model)
                logits = self.model(xc)
                loss_c = self.criterion(logits, yc)
                loss_c.backward()
                out[int(c)] = _flatten_grads(self.model).clone()
            _zero_grads(self.model)
        if not was_training:
            self.model.eval()
        return out
