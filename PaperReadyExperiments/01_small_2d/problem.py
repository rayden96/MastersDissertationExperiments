"""
2D mixture-of-quadratics — `InterferenceProblem` implementation.

K components, each pulling theta toward its own centre. Centres placed on a
circle of radius R. Full-batch optimum is the centroid of the centres
(= origin under symmetric placement).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Make sure `interference` package can be imported when this is run as a script.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from interference import InterferenceProblem  # noqa: E402


class Mixture2DProblem(InterferenceProblem):
    """K quadratic mixture components in 2D.

    Parameters are a single 2-vector theta. Each component k contributes a
    half-squared-distance to its centre mu_k; the full loss is the unweighted
    average over k.

    The "subgroups" for §03 inter-batch metrics are the mixture components.
    `per_subgroup_grads(batch)` returns one gradient per unique component
    represented in the batch — the gradient of the component's loss term
    evaluated at the current theta.
    """

    def __init__(
        self,
        *,
        n_groups: int = 10,
        R: float = 2.0,
        batch_size: int = 20,
        lr: float = 0.05,
        init_position: Tuple[float, float] = (3.0, 0.0),
        seed: int = 0,
    ):
        self.n_groups = int(n_groups)
        self.R = float(R)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)

        angles = np.linspace(0.0, 2.0 * np.pi, self.n_groups, endpoint=False)
        centres = self.R * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.centres = torch.tensor(centres, dtype=torch.float32)

        self.theta = torch.tensor(list(init_position), dtype=torch.float32)
        self._device = torch.device("cpu")  # tiny problem; CPU is fine

    # ---- InterferenceProblem interface -------------------------------
    def flatten_params(self) -> torch.Tensor:
        return self.theta.detach().clone()

    @property
    def device(self) -> torch.device:
        return self._device

    def compute_reference(self) -> Tuple[torch.Tensor, float]:
        # Closed-form: grad = theta - mean(centres),  loss = mean_k 0.5 ||theta - mu_k||^2
        mean_centre = self.centres.mean(dim=0)
        grad = self.theta - mean_centre
        diffs = self.theta.unsqueeze(0) - self.centres  # [K, 2]
        loss = 0.5 * (diffs ** 2).sum(dim=1).mean()
        return grad.clone(), float(loss.item())

    def per_subgroup_grads(self, batch: torch.Tensor) -> Dict[int, torch.Tensor]:
        # batch: 1D tensor of component indices
        out: Dict[int, torch.Tensor] = {}
        for g in torch.unique(batch).tolist():
            grad_g = self.theta - self.centres[int(g)]
            out[int(g)] = grad_g.clone()
        return out

    # ---- training-loop helpers ---------------------------------------
    def sample_batch(self) -> torch.Tensor:
        ids = self.rng.integers(0, self.n_groups, size=self.batch_size)
        return torch.tensor(ids, dtype=torch.long)

    def step_sgd(self) -> Tuple[torch.Tensor, float]:
        """One vanilla-SGD step on a sampled batch.
        Returns (batch_indices, batch_loss).
        """
        batch = self.sample_batch()
        diffs = self.theta.unsqueeze(0) - self.centres[batch]  # [B, 2]
        batch_grad = diffs.mean(dim=0)
        batch_loss = float((0.5 * (diffs ** 2).sum(dim=1).mean()).item())
        self.theta = self.theta - self.lr * batch_grad
        return batch, batch_loss

    def step_bograd(self, buffer, mode: str = "negative") -> Tuple[torch.Tensor, float]:
        """One BoGrad-projected SGD step on a sampled batch.

        `buffer` is a list-like (deque) of past *raw* SGD updates u_i = -lr * g_i.
        We project the candidate update sequentially against each buffered past
        update — Gram-Schmidt subtraction, matching production BoGrad's
        update-stage `negative`-mode recipe. In negative mode we only subtract
        the projection when the inner product is < 0 (destructive overlap).
        After applying, we append the PRE-projection raw update to the buffer.

        Returns (batch_indices, batch_loss).
        """
        batch = self.sample_batch()
        diffs = self.theta.unsqueeze(0) - self.centres[batch]
        batch_grad = diffs.mean(dim=0)
        batch_loss = float((0.5 * (diffs ** 2).sum(dim=1).mean()).item())

        # Candidate update — what vanilla SGD would apply.
        raw_delta = -self.lr * batch_grad
        delta = raw_delta.clone()

        # Sequential Gram-Schmidt against past updates.
        for past in buffer:
            denom = float(torch.dot(past, past).item())
            if denom < 1e-12:
                continue
            dot_val = float(torch.dot(delta, past).item())
            if mode == "negative" and dot_val >= 0:
                continue
            if mode == "positive" and dot_val <= 0:
                continue
            delta = delta - (dot_val / denom) * past

        self.theta = self.theta + delta
        # Buffer the RAW update (pre-projection), matching production convention.
        buffer.append(raw_delta.clone())
        return batch, batch_loss

    # ---- info ---------------------------------------------------------
    def config(self) -> Dict:
        return {
            "n_groups": self.n_groups,
            "R": self.R,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "init_position": self.theta.tolist(),
            "centres": self.centres.tolist(),
        }
