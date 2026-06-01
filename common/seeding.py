"""
common.seeding — deterministic seeding and paired data-order generation.

docs/experiment_design.md §7: for the Chapter-5-style bakeoff, every method
compared within a trial must see the *exact same* mini-batch partitioning and
order. `paired_shuffle` produces that order deterministically from
(seed, trial_index); a `PairedBatchSampler` turns it into a DataLoader sampler
so baseline / COSGD / BoGrad / dropout / GradDrop are perfectly aligned.

Entry points
------------
  seed_everything(seed, deterministic=True)
  trial_seed(base_seed, trial_index)          -> int
  paired_shuffle(n, epochs, seed, trial)      -> np.ndarray [epochs, n]
  PairedBatchSampler(...)                      torch Sampler yielding the same
                                               index stream for every method
  order_hash(order)                            -> str  (audit two methods matched)
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Iterator, List

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (CPU+CUDA). With deterministic=True also
    set cudnn deterministic / disable benchmark for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def trial_seed(base_seed: int, trial_index: int) -> int:
    """Per-trial seed = base_seed + trial_index (docs §7)."""
    return int(base_seed) + int(trial_index)


def paired_shuffle(n: int, epochs: int, seed: int, trial_index: int = 0) -> np.ndarray:
    """Return an [epochs, n] int array of shuffled indices.

    Deterministic in (seed, trial_index): all methods in trial k call this with
    the same arguments and therefore train on identical batches each epoch.
    Uses a dedicated numpy Generator so it never perturbs global RNG state.
    """
    rng = np.random.default_rng(trial_seed(seed, trial_index))
    out = np.empty((epochs, n), dtype=np.int64)
    base = np.arange(n)
    for e in range(epochs):
        perm = base.copy()
        rng.shuffle(perm)
        out[e] = perm
    return out


class PairedBatchSampler:
    """A torch-compatible batch sampler driven by a precomputed per-epoch order.

    Construct once per (seed, trial); call `set_epoch(e)` before each epoch.
    Yields lists of indices (batches). drop_last mirrors DataLoader semantics.
    Because the order array is shared across methods, two runs with the same
    sampler see identical batches — the paired-trial guarantee.
    """

    def __init__(self, order: np.ndarray, batch_size: int, drop_last: bool = False):
        self.order = np.asarray(order)
        if self.order.ndim != 2:
            raise ValueError("order must be [epochs, n]")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.epochs, self.n = self.order.shape
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not (0 <= epoch < self.epochs):
            raise IndexError(f"epoch {epoch} out of range [0,{self.epochs})")
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        idx = self.order[self._epoch]
        n_full = self.n // self.batch_size
        for b in range(n_full):
            yield idx[b * self.batch_size:(b + 1) * self.batch_size].tolist()
        if not self.drop_last and self.n % self.batch_size:
            yield idx[n_full * self.batch_size:].tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size


def order_hash(order: np.ndarray) -> str:
    """Stable hash of an index order — lets a test assert two methods used the
    exact same data stream within a trial."""
    return hashlib.sha256(np.ascontiguousarray(order, dtype=np.int64).tobytes()).hexdigest()[:12]


__all__ = [
    "seed_everything",
    "trial_seed",
    "paired_shuffle",
    "PairedBatchSampler",
    "order_hash",
]
