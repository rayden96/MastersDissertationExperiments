"""
Interference metrics module — paper-ready implementation.

Implements the measurements defined in PreDiscovery/FocusedWork/03_*.md and
PreDiscovery/FocusedWork/04_*.md, generalised to any objective that exposes:

- a way to compute per-subgroup gradients on a given batch (for §03 inter-batch);
- a way to compute the full-batch (or large-reference) gradient (for the
  useful/wasted decomposition and the loss-decrease deficit);
- a way to compute the loss on the reference set (for calibration).

The original `PreDiscovery/FocusedWork/diagnostics.py` was tied to torchvision
image classification (per-class subgradients on the current batch). This module
factors the measurement out of that assumption so the same metrics apply to
a 2D toy problem, a CIFAR-10 image classifier, and a CIFAR-100 ResNet.
"""

from interference.problem import InterferenceProblem
from interference.meter import InterferenceMeter
from interference.summary import correlate, summarize_run
from interference.torch_classification import TorchClassificationProblem

__all__ = [
    "InterferenceProblem",
    "InterferenceMeter",
    "TorchClassificationProblem",
    "correlate",
    "summarize_run",
]
