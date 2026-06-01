"""
Interference framework diagnostics.

  ClassProbeSet               (per_class_probe) — balanced probe set + per-class eval
  ForgettingTracker           (interference)    — per-class forgetting accumulator
  WastedWorkTracker           (interference)    — net-vs-summed-step efficiency ratio
  InterferenceTracker         (interference)    — main wrapper, aggregates everything

See research/01_interference_framework/framework.md for the formal definitions
and motivation. The framework operationalises the question "does short-horizon
gradient interference slow training?" into measurable quantities that can be
captured during any training run.
"""

from common.diagnostics.per_class_probe import ClassProbeSet
from common.diagnostics.interference import (
    ForgettingTracker,
    InterferenceTracker,
    PairwiseAlignmentTracker,
    WastedWorkTracker,
)

__all__ = [
    "ClassProbeSet",
    "ForgettingTracker",
    "InterferenceTracker",
    "PairwiseAlignmentTracker",
    "WastedWorkTracker",
]
