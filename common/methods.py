"""
common.methods — the single registry that builds every (base_optimizer × method)
cell used across the ablations and the bakeoff.

One place constructs methods, so every experiment is identical and the full
optimizer set meets the five method arms consistently.

Base optimizers
---------------
  sgd       torch.optim.SGD          (momentum optional)
  signsgd   common.optimizers.SignSGD (momentum optional)
  rmsprop   torch.optim.RMSprop
  adam      torch.optim.Adam

Method arms
-----------
  baseline  the base optimizer, untouched
  bograd    BoGrad wrap, project_stage="update" FIXED by design (FocusedWork §04:
            the between-batch objects are the applied updates u_t)
  cosgd     COSGD wrap (per-class Gram-Schmidt)        -> step_kind="per_class"
  graddrop  GradDrop wrap (per-class sign-purity)      -> step_kind="per_class"
  dropout   base optimizer + activation dropout in the model (model change)

`build_method(method, base, hp=...) -> MethodSpec`. The Trainer builds the model
first (applying `spec.model_kwargs`, e.g. dropout_p), then calls
`spec.optimizer_factory(model, criterion)`. `spec.step_kind` tells the loop
whether to use the standard `.step()` or the per-class `.step(x, y, unique(y))`.

`hp` overrides defaults (lr, momentum, weight_decay, betas, alpha, K,
projection_mode, dropout_p, cosgd_method, class_order, prenormalize, combine, …)
so common.tuning can drive the search without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from common.optimizers import BoGrad, COSGD, GradDrop, SignSGD

BASE_OPTIMIZERS = ("sgd", "signsgd", "rmsprop", "adam")
METHODS = ("baseline", "bograd", "cosgd", "graddrop", "dropout")

# Base optimizers that take a `momentum` kwarg.
_MOMENTUM_BASES = {"sgd", "signsgd"}


@dataclass
class MethodSpec:
    name: str                       # method arm
    base: str                       # base optimizer key
    step_kind: str                  # "standard" | "per_class"
    optimizer_factory: Callable[[torch.nn.Module, Optional[torch.nn.Module]], Optimizer]
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    hp: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base optimizer resolution
# ---------------------------------------------------------------------------
def _base_cls_and_kwargs(base: str, hp: Dict[str, Any]) -> Tuple[Type[Optimizer], Dict[str, Any]]:
    """Return (optimizer_cls, kwargs) for a base optimizer, honouring hp overrides.

    `lr` is always included. Momentum/weight_decay/betas/alpha are filled per
    family with sensible defaults that hp can override.
    """
    if base not in BASE_OPTIMIZERS:
        raise ValueError(f"Unknown base optimizer '{base}'. Available: {BASE_OPTIMIZERS}")

    lr = hp.get("lr", _default_lr(base))
    wd = hp.get("weight_decay", 0.0)

    if base == "sgd":
        return torch.optim.SGD, dict(
            lr=lr, momentum=hp.get("momentum", 0.9), weight_decay=wd,
            nesterov=hp.get("nesterov", False),
        )
    if base == "signsgd":
        return SignSGD, dict(lr=lr, momentum=hp.get("momentum", 0.0), weight_decay=wd)
    if base == "rmsprop":
        return torch.optim.RMSprop, dict(
            lr=lr, alpha=hp.get("alpha", 0.99), momentum=hp.get("momentum", 0.0),
            weight_decay=wd, eps=hp.get("eps", 1e-8),
        )
    # adam
    return torch.optim.Adam, dict(
        lr=lr, betas=hp.get("betas", (0.9, 0.999)), weight_decay=wd,
        eps=hp.get("eps", 1e-8),
    )


def _default_lr(base: str) -> float:
    """Reasonable default LRs (overridden by tuning). SGD/SignSGD higher,
    adaptive methods lower."""
    return {"sgd": 0.05, "signsgd": 1e-3, "rmsprop": 1e-3, "adam": 1e-3}[base]


# ---------------------------------------------------------------------------
# Method builders
# ---------------------------------------------------------------------------
def build_method(
    method: str,
    base: str,
    *,
    hp: Optional[Dict[str, Any]] = None,
) -> MethodSpec:
    """Construct a MethodSpec for one (method × base) cell."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {METHODS}")
    hp = dict(hp or {})
    base_cls, base_kwargs = _base_cls_and_kwargs(base, hp)
    label = f"{base}+{method}" if method != "baseline" else base

    # --- baseline -------------------------------------------------------
    if method == "baseline":
        def factory(model, criterion=None, _cls=base_cls, _kw=base_kwargs):
            return _cls(model.parameters(), **_kw)
        return MethodSpec(method, base, "standard", factory, {}, label, hp)

    # --- dropout (model change, base optimizer untouched) ---------------
    if method == "dropout":
        def factory(model, criterion=None, _cls=base_cls, _kw=base_kwargs):
            return _cls(model.parameters(), **_kw)
        model_kwargs = {"dropout_p": hp.get("dropout_p", 0.3)}
        return MethodSpec(method, base, "standard", factory, model_kwargs, label, hp)

    # --- bograd (update-stage fixed by design) --------------------------
    if method == "bograd":
        def factory(model, criterion=None, _cls=base_cls, _kw=base_kwargs, _hp=hp):
            return BoGrad(
                model.parameters(),
                base_optimizer_cls=_cls,
                buffer_size=_hp.get("K", 32),
                project_stage="update",                       # FIXED by design
                projection_mode=_hp.get("projection_mode", "negative"),
                orth_method=_hp.get("orth_method", "sequential"),
                projection_scope=_hp.get("projection_scope", "per_tensor"),
                projection_strength=_hp.get("projection_strength", 1.0),
                preserve_magnitude=_hp.get("preserve_magnitude", False),
                random_projection=_hp.get("random_projection", False),
                min_projection_dim=_hp.get("min_projection_dim", 0),
                buffer_dtype=_hp.get("buffer_dtype", None),
                collect_stats=_hp.get("collect_stats", False),
                **_kw,
            )
        return MethodSpec(method, base, "standard", factory, {}, label, hp)

    # --- cosgd (per-class Gram-Schmidt) ---------------------------------
    # Canonical defaults are the scrutiny winners (PreDiscovery/research/
    # 03_cosgd_scrutiny/FINDINGS.md): combine="freq" (best + LR-robust; "sum"
    # diverges), preserve_magnitude=True (decouples LR), and conflict_gate=True
    # with threshold ~0.1 (the round-2 breakthrough: orthogonalise only
    # structured-conflict steps -> flips COSGD from a net loss to a net win).
    if method == "cosgd":
        def factory(model, criterion, _cls=base_cls, _kw=base_kwargs, _hp=hp):
            return COSGD(
                model.parameters(),
                base_optimizer_cls=_cls,
                model=model, criterion=criterion,
                orthogonalization_method=_hp.get("cosgd_method", "modified_gs_negative"),
                step_method=_hp.get("step_method", "single_forward"),
                class_order=_hp.get("class_order", "fixed"),
                prenormalize=_hp.get("prenormalize", False),
                combine=_hp.get("combine", "freq"),
                preserve_magnitude=_hp.get("preserve_magnitude", True),
                conflict_gate=_hp.get("conflict_gate", True),
                conflict_threshold=_hp.get("conflict_threshold", 0.1),
                collect_timing=_hp.get("collect_timing", False),
                **_kw,
            )
        return MethodSpec(method, base, "per_class", factory, {}, label, hp)

    # --- graddrop (per-class sign-purity) -------------------------------
    if method == "graddrop":
        def factory(model, criterion, _cls=base_cls, _kw=base_kwargs, _hp=hp):
            return GradDrop(
                model.parameters(),
                base_optimizer_cls=_cls,
                model=model, criterion=criterion,
                step_method=_hp.get("step_method", "single_forward"),
                leak=_hp.get("leak", 0.0),
                collect_timing=_hp.get("collect_timing", False),
                **_kw,
            )
        return MethodSpec(method, base, "per_class", factory, {}, label, hp)

    raise AssertionError("unreachable")


def all_cells(methods=METHODS, bases=BASE_OPTIMIZERS):
    """Yield (method, base) pairs. `dropout` is a model regulariser and is
    base-agnostic in spirit, but we still cross it with every base so the
    bakeoff matrix is rectangular and comparable."""
    for base in bases:
        for method in methods:
            yield method, base


__all__ = ["MethodSpec", "build_method", "all_cells", "BASE_OPTIMIZERS", "METHODS"]
