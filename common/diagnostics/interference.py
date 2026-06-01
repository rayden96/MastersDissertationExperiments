"""
Interference framework — measurement library.

This module operationalises the metrics defined in
research/01_interference_framework/framework.md.

Three sub-trackers and one wrapper:

  ForgettingTracker     — per-class loss-increase events across steps.
  WastedWorkTracker     — net displacement vs summed step magnitudes (efficiency).
  InterferenceTracker   — top-level wrapper. Wires up cosine alignments, the
                          two trackers above, optional per-class probe
                          evaluations, and optional full-batch gradient
                          reference computation.

Usage:

    from common.diagnostics import InterferenceTracker, ClassProbeSet
    probe = ClassProbeSet(test_set, num_classes=10, n_per_class=64, device=device)
    tracker = InterferenceTracker(
        model, criterion,
        probe_set=probe,
        log_every=10,
        wasted_work_K=32,
        device=device,
    )

    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x); loss = criterion(logits, y); loss.backward()
        tracker.before_step()
        optimizer.step()
        tracker.after_step(loss=loss.item())

    history = tracker.get_history()    # list of per-step dicts
    summary = tracker.summary()        # aggregated metrics
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from common.diagnostics.per_class_probe import ClassProbeSet


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _flatten_params(model: nn.Module) -> torch.Tensor:
    """Concatenate all trainable parameters into one 1-D tensor (detached)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def _flatten_grads(model: nn.Module) -> Optional[torch.Tensor]:
    """Concatenate all gradients into one 1-D tensor. Returns None if no grads exist."""
    pieces = []
    for p in model.parameters():
        if p.grad is None or not p.requires_grad:
            continue
        pieces.append(p.grad.detach().reshape(-1))
    if not pieces:
        return None
    return torch.cat(pieces)


def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    na = a.norm()
    nb = b.norm()
    if na.item() <= eps or nb.item() <= eps:
        return float("nan")
    return float((a @ b / (na * nb)).item())


# ----------------------------------------------------------------------------
# Forgetting tracker
# ----------------------------------------------------------------------------
class ForgettingTracker:
    """Per-class forgetting-event accumulator.

    A forgetting event for class c at step t is defined as
        per_class_loss(c, t) > per_class_loss(c, t-1)
    with magnitude
        Δ_{t,c} = max(0, loss(c, t) - loss(c, t-1)).

    Aggregates:
      - event_count[c]               : # forgetting steps for class c
      - cumulative_magnitude[c]      : Σ Δ across all forgetting events for c
      - per_step_records             : list of {step, events, magnitude, per-class-deltas}

    Single-step ``update`` returns a small dict suitable for logging into
    a per-step trace.
    """

    def __init__(self):
        self.event_count: Dict[int, int] = {}
        self.cumulative_magnitude: Dict[int, float] = {}
        # In-batch / out-of-batch decomposition (populated only when
        # batch_classes is supplied to .update()).
        self.in_batch_event_count: Dict[int, int] = {}
        self.in_batch_cumulative_magnitude: Dict[int, float] = {}
        self.out_of_batch_event_count: Dict[int, int] = {}
        self.out_of_batch_cumulative_magnitude: Dict[int, float] = {}
        self.per_step_records: List[Dict[str, Any]] = []
        self._prev_per_class_loss: Optional[Dict[int, float]] = None

    def update(
        self,
        step: int,
        per_class_loss: Dict[int, float],
        batch_classes: Optional[set] = None,
    ) -> Dict[str, float]:
        """Update with the per-class loss after the current step.

        Parameters
        ----------
        step : int
            Global step index.
        per_class_loss : dict[int, float]
            Loss per class on the probe set.
        batch_classes : set[int], optional
            Classes that were present in the mini-batch at this step.
            When provided, forgetting is partitioned into:
              - **in-batch**: regression on a class the gradient saw. Could
                be from gradient noise on the class itself OR from this step
                being too aggressive. Ambiguous source.
              - **out-of-batch**: regression on a class the gradient had
                NO information about. Cannot be from noise on that class's
                loss estimate (the step's gradient saw nothing of class c);
                must be parameter drift caused by other classes' gradients.
                This is "interference proper".
            See framework.md §8 Finding F6 for motivation.
        """
        events_this_step = 0
        magnitude_this_step = 0.0
        events_in = events_out = 0
        mag_in = mag_out = 0.0
        per_class_delta: Dict[int, float] = {}

        if self._prev_per_class_loss is not None:
            for c, loss_now in per_class_loss.items():
                if c not in self._prev_per_class_loss:
                    continue
                loss_prev = self._prev_per_class_loss[c]
                if not (math.isfinite(loss_now) and math.isfinite(loss_prev)):
                    continue
                delta = loss_now - loss_prev
                per_class_delta[c] = delta
                if delta > 0:
                    self.event_count[c] = self.event_count.get(c, 0) + 1
                    self.cumulative_magnitude[c] = self.cumulative_magnitude.get(c, 0.0) + delta
                    events_this_step += 1
                    magnitude_this_step += delta
                    if batch_classes is not None:
                        if c in batch_classes:
                            events_in += 1
                            mag_in += delta
                            self.in_batch_event_count[c] = self.in_batch_event_count.get(c, 0) + 1
                            self.in_batch_cumulative_magnitude[c] = (
                                self.in_batch_cumulative_magnitude.get(c, 0.0) + delta
                            )
                        else:
                            events_out += 1
                            mag_out += delta
                            self.out_of_batch_event_count[c] = self.out_of_batch_event_count.get(c, 0) + 1
                            self.out_of_batch_cumulative_magnitude[c] = (
                                self.out_of_batch_cumulative_magnitude.get(c, 0.0) + delta
                            )

        self._prev_per_class_loss = dict(per_class_loss)

        if per_class_delta:
            rec: Dict[str, Any] = {
                "step": step,
                "forgetting_events": events_this_step,
                "forgetting_magnitude_total": magnitude_this_step,
                "per_class_delta": per_class_delta,
            }
            if batch_classes is not None:
                rec["forgetting_events_in_batch"] = events_in
                rec["forgetting_magnitude_in_batch"] = mag_in
                rec["forgetting_events_out_of_batch"] = events_out
                rec["forgetting_magnitude_out_of_batch"] = mag_out
                rec["batch_classes"] = sorted(int(c) for c in batch_classes)
            self.per_step_records.append(rec)

        out: Dict[str, float] = {
            "forgetting_events": float(events_this_step),
            "forgetting_magnitude_total": float(magnitude_this_step),
        }
        if batch_classes is not None:
            out["forgetting_events_in_batch"] = float(events_in)
            out["forgetting_magnitude_in_batch"] = float(mag_in)
            out["forgetting_events_out_of_batch"] = float(events_out)
            out["forgetting_magnitude_out_of_batch"] = float(mag_out)
        return out

    def summary(self) -> Dict[str, Any]:
        s: Dict[str, Any] = {
            "classes_tracked": sorted(self.event_count.keys()),
            "total_events_per_class": dict(self.event_count),
            "cumulative_magnitude_per_class": dict(self.cumulative_magnitude),
            "total_events": int(sum(self.event_count.values())),
            "total_magnitude": float(sum(self.cumulative_magnitude.values())),
            "n_step_records": len(self.per_step_records),
        }
        # Out-of-batch / in-batch summaries (only meaningful if batch_classes
        # was provided during update calls).
        if self.in_batch_event_count or self.out_of_batch_event_count:
            s["in_batch_total_events"] = int(sum(self.in_batch_event_count.values()))
            s["in_batch_total_magnitude"] = float(sum(self.in_batch_cumulative_magnitude.values()))
            s["out_of_batch_total_events"] = int(sum(self.out_of_batch_event_count.values()))
            s["out_of_batch_total_magnitude"] = float(sum(self.out_of_batch_cumulative_magnitude.values()))
        return s


# ----------------------------------------------------------------------------
# Wasted-work tracker
# ----------------------------------------------------------------------------
class WastedWorkTracker:
    """Step-efficiency ratio over a rolling window.

    For window size K:
        wasted_work_ratio_K(t) = ‖θ_t − θ_{t−K}‖ / Σ_{i=t−K+1..t} ‖step_i‖

    Ratio close to 1 ⇒ steps stack up productively (no cancellation).
    Ratio close to 0 ⇒ steps cancel each other (lots of back-and-forth).

    Stores parameter-vector snapshots over the window — memory cost is
    O(K × n_params). Typical K ≤ 64 is fine for CIFAR-scale models.
    """

    def __init__(self, window_K: int = 32):
        self.K = int(window_K)
        self.step_norms: Deque[float] = deque(maxlen=self.K)
        self.params_history: Deque[torch.Tensor] = deque(maxlen=self.K + 1)
        self.records: List[Dict[str, float]] = []

    def update(self, step: int, params_after: torch.Tensor, step_norm: float) -> Dict[str, float]:
        self.step_norms.append(float(step_norm))
        self.params_history.append(params_after.detach().clone())

        record: Dict[str, float] = {"step": float(step)}
        if len(self.params_history) >= 2 and len(self.step_norms) >= self.K:
            net = float((self.params_history[-1] - self.params_history[0]).norm().item())
            total_step = float(sum(self.step_norms))
            ratio = net / total_step if total_step > 1e-12 else float("nan")
            record["wasted_work_ratio"] = ratio
            record["net_displacement"] = net
            record["total_step_norm"] = total_step
            self.records.append(record)
        return record

    def summary(self) -> Dict[str, Any]:
        ratios = [r["wasted_work_ratio"] for r in self.records
                  if math.isfinite(r.get("wasted_work_ratio", float("nan")))]
        if not ratios:
            return {"mean_ratio": float("nan"), "n": 0, "window_K": self.K}
        mean = sum(ratios) / len(ratios)
        std = (sum((r - mean) ** 2 for r in ratios) / max(len(ratios) - 1, 1)) ** 0.5 if len(ratios) > 1 else 0.0
        return {
            "mean_ratio": mean,
            "std_ratio": std,
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "n": len(ratios),
            "window_K": self.K,
        }


# ----------------------------------------------------------------------------
# Pairwise alignment tracker — distribution of cos(x_t, x_{t-k}) for k = 1..K
# ----------------------------------------------------------------------------
class PairwiseAlignmentTracker:
    """Track the *distribution* of pairwise cosine alignment within a K-step window.

    For each logged step t, this computes cos(x_t, x_{t-k}) for every
    k = 1..K (where K is the buffer size), against either:
      - past gradients (passed in via ``grad=...``), or
      - past applied updates (passed in via ``update=...``),
    or both.

    Per-step aggregates produced (prefixed ``grad_`` or ``update_`` in the
    output dict, depending on which was passed):

      n_pairs            number of buffered vectors compared against
      n_positive         count of pairs with cos > 0
      n_negative         count of pairs with cos < 0
      frac_positive      n_positive / n_pairs
      frac_negative      n_negative / n_pairs
      mean_cos           average cosine across all K pairs
      mean_abs_cos       average |cos|
      mean_positive_cos  average cosine across only positive pairs
      mean_negative_cos  average cosine across only negative pairs
      max_cos, min_cos   distributional extremes

    Run-level summary then averages each of these across all logged steps.

    This generalises the lag-1-only metrics ``cos_g_prev`` and ``cos_u_prev``
    (which compare only consecutive steps) to the full K-step window — i.e.
    the same window BoGrad orthogonalises against. It directly answers
    "how often do gradients conflict or align, and how strongly?" across the
    actual buffer span, not just at lag 1.
    """

    def __init__(self, buffer_K: int = 32):
        self.K = int(buffer_K)
        self.grad_buffer: Deque[torch.Tensor] = deque(maxlen=self.K)
        self.update_buffer: Deque[torch.Tensor] = deque(maxlen=self.K)
        self.records: List[Dict[str, Any]] = []

    def update(
        self,
        step: int,
        grad: Optional[torch.Tensor] = None,
        update: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute pairwise stats vs the buffer, then push current vector.

        Returns a dict with keys ``pairwise_grad_<stat>`` and/or
        ``pairwise_update_<stat>`` suitable for inclusion in a per-step row.
        """
        record: Dict[str, Any] = {"step": step}
        out: Dict[str, float] = {}

        if grad is not None and grad.numel() > 0:
            stats = self._compute_pairwise_stats(grad, self.grad_buffer)
            for k, v in stats.items():
                record[f"grad_{k}"] = v
                out[f"pairwise_grad_{k}"] = v
            self.grad_buffer.append(grad.detach().clone())

        if update is not None and update.numel() > 0:
            stats = self._compute_pairwise_stats(update, self.update_buffer)
            for k, v in stats.items():
                record[f"update_{k}"] = v
                out[f"pairwise_update_{k}"] = v
            self.update_buffer.append(update.detach().clone())

        # Record only if something useful was computed.
        if any(k != "step" for k in record.keys()):
            self.records.append(record)
        return out

    def _compute_pairwise_stats(
        self, x: torch.Tensor, buffer: Deque[torch.Tensor],
    ) -> Dict[str, float]:
        if not buffer:
            return {}
        x_flat = x.flatten()
        x_norm = x_flat.norm()
        if x_norm.item() < 1e-12:
            return {}

        cos_values: List[float] = []
        for b in buffer:
            b_flat = b.flatten()
            if b_flat.numel() != x_flat.numel():
                continue
            b_norm = b_flat.norm()
            if b_norm.item() < 1e-12:
                continue
            cos_v = float((x_flat @ b_flat / (x_norm * b_norm)).item())
            cos_values.append(cos_v)

        if not cos_values:
            return {}

        n = len(cos_values)
        positive = [v for v in cos_values if v > 0]
        negative = [v for v in cos_values if v < 0]
        return {
            "n_pairs": float(n),
            "n_positive": float(len(positive)),
            "n_negative": float(len(negative)),
            "frac_positive": len(positive) / n,
            "frac_negative": len(negative) / n,
            "mean_cos": sum(cos_values) / n,
            "mean_abs_cos": sum(abs(v) for v in cos_values) / n,
            "mean_positive_cos": sum(positive) / len(positive) if positive else 0.0,
            "mean_negative_cos": sum(negative) / len(negative) if negative else 0.0,
            "max_cos": max(cos_values),
            "min_cos": min(cos_values),
        }

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"n_records": 0, "K": self.K}
        keys: set = set()
        for r in self.records:
            for k in r.keys():
                if k != "step":
                    keys.add(k)
        agg: Dict[str, float] = {}
        for k in sorted(keys):
            vals = [r[k] for r in self.records
                    if k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k])]
            if vals:
                agg[f"{k}_mean"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    m = agg[f"{k}_mean"]
                    agg[f"{k}_std"] = (
                        sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                    ) ** 0.5
        agg["n_records"] = len(self.records)
        agg["K"] = self.K
        return agg


# ----------------------------------------------------------------------------
# Top-level tracker
# ----------------------------------------------------------------------------
class InterferenceTracker:
    """Wraps every interference metric for a training run.

    Lifecycle:
      - Construct once per run with model + criterion (+ optional probe set,
        full-batch loader).
      - Call ``before_step()`` after ``loss.backward()`` and before ``optimizer.step()``.
      - Call ``after_step(loss=loss.item())`` after ``optimizer.step()``.
      - At end: ``get_history()`` for per-step rows, ``summary()`` for aggregated.

    Parameters
    ----------
    model : nn.Module
    criterion : callable taking (logits, y) -> loss tensor
    probe_set : ClassProbeSet, optional
        If provided, per-class probe evaluation runs every ``probe_every`` steps
        and feeds ForgettingTracker.
    full_batch_loader : DataLoader, optional
        If provided AND ``full_grad_every`` is set, the full-dataset gradient
        is computed every ``full_grad_every`` steps and ``cos(g_batch, g_full)``
        is logged. Expensive — opt in only for diagnostic-only runs.
    log_every : int, default 10
        Per-step row is recorded only on these steps. The cheap step-step
        cosine metrics still require this stride.
    probe_every : int, optional
        Default = ``log_every``. Set to a multiple of ``log_every`` to evaluate
        the probe less frequently than the cosine metrics. Set to ``None`` to
        disable per-class evaluation entirely (forgetting metrics also disabled).
    full_grad_every : int, optional
        Default ``None`` (disabled).
    wasted_work_K : int, default 32
        Window size for the wasted-work ratio.
    device : torch.device or str
    trajectory_lags : list of int, default [1, 4, 16]
        Lag offsets k for cos(Δθ_t, Δθ_{t-k}). k=1 duplicates ``cos_u_prev`` —
        kept for explicit comparison.
    pairwise_K : int, default 32
        Window size for the PairwiseAlignmentTracker. Per logged step,
        cos(x_t, x_{t-k}) for k = 1..K is computed against both gradient
        and update buffers; aggregates fraction-positive, fraction-negative,
        mean-positive-cos, mean-negative-cos, etc. Set to 0 to disable.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        *,
        probe_set: Optional[ClassProbeSet] = None,
        full_batch_loader: Optional[DataLoader] = None,
        log_every: int = 10,
        probe_every: Optional[int] = None,
        full_grad_every: Optional[int] = None,
        wasted_work_K: int = 32,
        device="cpu",
        trajectory_lags: Optional[List[int]] = None,
        pairwise_K: int = 32,
    ):
        self.model = model
        self.criterion = criterion
        self.probe_set = probe_set
        self.full_batch_loader = full_batch_loader
        self.log_every = int(log_every)
        self.probe_every = int(probe_every) if probe_every is not None else int(log_every)
        self.probe_disabled = probe_every is None and probe_set is None
        self.full_grad_every = int(full_grad_every) if full_grad_every is not None else None
        self.wasted_work_K = int(wasted_work_K)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.trajectory_lags = list(trajectory_lags or [1, 4, 16])

        # Per-step state
        self.step_idx = 0
        # Captured on every step (cheap) — used by WastedWorkTracker which needs
        # every step to compute a meaningful displacement-vs-step-norms ratio.
        self._params_before_step: Optional[torch.Tensor] = None
        # Captured only on logged steps — used by the more expensive metrics.
        self.params_before: Optional[torch.Tensor] = None
        self.grad_at_step: Optional[torch.Tensor] = None
        self.prev_grad: Optional[torch.Tensor] = None
        self.prev_update: Optional[torch.Tensor] = None
        max_lag = max(self.trajectory_lags) if self.trajectory_lags else 0
        self.update_history: Deque[torch.Tensor] = deque(maxlen=max_lag + 1)

        # Sub-trackers
        self.forgetting = ForgettingTracker()
        self.wasted_work = WastedWorkTracker(window_K=self.wasted_work_K)
        self.pairwise = (
            PairwiseAlignmentTracker(buffer_K=int(pairwise_K))
            if int(pairwise_K) > 0 else None
        )

        # Cumulative summaries used for run-level normalised metrics
        # (see framework.md §4.2 — forgetting-per-step / per-progress).
        self._cum_step_norm: float = 0.0
        self._cum_forgetting_magnitude: float = 0.0
        self._initial_loss: Optional[float] = None
        self._latest_loss: Optional[float] = None

        # Net displacement: snapshot of θ at run start so we can report
        # ‖θ_end − θ_start‖ per framework.md §4.3.
        self._theta_start: Optional[torch.Tensor] = None

        # Output
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal scheduling
    # ------------------------------------------------------------------
    def _should_log(self) -> bool:
        return (self.step_idx % self.log_every) == 0

    def _should_probe(self) -> bool:
        if self.probe_set is None:
            return False
        return (self.step_idx % self.probe_every) == 0

    def _should_full_grad(self) -> bool:
        return (
            self.full_batch_loader is not None
            and self.full_grad_every is not None
            and (self.step_idx % self.full_grad_every) == 0
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def before_step(self) -> None:
        """Snapshot params (every step, for wasted-work) + grads (only on log steps).

        WastedWorkTracker requires every-step measurements to make the ratio
        meaningful: it tracks net displacement over K steps vs. the sum of K
        per-step norms. If we only updated it on log steps, the numerator and
        denominator would be on different cadences (the displacement spans K *
        log_every actual steps; the denominator only sums K single-step norms),
        producing meaningless ratios > 1.
        """
        # Always capture for wasted-work (cheap — one tensor copy).
        self._params_before_step = _flatten_params(self.model)
        # Capture grads only on logged steps (only needed for cosine metrics).
        if self._should_log():
            self.params_before = self._params_before_step.clone()
            self.grad_at_step = _flatten_grads(self.model)

    @torch.no_grad()
    def after_step(
        self,
        loss: Optional[float] = None,
        batch_classes: Optional[set] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compute and record metrics. Call after ``optimizer.step()``.

        Parameters
        ----------
        loss : float, optional
            Training-set loss for this step (any normalisation is fine; used
            for the "loss progress" run-level metric).
        batch_classes : set[int], optional
            Set of class labels present in the current mini-batch. When
            provided, the per-class forgetting metric is partitioned into
            in-batch and out-of-batch components — the out-of-batch component
            is the cleaner "interference proper" signal (a class regressed
            even though the step had no gradient information about it).
            See framework.md §8 Finding F6.
        """
        params_after = _flatten_params(self.model)

        # Snapshot starting θ on first call (used for net-displacement summary).
        if self._theta_start is None:
            self._theta_start = (
                self._params_before_step.clone()
                if self._params_before_step is not None
                else params_after.clone()
            )

        # Always update wasted-work (every step).
        single_step_norm = 0.0
        if self._params_before_step is not None:
            single_step_norm = float((params_after - self._params_before_step).norm().item())
            ww_record_every = self.wasted_work.update(self.step_idx, params_after, single_step_norm)
            self._cum_step_norm += single_step_norm
        else:
            ww_record_every = {}
        self._params_before_step = None

        # Track latest loss every step so we can compute per-progress forgetting
        # at the end. Initial loss is captured the first time we get a value.
        if loss is not None and math.isfinite(float(loss)):
            if self._initial_loss is None:
                self._initial_loss = float(loss)
            self._latest_loss = float(loss)

        if not self._should_log():
            self.step_idx += 1
            return None

        update = (params_after - self.params_before) if self.params_before is not None else None
        grad = self.grad_at_step

        row: Dict[str, Any] = {
            "step": int(self.step_idx),
            "loss": float(loss) if loss is not None else float("nan"),
        }

        if grad is not None:
            row["g_norm"] = float(grad.norm().item())
        u_norm = float(update.norm().item()) if update is not None else 0.0
        if update is not None:
            row["u_norm"] = u_norm

        # Step-step alignments
        if grad is not None and self.prev_grad is not None and grad.numel() == self.prev_grad.numel():
            row["cos_g_prev"] = _safe_cos(grad, self.prev_grad)
        if (
            update is not None
            and self.prev_update is not None
            and update.numel() == self.prev_update.numel()
        ):
            row["cos_u_prev"] = _safe_cos(update, self.prev_update)
        if grad is not None and update is not None and grad.numel() == update.numel():
            row["cos_u_neg_g"] = _safe_cos(update, -grad)

        # Trajectory lags
        if update is not None:
            self.update_history.append(update.clone())
            for lag in self.trajectory_lags:
                if len(self.update_history) > lag:
                    earlier = self.update_history[-lag - 1]
                    if earlier.numel() == update.numel():
                        row[f"cos_u_lag{lag}"] = _safe_cos(update, earlier)

        # Pairwise alignment statistics across the K-step window — distribution of
        # cos(x_t, x_{t-k}) for k = 1..K against both grad and update buffers.
        if self.pairwise is not None:
            pw_stats = self.pairwise.update(self.step_idx, grad=grad, update=update)
            for k, v in pw_stats.items():
                row[k] = v

        # Per-class probe + forgetting
        if self._should_probe():
            per_class = self.probe_set.evaluate(self.model, self.criterion)
            per_class_loss = {c: v["loss"] for c, v in per_class.items()}
            per_class_acc = {c: v["acc"] for c, v in per_class.items()}
            row["per_class_loss"] = per_class_loss
            row["per_class_acc"] = per_class_acc
            forgetting_step = self.forgetting.update(
                self.step_idx, per_class_loss, batch_classes=batch_classes,
            )
            row.update(forgetting_step)
            valid_losses = [v for v in per_class_loss.values() if math.isfinite(v)]
            valid_accs = [v for v in per_class_acc.values() if math.isfinite(v)]
            if valid_losses:
                row["probe_loss_mean"] = sum(valid_losses) / len(valid_losses)
            if valid_accs:
                row["probe_acc_mean"] = sum(valid_accs) / len(valid_accs)

            # Step-magnitude-normalised forgetting (framework.md §4.2):
            #   Δ^per-step_t = Σ_c Δ_{t,c} / ‖u_t‖.
            # Only meaningful on logged probe-steps and when we have a
            # non-trivial step.
            mag = forgetting_step.get("forgetting_magnitude_total", 0.0)
            self._cum_forgetting_magnitude += mag
            if single_step_norm > 1e-12:
                row["forgetting_per_unit_step"] = mag / single_step_norm

        # Full-batch gradient reference
        if self._should_full_grad():
            full_grad = self._compute_full_batch_grad()
            if full_grad is not None and grad is not None and full_grad.numel() == grad.numel():
                row["cos_g_full"] = _safe_cos(grad, full_grad)
                row["full_grad_norm"] = float(full_grad.norm().item())
                full_norm = full_grad.norm()
                if full_norm.item() > 1e-12:
                    f_unit = full_grad / full_norm
                    parallel = (grad @ f_unit) * f_unit
                    perp = grad - parallel
                    row["g_parallel_to_full_norm"] = float(parallel.norm().item())
                    row["g_perp_to_full_norm"] = float(perp.norm().item())

        # Wasted-work — already updated above (unconditional). Just stamp the
        # most recent record onto this logging row for convenience.
        for k, v in ww_record_every.items():
            if k == "step":
                continue
            row[f"ww_{k}"] = v

        # Carry state forward
        if grad is not None:
            self.prev_grad = grad.clone()
        if update is not None:
            self.prev_update = update.clone()
        self.params_before = None
        self.grad_at_step = None

        self.history.append(row)
        self.step_idx += 1
        return row

    # ------------------------------------------------------------------
    # Full-batch gradient reference (optional, expensive)
    # ------------------------------------------------------------------
    def _compute_full_batch_grad(self) -> Optional[torch.Tensor]:
        """Compute the full-dataset gradient at current parameters.

        The returned tensor is the mean per-example gradient. Restores the
        model's training/eval state and zero-grads at exit so the next training
        step is unaffected.
        """
        if self.full_batch_loader is None:
            return None
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        n_total = 0
        with torch.enable_grad():
            for x, y in self.full_batch_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                bs = y.size(0)
                # Weight by batch size; we'll divide by total at end to get mean.
                (loss * bs).backward()
                n_total += bs
            full_grad = _flatten_grads(self.model)
            if full_grad is not None and n_total > 0:
                full_grad = full_grad / n_total

        if was_training:
            self.model.train()
        # Re-zero so this doesn't pollute the next training step.
        self.model.zero_grad(set_to_none=True)
        return full_grad

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.history)

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}

        # Aggregate finite scalar metrics across all logged rows.
        scalar_keys: set = set()
        for r in self.history:
            for k, v in r.items():
                if k == "step":
                    continue
                if isinstance(v, (int, float)) and math.isfinite(v):
                    scalar_keys.add(k)

        agg: Dict[str, Any] = {}
        for k in sorted(scalar_keys):
            vals = [r[k] for r in self.history
                    if k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k])]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            agg[f"{k}_mean"] = mean
            if len(vals) > 1:
                std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                agg[f"{k}_std"] = std
            agg[f"{k}_n"] = len(vals)

        agg["forgetting_summary"] = self.forgetting.summary()
        agg["wasted_work_summary"] = self.wasted_work.summary()
        if self.pairwise is not None:
            agg["pairwise_summary"] = self.pairwise.summary()
        agg["n_logged_steps"] = len(self.history)

        # Run-level normalised metrics (framework.md §4.2, §4.3).
        agg["cum_step_norm"] = self._cum_step_norm
        agg["cum_forgetting_magnitude"] = self._cum_forgetting_magnitude

        # Pull OOB forgetting from the sub-tracker (only present when
        # batch_classes was supplied during update calls).
        forget_summary = self.forgetting.summary()
        oob_total = float(forget_summary.get("out_of_batch_total_magnitude", 0.0))
        agg["cum_forgetting_oob_magnitude"] = oob_total

        if self._cum_step_norm > 1e-12:
            # Σ Δ / Σ ‖u‖ — total wasted work per total parameter movement.
            agg["forgetting_per_unit_step_run"] = (
                self._cum_forgetting_magnitude / self._cum_step_norm
            )
            # OOB version — the cleaner cross-class interference signal,
            # normalised by total movement so we can compare across methods
            # with different step magnitudes (per F4).
            if oob_total > 0:
                agg["forgetting_oob_per_unit_step"] = oob_total / self._cum_step_norm

        if (
            self._initial_loss is not None
            and self._latest_loss is not None
            and abs(self._initial_loss - self._latest_loss) > 1e-12
        ):
            # Σ Δ / |L_init - L_final| — total wasted work per unit useful progress.
            # If the run regresses on training loss the denominator goes negative;
            # we report magnitude of progress.
            progress = abs(self._initial_loss - self._latest_loss)
            agg["forgetting_per_progress"] = self._cum_forgetting_magnitude / progress
            agg["loss_progress"] = self._initial_loss - self._latest_loss
            # OOB-per-progress: the most direct test of (W) — total
            # cross-class interference per unit useful progress.
            if oob_total > 0:
                agg["forgetting_oob_per_progress"] = oob_total / progress
        # Net displacement (θ_end − θ_start).
        if self._theta_start is not None:
            theta_now = _flatten_params(self.model)
            if theta_now.numel() == self._theta_start.numel():
                agg["net_displacement"] = float((theta_now - self._theta_start).norm().item())

        return agg


__all__ = [
    "InterferenceTracker",
    "ForgettingTracker",
    "WastedWorkTracker",
    "PairwiseAlignmentTracker",
]
