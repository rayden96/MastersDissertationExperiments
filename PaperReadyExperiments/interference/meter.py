"""
InterferenceMeter — paper-ready implementation of the §03 + §04 measurements
from PreDiscovery/FocusedWork.

Computes, on a logged step:

  §03 inter-batch (within current batch, across per-subgroup subgradients):
    - cancellation index   I_inter = ||sum_c g_{t,c}|| / sum_c ||g_{t,c}||
    - pairwise cosine stats (frac_neg, mean cos)
    - magnitude stats       (mean, std, max/min ratio of class norms)
    - useful / wasted       decomposition of each subgrad against ref direction
    - useful descent frac   of the *batch* gradient against ref direction

  §04 between-batch (within a rolling K-window of applied updates):
    - cancellation index   I_between_K = ||theta_t - theta_{t-K}|| / sum ||u_i||
    - pairwise cosine stats within the window (pooled; a lag-resolved
      cos(u_t,u_{t-k}) vs k profile is a planned addition — see
      DISSERTATION_OUTLINE.md D.4)
    - useful path frac     against the window-start reference direction

  Per-step first-order loss-decrease deficit (>= 0; larger = more interference):
    D_t = <g_tilde, u_t> + lr * ||g_tilde||^2
    i.e. the first-order reference-loss change of the applied update u_t minus
    that of the ideal SGD step u_ideal = -lr*g_tilde. 0 when u_t is as
    descent-effective as the ideal; grows as interference steals descent.

  Sparse measured-loss calibration:
    reference loss recorded at each ref refresh.

Reference policy:
    The reference gradient is refreshed every `ref_refresh_every` logged
    steps. Between refreshes the same (slightly stale) reference is used.
    At step t, the deficit D_t is computed using the most recent reference
    that was active before step t (so we never use a reference computed at
    theta_t to evaluate the step that produced theta_{t+1}).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from interference.problem import InterferenceProblem


class InterferenceMeter:
    def __init__(
        self,
        problem: InterferenceProblem,
        lr: float,
        *,
        K_values: Sequence[int] = (4, 32, 128),
        log_every: int = 10,
        ref_refresh_every: int = 50,
    ):
        self.problem = problem
        self.lr = float(lr)
        self.K_values = list(K_values)
        self.K_max = max(self.K_values)
        self.log_every = int(log_every)
        self.ref_refresh_every = int(ref_refresh_every)

        self.ref_grad: Optional[torch.Tensor] = None
        self.ref_loss: Optional[float] = None
        self.last_ref_step: int = -10**9

        # Rolling buffers for between-batch analysis.
        # update_buffer[-1] is the most recent applied update u_t.
        # ref_grad_at_step[i] is the ref grad that was active just before the
        # i-th most-recent update was applied (aligned 1-1 with update_buffer).
        self.update_buffer: deque = deque(maxlen=self.K_max)
        self.ref_grad_at_step: deque = deque(maxlen=self.K_max + 1)

        # Per-step before-step state.
        self._params_before: Optional[torch.Tensor] = None

        # Cumulative deficit (every step, cheap).
        # `cum_deficit` uses the SGD yardstick ideal (u_ideal = -lr * g_tilde),
        # the cross-method common scale. `cum_deficit_precond` uses the training
        # optimiser's own response to g_tilde (problem.preconditioned_ideal),
        # isolating cancellation from preconditioning; it stays at 0 / count 0
        # unless the problem implements the hook.
        self.cum_deficit: float = 0.0
        self.cum_deficit_count: int = 0
        self.cum_deficit_precond: float = 0.0
        self.cum_deficit_precond_count: int = 0

        # Logs.
        self.logs: List[Dict[str, Any]] = []
        self.calibration_logs: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    def initialize(self, append_calibration: bool = True, at_step: int = 0) -> None:
        """Compute the initial reference gradient at the current parameters.

        On a fresh run, call with defaults. On *resume*, call with
        `append_calibration=False` (and `at_step=<resumed step>`) so the
        reference is recomputed for the restored model without appending a
        duplicate step-0 calibration entry to a restored log.
        """
        g, l = self.problem.compute_reference()
        self.ref_grad = g
        self.ref_loss = float(l)
        self.last_ref_step = at_step
        if append_calibration:
            self.calibration_logs.append({"step": at_step, "ref_loss": float(l)})

    def state_dict(self) -> Dict[str, Any]:
        """Lightweight, JSON-friendly meter state for checkpoint/resume.

        Persists the accumulated logs + cumulative deficits (cheap dicts/floats).
        The heavy tensors (ref_grad, update_buffer) are NOT persisted — on resume
        the ref grad is recomputed via initialize(append_calibration=False) and
        the between-batch window buffers restart (a few windows near the resume
        point are skipped, which is documented and immaterial for run-level stats)."""
        return {
            "logs": self.logs,
            "calibration_logs": self.calibration_logs,
            "cum_deficit": self.cum_deficit,
            "cum_deficit_count": self.cum_deficit_count,
            "cum_deficit_precond": self.cum_deficit_precond,
            "cum_deficit_precond_count": self.cum_deficit_precond_count,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.logs = sd.get("logs", [])
        self.calibration_logs = sd.get("calibration_logs", [])
        self.cum_deficit = sd.get("cum_deficit", 0.0)
        self.cum_deficit_count = sd.get("cum_deficit_count", 0)
        self.cum_deficit_precond = sd.get("cum_deficit_precond", 0.0)
        self.cum_deficit_precond_count = sd.get("cum_deficit_precond_count", 0)

    def before_step(self) -> None:
        self._params_before = self.problem.flatten_params()

    # -----------------------------------------------------------------
    def after_step(
        self,
        step: int,
        batch: Any,
        applied_loss: float,
    ) -> None:
        params_after = self.problem.flatten_params()
        u_t = params_after - self._params_before
        self.update_buffer.append(u_t.clone())

        # Per-step deficit using the *current* (potentially stale) ref_grad.
        #
        # D_t = <g_tilde, u_t> - <g_tilde, u_ideal>   (>= 0; larger = worse):
        # the first-order reference-loss decrease the applied update u_t FAILED
        # to achieve relative to the ideal full-batch update u_ideal. <g_tilde,u>
        # is the first-order change in reference loss under update u; the ideal
        # step is the most descent-effective, so the actual step's change is >=
        # the ideal's and the (non-negative) gap is the interference deficit.
        #
        # SGD yardstick: u_ideal = -lr*g_tilde  =>  D_t = g_dot_u + lr*||g~||^2.
        # (Sign convention: positive = hurt; matches FocusedWork/03 prose.)
        D_t = float("nan")
        D_t_precond = float("nan")
        if self.ref_grad is not None:
            g_dot_u = torch.dot(self.ref_grad, u_t).item()
            ref_sq = torch.dot(self.ref_grad, self.ref_grad).item()
            # SGD-yardstick ideal: u_ideal = -lr * g_tilde
            D_t = g_dot_u + self.lr * ref_sq
            self.cum_deficit += D_t
            self.cum_deficit_count += 1

            # Optional per-optimiser ideal (isolates preconditioning); same sign
            # convention with u_ideal = the optimiser's own response to g_tilde.
            u_ideal = self.problem.preconditioned_ideal(self.ref_grad)
            if u_ideal is not None:
                D_t_precond = g_dot_u - torch.dot(self.ref_grad, u_ideal).item()
                self.cum_deficit_precond += D_t_precond
                self.cum_deficit_precond_count += 1

        # Record which reference was active when this update was applied
        # (aligned with the just-appended update_buffer entry).
        self.ref_grad_at_step.append(
            self.ref_grad.clone() if self.ref_grad is not None else None
        )

        # Refresh reference (for use on subsequent steps).
        if step - self.last_ref_step >= self.ref_refresh_every:
            new_ref, new_loss = self.problem.compute_reference()
            self.ref_grad = new_ref
            self.ref_loss = float(new_loss)
            self.last_ref_step = step
            self.calibration_logs.append({
                "step": step,
                "ref_loss": float(new_loss),
                "cum_deficit_so_far": self.cum_deficit,
            })

        # Heavy diagnostics only on logged steps.
        if step % self.log_every != 0:
            return

        log: Dict[str, Any] = {
            "step": step,
            "applied_loss": float(applied_loss),
            "u_norm": float(torch.norm(u_t).item()),
            "D_t": float(D_t),
            "D_t_precond": float(D_t_precond),
            "cum_deficit": float(self.cum_deficit),
            "cum_deficit_precond": float(self.cum_deficit_precond),
        }

        self._log_inter_batch(log, batch)
        self._log_between_batch(log)
        self.logs.append(log)

    # -----------------------------------------------------------------
    def _log_inter_batch(self, log: Dict[str, Any], batch: Any) -> None:
        per_class = self.problem.per_subgroup_grads(batch)
        if len(per_class) < 2:
            return
        grads = list(per_class.values())
        norms = [float(torch.norm(g).item()) for g in grads]

        # (1) cancellation index
        summed = torch.stack(grads).sum(dim=0)
        summed_norm = float(torch.norm(summed).item())
        norm_sum = sum(norms)
        log["I_inter"] = (summed_norm / norm_sum) if norm_sum > 0 else float("nan")

        # (2a) pairwise cosines
        cos_list: List[float] = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                if norms[i] > 0 and norms[j] > 0:
                    c = torch.dot(grads[i], grads[j]).item() / (norms[i] * norms[j])
                    cos_list.append(c)
        if cos_list:
            log["inter_n_pairs"] = len(cos_list)
            log["inter_frac_neg"] = float(sum(1 for c in cos_list if c < 0) / len(cos_list))
            log["inter_mean_cos"] = float(np.mean(cos_list))

        # (2b) magnitude stats
        log["inter_mean_grad_norm"] = float(np.mean(norms))
        log["inter_std_grad_norm"] = float(np.std(norms))
        min_n = min(norms)
        log["inter_max_min_ratio"] = float(max(norms) / min_n) if min_n > 1e-12 else float("nan")

        # (3) useful/wasted decomposition vs reference direction
        if self.ref_grad is not None and torch.norm(self.ref_grad).item() > 1e-12:
            ref_unit = self.ref_grad / torch.norm(self.ref_grad)
            useful_per_class = []
            wasted_per_class = []
            for g in grads:
                u_c = torch.dot(g, ref_unit).item()
                useful_per_class.append(u_c)
                perp = g - u_c * ref_unit
                wasted_per_class.append(float(torch.norm(perp).item()))
            log["inter_useful_mass"] = float(sum(useful_per_class))
            log["inter_wasted_mass"] = float(sum(wasted_per_class))
            g_batch = torch.stack(grads).mean(dim=0)
            g_batch_norm = float(torch.norm(g_batch).item())
            if g_batch_norm > 1e-12:
                log["inter_useful_descent_frac"] = float(
                    torch.dot(g_batch, ref_unit).item() / g_batch_norm
                )

    # -----------------------------------------------------------------
    def _log_between_batch(self, log: Dict[str, Any]) -> None:
        for K in self.K_values:
            if len(self.update_buffer) < K:
                continue
            window = list(self.update_buffer)[-K:]
            net = torch.stack(window).sum(dim=0)
            net_norm = float(torch.norm(net).item())
            norms_w = [float(torch.norm(u).item()) for u in window]
            norm_sum_w = sum(norms_w)
            log[f"I_between_K{K}"] = (
                (net_norm / norm_sum_w) if norm_sum_w > 0 else float("nan")
            )

            # Pairwise within window (subsample if too many).
            cos_list_w: List[float] = []
            max_pairs = 1024
            n = len(window)
            if n * (n - 1) // 2 > max_pairs:
                rng = np.random.default_rng(seed=log["step"] + K)
                pairs = set()
                while len(pairs) < max_pairs:
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(0, n))
                    if i < j:
                        pairs.add((i, j))
                pair_iter = iter(pairs)
            else:
                pair_iter = ((i, j) for i in range(n) for j in range(i + 1, n))
            for i, j in pair_iter:
                if norms_w[i] > 0 and norms_w[j] > 0:
                    c = torch.dot(window[i], window[j]).item() / (norms_w[i] * norms_w[j])
                    cos_list_w.append(c)
            if cos_list_w:
                log[f"between_K{K}_n_pairs"] = len(cos_list_w)
                log[f"between_K{K}_frac_neg"] = float(
                    sum(1 for c in cos_list_w if c < 0) / len(cos_list_w)
                )
                log[f"between_K{K}_mean_cos"] = float(np.mean(cos_list_w))

            # Magnitude stats over window.
            log[f"between_K{K}_mean_u_norm"] = float(np.mean(norms_w))
            log[f"between_K{K}_std_u_norm"] = float(np.std(norms_w))

            # Useful/wasted vs window-start reference.
            if len(self.ref_grad_at_step) >= K + 1:
                start_ref = self.ref_grad_at_step[-K - 1]
                if start_ref is not None and torch.norm(start_ref).item() > 1e-12:
                    ref_unit_w = start_ref / torch.norm(start_ref)
                    useful_window: List[float] = []
                    wasted_window: List[float] = []
                    for u in window:
                        # descent direction in step space is -ref_unit
                        signed_useful = -float(torch.dot(u, ref_unit_w).item())
                        useful_window.append(signed_useful)
                        perp = u + signed_useful * ref_unit_w
                        wasted_window.append(float(torch.norm(perp).item()))
                    log[f"between_K{K}_useful_path"] = float(sum(useful_window))
                    log[f"between_K{K}_wasted_path"] = float(sum(wasted_window))
                    if norm_sum_w > 0:
                        log[f"between_K{K}_useful_path_frac"] = float(
                            abs(sum(useful_window)) / norm_sum_w
                        )
