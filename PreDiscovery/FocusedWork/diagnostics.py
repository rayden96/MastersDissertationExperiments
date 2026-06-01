"""
Diagnostics for the FocusedWork measurement framework (§03 + §04).

Implements:
- Reference gradient sampling on a fixed reference set, refreshed every N steps.
- Per-class subgradient computation on the current training batch.
- §03 inter-batch metrics: cancellation index, pairwise cosine stats, magnitude
  stats, useful/wasted decomposition against the full-batch reference.
- §04 between-batch metrics: K-window cancellation index, pairwise cosines and
  magnitude stats within the K-window, useful/wasted decomposition against the
  window-start reference.
- Per-step first-order loss-decrease deficit D_t = lr * (<ref_grad, eff_grad> - |ref_grad|^2)
  where eff_grad = -u_t / lr (the "effective gradient" the optimiser implicitly
  applied, recovered from the parameter delta).

Scope: SGD (with optional momentum). For Adam/RMSprop the deficit accounting
needs separate derivation; the geometric metrics still apply.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters() if p.requires_grad])


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


def _compute_full_grad(
    model: nn.Module,
    criterion: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Gradient of the average loss over `loader`, plus that average loss."""
    was_training = model.training
    model.train()
    _zero_grads(model)
    n_total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        # backward of (loss * batch_size) accumulates the gradient of the SUM
        (loss * x.size(0)).backward()
        n_total += x.size(0)
        loss_sum += loss.item() * x.size(0)
    grad = _flatten_grads(model) / max(n_total, 1)
    _zero_grads(model)
    if not was_training:
        model.eval()
    return grad, loss_sum / max(n_total, 1)


def _per_class_grads_on_batch(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """Compute per-class subgradients on a single batch.
    Returns dict: class_label -> flat gradient tensor (mean of per-sample losses
    over class-c samples in this batch).
    """
    was_training = model.training
    model.train()
    classes = torch.unique(y).cpu().tolist()
    out: Dict[int, torch.Tensor] = {}
    for c in classes:
        mask = (y == c)
        n_c = int(mask.sum().item())
        if n_c == 0:
            continue
        x_c, y_c = x[mask], y[mask]
        _zero_grads(model)
        logits = model(x_c)
        loss_c = criterion(logits, y_c)
        loss_c.backward()
        out[int(c)] = _flatten_grads(model).clone()
    _zero_grads(model)
    if not was_training:
        model.eval()
    return out


# ---------------------------------------------------------------------------
# meter
# ---------------------------------------------------------------------------
class InterferenceMeter:
    """
    Tracks the §03 / §04 measurements during training.

    Usage (SGD-style training loop):

        meter = InterferenceMeter(model, criterion, ref_loader, num_classes,
                                  device, lr=0.05)
        meter.initialize()  # one-shot reference compute at θ_0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()

            meter.before_step()
            optimizer.step()
            meter.after_step(step, x, y, applied_loss=loss.item())

        summary = meter.summarize()

    For COSGD (which does its own forward+backward inside step()):
        meter.before_step()
        loss_value = optimizer.step(x, y, torch.unique(y))
        meter.after_step(step, x, y, applied_loss=loss_value)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        ref_loader,
        num_classes: int,
        device: torch.device,
        lr: float,
        K_values: Sequence[int] = (4, 32, 128),
        log_every: int = 10,
        ref_refresh_every: int = 50,
    ):
        self.model = model
        self.criterion = criterion
        self.ref_loader = ref_loader
        self.num_classes = num_classes
        self.device = device
        self.lr = float(lr)
        self.K_values = list(K_values)
        self.K_max = max(self.K_values)
        self.log_every = int(log_every)
        self.ref_refresh_every = int(ref_refresh_every)

        self.ref_grad: Optional[torch.Tensor] = None
        self.ref_loss: Optional[float] = None
        self.last_ref_step: int = -10**9

        # rolling buffer of applied updates for between-batch K-window analysis
        self.update_buffer: deque = deque(maxlen=self.K_max)

        # for each entry in update_buffer at the same index, what was the
        # reference gradient stored at that step?  Aligned 1-1.
        self.ref_grad_at_step: deque = deque(maxlen=self.K_max + 1)

        # before_step state
        self._params_before: Optional[torch.Tensor] = None

        # cumulative deficit
        self.cum_deficit: float = 0.0
        self.cum_deficit_count: int = 0

        # logs
        self.logs: List[Dict] = []
        self.calibration_logs: List[Dict] = []

    # ----------------------------------------------------------------
    def initialize(self) -> None:
        """Compute initial reference gradient at θ_0."""
        g, l = _compute_full_grad(self.model, self.criterion, self.ref_loader, self.device)
        self.ref_grad = g
        self.ref_loss = l
        self.last_ref_step = 0
        self.calibration_logs.append({"step": 0, "ref_loss": l})
        # seed the alignment deque with the initial reference grad
        self.ref_grad_at_step.append(self.ref_grad.clone())

    # ----------------------------------------------------------------
    def before_step(self) -> None:
        self._params_before = _flatten_params(self.model)

    # ----------------------------------------------------------------
    def after_step(
        self,
        step: int,
        x: torch.Tensor,
        y: torch.Tensor,
        applied_loss: float,
    ) -> None:
        params_after = _flatten_params(self.model)
        u_t = params_after - self._params_before
        self.update_buffer.append(u_t.clone())

        # --- per-step deficit (uses the CURRENT stored ref_grad, which is at
        #     the previous refresh point — at most ref_refresh_every steps stale) ---
        D_t = float("nan")
        if self.ref_grad is not None:
            eff_grad_t = -u_t / self.lr  # what SGD would have used
            inner = torch.dot(self.ref_grad, eff_grad_t).item()
            ref_norm_sq = torch.dot(self.ref_grad, self.ref_grad).item()
            D_t = self.lr * (inner - ref_norm_sq)
            self.cum_deficit += D_t
            self.cum_deficit_count += 1

        # --- align ref_grad_at_step with update_buffer ---
        # store the ref_grad that was active when this update was applied
        self.ref_grad_at_step.append(self.ref_grad.clone() if self.ref_grad is not None else None)

        # --- maybe refresh reference for future steps ---
        if step - self.last_ref_step >= self.ref_refresh_every:
            new_ref, new_loss = _compute_full_grad(
                self.model, self.criterion, self.ref_loader, self.device
            )
            self.ref_grad = new_ref
            self.ref_loss = new_loss
            self.last_ref_step = step
            self.calibration_logs.append({
                "step": step,
                "ref_loss": new_loss,
                "cum_deficit_so_far": self.cum_deficit,
            })

        # --- decide whether to do heavy diagnostics this step ---
        if step % self.log_every != 0:
            return

        log: Dict = {
            "step": step,
            "applied_loss": float(applied_loss),
            "u_norm": float(torch.norm(u_t).item()),
            "D_t": float(D_t),
            "cum_deficit": float(self.cum_deficit),
        }

        # --- §03 inter-batch on the current batch ---
        per_class = _per_class_grads_on_batch(self.model, self.criterion, x, y)
        if len(per_class) >= 2:
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
                log["inter_mean_cos"] = float(sum(cos_list) / len(cos_list))

            # (2b) magnitude stats
            log["inter_mean_class_norm"] = float(np.mean(norms))
            log["inter_std_class_norm"] = float(np.std(norms))
            min_n = min(norms)
            log["inter_max_min_ratio"] = float(max(norms) / min_n) if min_n > 1e-12 else float("nan")

            # (3) useful/wasted decomposition vs ref grad
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
                # batch gradient as the unweighted mean of per-class subgrads
                # (ignores n_c weighting; close enough for balanced batches)
                g_batch = torch.stack(grads).mean(dim=0)
                g_batch_norm = float(torch.norm(g_batch).item())
                if g_batch_norm > 1e-12:
                    log["inter_useful_descent_frac"] = float(
                        torch.dot(g_batch, ref_unit).item() / g_batch_norm
                    )

        # --- §04 between-batch K-window analysis ---
        for K in self.K_values:
            if len(self.update_buffer) < K:
                continue
            window = list(self.update_buffer)[-K:]
            net = torch.stack(window).sum(dim=0)
            net_norm = float(torch.norm(net).item())
            norms_w = [float(torch.norm(u).item()) for u in window]
            norm_sum_w = sum(norms_w)
            log[f"I_between_K{K}"] = (net_norm / norm_sum_w) if norm_sum_w > 0 else float("nan")

            # pairwise within window — sample a subset if K is large to bound cost
            cos_list_w: List[float] = []
            max_pairs = 1024
            n = len(window)
            if n * (n - 1) // 2 > max_pairs:
                # subsample pairs
                rng = np.random.default_rng(seed=step + K)
                sampled = set()
                while len(sampled) < max_pairs:
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(0, n))
                    if i < j and (i, j) not in sampled:
                        sampled.add((i, j))
                pair_iter = iter(sampled)
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
                log[f"between_K{K}_mean_cos"] = float(sum(cos_list_w) / len(cos_list_w))

            # magnitude stats over window
            log[f"between_K{K}_mean_u_norm"] = float(np.mean(norms_w))
            log[f"between_K{K}_std_u_norm"] = float(np.std(norms_w))

            # useful/wasted decomp vs window-start reference
            # The window-start reference is the ref_grad active K updates ago.
            # ref_grad_at_step has length up to K_max+1; element [-K-1] would
            # be the ref active before the FIRST update of the window (i.e. at
            # the window's start). If we have fewer entries, skip.
            if len(self.ref_grad_at_step) >= K + 1:
                start_ref = self.ref_grad_at_step[-K - 1]
                if start_ref is not None and torch.norm(start_ref).item() > 1e-12:
                    ref_unit_w = start_ref / torch.norm(start_ref)
                    useful_window = []
                    wasted_window = []
                    for u in window:
                        # descent direction is -ref_unit, so positive useful = u·(-ref_unit)
                        signed_useful = -float(torch.dot(u, ref_unit_w).item())
                        useful_window.append(signed_useful)
                        # perpendicular component magnitude
                        perp = u + signed_useful * ref_unit_w  # u - (-signed_useful)*ref_unit
                        wasted_window.append(float(torch.norm(perp).item()))
                    log[f"between_K{K}_useful_path"] = float(sum(useful_window))
                    log[f"between_K{K}_wasted_path"] = float(sum(wasted_window))
                    if norm_sum_w > 0:
                        log[f"between_K{K}_useful_path_frac"] = float(
                            abs(sum(useful_window)) / norm_sum_w
                        )

        self.logs.append(log)

    # ----------------------------------------------------------------
    def summarize(self) -> Dict:
        if not self.logs:
            return {"n_logs": 0}

        def fvals(key):
            return [
                l[key] for l in self.logs
                if key in l and isinstance(l[key], (int, float)) and math.isfinite(l[key])
            ]

        def safe_mean(key):
            v = fvals(key)
            return float(np.mean(v)) if v else float("nan")

        def safe_std(key):
            v = fvals(key)
            return float(np.std(v)) if len(v) > 1 else 0.0

        def correlation(key1, key2):
            pairs = [
                (l[key1], l[key2]) for l in self.logs
                if key1 in l and key2 in l
                and isinstance(l[key1], (int, float)) and isinstance(l[key2], (int, float))
                and math.isfinite(l[key1]) and math.isfinite(l[key2])
            ]
            if len(pairs) < 5:
                return float("nan")
            xs = np.array([p[0] for p in pairs])
            ys = np.array([p[1] for p in pairs])
            if xs.std() == 0 or ys.std() == 0:
                return float("nan")
            return float(np.corrcoef(xs, ys)[0, 1])

        summary: Dict = {
            "n_logs": len(self.logs),
            "n_steps_total": self.cum_deficit_count,
            "cum_deficit": self.cum_deficit,
            "mean_deficit_per_step": (
                self.cum_deficit / max(self.cum_deficit_count, 1)
            ),
            "mean_D_t_at_logged_steps": safe_mean("D_t"),
            # §03 inter-batch
            "I_inter_mean": safe_mean("I_inter"),
            "I_inter_std": safe_std("I_inter"),
            "inter_frac_neg_mean": safe_mean("inter_frac_neg"),
            "inter_mean_cos_mean": safe_mean("inter_mean_cos"),
            "inter_mean_class_norm_mean": safe_mean("inter_mean_class_norm"),
            "inter_max_min_ratio_mean": safe_mean("inter_max_min_ratio"),
            "inter_useful_mass_mean": safe_mean("inter_useful_mass"),
            "inter_wasted_mass_mean": safe_mean("inter_wasted_mass"),
            "inter_useful_descent_frac_mean": safe_mean("inter_useful_descent_frac"),
        }

        # §04 between-batch (one row per K)
        for K in self.K_values:
            summary[f"I_between_K{K}_mean"] = safe_mean(f"I_between_K{K}")
            summary[f"I_between_K{K}_std"] = safe_std(f"I_between_K{K}")
            summary[f"between_K{K}_frac_neg_mean"] = safe_mean(f"between_K{K}_frac_neg")
            summary[f"between_K{K}_mean_cos_mean"] = safe_mean(f"between_K{K}_mean_cos")
            summary[f"between_K{K}_useful_path_frac_mean"] = safe_mean(
                f"between_K{K}_useful_path_frac"
            )
            summary[f"between_K{K}_mean_u_norm_mean"] = safe_mean(f"between_K{K}_mean_u_norm")

        # correlations: do the geometric summaries predict per-step training-hurt?
        summary["corr_I_inter_vs_Dt"] = correlation("I_inter", "D_t")
        summary["corr_inter_mean_cos_vs_Dt"] = correlation("inter_mean_cos", "D_t")
        summary["corr_inter_max_min_ratio_vs_Dt"] = correlation("inter_max_min_ratio", "D_t")
        summary["corr_I_between_K32_vs_Dt"] = correlation("I_between_K32", "D_t")
        summary["corr_between_K32_mean_cos_vs_Dt"] = correlation("between_K32_mean_cos", "D_t")

        # measured-loss calibration: drop in ref_loss at each refresh
        if len(self.calibration_logs) >= 2:
            initial = self.calibration_logs[0]["ref_loss"]
            final = self.calibration_logs[-1]["ref_loss"]
            summary["ref_loss_initial"] = float(initial)
            summary["ref_loss_final"] = float(final)
            summary["ref_loss_drop"] = float(initial - final)

        return summary
