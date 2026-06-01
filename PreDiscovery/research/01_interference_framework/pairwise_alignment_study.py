"""
Multi-trial pairwise alignment study on standard (interleaved) CIFAR-10.

Question being answered
-----------------------
For each of {vanilla SGD, SGD+momentum, Adam} (with and without BoGrad),
under STANDARD random-shuffle CIFAR-10 batching, what is the *distribution*
of pairwise cosine alignments across a K-step buffer? Specifically:

  - What fraction of pairs are positively aligned (cos > 0)?
  - What fraction are negatively aligned (cos < 0)?
  - What are the typical magnitudes of each?
  - How do these change with vs without BoGrad?
  - How do these change at the gradient layer vs the update layer?

This characterises *normal training* (no class-disjoint or contrived
batches) — the regime the thesis chapter ultimately needs to describe.

Multi-trial design
------------------
3 trials per configuration (default), each with a different seed but the
SAME data-shuffle order across configurations within a trial (paired
comparison). Reports mean ± std across trials.

Six configurations:
  1. SGD vanilla (no momentum, no BoGrad)
  2. SGD+momentum (no BoGrad)
  3. Adam (no BoGrad)
  4. SGD vanilla + BoGrad gradient-stage K=8 negative
  5. SGD+momentum + BoGrad update-stage K=32 negative
  6. Adam + BoGrad update-stage K=128 negative

Output
------
research/01_interference_framework/results/pairwise_alignment_study/run_<timestamp>/
  config.json
  per_trial/<variant>__t<trial>.json    (full diagnostic history for each trial)
  summary.json                          (aggregated across trials)
  summary_table.txt
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure stdout can handle unicode on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.diagnostics import ClassProbeSet, InterferenceTracker  # noqa: E402
from common.optimizers import BoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, build_cifar10, evaluate, make_test_loader, make_train_loader,
)


# ============================================================================
# Variant registry
# ============================================================================
@dataclass
class Variant:
    name: str
    label: str
    factory: Callable[[nn.Module, float], torch.optim.Optimizer]
    lr: float
    pairwise_K: int


def build_variants(*, bograd_K_vanilla: int = 8, bograd_K_momentum: int = 32,
                    bograd_K_adam: int = 128, pairwise_K: int = 32) -> List[Variant]:
    """Return the standard six configurations.

    `pairwise_K` is the buffer size used by PairwiseAlignmentTracker. We use
    a *common* K=32 across all variants to make alignment-distribution
    comparisons apples-to-apples — independent of each variant's BoGrad K.
    """
    return [
        Variant(
            "sgd_vanilla", "SGD vanilla (lr=0.05, no mom)",
            lambda m, lr: torch.optim.SGD(m.parameters(), lr=lr),
            lr=0.05, pairwise_K=pairwise_K,
        ),
        Variant(
            "sgd_momentum", "SGD+momentum (lr=0.05, mu=0.9)",
            lambda m, lr: torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9),
            lr=0.05, pairwise_K=pairwise_K,
        ),
        Variant(
            "adam", "Adam (lr=1e-3)",
            lambda m, lr: torch.optim.Adam(m.parameters(), lr=lr),
            lr=1e-3, pairwise_K=pairwise_K,
        ),
        Variant(
            "sgd_vanilla_bograd", f"SGD vanilla + BoGrad grad-K={bograd_K_vanilla} neg",
            lambda m, lr: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=bograd_K_vanilla, project_stage="gradient",
                projection_mode="negative", orth_method="sequential",
                lr=lr,
            ),
            lr=0.05, pairwise_K=pairwise_K,
        ),
        Variant(
            "sgd_momentum_bograd", f"SGD+mom + BoGrad upd-K={bograd_K_momentum} neg",
            lambda m, lr: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=bograd_K_momentum, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=lr, momentum=0.9,
            ),
            lr=0.05, pairwise_K=pairwise_K,
        ),
        Variant(
            "adam_bograd", f"Adam + BoGrad upd-K={bograd_K_adam} neg",
            lambda m, lr: BoGrad(
                m.parameters(), torch.optim.Adam,
                buffer_size=bograd_K_adam, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=lr,
            ),
            lr=1e-3, pairwise_K=pairwise_K,
        ),
    ]


# ============================================================================
# Run helper
# ============================================================================
@dataclass
class TrialResult:
    variant: str
    trial: int
    final_acc: float
    summary: Dict[str, Any]
    history: List[Dict[str, Any]]


def run_trial(
    variant: Variant,
    trial: int,
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe: ClassProbeSet,
    device: torch.device,
    epochs: int,
    log_every: int,
    probe_every: int,
) -> TrialResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    optimizer = variant.factory(model, variant.lr)
    criterion = nn.CrossEntropyLoss()

    tracker = InterferenceTracker(
        model, criterion,
        probe_set=probe,
        log_every=log_every,
        probe_every=probe_every,
        wasted_work_K=32,
        device=device,
        trajectory_lags=[1, 4, 16],
        pairwise_K=variant.pairwise_K,
    )

    print(f"  trial {trial} seed={seed}", end="", flush=True)
    t0 = time.time()
    epoch_acc: List[float] = []

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            tracker.before_step()
            optimizer.step()
            tracker.after_step(loss=loss.item())

        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        print(f"  e{epoch + 1}={acc:.4f}", end="", flush=True)

    elapsed = time.time() - t0
    print(f"  ({elapsed:.0f}s)")

    return TrialResult(
        variant=variant.name,
        trial=trial,
        final_acc=epoch_acc[-1] if epoch_acc else float("nan"),
        summary=tracker.summary(),
        history=tracker.get_history(),
    )


# ============================================================================
# Aggregation
# ============================================================================
def aggregate_across_trials(trials: List[TrialResult], key: str) -> Dict[str, float]:
    """Pull `summary[key]` across trials, return mean+std."""
    values: List[float] = []
    for t in trials:
        v = t.summary.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            values.append(float(v))
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def aggregate_pairwise(trials: List[TrialResult], side: str, stat: str) -> Dict[str, float]:
    """Pull pairwise summary stat across trials. side ∈ {grad, update}."""
    values: List[float] = []
    for t in trials:
        pw = t.summary.get("pairwise_summary", {}) or {}
        # Stats are stored as e.g. "grad_frac_positive_mean" in pairwise summary.
        v = pw.get(f"{side}_{stat}_mean")
        if isinstance(v, (int, float)) and math.isfinite(v):
            values.append(float(v))
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--probe-every", type=int, default=20,
                    help="Probe set evaluation cadence. Higher = cheaper. "
                         "Pairwise alignment doesn't need the probe; only "
                         "forgetting does.")
    ap.add_argument("--pairwise-K", type=int, default=32,
                    help="Buffer size for pairwise alignment tracker. "
                         "Common value across all variants for apples-to-apples.")
    ap.add_argument("--bograd-K-vanilla", type=int, default=8)
    ap.add_argument("--bograd-K-momentum", type=int, default=32)
    ap.add_argument("--bograd-K-adam", type=int, default=128)
    ap.add_argument("--base-seed", type=int, default=2026)
    ap.add_argument("--variants", type=str, default=None,
                    help="Comma-separated subset of variant names (default: all 6)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Trials: {args.trials} | Epochs: {args.epochs} | "
          f"pairwise_K: {args.pairwise_K} | log_every: {args.log_every}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    print("Building probe set...")
    probe = ClassProbeSet(test_ds, num_classes=10, n_per_class=64,
                          seed=args.base_seed, device=device)

    variants = build_variants(
        bograd_K_vanilla=args.bograd_K_vanilla,
        bograd_K_momentum=args.bograd_K_momentum,
        bograd_K_adam=args.bograd_K_adam,
        pairwise_K=args.pairwise_K,
    )
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",")}
        variants = [v for v in variants if v.name in wanted]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "research" / "01_interference_framework" / "results" / "pairwise_alignment_study" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial_dir = out_dir / "per_trial"
    per_trial_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}\n")

    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id}, fh, indent=2)

    # results[variant_name] = list of TrialResult
    results: Dict[str, List[TrialResult]] = {v.name: [] for v in variants}

    total_runs = len(variants) * args.trials
    run_idx = 0

    for trial in range(args.trials):
        trial_seed = args.base_seed + trial * 1000
        print(f"=== Trial {trial + 1}/{args.trials} (seed={trial_seed}) ===")

        for variant in variants:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {variant.label}")
            # Fresh loader per (trial, variant) — same seed within trial = paired data order.
            train_loader = make_train_loader(
                train_ds, args.batch_size, args.num_workers,
                generator_seed=trial_seed,
                pin_memory=(device.type == "cuda"),
            )
            try:
                r = run_trial(
                    variant, trial, trial_seed,
                    train_loader, test_loader, probe, device,
                    epochs=args.epochs, log_every=args.log_every,
                    probe_every=args.probe_every,
                )
            except Exception as exc:
                print(f"\n    !! FAILED: {exc}")
                continue
            results[variant.name].append(r)
            with (per_trial_dir / f"{variant.name}__t{trial}.json").open("w") as fh:
                json.dump({
                    "variant": variant.name,
                    "trial": trial,
                    "seed": trial_seed,
                    "final_acc": r.final_acc,
                    "summary": r.summary,
                    "history": r.history,
                }, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Aggregated summary
    # ------------------------------------------------------------------
    summary_data: Dict[str, Any] = {"variants": {}}

    for variant in variants:
        trials = results[variant.name]
        if not trials:
            continue
        accs = [t.final_acc for t in trials if math.isfinite(t.final_acc)]
        acc_agg = {
            "mean": statistics.mean(accs) if accs else float("nan"),
            "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "n": len(accs),
        }

        summary_data["variants"][variant.name] = {
            "label": variant.label,
            "lr": variant.lr,
            "pairwise_K": variant.pairwise_K,
            "n_trials": len(trials),
            "final_acc": acc_agg,
            # Cheap step-step alignments
            "cos_g_prev": aggregate_across_trials(trials, "cos_g_prev_mean"),
            "cos_u_prev": aggregate_across_trials(trials, "cos_u_prev_mean"),
            "ww_K32": aggregate_across_trials(trials, "ww_wasted_work_ratio_mean"),
            # Pairwise — gradient buffer
            "grad_frac_positive": aggregate_pairwise(trials, "grad", "frac_positive"),
            "grad_frac_negative": aggregate_pairwise(trials, "grad", "frac_negative"),
            "grad_mean_positive_cos": aggregate_pairwise(trials, "grad", "mean_positive_cos"),
            "grad_mean_negative_cos": aggregate_pairwise(trials, "grad", "mean_negative_cos"),
            "grad_mean_abs_cos": aggregate_pairwise(trials, "grad", "mean_abs_cos"),
            "grad_max_cos": aggregate_pairwise(trials, "grad", "max_cos"),
            "grad_min_cos": aggregate_pairwise(trials, "grad", "min_cos"),
            # Pairwise — update buffer
            "update_frac_positive": aggregate_pairwise(trials, "update", "frac_positive"),
            "update_frac_negative": aggregate_pairwise(trials, "update", "frac_negative"),
            "update_mean_positive_cos": aggregate_pairwise(trials, "update", "mean_positive_cos"),
            "update_mean_negative_cos": aggregate_pairwise(trials, "update", "mean_negative_cos"),
            "update_mean_abs_cos": aggregate_pairwise(trials, "update", "mean_abs_cos"),
        }

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary_data, fh, indent=2)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 130)
    print(f"PAIRWISE ALIGNMENT STUDY — {args.trials} trials × {args.epochs} epochs CIFAR-10 standard batching")
    print("=" * 130)
    print(f"  Pairwise buffer K = {args.pairwise_K}, log_every = {args.log_every}")
    print()

    def _fmt(d: Dict[str, float], fmt: str = ".3f") -> str:
        if d["n"] == 0:
            return "n/a"
        return f"{d['mean']:{fmt}}±{d['std']:{fmt}}"

    # GRADIENT-buffer table
    print("--- GRADIENT buffer pairwise stats (cos(g_t, g_{t-k}) for k=1..K) ---")
    print(f"{'variant':30s} {'acc':>11s} {'%pos':>11s} {'%neg':>11s} "
          f"{'⟨cos+⟩':>11s} {'⟨cos-⟩':>11s} {'⟨|cos|⟩':>11s} {'min_cos':>9s} {'max_cos':>9s}")
    print("-" * 130)
    for variant in variants:
        d = summary_data["variants"].get(variant.name)
        if d is None:
            continue
        print(
            f"{variant.name:30s} "
            f"{_fmt(d['final_acc']):>11s} "
            f"{_fmt(d['grad_frac_positive']):>11s} "
            f"{_fmt(d['grad_frac_negative']):>11s} "
            f"{_fmt(d['grad_mean_positive_cos']):>11s} "
            f"{_fmt(d['grad_mean_negative_cos']):>11s} "
            f"{_fmt(d['grad_mean_abs_cos']):>11s} "
            f"{d['grad_min_cos']['mean']:>+9.3f} "
            f"{d['grad_max_cos']['mean']:>+9.3f} "
        )

    print()
    # UPDATE-buffer table
    print("--- UPDATE buffer pairwise stats (cos(u_t, u_{t-k}) for k=1..K) ---")
    print(f"{'variant':30s} {'WW_K32':>11s} {'%pos':>11s} {'%neg':>11s} "
          f"{'⟨cos+⟩':>11s} {'⟨cos-⟩':>11s} {'⟨|cos|⟩':>11s}")
    print("-" * 130)
    for variant in variants:
        d = summary_data["variants"].get(variant.name)
        if d is None:
            continue
        print(
            f"{variant.name:30s} "
            f"{_fmt(d['ww_K32']):>11s} "
            f"{_fmt(d['update_frac_positive']):>11s} "
            f"{_fmt(d['update_frac_negative']):>11s} "
            f"{_fmt(d['update_mean_positive_cos']):>11s} "
            f"{_fmt(d['update_mean_negative_cos']):>11s} "
            f"{_fmt(d['update_mean_abs_cos']):>11s} "
        )

    print()
    print("=" * 130)
    print(f"Done. Results at {out_dir}")


if __name__ == "__main__":
    main()
