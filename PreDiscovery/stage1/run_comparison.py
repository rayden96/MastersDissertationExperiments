"""
Stage 1 — three-method comparison with both interference metrics.

Runs:
  1. Baseline (SGD+momentum)
  2. + BoGrad (between-batch interference reducer, update-stage K=32 neg)
  3. + COSGD (inter-batch interference reducer, single_forward + GS)

Captures per step (or per N steps):
  - inter-batch interference: pairwise cos(g_c, g_{c'}) within probe
    (via stage1.inter_batch_metric.measure_inter_batch_interference)
  - between-batch interference: OOB forgetting magnitude
    (via common.diagnostics.InterferenceTracker)
  - standard training: train loss, test acc per epoch
  - WW_K, pairwise update alignment

Output: a per-method JSON history + a summary table that's the headline
chart for tomorrow's meeting.

Usage
-----
    python stage1/run_comparison.py
    python stage1/run_comparison.py --quick
    python stage1/run_comparison.py --epochs 2 --inter-batch-every 25
    python stage1/run_comparison.py --only baseline,bograd
"""

from __future__ import annotations

import argparse
import json
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.diagnostics import ClassProbeSet, InterferenceTracker  # noqa: E402
from common.optimizers import BoGrad, COSGD  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, build_cifar10, evaluate, make_test_loader, make_train_loader,
)
from stage1.inter_batch_metric import measure_inter_batch_interference  # noqa: E402


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    label: str
    kind: str   # "standard" or "cosgd" — controls training-loop dispatch
    factory: Callable[[nn.Module, nn.Module], torch.optim.Optimizer]


def build_variants(*, lr: float, momentum: float, bograd_K: int) -> List[Variant]:
    return [
        Variant(
            "baseline", "SGD+momentum baseline",
            kind="standard",
            factory=lambda m, c: torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum),
        ),
        Variant(
            "bograd", f"SGD+momentum + BoGrad upd-K={bograd_K} neg",
            kind="standard",
            factory=lambda m, c: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=bograd_K, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=lr, momentum=momentum,
            ),
        ),
        Variant(
            "cosgd", "COSGD (modified GS, single_forward)",
            kind="cosgd",
            factory=lambda m, c: COSGD(
                m.parameters(), lr=lr,
                model=m, criterion=c,
                orthogonalization_method="modified_gs_normal",
                step_method="single_forward",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Per-variant training run
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    variant: str
    label: str
    final_acc: float
    epoch_acc: List[float]
    history: List[Dict[str, Any]]                  # InterferenceTracker
    inter_batch_history: List[Dict[str, Any]]      # per-measurement inter-batch records
    summary: Dict[str, Any]
    forgetting: Dict[str, Any]
    wasted_work: Dict[str, Any]
    wall_clock_s: float


def run_variant(
    variant: Variant,
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe_set: ClassProbeSet,
    device: torch.device,
    *,
    epochs: int,
    log_every: int,
    probe_every: int,
    inter_batch_every: int,
    pairwise_K: int,
    seed: int,
) -> RunResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = variant.factory(model, criterion)

    tracker = InterferenceTracker(
        model, criterion,
        probe_set=probe_set,
        log_every=log_every,
        probe_every=probe_every,
        wasted_work_K=32,
        device=device,
        trajectory_lags=[1, 4, 16],
        pairwise_K=pairwise_K,
    )

    inter_batch_history: List[Dict[str, Any]] = []

    print(f"\n=== {variant.label} ===")
    t0 = time.time()
    epoch_acc: List[float] = []
    step_count = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ----- Optionally measure inter-batch interference -----
            # This calls the model independently with class-restricted probe
            # data — does not modify training state when we zero grads after.
            if step_count % inter_batch_every == 0:
                ib_stats = measure_inter_batch_interference(model, criterion, probe_set)
                ib_stats["step"] = step_count
                inter_batch_history.append(ib_stats)

            # ----- Training step (dispatched by variant kind) -----
            if variant.kind == "standard":
                # Standard PyTorch optimizer interface
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                tracker.before_step()
                optimizer.step()
                batch_classes = set(y.unique().cpu().tolist())
                tracker.after_step(loss=loss.item(), batch_classes=batch_classes)

            elif variant.kind == "cosgd":
                # COSGD does the per-class fwd+bwd + projection internally.
                # The .step() call returns the running loss across the per-class
                # gradients. We still need to interface with InterferenceTracker:
                #   - InterferenceTracker.before_step expects grads to be set.
                #   - But COSGD does its own grad management and calls .step()
                #     which already applied parameter updates.
                # So we wrap differently: snapshot params before, call COSGD step,
                # then call after_step with empty grad info.
                params_before = [p.detach().clone() for p in model.parameters()
                                 if p.requires_grad]
                # COSGD's step expects (data, labels, unique_labels)
                unique_labels = y.unique()
                loss_val = optimizer.step(x, y, unique_labels)
                # Manually populate tracker state for after_step.
                # We don't have a single "gradient" for COSGD (it has per-class
                # ones), so we leave grad fields None — only update-related
                # metrics will be valid.
                # Hack: call before_step and after_step in sequence with
                # whatever state we can reconstruct.
                tracker._params_before_step = torch.cat(
                    [p.reshape(-1) for p in params_before]
                )
                # No grad info available cleanly; tracker.after_step will
                # compute update from params_before and current params.
                tracker.after_step(
                    loss=float(loss_val) if loss_val is not None else float("nan"),
                    batch_classes=set(y.unique().cpu().tolist()),
                )
                del params_before
            else:
                raise ValueError(f"Unknown variant.kind {variant.kind}")

            step_count += 1

        # Eval once per epoch
        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{epochs}  test_acc={acc:.4f}  steps={step_count}  ({elapsed:.0f}s)")

    return RunResult(
        variant=variant.name,
        label=variant.label,
        final_acc=epoch_acc[-1] if epoch_acc else float("nan"),
        epoch_acc=epoch_acc,
        history=tracker.get_history(),
        inter_batch_history=inter_batch_history,
        summary=tracker.summary(),
        forgetting=tracker.forgetting.summary(),
        wasted_work=tracker.wasted_work.summary(),
        wall_clock_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bograd-K", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--probe-n-per-class", type=int, default=64)
    ap.add_argument("--inter-batch-every", type=int, default=25,
                    help="Measure inter-batch interference every N training steps.")
    ap.add_argument("--pairwise-K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated subset of {baseline, bograd, cosgd}")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"epochs={args.epochs}  lr={args.lr}  mu={args.momentum}  "
          f"batch={args.batch_size}  log_every={args.log_every}  "
          f"inter_batch_every={args.inter_batch_every}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    print("Building probe set...")
    probe = ClassProbeSet(test_ds, num_classes=10,
                          n_per_class=args.probe_n_per_class,
                          seed=args.seed, device=device)

    variants = build_variants(lr=args.lr, momentum=args.momentum, bograd_K=args.bograd_K)
    if args.only:
        wanted = {v.strip() for v in args.only.split(",") if v.strip()}
        variants = [v for v in variants if v.name in wanted]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "stage1" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id}, fh, indent=2)

    results: Dict[str, RunResult] = {}
    for variant in variants:
        train_loader = make_train_loader(
            train_ds, args.batch_size, args.num_workers,
            generator_seed=args.seed,
            pin_memory=(device.type == "cuda"),
        )
        try:
            r = run_variant(
                variant, train_loader, test_loader, probe, device,
                epochs=args.epochs, log_every=args.log_every,
                probe_every=args.probe_every,
                inter_batch_every=args.inter_batch_every,
                pairwise_K=args.pairwise_K,
                seed=args.seed,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"\n  !! {variant.name} failed: {exc}")
            continue

        results[variant.name] = r

        sub = out_dir / variant.name
        sub.mkdir(exist_ok=True)
        with (sub / "history.json").open("w") as fh:
            json.dump(r.history, fh, indent=2)
        with (sub / "inter_batch_history.json").open("w") as fh:
            json.dump(r.inter_batch_history, fh, indent=2)
        with (sub / "summary.json").open("w") as fh:
            json.dump(r.summary, fh, indent=2, default=str)
        with (sub / "result.json").open("w") as fh:
            json.dump({
                "variant": r.variant,
                "label": r.label,
                "final_acc": r.final_acc,
                "epoch_acc": r.epoch_acc,
                "wall_clock_s": r.wall_clock_s,
                "forgetting": r.forgetting,
                "wasted_work": r.wasted_work,
            }, fh, indent=2, default=str)

    # ---------- Console summary ----------
    print("\n" + "=" * 130)
    print("STAGE 1 — both-types-of-interference comparison")
    print("=" * 130)
    print(f"{'method':40s} {'final_acc':>10s} {'IB_%neg':>9s} {'IB_⟨cos⟩':>10s} "
          f"{'OOB_total':>10s} {'OOB_per_step':>13s} {'WW_K':>7s}")
    print("-" * 130)

    for vname, r in results.items():
        # Inter-batch summary: average frac_negative and mean_cos across all
        # measurements during training
        ib_records = [rec for rec in r.inter_batch_history if rec.get("n_pairs", 0) > 0]
        if ib_records:
            ib_frac_neg = sum(rec["frac_negative"] for rec in ib_records) / len(ib_records)
            ib_mean_cos = sum(rec["mean_cos"] for rec in ib_records) / len(ib_records)
        else:
            ib_frac_neg = float("nan")
            ib_mean_cos = float("nan")

        oob_total = r.forgetting.get("out_of_batch_total_magnitude", float("nan"))
        oob_per_step = r.summary.get("forgetting_oob_per_unit_step", float("nan"))
        ww = r.summary.get("ww_wasted_work_ratio_mean", float("nan"))

        print(f"{r.label:40s} {r.final_acc:>10.4f} {ib_frac_neg:>9.3f} {ib_mean_cos:>+10.3f} "
              f"{oob_total:>10.3f} {oob_per_step:>13.4f} {ww:>7.3f}")

    print("\nReading guide:")
    print("  IB_%neg     : fraction of in-batch class-pair gradient cosines that are NEGATIVE")
    print("                = direct measure of intra-batch class-gradient conflict (COSGD targets this)")
    print("  IB_⟨cos⟩    : mean of all in-batch class-pair cosines (positive = mostly aligned)")
    print("  OOB_total   : total out-of-batch forgetting magnitude across run (BoGrad targets this)")
    print("  OOB_per_step: OOB forgetting per unit parameter movement (step-magnitude-normalised)")
    print("  WW_K        : wasted-work ratio (1 = perfectly efficient, lower = more cancellation)")
    print()
    print(f"Done. Results at {out_dir}")


if __name__ == "__main__":
    main()
