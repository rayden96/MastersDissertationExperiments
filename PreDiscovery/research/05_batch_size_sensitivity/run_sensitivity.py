"""
Batch-size sensitivity — does inter-batch interference shrink or grow
with batch size?

Single optimiser (SGD+momentum) sweeps batch size ∈ {16, 32, 64, 128,
256, 512}. LR scales linearly (`lr = lr_base * batch / 128`) so
per-step useful displacement is comparable.

Same metric pipeline as Stage 1 / Stage 4: InterferenceTracker
(WW_K, OOB), per-call inter-batch metric (IB_%neg, IB_<cos>),
test accuracy per epoch.

Usage
-----
    python research/05_batch_size_sensitivity/run_sensitivity.py
    python research/05_batch_size_sensitivity/run_sensitivity.py --quick
    python research/05_batch_size_sensitivity/run_sensitivity.py \
        --batch-sizes 32,128,512 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
from testing._common import (  # noqa: E402
    SmallCNN, build_cifar10, evaluate, make_test_loader, make_train_loader,
)
from stage1.inter_batch_metric import measure_inter_batch_interference  # noqa: E402


# ---------------------------------------------------------------------------
# Per-batch-size run
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    batch_size: int
    lr: float
    final_acc: float
    epoch_acc: List[float]
    inter_batch_history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    forgetting: Dict[str, Any]
    wasted_work: Dict[str, Any]
    n_steps: int
    wall_clock_s: float


def run_one(
    batch_size: int,
    lr: float,
    momentum: float,
    train_ds,
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
    num_workers: int,
) -> RunResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_loader = make_train_loader(
        train_ds, batch_size, num_workers,
        generator_seed=seed, pin_memory=(device.type == "cuda"),
    )

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

    print(f"\n=== batch={batch_size}  lr={lr:.4f} ===")
    t0 = time.time()
    epoch_acc: List[float] = []
    step_count = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if step_count % inter_batch_every == 0:
                ib = measure_inter_batch_interference(model, criterion, probe_set)
                ib["step"] = step_count
                inter_batch_history.append(ib)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            tracker.before_step()
            optimizer.step()
            tracker.after_step(loss=loss.item(),
                               batch_classes=set(y.unique().cpu().tolist()))
            step_count += 1

        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{epochs}  test_acc={acc:.4f}  "
              f"steps={step_count}  ({elapsed:.0f}s)")

    return RunResult(
        batch_size=batch_size,
        lr=lr,
        final_acc=epoch_acc[-1] if epoch_acc else float("nan"),
        epoch_acc=epoch_acc,
        inter_batch_history=inter_batch_history,
        summary=tracker.summary(),
        forgetting=tracker.forgetting.summary(),
        wasted_work=tracker.wasted_work.summary(),
        n_steps=step_count,
        wall_clock_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr-base", type=float, default=0.05,
                    help="LR at the reference batch (scaled linearly)")
    ap.add_argument("--lr-reference-batch", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-sizes", type=str, default="16,32,64,128,256,512")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--probe-n-per-class", type=int, default=64)
    ap.add_argument("--inter-batch-every", type=int, default=25)
    ap.add_argument("--pairwise-K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    batch_sizes: List[int] = [int(s) for s in args.batch_sizes.split(",") if s.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"epochs={args.epochs}  batches={batch_sizes}  "
          f"lr_base={args.lr_base}  ref_batch={args.lr_reference_batch}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(
        test_ds, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Building probe set...")
    probe = ClassProbeSet(
        test_ds, num_classes=10,
        n_per_class=args.probe_n_per_class,
        seed=args.seed, device=device,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "research" / "05_batch_size_sensitivity" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id},
                  fh, indent=2)

    results: Dict[int, RunResult] = {}
    for bs in batch_sizes:
        lr = args.lr_base * bs / args.lr_reference_batch
        try:
            r = run_one(
                bs, lr, args.momentum, train_ds, test_loader, probe, device,
                epochs=args.epochs,
                log_every=args.log_every, probe_every=args.probe_every,
                inter_batch_every=args.inter_batch_every,
                pairwise_K=args.pairwise_K,
                seed=args.seed, num_workers=args.num_workers,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"\n  !! batch={bs} failed: {exc}")
            continue

        results[bs] = r

        sub = out_dir / f"batch_{bs:04d}"
        sub.mkdir(exist_ok=True)
        with (sub / "inter_batch_history.json").open("w") as fh:
            json.dump(r.inter_batch_history, fh, indent=2)
        with (sub / "summary.json").open("w") as fh:
            json.dump(r.summary, fh, indent=2, default=str)
        with (sub / "result.json").open("w") as fh:
            json.dump({
                "batch_size": r.batch_size,
                "lr": r.lr,
                "final_acc": r.final_acc,
                "epoch_acc": r.epoch_acc,
                "n_steps": r.n_steps,
                "wall_clock_s": r.wall_clock_s,
                "forgetting": r.forgetting,
                "wasted_work": r.wasted_work,
            }, fh, indent=2, default=str)

    # ---------- Console summary ----------
    print("\n" + "=" * 100)
    print("BATCH-SIZE SENSITIVITY — IB_%neg and WW_K vs batch size")
    print("=" * 100)
    print(f"{'batch':>6s} {'lr':>8s} {'final_acc':>10s} {'IB_%neg':>9s} "
          f"{'IB_<cos>':>10s} {'WW_K':>7s} {'n_steps':>8s}")
    print("-" * 100)
    for bs in sorted(results.keys()):
        r = results[bs]
        ib = [rec for rec in r.inter_batch_history if rec.get("n_pairs", 0) > 0]
        ib_pct_neg = sum(rec["frac_negative"] for rec in ib) / len(ib) if ib else float("nan")
        ib_mean_cos = sum(rec["mean_cos"] for rec in ib) / len(ib) if ib else float("nan")
        ww = r.summary.get("ww_wasted_work_ratio_mean", float("nan"))
        print(f"{bs:>6d} {r.lr:>8.4f} {r.final_acc:>10.4f} {ib_pct_neg:>9.3f} "
              f"{ib_mean_cos:>+10.3f} {ww:>7.3f} {r.n_steps:>8d}")

    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
