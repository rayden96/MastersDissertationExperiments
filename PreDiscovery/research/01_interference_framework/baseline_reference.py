"""
Framework validation run — single-epoch SmallCNN on CIFAR-10 with full
interference diagnostics.

Purpose:
  1. Verify InterferenceTracker, ClassProbeSet, ForgettingTracker, and
     WastedWorkTracker all wire up and produce sane numbers.
  2. Establish a *reference profile* for the SGD+momentum baseline that
     subsequent experiments can compare against.
  3. Sanity-check synthetic-experiment expectations: the metrics on a
     normal CIFAR-10 run should be qualitatively reasonable (cos_g_prev
     should be small but non-zero, forgetting should occur sometimes
     but not on every step, etc.).

Output:
  research/01_interference_framework/results/baseline_reference/run_<timestamp>/
    history.json    — per-step diagnostic rows
    summary.json    — aggregated metrics
    forgetting.json — ForgettingTracker.summary()
    wasted_work.json — WastedWorkTracker.summary()
    config.json     — run config + git sha if available

Usage:
    python research/01_interference_framework/baseline_reference.py
    python research/01_interference_framework/baseline_reference.py --epochs 2 --probe-every 5
    python research/01_interference_framework/baseline_reference.py --quick
    python research/01_interference_framework/baseline_reference.py --include-bograd
        --include-bograd: run a paired BoGrad run with the same diagnostics for comparison
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


def run_one_diagnostic_run(
    name: str,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe_set: ClassProbeSet,
    device: torch.device,
    epochs: int,
    log_every: int,
    probe_every: int,
    wasted_work_K: int,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    optimizer = optimizer_factory(model)
    criterion = nn.CrossEntropyLoss()

    tracker = InterferenceTracker(
        model, criterion,
        probe_set=probe_set,
        log_every=log_every,
        probe_every=probe_every,
        wasted_work_K=wasted_work_K,
        device=device,
    )

    print(f"\n=== {name} ===")
    t0 = time.time()
    epoch_acc: List[float] = []
    step_count = 0

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
            step_count += 1

        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{epochs}  test_acc={acc:.4f}  steps={step_count}  ({elapsed:.0f}s)")

    return {
        "name": name,
        "history": tracker.get_history(),
        "summary": tracker.summary(),
        "forgetting": tracker.forgetting.summary(),
        "wasted_work": tracker.wasted_work.summary(),
        "epoch_test_acc": epoch_acc,
        "wall_clock_s": time.time() - t0,
        "total_steps": step_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1,
                    help="How many epochs to run for the diagnostic capture")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="1/4 of CIFAR-10 train set")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--probe-n-per-class", type=int, default=64)
    ap.add_argument("--wasted-work-K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--include-bograd", action="store_true",
                    help="Also run BoGrad on both SGD-vanilla and SGD+momentum")
    ap.add_argument("--bograd-K-momentum", type=int, default=32,
                    help="K for BoGrad on SGD+momentum (update-stage)")
    ap.add_argument("--bograd-K-vanilla", type=int, default=8,
                    help="K for BoGrad on SGD-vanilla (gradient-stage)")
    ap.add_argument("--skip-vanilla", action="store_true",
                    help="Skip the SGD-vanilla (no momentum) baseline run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Config: lr={args.lr} momentum={args.momentum} batch={args.batch_size} "
          f"epochs={args.epochs} log_every={args.log_every} probe_every={args.probe_every}")

    # Data
    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    def fresh_train_loader():
        """Re-create the loader with the same generator seed so each run
        sees the same batch order — essential for paired comparisons."""
        return make_train_loader(
            train_ds, args.batch_size, args.num_workers,
            generator_seed=args.seed, pin_memory=(device.type == "cuda"),
        )

    # Probe set: balanced per-class subset of test set
    print("Building per-class probe set...")
    probe = ClassProbeSet(
        test_ds, num_classes=10,
        n_per_class=args.probe_n_per_class,
        seed=args.seed, device=device,
    )
    print(f"  Probe size: {probe.total_size()} examples ({args.probe_n_per_class} per class)")

    # Output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "research" / "01_interference_framework" / "results" / "baseline_reference"
    out_dir = out_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Save config
    with (out_dir / "config.json").open("w") as fh:
        json.dump({
            "args": vars(args),
            "device": str(device),
            "torch_version": torch.__version__,
            "run_id": run_id,
        }, fh, indent=2)

    runs: Dict[str, Any] = {}

    # Build the run list. Each entry: (key, display_name, optimizer_factory).
    runs_to_do: List[tuple] = []

    if not args.skip_vanilla:
        runs_to_do.append((
            "sgd_vanilla",
            "SGD vanilla (no momentum)",
            lambda m: torch.optim.SGD(m.parameters(), lr=args.lr),
        ))

    runs_to_do.append((
        "sgd_momentum",
        "SGD+momentum baseline",
        lambda m: torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum),
    ))

    if args.include_bograd:
        if not args.skip_vanilla:
            runs_to_do.append((
                "bograd_sgd_vanilla",
                f"BoGrad+SGD vanilla (gradient-stage K={args.bograd_K_vanilla} neg)",
                lambda m: BoGrad(
                    m.parameters(), torch.optim.SGD,
                    buffer_size=args.bograd_K_vanilla, project_stage="gradient",
                    projection_mode="negative", orth_method="sequential",
                    lr=args.lr,
                ),
            ))
        runs_to_do.append((
            "bograd_sgd_momentum",
            f"BoGrad+SGD+momentum (update-stage K={args.bograd_K_momentum} neg)",
            lambda m: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=args.bograd_K_momentum, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=args.lr, momentum=args.momentum,
            ),
        ))

    for run_key, display_name, factory in runs_to_do:
        runs[run_key] = run_one_diagnostic_run(
            display_name, factory,
            fresh_train_loader(), test_loader, probe, device,
            epochs=args.epochs, log_every=args.log_every, probe_every=args.probe_every,
            wasted_work_K=args.wasted_work_K, seed=args.seed,
        )

    # Save outputs
    for run_name, run_data in runs.items():
        run_subdir = out_dir / run_name
        run_subdir.mkdir(exist_ok=True)
        with (run_subdir / "history.json").open("w") as fh:
            json.dump(run_data["history"], fh, indent=2)
        with (run_subdir / "summary.json").open("w") as fh:
            json.dump(run_data["summary"], fh, indent=2, default=str)
        with (run_subdir / "forgetting.json").open("w") as fh:
            json.dump(run_data["forgetting"], fh, indent=2, default=str)
        with (run_subdir / "wasted_work.json").open("w") as fh:
            json.dump(run_data["wasted_work"], fh, indent=2)
        with (run_subdir / "epoch_acc.json").open("w") as fh:
            json.dump(run_data["epoch_test_acc"], fh, indent=2)

    # Console summary
    print("\n" + "=" * 92)
    print("BASELINE REFERENCE — diagnostic summary")
    print("=" * 92)
    for run_name, run_data in runs.items():
        s = run_data["summary"]
        f = run_data["forgetting"]
        w = run_data["wasted_work"]
        print(f"\n--- {run_name} ---")
        print(f"  steps logged: {s.get('n_logged_steps', 0)}")
        print(f"  final acc: {run_data['epoch_test_acc'][-1] if run_data['epoch_test_acc'] else float('nan'):.4f}")

        cos_g = s.get("cos_g_prev_mean", float("nan"))
        cos_u = s.get("cos_u_prev_mean", float("nan"))
        cos_ug = s.get("cos_u_neg_g_mean", float("nan"))
        print(f"  geometric:    cos(g,g_prev)={cos_g:>+.4f}  "
              f"cos(u,u_prev)={cos_u:>+.4f}  cos(u,-g)={cos_ug:>+.4f}")

        fwd_evs = s.get("forgetting_events_mean", float("nan"))
        fwd_mag = s.get("forgetting_magnitude_total_mean", float("nan"))
        print(f"  forgetting:   per-step events_mean={fwd_evs:.3f}  "
              f"magnitude_mean={fwd_mag:.4f}  total_events={f.get('total_events', 0)}")

        ww = s.get("ww_wasted_work_ratio_mean", float("nan"))
        net_disp = s.get("net_displacement", float("nan"))
        print(f"  wasted-work:  ratio_mean={ww:.4f}  (window K={w.get('window_K', '?')})  "
              f"net_disp={net_disp:.3f}")

        # Run-level normalised forgetting (framework.md §4.2)
        f_per_step = s.get("forgetting_per_unit_step_run", float("nan"))
        f_per_progress = s.get("forgetting_per_progress", float("nan"))
        progress = s.get("loss_progress", float("nan"))
        print(f"  normalised:   Δ/Σ‖u‖={f_per_step:.4f}  Δ/progress={f_per_progress:.4f}  "
              f"(progress={progress:+.3f})")

        for k in (1, 4, 16):
            traj = s.get(f"cos_u_lag{k}_mean", None)
            if traj is not None:
                print(f"  trajectory:   cos(u_t, u_{{t-{k}}})={traj:>+.4f}")

    print("\n" + "=" * 92)
    print(f"Done. Results at {out_dir}")


if __name__ == "__main__":
    main()
