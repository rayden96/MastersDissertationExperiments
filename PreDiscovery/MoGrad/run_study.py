"""
MoGrad study — multi-trial sweep over projection mode and start_step.

For each (mode, start_step) combo: 3 trials × 1 epoch CIFAR-10. Plus
SGD baseline and SGD+momentum baseline as references (3 trials each).

Reports mean ± std final accuracy. Tracks the projection-rate (how often
projection actually fires after warmup) for each MoGrad variant.

Usage
-----
    python MoGrad/run_study.py
    python MoGrad/run_study.py --modes negative --start-steps 100,200
    python MoGrad/run_study.py --quick
    python MoGrad/run_study.py --trials 5
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
from typing import Any, Callable, Dict, List

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

from MoGrad.mograd import MoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, build_cifar10, evaluate, make_test_loader, make_train_loader,
)


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
    label: str
    final_acc: float
    epoch_acc: List[float]
    projection_rate: float
    wall_clock_s: float


def train_one(
    label: str,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int,
) -> TrialResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    optimizer = optimizer_factory(model)
    criterion = nn.CrossEntropyLoss()

    print(f"  {label}", end="", flush=True)
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
            optimizer.step()
        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        print(f"  e{epoch + 1}={acc:.4f}", end="", flush=True)

    elapsed = time.time() - t0
    proj_rate = optimizer.projection_rate() if hasattr(optimizer, "projection_rate") else float("nan")
    print(f"  proj_rate={proj_rate:.3f}  ({elapsed:.0f}s)")

    return TrialResult(
        label=label,
        final_acc=epoch_acc[-1] if epoch_acc else float("nan"),
        epoch_acc=epoch_acc,
        projection_rate=proj_rate,
        wall_clock_s=elapsed,
    )


def aggregate(values: List[float]) -> Dict[str, float]:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(finite) == 1:
        return {"mean": float(finite[0]), "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(finite),
        "std": statistics.stdev(finite),
        "n": len(finite),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--modes", type=str, default="full,negative,positive")
    ap.add_argument("--start-steps", type=str, default="50,100,200,500,1000")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum-beta", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()

    modes: List[str] = [s.strip() for s in args.modes.split(",") if s.strip()]
    start_steps: List[int] = [int(s) for s in args.start_steps.split(",") if s.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Trials: {args.trials} | Epochs: {args.epochs} | "
          f"modes: {modes} | start_steps: {start_steps}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "MoGrad" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id}, fh, indent=2)

    # results[config_label] = list of TrialResult
    results: Dict[str, List[TrialResult]] = {}

    n_baselines = 2
    n_mograd = len(modes) * len(start_steps)
    total = (n_baselines + n_mograd) * args.trials
    run_idx = 0

    for trial in range(args.trials):
        seed = args.base_seed + trial * 1000
        print(f"=== Trial {trial + 1}/{args.trials} (seed={seed}) ===")

        # Baselines
        for label, factory in [
            ("sgd_vanilla", lambda m: torch.optim.SGD(m.parameters(), lr=args.lr)),
            ("sgd_momentum", lambda m: torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum_beta)),
        ]:
            train_loader = make_train_loader(
                train_ds, args.batch_size, args.num_workers,
                generator_seed=seed, pin_memory=(device.type == "cuda"),
            )
            run_idx += 1
            r = train_one(
                f"[{run_idx}/{total}] {label}",
                factory, train_loader, test_loader, device,
                epochs=args.epochs, seed=seed,
            )
            results.setdefault(label, []).append(r)
            with (out_dir / f"{label}__t{trial}.json").open("w") as fh:
                json.dump(vars(r), fh, indent=2)

        # MoGrad variants
        for mode in modes:
            for start_step in start_steps:
                cfg = f"mograd_{mode}_start{start_step}"
                train_loader = make_train_loader(
                    train_ds, args.batch_size, args.num_workers,
                    generator_seed=seed, pin_memory=(device.type == "cuda"),
                )
                run_idx += 1
                r = train_one(
                    f"[{run_idx}/{total}] {cfg}",
                    lambda m, mode=mode, start_step=start_step: MoGrad(
                        m.parameters(), lr=args.lr,
                        momentum_beta=args.momentum_beta,
                        start_step=start_step,
                        projection_mode=mode,
                    ),
                    train_loader, test_loader, device,
                    epochs=args.epochs, seed=seed,
                )
                results.setdefault(cfg, []).append(r)
                with (out_dir / f"{cfg}__t{trial}.json").open("w") as fh:
                    json.dump(vars(r), fh, indent=2)

    # ---------- Summary ----------
    summary: Dict[str, Any] = {}
    for cfg, trials in results.items():
        accs = [t.final_acc for t in trials]
        proj = [t.projection_rate for t in trials]
        summary[cfg] = {
            "final_acc": aggregate(accs),
            "projection_rate": aggregate(proj),
            "n_trials": len(trials),
        }
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Console table
    print("\n" + "=" * 92)
    print(f"MoGrad STUDY — {args.trials} trials × {args.epochs} epochs CIFAR-10")
    print("=" * 92)
    print(f"{'config':32s} {'final_acc':>20s} {'proj_rate':>15s} {'Δvs.SGD+mom':>14s}")
    print("-" * 92)

    base_mom = summary.get("sgd_momentum", {}).get("final_acc", {}).get("mean", float("nan"))

    # Print baselines first
    for label in ("sgd_vanilla", "sgd_momentum"):
        if label not in summary:
            continue
        s = summary[label]
        acc = s["final_acc"]
        delta = acc["mean"] - base_mom if (acc["n"] and math.isfinite(base_mom)) else float("nan")
        print(f"{label:32s} {acc['mean']:>10.4f}±{acc['std']:.4f}    "
              f"{'—':>15s} {delta:>+14.4f}")
    print("-" * 92)

    # Then MoGrad variants
    for mode in modes:
        for start_step in start_steps:
            cfg = f"mograd_{mode}_start{start_step}"
            if cfg not in summary:
                continue
            s = summary[cfg]
            acc = s["final_acc"]
            proj = s["projection_rate"]
            delta = acc["mean"] - base_mom if (acc["n"] and math.isfinite(base_mom)) else float("nan")
            print(f"{cfg:32s} {acc['mean']:>10.4f}±{acc['std']:.4f}    "
                  f"{proj['mean']:>10.3f}±{proj['std']:.3f} {delta:>+14.4f}")
        print("-" * 92)

    print(f"\nReading guide:")
    print(f"  proj_rate    : fraction of post-warmup steps where projection actually fired.")
    print(f"                 'full' should be ~1.0 (always fires when ‖m‖>0).")
    print(f"                 'negative' fires when cos(g, m) < 0 — typically <50%.")
    print(f"                 'positive' fires when cos(g, m) > 0 — typically >50%.")
    print(f"  Δvs.SGD+mom  : final_acc mean - SGD+momentum baseline mean.")
    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
