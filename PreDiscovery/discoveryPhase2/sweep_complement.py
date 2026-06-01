"""
Multi-trial parameter sweep over Complement-variant `perp_weight` (γ).

Tests four base optimisers:
  - SGD + momentum
  - RMSprop  + momentum
  - Adam
  - SignSGD + momentum

For each base optimiser we run:
  - the BASELINE (no complement)
  - several γ values via the matching ComplementXxx optimiser

Each (optimiser × γ) config is run for `--trials` independent seeds. Within a
trial we use the same data-shuffling generator across all configs so that
variant comparisons within a trial are paired (same minibatches, same model
init seed).

Output
------
    discoveryPhase2/results/sweep_<timestamp>/
        results.json              everything
        per_trial/<config>.json   per-trial epoch curves

Console output is grouped by optimiser family with γ rows showing
mean ± std across trials and the delta vs baseline.

Usage
-----
    python discoveryPhase2/sweep_complement.py                       # defaults
    python discoveryPhase2/sweep_complement.py --epochs 5 --trials 3
    python discoveryPhase2/sweep_complement.py --gammas 0.25,0.5,1.0,2.0
    python discoveryPhase2/sweep_complement.py --quick                # 1/4 train set
    python discoveryPhase2/sweep_complement.py --bases sgd_momentum,rmsprop
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from discoveryPhase2.enhanced_variants import (  # noqa: E402
    ComplementAdam,
    ComplementMomentumSGD,
    ComplementRMSprop,
    ComplementSignSGD,
    SignSGD,
)


# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_cifar10(data_root: Path, download: bool, quick: bool):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(str(data_root), train=True, download=download, transform=tf)
    test = datasets.CIFAR10(str(data_root), train=False, download=download, transform=tf)
    if quick:
        train = Subset(train, list(range(0, len(train), 4)))
        test = Subset(test, list(range(0, len(test), 2)))
    return train, test


def make_train_loader(dataset, batch_size, num_workers, generator_seed, pin_memory):
    g = torch.Generator()
    g.manual_seed(generator_seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, generator=g,
    )


# ---------------------------------------------------------------------------
# Per-base optimiser families
# ---------------------------------------------------------------------------
@dataclass
class OptFamily:
    name: str
    lr: float
    base_factory: Callable[[Any], torch.optim.Optimizer]
    complement_factory: Callable[[Any, float], torch.optim.Optimizer]   # (model, gamma) -> opt
    note: str = ""


def build_families() -> Dict[str, OptFamily]:
    return {
        "sgd_momentum": OptFamily(
            name="sgd_momentum",
            lr=0.05,
            base_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9),
            complement_factory=lambda m, gamma: ComplementMomentumSGD(
                m.parameters(), lr=0.05, momentum=0.9, perp_weight=gamma,
            ),
            note="lr=0.05, momentum=0.9",
        ),
        "rmsprop": OptFamily(
            name="rmsprop",
            lr=1e-3,
            base_factory=lambda m: torch.optim.RMSprop(
                m.parameters(), lr=1e-3, alpha=0.99, momentum=0.9,
            ),
            complement_factory=lambda m, gamma: ComplementRMSprop(
                m.parameters(), lr=1e-3, alpha=0.99, momentum=0.9, perp_weight=gamma,
            ),
            note="lr=1e-3, alpha=0.99, momentum=0.9",
        ),
        "adam": OptFamily(
            name="adam",
            lr=1e-3,
            base_factory=lambda m: torch.optim.Adam(m.parameters(), lr=1e-3),
            complement_factory=lambda m, gamma: ComplementAdam(
                m.parameters(), lr=1e-3, perp_weight=gamma,
            ),
            note="lr=1e-3 (Fix A: decompose against m̂)",
        ),
        "signsgd_momentum": OptFamily(
            name="signsgd_momentum",
            lr=1e-3,
            base_factory=lambda m: SignSGD(m.parameters(), lr=1e-3, momentum=0.9),
            complement_factory=lambda m, gamma: ComplementSignSGD(
                m.parameters(), lr=1e-3, momentum=0.9, perp_weight=gamma,
            ),
            note="lr=1e-3, momentum=0.9",
        ),
    }


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_run(
    label: str,
    optimizer_factory: Callable[[Any], torch.optim.Optimizer],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int,
) -> Dict[str, Any]:
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
    print(f"  ({elapsed:.0f}s)")

    return {
        "final_test_acc": epoch_acc[-1] if epoch_acc else float("nan"),
        "best_test_acc": max(epoch_acc) if epoch_acc else float("nan"),
        "epoch_test_acc": epoch_acc,
        "wall_clock_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--gammas", type=str, default="0.25,0.5,1.0,2.0",
                    help="Comma-separated perp_weight values to sweep")
    ap.add_argument("--bases", type=str, default="sgd_momentum,rmsprop,adam,signsgd_momentum",
                    help="Comma-separated optimiser families to test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="1/4 of CIFAR-10 train set for fast iteration")
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()

    gammas: List[float] = [float(s) for s in args.gammas.split(",") if s.strip()]
    base_names: List[str] = [s.strip() for s in args.bases.split(",") if s.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Trials: {args.trials} | Epochs: {args.epochs} | gammas: {gammas} | bases: {base_names}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    families = build_families()
    selected = [families[n] for n in base_names if n in families]
    if not selected:
        print(f"No valid families in {base_names}. Choices: {list(families.keys())}")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "discoveryPhase2" / "results" / f"sweep_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial_dir = out_dir / "per_trial"
    per_trial_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}\n")

    # results[fam_name][config_label] = list of trial dicts
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    total_runs = len(selected) * (1 + len(gammas)) * args.trials
    run_idx = 0

    for trial in range(args.trials):
        trial_seed = args.base_seed + trial * 1000
        print(f"=== Trial {trial + 1}/{args.trials} (seed={trial_seed}) ===")

        for fam in selected:
            print(f"  -- {fam.name} ({fam.note}) --")
            results.setdefault(fam.name, {})

            # Baseline
            train_loader = make_train_loader(
                train_ds, args.batch_size, args.num_workers,
                generator_seed=trial_seed, pin_memory=(device.type == "cuda"),
            )
            run_idx += 1
            label = f"[{run_idx}/{total_runs}] baseline"
            r = train_run(
                label, fam.base_factory, train_loader, test_loader,
                device, args.epochs, seed=trial_seed,
            )
            r["trial"] = trial
            r["gamma"] = 0.0
            results[fam.name].setdefault("baseline", []).append(r)
            with (per_trial_dir / f"{fam.name}__baseline__t{trial}.json").open("w") as fh:
                json.dump(r, fh, indent=2)

            # Gamma sweep
            for gamma in gammas:
                train_loader = make_train_loader(
                    train_ds, args.batch_size, args.num_workers,
                    generator_seed=trial_seed, pin_memory=(device.type == "cuda"),
                )
                run_idx += 1
                cfg_label = f"complement_g{gamma}"
                label = f"[{run_idx}/{total_runs}] {cfg_label}"
                r = train_run(
                    label, lambda m, g=gamma: fam.complement_factory(m, g),
                    train_loader, test_loader, device, args.epochs, seed=trial_seed,
                )
                r["trial"] = trial
                r["gamma"] = gamma
                results[fam.name].setdefault(cfg_label, []).append(r)
                with (per_trial_dir / f"{fam.name}__{cfg_label}__t{trial}.json").open("w") as fh:
                    json.dump(r, fh, indent=2)

    # Save complete results.
    summary = {
        "config": vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "gammas": gammas,
        "bases": base_names,
        "results": results,
    }
    with (out_dir / "results.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- Aggregated summary table ----
    print("\n" + "=" * 92)
    print("SWEEP SUMMARY — final test accuracy (mean ± std across trials)")
    print("=" * 92)

    for fam_name in base_names:
        if fam_name not in results:
            continue
        fam_results = results[fam_name]
        baseline_finals = [t["final_test_acc"] for t in fam_results.get("baseline", [])]
        baseline_agg = aggregate(baseline_finals)

        print(f"\n--- {fam_name} ({families[fam_name].note}) ---")
        print(f"{'γ':>6s}  {'mean':>7s}  {'std':>7s}  {'best':>7s}  {'Δvs.base':>10s}  {'trials'}")
        # baseline row
        baseline_bests = [max(t["epoch_test_acc"]) if t["epoch_test_acc"] else float("nan")
                          for t in fam_results.get("baseline", [])]
        print(
            f"{'0.00':>6s}  "
            f"{baseline_agg['mean']:>7.4f}  "
            f"{baseline_agg['std']:>7.4f}  "
            f"{aggregate(baseline_bests)['mean']:>7.4f}  "
            f"{'(baseline)':>10s}  "
            f"{baseline_agg['n']}"
        )
        for gamma in gammas:
            cfg_label = f"complement_g{gamma}"
            trials = fam_results.get(cfg_label, [])
            finals = [t["final_test_acc"] for t in trials]
            bests = [max(t["epoch_test_acc"]) if t["epoch_test_acc"] else float("nan") for t in trials]
            agg = aggregate(finals)
            delta = agg["mean"] - baseline_agg["mean"] if math.isfinite(agg["mean"]) and math.isfinite(baseline_agg["mean"]) else float("nan")
            marker = ""
            if math.isfinite(delta):
                if delta > 0.01:
                    marker = "++"
                elif delta > 0.0:
                    marker = "+"
                elif delta < -0.01:
                    marker = "--"
                else:
                    marker = "-"
            print(
                f"{gamma:>6.2f}  "
                f"{agg['mean']:>7.4f}  "
                f"{agg['std']:>7.4f}  "
                f"{aggregate(bests)['mean']:>7.4f}  "
                f"{delta:>+9.4f}{marker:>1s}  "
                f"{agg['n']}"
            )

    print("\n" + "=" * 92)
    print(f"Markers: ++ >+1pt | + 0..+1pt | - −1..0pt | -- <−1pt vs baseline mean")
    print(f"Done. Results at {out_dir}")


if __name__ == "__main__":
    main()
