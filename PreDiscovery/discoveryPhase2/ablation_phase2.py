"""
Phase-2 ablation: do any of the five enhancements actually help?

Bar to clear: each variant must beat its base optimiser's BASELINE
(SGD-vanilla, SGD+momentum, or Adam) — not just vanilla SGD. Phase 1 showed
that's where BoGrad actually needs to add value.

Same diagnostics as Phase 1, same model (SmallCNN), same dataset (CIFAR-10),
seeded for reproducibility.

Usage
-----
    python discoveryPhase2/ablation_phase2.py --epochs 5
    python discoveryPhase2/ablation_phase2.py --epochs 5 --quick
    python discoveryPhase2/ablation_phase2.py --regimes sgd_momentum,adam --epochs 5
    python discoveryPhase2/ablation_phase2.py --only sgd_momentum_complement --epochs 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.optimizers import BoGrad  # noqa: E402
from discovery.diagnostics import DiagnosticHarness  # noqa: E402
from discoveryPhase2.enhanced_variants import (  # noqa: E402
    AdaptiveTriggerBoGrad,
    ComplementAdam,
    ComplementMomentumSGD,
    GradientDifferenceBoGrad,
    MultiScaleBoGrad,
)


# ---------------------------------------------------------------------------
# Model + data (mirrors Phase 1 for direct comparability)
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


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    regime: str
    factory: Callable[[Any], torch.optim.Optimizer]
    note: str = ""


def build_variants(
    *, K_short: int = 4, K: int = 8, K_long: int = 32,
    perp_weight: float = 1.0, trigger_threshold: float = -0.3,
) -> List[Variant]:
    variants: List[Variant] = []

    # ---------------- SGD vanilla ----------------
    variants += [
        Variant(
            "sgd_vanilla_baseline", "sgd_vanilla",
            lambda m: torch.optim.SGD(m.parameters(), lr=0.05),
        ),
        Variant(
            "sgd_vanilla_trajectory_long", "sgd_vanilla",
            lambda m: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=K_long, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=0.05,
            ),
            note=f"V1: long (K={K_long}) trajectory buffer of applied deltas",
        ),
        # V2 (complement-aware) skipped for vanilla SGD — nothing to complement.
        Variant(
            "sgd_vanilla_adaptive", "sgd_vanilla",
            lambda m: AdaptiveTriggerBoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=K, projection_mode="negative",
                trigger_threshold=trigger_threshold,
                lr=0.05,
            ),
            note=f"V3: adaptive trigger (cos<{trigger_threshold})",
        ),
        Variant(
            "sgd_vanilla_grad_diff", "sgd_vanilla",
            lambda m: GradientDifferenceBoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=K, projection_mode="negative",
                lr=0.05,
            ),
            note="V4: buffer = gradient differences",
        ),
        Variant(
            "sgd_vanilla_multiscale", "sgd_vanilla",
            lambda m: MultiScaleBoGrad(
                m.parameters(), torch.optim.SGD,
                short_buffer_size=K_short, long_buffer_size=K_long,
                projection_mode="negative",
                lr=0.05,
            ),
            note=f"V5: short K={K_short} + long K={K_long}",
        ),
    ]

    # ---------------- SGD + momentum (β=0.9) ----------------
    variants += [
        Variant(
            "sgd_momentum_baseline", "sgd_momentum",
            lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9),
        ),
        Variant(
            "sgd_momentum_trajectory_long", "sgd_momentum",
            lambda m: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=K_long, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=0.05, momentum=0.9,
            ),
            note=f"V1: long (K={K_long}) trajectory of applied deltas",
        ),
        Variant(
            "sgd_momentum_complement", "sgd_momentum",
            lambda m: ComplementMomentumSGD(
                m.parameters(),
                lr=0.05, momentum=0.9, perp_weight=perp_weight,
            ),
            note=f"V2: momentum + γ={perp_weight} perp-to-velocity boost",
        ),
        Variant(
            "sgd_momentum_adaptive", "sgd_momentum",
            lambda m: AdaptiveTriggerBoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=K, projection_mode="negative",
                trigger_threshold=trigger_threshold,
                lr=0.05, momentum=0.9,
            ),
            note=f"V3: adaptive trigger (cos<{trigger_threshold})",
        ),
        Variant(
            "sgd_momentum_grad_diff", "sgd_momentum",
            lambda m: GradientDifferenceBoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=K, projection_mode="negative",
                lr=0.05, momentum=0.9,
            ),
            note="V4: buffer = gradient differences",
        ),
        Variant(
            "sgd_momentum_multiscale", "sgd_momentum",
            lambda m: MultiScaleBoGrad(
                m.parameters(), torch.optim.SGD,
                short_buffer_size=K_short, long_buffer_size=K_long,
                projection_mode="negative",
                lr=0.05, momentum=0.9,
            ),
            note=f"V5: short K={K_short} + long K={K_long}",
        ),
    ]

    # ---------------- Adam ----------------
    variants += [
        Variant(
            "adam_baseline", "adam",
            lambda m: torch.optim.Adam(m.parameters(), lr=1e-3),
        ),
        Variant(
            "adam_trajectory_long", "adam",
            lambda m: BoGrad(
                m.parameters(), torch.optim.Adam,
                buffer_size=K_long, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=1e-3,
            ),
            note=f"V1: long (K={K_long}) trajectory of applied deltas",
        ),
        Variant(
            "adam_complement", "adam",
            lambda m: ComplementAdam(
                m.parameters(),
                lr=1e-3, perp_weight=perp_weight,
            ),
            note=f"V2: Adam + γ={perp_weight} perp-to-update preconditioned boost",
        ),
        Variant(
            "adam_adaptive", "adam",
            lambda m: AdaptiveTriggerBoGrad(
                m.parameters(), torch.optim.Adam,
                buffer_size=K, projection_mode="negative",
                trigger_threshold=trigger_threshold,
                lr=1e-3,
            ),
            note=f"V3: adaptive trigger (cos<{trigger_threshold})",
        ),
        Variant(
            "adam_grad_diff", "adam",
            lambda m: GradientDifferenceBoGrad(
                m.parameters(), torch.optim.Adam,
                buffer_size=K, projection_mode="negative",
                lr=1e-3,
            ),
            note="V4: buffer = gradient differences",
        ),
        Variant(
            "adam_multiscale", "adam",
            lambda m: MultiScaleBoGrad(
                m.parameters(), torch.optim.Adam,
                short_buffer_size=K_short, long_buffer_size=K_long,
                projection_mode="negative",
                lr=1e-3,
            ),
            note=f"V5: short K={K_short} + long K={K_long}",
        ),
    ]

    return variants


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


def train_variant(
    variant: Variant,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    log_every: int,
    seed: int = 2026,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    optimizer = variant.factory(model)
    criterion = nn.CrossEntropyLoss()
    harness = DiagnosticHarness(model, log_every=log_every)

    label = f"[{variant.regime}] {variant.name}"
    if variant.note:
        label += f"  ({variant.note})"
    print(f"\n=== {label} ===")
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
            harness.before_step()
            optimizer.step()
            harness.after_step(loss=loss.item())

        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{epochs}  test_acc={acc:.4f}  elapsed={elapsed:.1f}s")

    final = epoch_acc[-1] if epoch_acc else 0.0
    best = max(epoch_acc) if epoch_acc else 0.0
    n = len(harness.history)
    late = harness.aggregate_window(last_k=max(n // 4, 1))

    extra: Dict[str, float] = {}
    if isinstance(optimizer, AdaptiveTriggerBoGrad):
        extra["trigger_rate"] = optimizer.trigger_rate()

    return {
        "variant": variant.name,
        "regime": variant.regime,
        "note": variant.note,
        "final_test_acc": final,
        "best_test_acc": best,
        "epoch_test_acc": epoch_acc,
        "wall_clock_s": time.time() - t0,
        "diagnostics_late": late,
        "extra": extra,
        "history": harness.history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--K-short", type=int, default=4)
    ap.add_argument("--K-long", type=int, default=32)
    ap.add_argument("--perp-weight", type=float, default=1.0)
    ap.add_argument("--trigger-threshold", type=float, default=-0.3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--quick", action="store_true",
                    help="1/4 of CIFAR-10 train set for fast iteration")
    ap.add_argument("--regimes", type=str, default=None,
                    help="Comma-separated subset of {sgd_vanilla, sgd_momentum, adam}")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated variant names (overrides --regimes)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    variants = build_variants(
        K_short=args.K_short, K=args.K, K_long=args.K_long,
        perp_weight=args.perp_weight, trigger_threshold=args.trigger_threshold,
    )
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        variants = [v for v in variants if v.name in wanted]
    elif args.regimes:
        wanted = {s.strip() for s in args.regimes.split(",") if s.strip()}
        variants = [v for v in variants if v.regime in wanted]

    if not variants:
        print("No variants selected. Exiting.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "discoveryPhase2" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_dir = out_dir / "histories"
    history_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}")

    results: List[Dict[str, Any]] = []
    for v in variants:
        try:
            r = train_variant(
                v, train_loader, test_loader, device,
                epochs=args.epochs, log_every=args.log_every, seed=args.seed,
            )
            results.append(r)
        except Exception as exc:
            print(f"  !! variant {v.name} failed: {exc}")
            results.append({"variant": v.name, "regime": v.regime, "error": str(exc)})

        if "history" in (results[-1] if results else {}):
            with (history_dir / f"{v.name}.json").open("w", encoding="utf-8") as fh:
                json.dump(results[-1]["history"], fh, indent=2)
            results[-1].pop("history")

    summary = {
        "config": vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "results": results,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ---- summary table, grouped by regime, sorted by accuracy within regime ----
    print("\n" + "=" * 92)
    print(f"{'Variant':45s} {'Final':>8s} {'Best':>8s} {'cos_g':>8s} {'cos_u':>8s} {'cos_ug':>8s} {'extra':>8s}")
    print("=" * 92)

    # group by regime, baseline first within group
    by_regime: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_regime.setdefault(r["regime"], []).append(r)

    for regime in ("sgd_vanilla", "sgd_momentum", "adam"):
        rows = by_regime.get(regime, [])
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: (0 if r["variant"].endswith("_baseline") else 1, r["variant"]))
        baseline_acc = next(
            (r.get("final_test_acc", float("nan"))
             for r in rows if r["variant"].endswith("_baseline") and "error" not in r),
            None,
        )
        for r in rows:
            if "error" in r:
                print(f"{r['variant']:45s}  ERROR: {r['error']}")
                continue
            late = r.get("diagnostics_late", {})
            cos_g = late.get("cos_g_prev_mean", float("nan"))
            cos_u = late.get("cos_u_prev_mean", float("nan"))
            cos_ug = late.get("cos_u_neg_g_mean", float("nan"))
            extra = r.get("extra", {})
            extra_str = ""
            if "trigger_rate" in extra:
                extra_str = f"t={extra['trigger_rate']:.2f}"
            marker = "  "
            if baseline_acc is not None and not r["variant"].endswith("_baseline"):
                delta = r["final_test_acc"] - baseline_acc
                if delta > 0.005:
                    marker = "++"
                elif delta < -0.01:
                    marker = "--"
                else:
                    marker = " ~"
            print(
                f"{r['variant']:45s} "
                f"{r['final_test_acc']:>8.4f} "
                f"{r['best_test_acc']:>8.4f} "
                f"{cos_g:>8.3f} "
                f"{cos_u:>8.3f} "
                f"{cos_ug:>8.3f} "
                f"{extra_str:>8s}  {marker}"
            )
        print("-" * 92)

    print(f"Done. Results at {out_dir}")
    print("Markers: ++ beats baseline by >0.5pt | ~ within 0.5pt | -- worse by >1pt")


if __name__ == "__main__":
    main()
