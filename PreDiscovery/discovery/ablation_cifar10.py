"""
BoGrad ablation on CIFAR-10 — does any variant actually help?

Three optimiser regimes are tested independently:
  - SGD vanilla (no momentum, no weight decay)
  - SGD + momentum (β=0.9)
  - Adam

For each regime, several BoGrad variants are run alongside the baseline.
Adam additionally gets two flanking-LR baselines, so we can tell whether any
"BoGrad helps Adam" effect is the projection or just the implicit LR boost
that comes from norm-reduction shrinking v_t.

Diagnostics (cos(g_t,g_{t-1}), cos(u_t,u_{t-1}), descent quality, norms) are
captured every `log_every` steps and saved per-variant. The final summary
table prints test accuracy and the late-training average of each diagnostic
so you can see whether a variant that improved accuracy also actually
decorrelated the *update* trajectory (vs only the gradient trajectory).

Usage
-----
    python discovery/ablation_cifar10.py
    python discovery/ablation_cifar10.py --epochs 3 --quick
    python discovery/ablation_cifar10.py --only adam_baseline,adam_bograd_natural_neg
    python discovery/ablation_cifar10.py --regimes sgd_momentum,adam --epochs 5
"""

from __future__ import annotations

import argparse
import json
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

from common.optimizers import BoGrad  # noqa: E402
from discovery.bograd_variants import AdamWithBoGrad  # noqa: E402
from discovery.diagnostics import DiagnosticHarness  # noqa: E402


# ---------------------------------------------------------------------------
# Model
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_cifar10(data_root: Path, download: bool, quick: bool):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(str(data_root), train=True, download=download, transform=tf)
    test = datasets.CIFAR10(str(data_root), train=False, download=download, transform=tf)
    if quick:
        # 1/4 of the train set, 1/2 of the test set — enough to compare ranks fast.
        train = Subset(train, list(range(0, len(train), 4)))
        test = Subset(test, list(range(0, len(test), 2)))
    return train, test


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    regime: str  # "sgd_vanilla", "sgd_momentum", "adam"
    factory: Callable[[Any], torch.optim.Optimizer]
    note: str = ""


def build_variants(buffer_size: int) -> List[Variant]:
    K = buffer_size
    variants: List[Variant] = []

    # ---- SGD vanilla (no momentum, no weight decay) ----
    variants += [
        Variant(
            "sgd_vanilla_baseline", "sgd_vanilla",
            lambda m: torch.optim.SGD(m.parameters(), lr=0.05),
        ),
        Variant(
            "sgd_vanilla_bograd_grad_full", "sgd_vanilla",
            lambda m: BoGrad(m.parameters(), torch.optim.SGD,
                             buffer_size=K, project_stage="gradient",
                             projection_mode="full", orth_method="sequential",
                             lr=0.05),
            note="Original BOSGD, full projection",
        ),
        Variant(
            "sgd_vanilla_bograd_grad_neg", "sgd_vanilla",
            lambda m: BoGrad(m.parameters(), torch.optim.SGD,
                             buffer_size=K, project_stage="gradient",
                             projection_mode="negative", orth_method="sequential",
                             lr=0.05),
            note="Asymmetric (PCGrad-style); only kills destructive interference",
        ),
    ]

    # ---- SGD + momentum (β=0.9) ----
    variants += [
        Variant(
            "sgd_momentum_baseline", "sgd_momentum",
            lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9),
        ),
        Variant(
            "sgd_momentum_bograd_grad_full", "sgd_momentum",
            lambda m: BoGrad(m.parameters(), torch.optim.SGD,
                             buffer_size=K, project_stage="gradient",
                             projection_mode="full", orth_method="sequential",
                             lr=0.05, momentum=0.9),
            note="Project g, then momentum picks it up",
        ),
        Variant(
            "sgd_momentum_bograd_grad_neg", "sgd_momentum",
            lambda m: BoGrad(m.parameters(), torch.optim.SGD,
                             buffer_size=K, project_stage="gradient",
                             projection_mode="negative", orth_method="sequential",
                             lr=0.05, momentum=0.9),
        ),
        Variant(
            "sgd_momentum_bograd_update_full", "sgd_momentum",
            lambda m: BoGrad(m.parameters(), torch.optim.SGD,
                             buffer_size=K, project_stage="update",
                             projection_mode="full", orth_method="sequential",
                             lr=0.05, momentum=0.9),
            note="Project the momentum-applied delta (≡ projecting velocity)",
        ),
        Variant(
            "sgd_momentum_bograd_update_neg", "sgd_momentum",
            lambda m: BoGrad(m.parameters(), torch.optim.SGD,
                             buffer_size=K, project_stage="update",
                             projection_mode="negative", orth_method="sequential",
                             lr=0.05, momentum=0.9),
        ),
    ]

    # ---- Adam ----
    variants += [
        Variant(
            "adam_baseline", "adam",
            lambda m: torch.optim.Adam(m.parameters(), lr=1e-3),
        ),
        Variant(
            "adam_baseline_lr_low", "adam",
            lambda m: torch.optim.Adam(m.parameters(), lr=5e-4),
            note="Flanking-LR control (rules out implicit LR boost from norm reduction)",
        ),
        Variant(
            "adam_baseline_lr_high", "adam",
            lambda m: torch.optim.Adam(m.parameters(), lr=2e-3),
            note="Flanking-LR control",
        ),
        Variant(
            "adam_bograd_grad_full", "adam",
            lambda m: AdamWithBoGrad(m.parameters(), lr=1e-3,
                                     buffer_size=K, projection_point="grad",
                                     projection_mode="full"),
            note="Geometrically incoherent; reference",
        ),
        Variant(
            "adam_bograd_grad_neg", "adam",
            lambda m: AdamWithBoGrad(m.parameters(), lr=1e-3,
                                     buffer_size=K, projection_point="grad",
                                     projection_mode="negative"),
        ),
        Variant(
            "adam_bograd_momentum_neg", "adam",
            lambda m: AdamWithBoGrad(m.parameters(), lr=1e-3,
                                     buffer_size=K, projection_point="momentum",
                                     projection_mode="negative"),
            note="Project m_t (the EMA), don't overwrite EMA state",
        ),
        Variant(
            "adam_bograd_natural_neg", "adam",
            lambda m: AdamWithBoGrad(m.parameters(), lr=1e-3,
                                     buffer_size=K, projection_point="natural",
                                     projection_mode="negative"),
            note="Project g_t in v-weighted inner product (Adam's natural metric)",
        ),
        Variant(
            "adam_bograd_update_neg", "adam",
            lambda m: AdamWithBoGrad(m.parameters(), lr=1e-3,
                                     buffer_size=K, projection_point="update",
                                     projection_mode="negative"),
            note="Project the final Adam update u_t",
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

    # Late-training diagnostic average (last quarter of logged steps)
    n = len(harness.history)
    late = harness.aggregate_window(last_k=max(n // 4, 1))

    return {
        "variant": variant.name,
        "regime": variant.regime,
        "note": variant.note,
        "final_test_acc": final,
        "best_test_acc": best,
        "epoch_test_acc": epoch_acc,
        "wall_clock_s": time.time() - t0,
        "diagnostics_late": late,
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
    ap.add_argument("--buffer-size", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--quick", action="store_true",
                    help="Use 1/4 of CIFAR-10 train set for fast sanity checks")
    ap.add_argument("--regimes", type=str, default=None,
                    help="Comma-separated subset of {sgd_vanilla, sgd_momentum, adam}")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated variant names to run (overrides --regimes)")
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

    variants = build_variants(buffer_size=args.buffer_size)
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
    out_dir = ROOT / "discovery" / "results" / f"run_{run_id}"
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

        # Save history per variant immediately so a crash mid-run doesn't lose data.
        if "history" in (results[-1] if results else {}):
            with (history_dir / f"{v.name}.json").open("w", encoding="utf-8") as fh:
                json.dump(results[-1]["history"], fh, indent=2)
            results[-1].pop("history")

    # Save aggregated results.
    summary = {
        "config": vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "results": results,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ---- Pretty-print summary ----
    print("\n" + "=" * 80)
    print(f"{'Variant':45s} {'Final':>8s} {'Best':>8s} {'cos_g':>8s} {'cos_u':>8s} {'cos_ug':>8s}")
    print("=" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['variant']:45s}  ERROR: {r['error']}")
            continue
        late = r.get("diagnostics_late", {})
        cos_g = late.get("cos_g_prev_mean", float("nan"))
        cos_u = late.get("cos_u_prev_mean", float("nan"))
        cos_ug = late.get("cos_u_neg_g_mean", float("nan"))
        print(
            f"{r['variant']:45s} "
            f"{r['final_test_acc']:>8.4f} "
            f"{r['best_test_acc']:>8.4f} "
            f"{cos_g:>8.3f} "
            f"{cos_u:>8.3f} "
            f"{cos_ug:>8.3f}"
        )
    print("=" * 80)
    print(f"Done. Results at {out_dir}")


if __name__ == "__main__":
    main()
