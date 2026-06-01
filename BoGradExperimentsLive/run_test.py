"""
First test in BoGradExperimentsLive: BoGrad (update stage, K=32, negative mode)
applied to six base optimisers, on CIFAR-10 and MNIST. Single shared LR across
all methods.

Methods (each run twice — baseline and with BoGrad):
  - sgd
  - sgd_momentum
  - adam
  - rmsprop
  - signsgd
  - signsgd_momentum

12 method-rows × 2 datasets = 24 runs. 2 epochs each. Small no-BN CNNs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.optimizers import BoGrad  # noqa: E402
from discoveryPhase2.enhanced_variants import SignSGD  # noqa: E402


# ---------------------------------------------------------------------------
# Models — match FocusedWork conventions (no BN, no dropout)
# ---------------------------------------------------------------------------
class SmallCIFARCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------
def build_cifar10(data_root: Path):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(str(data_root), train=True, download=True, transform=tf)
    test = datasets.CIFAR10(str(data_root), train=False, download=True, transform=tf)
    return train, test


def build_mnist(data_root: Path):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(str(data_root), train=True, download=True, transform=tf)
    test = datasets.MNIST(str(data_root), train=False, download=True, transform=tf)
    return train, test


# ---------------------------------------------------------------------------
# Per-method default learning rates. Standard family-appropriate values, NOT
# individually tuned. Override with --lr_<method> on the CLI.
# ---------------------------------------------------------------------------
DEFAULT_LRS: Dict[str, float] = {
    "sgd":              0.01,
    "sgd_momentum":     0.01,
    "adam":             0.001,
    "rmsprop":          0.001,
    "signsgd":          0.001,
    "signsgd_momentum": 0.001,
}


# ---------------------------------------------------------------------------
# Optimiser factory — returns (base_optimizer_cls, base_kwargs)
# ---------------------------------------------------------------------------
def get_base_optimizer(method: str, lr: float):
    if method == "sgd":
        return torch.optim.SGD, {"lr": lr, "momentum": 0.0}
    if method == "sgd_momentum":
        return torch.optim.SGD, {"lr": lr, "momentum": 0.6}
    if method == "adam":
        return torch.optim.Adam, {"lr": lr}
    if method == "rmsprop":
        return torch.optim.RMSprop, {"lr": lr}
    if method == "signsgd":
        return SignSGD, {"lr": lr, "momentum": 0.0}
    if method == "signsgd_momentum":
        return SignSGD, {"lr": lr, "momentum": 0.6}
    raise ValueError(f"unknown method {method!r}")


def make_optimizer(method: str, model: nn.Module, lr: float, with_bograd: bool, K: int):
    base_cls, base_kwargs = get_base_optimizer(method, lr)
    if with_bograd:
        return BoGrad(
            model.parameters(),
            base_optimizer_cls=base_cls,
            buffer_size=K,
            project_stage="update",
            projection_mode="negative",
            orth_method="sequential",
            collect_stats=False,  # cheaper
            **base_kwargs,
        )
    return base_cls(model.parameters(), **base_kwargs)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


def train_one_run(
    *,
    method: str,
    with_bograd: bool,
    dataset_name: str,
    train_dataset,
    test_dataset,
    model_factory: Callable[[], nn.Module],
    device: torch.device,
    lr: float,
    K: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(method, model, lr, with_bograd, K)

    g = torch.Generator()
    g.manual_seed(seed)
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=pin, persistent_workers=True, generator=g,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=2, pin_memory=pin, persistent_workers=True,
    )

    label = f"{method}{'+bograd' if with_bograd else ''}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{label} | {dataset_name}] lr={lr:g}  K={K if with_bograd else '-'}  "
          f"batch={batch_size}  epochs={epochs}  params={n_params}  seed={seed}")
    t0 = time.time()

    epoch_test_acc: List[float] = []
    epoch_train_loss: List[float] = []
    diverged = False
    diverged_at_step = -1
    n_batches_per_epoch = max(1, len(train_loader))
    # Print a dot at every ~10% of batches within an epoch
    dot_every = max(1, n_batches_per_epoch // 10)

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        running_recent = 0.0  # exp-moving-average loss for liveness
        ema_alpha = 0.05
        epoch_t0 = time.time()
        print(f"  e{epoch + 1}: ", end="", flush=True)

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if not (loss_val == loss_val):  # NaN check
                diverged = True
                diverged_at_step = epoch * n_batches_per_epoch + batch_idx
                break
            epoch_loss_sum += loss_val
            n_batches += 1
            running_recent = loss_val if batch_idx == 0 else (
                (1 - ema_alpha) * running_recent + ema_alpha * loss_val
            )

            if (batch_idx + 1) % dot_every == 0:
                print(".", end="", flush=True)

        train_loss = epoch_loss_sum / max(n_batches, 1)
        epoch_train_loss.append(train_loss)
        epoch_dt = time.time() - epoch_t0

        if diverged:
            epoch_test_acc.append(0.1)
            print(f" DIVERGED @ step {diverged_at_step}, last_loss={running_recent:.4g} "
                  f"({epoch_dt:.1f}s)")
            break

        acc = evaluate(model, test_loader, device)
        epoch_test_acc.append(acc)
        print(f" train_loss={train_loss:.4f}  test_acc={acc:.4f}  ({epoch_dt:.1f}s)")

    elapsed = time.time() - t0
    if epoch_test_acc:
        best = max(epoch_test_acc)
        final = epoch_test_acc[-1]
        print(f"  done: final_acc={final:.4f}  best_acc={best:.4f}  total={elapsed:.1f}s")
    else:
        print(f"  done: NO EPOCHS COMPLETED  total={elapsed:.1f}s")

    return {
        "method": method,
        "with_bograd": with_bograd,
        "dataset": dataset_name,
        "lr": lr,
        "K": K if with_bograd else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "diverged": diverged,
        "final_test_acc": epoch_test_acc[-1] if epoch_test_acc else float("nan"),
        "epoch_test_acc": epoch_test_acc,
        "epoch_train_loss": epoch_train_loss,
        "wall_clock_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=[
        "sgd", "sgd_momentum", "adam", "rmsprop", "signsgd", "signsgd_momentum",
    ])
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "mnist"])
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    # Per-method LR overrides. If not given, uses DEFAULT_LRS[method].
    for m in DEFAULT_LRS:
        parser.add_argument(f"--lr_{m}", type=float, default=None,
                            help=f"override LR for {m} (default {DEFAULT_LRS[m]})")
    args = parser.parse_args()

    # Resolve per-method LRs
    lrs: Dict[str, float] = {}
    for m in args.methods:
        override = getattr(args, f"lr_{m}", None)
        lrs[m] = override if override is not None else DEFAULT_LRS[m]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perf: enable cudnn autotuner (picks fastest conv algorithms for our
    # input shapes after the first batch) and TF32 on Ampere+ if applicable.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # use TF32 where allowed

    print(f"device: {device}, K={args.K}, epochs={args.epochs}")
    print("LRs: " + ", ".join(f"{m}={lrs[m]}" for m in args.methods))

    data_root = REPO / "data"
    here = Path(__file__).resolve().parent
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = here / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    for dataset_name in args.datasets:
        if dataset_name == "cifar10":
            train_ds, test_ds = build_cifar10(data_root)
            model_factory = lambda: SmallCIFARCNN(10)
        elif dataset_name == "mnist":
            train_ds, test_ds = build_mnist(data_root)
            model_factory = lambda: SmallMNISTCNN(10)
        else:
            raise ValueError(dataset_name)

        for method in args.methods:
            for with_bograd in (False, True):
                try:
                    res = train_one_run(
                        method=method,
                        with_bograd=with_bograd,
                        dataset_name=dataset_name,
                        train_dataset=train_ds,
                        test_dataset=test_ds,
                        model_factory=model_factory,
                        device=device,
                        lr=lrs[method],
                        K=args.K,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        seed=args.seed,
                    )
                    all_results.append(res)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"  [{method}{'+bograd' if with_bograd else ''} | {dataset_name}] ERROR: {e}")
                    print(tb)
                    all_results.append({
                        "method": method, "with_bograd": with_bograd,
                        "dataset": dataset_name, "error": str(e), "traceback": tb,
                    })

    # save full results
    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # === Comparison table ===
    by_key = {(r.get("method"), r.get("dataset"), r.get("with_bograd")): r for r in all_results}

    def fmt_acc(r):
        if not r:
            return "       -    "
        if r.get("error"):
            return "    ERROR   "
        if r.get("diverged"):
            return "  diverged  "
        v = r.get("final_test_acc", float("nan"))
        return f"   {v:.4f}   " if v == v else "      NaN   "

    def fmt_delta(base, bg):
        if not base or not bg or base.get("error") or bg.get("error"):
            return "    -   "
        if base.get("diverged") or bg.get("diverged"):
            return "    -   "
        a = base.get("final_test_acc", float("nan"))
        b = bg.get("final_test_acc", float("nan"))
        if not (a == a) or not (b == b):
            return "    -   "
        d = b - a
        marker = "++" if d > 0.02 else ("+" if d > 0.005 else ("--" if d < -0.02 else ("-" if d < -0.005 else "~")))
        return f"{d:+.4f}{marker:>3}"

    for ds in args.datasets:
        print(f"\n=== {ds.upper()} — final test accuracy (lr per method) ===")
        header = f"{'method':<22} {'lr':>7}   {'baseline':^12} {'+BoGrad K=' + str(args.K):^14}   {'Δ (BG − base)':^14}"
        print(header)
        print("-" * len(header))
        for method in args.methods:
            base = by_key.get((method, ds, False))
            bg = by_key.get((method, ds, True))
            lr = lrs[method]
            print(
                f"{method:<22} {lr:>7g}   "
                f"{fmt_acc(base):^12} {fmt_acc(bg):^14}   {fmt_delta(base, bg):^14}"
            )

    # Also a summary by method showing both datasets for quick scanning
    print("\n=== Summary: BoGrad effect (Δ accuracy) by method × dataset ===")
    print(f"{'method':<22} {'cifar10':>16} {'mnist':>16}")
    print("-" * 56)
    for method in args.methods:
        c_delta = fmt_delta(by_key.get((method, 'cifar10', False)), by_key.get((method, 'cifar10', True)))
        m_delta = fmt_delta(by_key.get((method, 'mnist', False)), by_key.get((method, 'mnist', True)))
        print(f"{method:<22} {c_delta:>16} {m_delta:>16}")

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
