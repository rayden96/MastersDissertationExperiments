"""
Empirical run for the FocusedWork measurements.

Methods (vanilla-SGD scope per §01):
- sgd            : plain SGD
- cosgd          : COSGD (per-class within-batch orthogonalisation)
- bograd         : BoGrad over SGD (between-batch projection, K=32, mode=negative)

Datasets: CIFAR-10 and MNIST. Architectures are small CNNs without BatchNorm or
dropout (so per-class subgradient computation in diagnostics doesn't perturb
running stats / random masks).

Outputs JSON files under FocusedWork/results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# add repo root to sys.path so common.optimizers imports work
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent.parent.parent))  # for the worktree case

# We're in a worktree; the optimizers live in the main repo.
MAIN_REPO_ROOT = Path(r"C:\Masters Work\Dissertation&Experiments\MastersDissertationExperiments")
if str(MAIN_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_REPO_ROOT))

from common.optimizers import BoGrad, COSGD  # noqa: E402

# import diagnostics (sibling module)
from diagnostics import InterferenceMeter  # noqa: E402


# ---------------------------------------------------------------------------
# Architectures (no BN/dropout — see header)
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
def build_cifar10(data_root: Path, quick: bool = False):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(str(data_root), train=True, download=True, transform=tf)
    test = datasets.CIFAR10(str(data_root), train=False, download=True, transform=tf)
    if quick:
        train = Subset(train, list(range(0, len(train), 4)))
        test = Subset(test, list(range(0, len(test), 2)))
    return train, test


def build_mnist(data_root: Path, quick: bool = False):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(str(data_root), train=True, download=True, transform=tf)
    test = datasets.MNIST(str(data_root), train=False, download=True, transform=tf)
    if quick:
        train = Subset(train, list(range(0, len(train), 4)))
        test = Subset(test, list(range(0, len(test), 2)))
    return train, test


def build_balanced_reference_subset(dataset, num_classes: int, n_per_class: int, seed: int = 2026):
    """Pick n_per_class samples per class (balanced) for the reference set."""
    rng = np.random.default_rng(seed)
    # for Subset wrappers, walk through underlying targets; simplest: iterate dataset
    indices_by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    # fast path for raw datasets
    targets = None
    if hasattr(dataset, "targets"):
        try:
            tgts = dataset.targets
            if isinstance(tgts, torch.Tensor):
                tgts = tgts.tolist()
            elif isinstance(tgts, np.ndarray):
                tgts = tgts.tolist()
            targets = list(tgts)
        except Exception:
            targets = None
    if targets is None:
        # slow path
        targets = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            targets.append(int(y))
    for i, y in enumerate(targets):
        if y < num_classes:
            indices_by_class[int(y)].append(i)
    chosen: List[int] = []
    for c in range(num_classes):
        pool = indices_by_class[c]
        rng.shuffle(pool)
        chosen.extend(pool[:n_per_class])
    return Subset(dataset, chosen)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
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


# ---------------------------------------------------------------------------
# Optimiser factories
# ---------------------------------------------------------------------------
def make_optimizer(method: str, model: nn.Module, criterion: nn.Module, lr: float):
    if method == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    if method == "cosgd":
        # vanilla SGD-flavoured COSGD; orth method modified_gs_negative is the
        # softer Stage-1 variant
        return COSGD(
            model.parameters(),
            lr=lr,
            model=model,
            criterion=criterion,
            orthogonalization_method="modified_gs_negative",
            step_method="single_forward",
        )
    if method == "bograd":
        return BoGrad(
            model.parameters(),
            base_optimizer_cls=torch.optim.SGD,
            buffer_size=32,
            project_stage="update",
            projection_mode="negative",
            orth_method="sequential",
            lr=lr,
            momentum=0.0,
        )
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_one(
    *,
    method: str,
    dataset_name: str,
    train_dataset,
    test_dataset,
    ref_dataset,
    num_classes: int,
    model_factory: Callable[[], nn.Module],
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    log_every: int,
    ref_refresh_every: int,
    K_values: List[int],
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(method, model, criterion, lr)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, generator=g,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0,
    )
    ref_loader = DataLoader(
        ref_dataset, batch_size=256, shuffle=False, num_workers=0,
    )

    meter = InterferenceMeter(
        model=model,
        criterion=criterion,
        ref_loader=ref_loader,
        num_classes=num_classes,
        device=device,
        lr=lr,
        K_values=K_values,
        log_every=log_every,
        ref_refresh_every=ref_refresh_every,
    )
    meter.initialize()

    epoch_test_acc: List[float] = []

    print(f"  [{method} | {dataset_name}]", end="", flush=True)
    t0 = time.time()

    step = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if method == "cosgd":
                # COSGD does its own forward+backward inside .step() and
                # returns the average per-class loss
                meter.before_step()
                loss_val = optimizer.step(x, y, torch.unique(y))
                if not isinstance(loss_val, (int, float)):
                    loss_val = float("nan")
                meter.after_step(step, x, y, applied_loss=float(loss_val))
            else:
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                meter.before_step()
                optimizer.step()
                meter.after_step(step, x, y, applied_loss=loss.item())

            step += 1

        acc = evaluate(model, test_loader, device)
        epoch_test_acc.append(acc)
        print(f"  e{epoch + 1}={acc:.4f}", end="", flush=True)

    elapsed = time.time() - t0
    print(f"  ({elapsed:.0f}s)")

    summary = meter.summarize()
    return {
        "method": method,
        "dataset": dataset_name,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "log_every": log_every,
        "ref_refresh_every": ref_refresh_every,
        "K_values": K_values,
        "final_test_acc": epoch_test_acc[-1] if epoch_test_acc else float("nan"),
        "epoch_test_acc": epoch_test_acc,
        "wall_clock_s": elapsed,
        "summary": summary,
        "logs": meter.logs,  # full per-step logs
        "calibration_logs": meter.calibration_logs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "mnist"])
    parser.add_argument("--methods", nargs="+", default=["sgd", "cosgd", "bograd"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_cifar10", type=float, default=0.05)
    parser.add_argument("--lr_mnist", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--ref_refresh_every", type=int, default=50)
    parser.add_argument("--K_values", type=int, nargs="+", default=[4, 32, 128])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ref_n_per_class", type=int, default=200)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out_dir", type=str, default=str(HERE / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_root = MAIN_REPO_ROOT / "data"

    all_results: List[Dict[str, Any]] = []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_subdir = out_dir / f"run_{run_id}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        if dataset_name == "cifar10":
            train_ds, test_ds = build_cifar10(data_root, quick=args.quick)
            num_classes = 10
            model_factory = lambda: SmallCIFARCNN(num_classes=10)
            lr = args.lr_cifar10
        elif dataset_name == "mnist":
            train_ds, test_ds = build_mnist(data_root, quick=args.quick)
            num_classes = 10
            model_factory = lambda: SmallMNISTCNN(num_classes=10)
            lr = args.lr_mnist
        else:
            raise ValueError(f"unknown dataset {dataset_name!r}")

        ref_ds = build_balanced_reference_subset(
            train_ds, num_classes=num_classes,
            n_per_class=args.ref_n_per_class, seed=args.seed,
        )
        print(f"[{dataset_name}] train={len(train_ds)} test={len(test_ds)} ref={len(ref_ds)}")

        for method in args.methods:
            try:
                res = run_one(
                    method=method,
                    dataset_name=dataset_name,
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    ref_dataset=ref_ds,
                    num_classes=num_classes,
                    model_factory=model_factory,
                    device=device,
                    lr=lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    log_every=args.log_every,
                    ref_refresh_every=args.ref_refresh_every,
                    K_values=args.K_values,
                    seed=args.seed,
                )
                all_results.append(res)
                # save individual run
                fname = f"{dataset_name}_{method}.json"
                with open(out_subdir / fname, "w") as f:
                    json.dump(res, f, indent=2, default=str)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"  [{method} | {dataset_name}] FAILED: {e}")
                print(tb)
                all_results.append({
                    "method": method,
                    "dataset": dataset_name,
                    "error": str(e),
                    "traceback": tb,
                })

    # combined summary file (just the headline summaries, no logs)
    headline = []
    for r in all_results:
        if "error" in r:
            headline.append({"method": r["method"], "dataset": r["dataset"], "error": r["error"]})
            continue
        s = r["summary"]
        headline.append({
            "method": r["method"],
            "dataset": r["dataset"],
            "final_test_acc": r["final_test_acc"],
            "wall_clock_s": r["wall_clock_s"],
            **{k: s[k] for k in s if not k.startswith("corr_")},
            **{k: s[k] for k in s if k.startswith("corr_")},
        })
    with open(out_subdir / "headline_summary.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)

    print(f"\nResults written to {out_subdir}")


if __name__ == "__main__":
    main()
