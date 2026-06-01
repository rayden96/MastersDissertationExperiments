"""
Shared utilities for testing/ scripts.

Includes:
  - SmallCNN, ResNet8, ResNet18CIFAR (architectures)
  - CIFAR-10 builder, train-loader factory
  - evaluate, train_run, aggregate
  - print_table helper for test summaries

Each test in testing/<NN>_*/run.py imports from this module rather than
re-defining helpers.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """The 3-conv CNN from the IJCNN experiments. ~93k parameters."""

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


class _BasicBlock(nn.Module):
    """Standard ResNet basic block."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNet8(nn.Module):
    """Small ResNet for CIFAR — 3 BasicBlocks (≈ResNet-8 in depth). ~250k params."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = _BasicBlock(16, 16, stride=1)
        self.layer2 = _BasicBlock(16, 32, stride=2)
        self.layer3 = _BasicBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR input (no maxpool, smaller stem). ~11M params."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


ARCHITECTURES: Dict[str, Callable[[], nn.Module]] = {
    "small_cnn": SmallCNN,
    "resnet8": ResNet8,
    "resnet18": ResNet18CIFAR,
}


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_cifar10(data_root: Path, download: bool = True, quick: bool = False):
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


def make_test_loader(dataset, batch_size=256, num_workers=2, pin_memory=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


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
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int,
    *,
    quiet: bool = False,
) -> Dict[str, Any]:
    """One training run. Returns dict with epoch_test_acc, final/best, wall_clock."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = model_factory().to(device)
    optimizer = optimizer_factory(model)
    criterion = nn.CrossEntropyLoss()

    if not quiet:
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
        if not quiet:
            print(f"  e{epoch + 1}={acc:.4f}", end="", flush=True)

    elapsed = time.time() - t0
    if not quiet:
        print(f"  ({elapsed:.0f}s)")

    return {
        "final_test_acc": epoch_acc[-1] if epoch_acc else float("nan"),
        "best_test_acc": max(epoch_acc) if epoch_acc else float("nan"),
        "epoch_test_acc": epoch_acc,
        "wall_clock_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
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


def delta_marker(delta: float, big: float = 0.01, mid: float = 0.0) -> str:
    if not math.isfinite(delta):
        return ""
    if delta > big:
        return "++"
    if delta > mid:
        return "+"
    if delta < -big:
        return "--"
    if delta < -mid:
        return "-"
    return "~"
