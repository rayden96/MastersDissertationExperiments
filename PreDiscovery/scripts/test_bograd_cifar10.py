"""
Sanity check for BoGrad (scrutinised) on CIFAR-10 with a standard small CNN.

Not a real experiment — a fast correctness + integration test. Runs five short
configs, prints per-epoch test accuracy, and for the BoGrad runs prints a few
stat snapshots so we can eyeball that the projection is doing something
reasonable.

Usage
-----
    python scripts/test_bograd_cifar10.py
    python scripts/test_bograd_cifar10.py --epochs 5 --skip-download

Runs reasonably fast on a T4 or better; CPU works but is slow. Uses torchvision
to download CIFAR-10 to ./data on first run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.optimizers import BoGrad  # noqa: E402


# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """The same 3-conv CNN used in the IJCNN CIFAR-10 experiments."""

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


def build_cifar10(data_root: Path, download: bool) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(root=str(data_root), train=True, download=download, transform=tf)
    test = datasets.CIFAR10(root=str(data_root), train=False, download=download, transform=tf)
    return train, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_run(
    name: str,
    optimizer_factory,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int = 2026,
    stats_every: int = 200,
) -> float:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    optimizer = optimizer_factory(model)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== {name} ===")
    has_stats = hasattr(optimizer, "get_last_step_stats")
    t0 = time.time()
    final_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            if has_stats and step and step % stats_every == 0:
                stats = optimizer.get_last_step_stats()
                if stats:
                    compact = " ".join(f"{k}={v:.3f}" for k, v in stats.items())
                    print(f"    step {step:5d}  loss={loss.item():.3f}  {compact}")

        final_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{epochs}  test_acc={final_acc:.4f}  elapsed={elapsed:.1f}s")

    return final_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true", help="Do not attempt to download CIFAR-10")
    ap.add_argument("--only", type=str, default=None, help="Comma-separated names of runs to include")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch {torch.__version__}")

    train_ds, test_ds = build_cifar10(Path(args.data_root), download=not args.skip_download)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    def run(name):
        if args.only is None:
            return True
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        return name in wanted

    results: dict[str, float] = {}

    if run("sgd"):
        results["sgd"] = train_run(
            "SGD baseline (lr=0.05, momentum=0.9)",
            lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9),
            train_loader, test_loader, device, args.epochs,
        )

    if run("sgd_bograd_grad"):
        results["sgd_bograd_grad"] = train_run(
            "SGD + BoGrad (project_stage=gradient, K=8)",
            lambda m: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=8, project_stage="gradient",
                lr=0.05, momentum=0.9,
            ),
            train_loader, test_loader, device, args.epochs,
        )

    if run("sgd_bograd_update"):
        results["sgd_bograd_update"] = train_run(
            "SGD + BoGrad (project_stage=update, K=8)   [recommended for SGD+momentum]",
            lambda m: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=8, project_stage="update",
                lr=0.05, momentum=0.9,
            ),
            train_loader, test_loader, device, args.epochs,
        )

    if run("adam"):
        results["adam"] = train_run(
            "Adam baseline (lr=1e-3)",
            lambda m: torch.optim.Adam(m.parameters(), lr=1e-3),
            train_loader, test_loader, device, args.epochs,
        )

    if run("adam_bograd_update"):
        results["adam_bograd_update"] = train_run(
            "Adam + BoGrad (project_stage=update, K=8)   [principled for Adam]",
            lambda m: BoGrad(
                m.parameters(), torch.optim.Adam,
                buffer_size=8, project_stage="update",
                lr=1e-3,
            ),
            train_loader, test_loader, device, args.epochs,
        )

    if run("adam_bograd_grad"):
        results["adam_bograd_grad"] = train_run(
            "Adam + BoGrad (project_stage=gradient, K=8)   [known-incoherent, for comparison]",
            lambda m: BoGrad(
                m.parameters(), torch.optim.Adam,
                buffer_size=8, project_stage="gradient",
                lr=1e-3,
            ),
            train_loader, test_loader, device, args.epochs,
        )

    print("\n" + "=" * 60)
    print("Final test accuracies")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:28s}  {acc:.4f}")


if __name__ == "__main__":
    main()
