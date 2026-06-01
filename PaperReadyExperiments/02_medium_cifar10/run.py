"""
Medium: CIFAR-10 with a small no-BN CNN. Vanilla SGD+momentum baseline,
multi-seed, interference metrics tracked throughout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
_REPO = _PARENT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from interference.image_runner import run_image_experiment  # noqa: E402
from common.optimizers import BoGrad, COSGD  # noqa: E402


class SmallCIFARCNN(nn.Module):
    """3-conv CNN, ~93k params, no BN/dropout."""

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


def build_cifar10(data_root: Path):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(str(data_root), train=True, download=True, transform=tf)
    test = datasets.CIFAR10(str(data_root), train=False, download=True, transform=tf)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "bograd", "cosgd"],
                        default="baseline",
                        help="baseline = SGD+momentum; bograd = BoGrad wrap "
                             "(K=32, update-stage, negative); cosgd = per-class "
                             "Gram-Schmidt within batch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--K", type=int, default=32, help="BoGrad buffer size")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ref_refresh_every", type=int, default=100)
    parser.add_argument("--K_values", type=int, nargs="+", default=[4, 32, 128])
    parser.add_argument("--ref_n_per_class", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_ds, test_ds = build_cifar10(_REPO / "data")
    criterion = nn.CrossEntropyLoss()

    is_cosgd = (args.method == "cosgd")

    if args.method == "baseline":
        def opt_factory(model):
            return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.method == "bograd":
        def opt_factory(model):
            return BoGrad(
                model.parameters(),
                base_optimizer_cls=torch.optim.SGD,
                buffer_size=args.K,
                project_stage="update",
                projection_mode="negative",
                orth_method="sequential",
                collect_stats=False,
                lr=args.lr, momentum=args.momentum,
            )
    elif args.method == "cosgd":
        def opt_factory(model):
            return COSGD(
                model.parameters(),
                lr=args.lr,
                model=model, criterion=criterion,
                orthogonalization_method="modified_gs_negative",
                step_method="single_forward",
            )

    run_image_experiment(
        experiment_name=f"02_medium_cifar10_{args.method}",
        model_factory=lambda: SmallCIFARCNN(num_classes=10),
        train_dataset=train_ds,
        test_dataset=test_ds,
        num_classes=10,
        out_root=_HERE / "results",
        optimizer_factory=opt_factory,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=list(args.seeds),
        log_every=args.log_every,
        ref_refresh_every=args.ref_refresh_every,
        K_values=list(args.K_values),
        ref_n_per_class=args.ref_n_per_class,
        num_workers=args.num_workers,
        is_cosgd=is_cosgd,
        method_label=args.method,
    )


if __name__ == "__main__":
    main()
