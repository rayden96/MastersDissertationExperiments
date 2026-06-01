"""
Large: CIFAR-100 with ResNet18 (has BatchNorm). Vanilla SGD+momentum baseline,
multi-seed, interference metrics tracked throughout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# ResNet18 adapted for CIFAR-100 (32x32 input, no maxpool, 100-class head).
# Self-contained — does not depend on PreDiscovery/ (which is an archive).
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 100):
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_cifar100(data_root: Path):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train = datasets.CIFAR100(str(data_root), train=True, download=True, transform=tf)
    test = datasets.CIFAR100(str(data_root), train=False, download=True, transform=tf)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "bograd", "cosgd"],
                        default="baseline",
                        help="baseline = SGD+momentum; bograd = BoGrad wrap "
                             "(K=32, update-stage, negative); cosgd = per-class "
                             "Gram-Schmidt within batch (slow at 100 classes!)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--K", type=int, default=32, help="BoGrad buffer size")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ref_refresh_every", type=int, default=200)
    parser.add_argument("--K_values", type=int, nargs="+", default=[4, 32, 128])
    parser.add_argument("--ref_n_per_class", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_ds, test_ds = build_cifar100(_REPO / "data")
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
        experiment_name=f"03_large_cifar100_{args.method}",
        model_factory=lambda: ResNet18CIFAR(num_classes=100),
        train_dataset=train_ds,
        test_dataset=test_ds,
        num_classes=100,
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
