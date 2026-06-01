"""
BoGrad sweep on CIFAR-100 with ResNet18 (~11.2M params, with BN).

The "semi-production" run: harder dataset (100 classes vs 10), bigger model,
more epochs. Same six base optimisers and same {baseline, +BoGrad K=32}
treatment as the smaller runs. Note ResNet18 has BatchNorm; BoGrad still
projects only the parameter delta (running BN statistics are buffers, not
tracked by the optimiser).

Per-method LRs from `_common.DEFAULT_LRS`. Saves JSON + plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets, transforms

from _common import (
    ALL_METHODS, DEFAULT_LRS, REPO, WRAPPERS,
    ResNet18CIFAR, run_sweep,
)


# CIFAR-100 normalisation stats (population mean/std on the train split).
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
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        help=f"subset of {WRAPPERS} (default: all)")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    for m in DEFAULT_LRS:
        parser.add_argument(f"--lr_{m}", type=float, default=None)
    args = parser.parse_args()

    lrs = {m: (getattr(args, f"lr_{m}", None) or DEFAULT_LRS[m]) for m in args.methods}

    here = Path(__file__).resolve().parent
    train_ds, test_ds = build_cifar100(REPO / "data")

    run_sweep(
        dataset_name="cifar100",
        train_dataset=train_ds,
        test_dataset=test_ds,
        model_factory=lambda: ResNet18CIFAR(num_classes=100),
        methods=list(args.methods),
        wrappers=list(args.wrappers),
        lrs=lrs,
        K=args.K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        out_root=here / "results" / "cifar100_resnet18",
    )


if __name__ == "__main__":
    main()
