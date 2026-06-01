"""
BoGrad sweep on MNIST with the small 2-conv CNN (~5k params, no BN).

Runs all six base optimisers (sgd, sgd_momentum, adam, rmsprop, signsgd,
signsgd_momentum) twice each — once baseline, once with BoGrad K=32 update-stage
negative. Per-method LRs from `_common.DEFAULT_LRS`. Saves JSON + plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets, transforms

from _common import (
    ALL_METHODS, DEFAULT_LRS, REPO, WRAPPERS,
    SmallMNISTCNN, run_sweep,
)


def build_mnist(data_root: Path):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(str(data_root), train=True, download=True, transform=tf)
    test = datasets.MNIST(str(data_root), train=False, download=True, transform=tf)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        help=f"subset of {WRAPPERS} (default: all)")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    for m in DEFAULT_LRS:
        parser.add_argument(f"--lr_{m}", type=float, default=None)
    args = parser.parse_args()

    lrs = {m: (getattr(args, f"lr_{m}", None) or DEFAULT_LRS[m]) for m in args.methods}

    here = Path(__file__).resolve().parent
    train_ds, test_ds = build_mnist(REPO / "data")

    run_sweep(
        dataset_name="mnist",
        train_dataset=train_ds,
        test_dataset=test_ds,
        model_factory=lambda: SmallMNISTCNN(num_classes=10),
        methods=list(args.methods),
        wrappers=list(args.wrappers),
        lrs=lrs,
        K=args.K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        out_root=here / "results" / "mnist_smallcnn",
    )


if __name__ == "__main__":
    main()
