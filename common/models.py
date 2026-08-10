"""
common.models — model registry for the paper-ready experiments.

`get_model(name, num_classes=..., **overrides) -> nn.Module`.

Centralises every architecture so experiments don't redefine them (CLAUDE.md:
shared code in common/, no copy-paste). `SmallCIFARCNN` and `ResNet18CIFAR`
were previously defined inline in 02/03 run.py — they live here now and those
scripts should import them.

Registered names
----------------
  grayscale_cnn   1-channel small CNN     MNIST, EMNIST-Balanced
  small_cifar_cnn 3-conv no-BN CNN ~93k   CIFAR-10  (the ablation workhorse)
  resnet18_cifar  ResNet-18 (CIFAR stem)  CIFAR-100
  resnet34_cifar  ResNet-34 (CIFAR stem)  CIFAR-100 scale study
  mlp             tabular MLP             Covertype
  text_cnn        embedding + 1D conv     Yahoo! Answers (dense grads)

`small_cifar_cnn` and `grayscale_cnn` accept a `width_mult` for the model-scale
study (5.7a). `dropout_p` toggles activation dropout (for the dropout comparator
arm) — 0.0 by default so the baseline matches the existing no-dropout models.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Image — small CNNs
# ---------------------------------------------------------------------------
class GrayscaleCNN(nn.Module):
    """Small 2-conv CNN for 1-channel 28x28 (MNIST / EMNIST). No BN."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        c1, c2 = int(32 * width_mult), int(64 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.classifier = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(self.drop(x))


class SmallCIFARCNN(nn.Module):
    """3-conv CNN (~93k params at width 1.0), no BN. The ablation workhorse.

    Moved verbatim from PaperReadyExperiments/02_medium_cifar10/run.py, plus a
    `width_mult` for the model-scale study and an optional activation `dropout_p`.
    """

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        c1, c2, c3 = int(32 * width_mult), int(64 * width_mult), int(128 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(self.drop(x))


# ---------------------------------------------------------------------------
# Image — ResNet (CIFAR stem)
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


class ResNetCIFAR(nn.Module):
    """ResNet with a 3x3 stride-1 stem and no initial maxpool (CIFAR-appropriate).

    `blocks_per_stage` selects depth: (2,2,2,2)=ResNet-18, (3,4,6,3)=ResNet-34.
    ResNet18CIFAR was moved verbatim from 03_large_cifar100/run.py.

    `dropout_p` adds activation dropout after global pooling, before the
    classifier, matching the placement used by the other models in this registry
    so the dropout comparator arm is defined identically across benchmarks.
    """

    def __init__(self, num_classes: int = 100, blocks_per_stage=(2, 2, 2, 2),
                 dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(64, 128, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks_per_stage[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks_per_stage[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


def ResNet18CIFAR(num_classes: int = 100, dropout_p: float = 0.0):
    return ResNetCIFAR(num_classes, blocks_per_stage=(2, 2, 2, 2), dropout_p=dropout_p)


def ResNet34CIFAR(num_classes: int = 100, dropout_p: float = 0.0):
    return ResNetCIFAR(num_classes, blocks_per_stage=(3, 4, 6, 3), dropout_p=dropout_p)


# ---------------------------------------------------------------------------
# Tabular — MLP
# ---------------------------------------------------------------------------
class TabularMLP(nn.Module):
    """MLP for tabular data (Covertype: 54 features -> 7 classes)."""

    def __init__(self, num_classes: int = 7, in_features: int = 54,
                 hidden=(128, 64), dropout_p: float = 0.0):
        super().__init__()
        dims = [in_features, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        return self.head(self.body(x))


# ---------------------------------------------------------------------------
# Text — Text-CNN (dense gradients, so BoGrad/COSGD/GradDrop all apply)
# ---------------------------------------------------------------------------
class TextCNN(nn.Module):
    """Kim-style 1D-conv text classifier over a learned embedding.

    Dense embedding gradients (not EmbeddingBag's sparse ones), so the
    gradient-transform methods work without sparse-grad special-casing.
    Input: LongTensor [batch, seq_len] of token ids (0 = pad).
    """

    def __init__(self, num_classes: int = 10, vocab_size: int = 30000,
                 embed_dim: int = 128, kernel_sizes=(3, 4, 5), num_filters: int = 100,
                 dropout_p: float = 0.0, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes
        ])
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        e = self.embedding(x).transpose(1, 2)            # [B, embed, seq]
        feats = [F.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "grayscale_cnn": GrayscaleCNN,
    "small_cifar_cnn": SmallCIFARCNN,
    "resnet18_cifar": ResNet18CIFAR,
    "resnet34_cifar": ResNet34CIFAR,
    "mlp": TabularMLP,
    "text_cnn": TextCNN,
}


def get_model(name: str, num_classes: int, **overrides) -> nn.Module:
    """Build a registered model. `overrides` are forwarded to the constructor
    (e.g. width_mult, dropout_p, in_features, vocab_size)."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes, **overrides)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "get_model", "count_params",
    "GrayscaleCNN", "SmallCIFARCNN", "ResNetCIFAR", "ResNet18CIFAR", "ResNet34CIFAR",
    "TabularMLP", "TextCNN",
]
