"""
common.datasets — dataset registry for the paper-ready experiments.

`get_dataset(name) -> DatasetBundle(train, val, test, meta)`.

Spans three modalities:
  mnist            10 classes   image   1x28x28   torchvision
  emnist_balanced  47 classes   image   1x28x28   torchvision (EMNIST 'balanced')
  cifar10          10 classes   image   3x32x32   torchvision
  cifar100        100 classes   image   3x32x32   torchvision
  covertype          7 classes  tabular 54 feats  sklearn fetch_covtype
  yahoo_answers     10 classes  text    token ids HuggingFace datasets (subsampled)

Two things the existing 02/03 runners lacked, added here:
  1. A real **train/val split** (carved from train) so tuning happens on val and
     test stays held out — required for the bakeoff's independent HP tuning.
  2. A **synthetic sub-class split** of CIFAR-10 into {2,5,10,20,50,100} pseudo
     classes (label sub-hashing) for the COSGD class-count scalability study (20.07).

`meta` carries everything the Trainer/model registry need:
  num_classes, input_kind ∈ {image, tabular, text}, model name + kwargs,
  default epochs/batch_size, normalisation, and (text) vocab_size/seq_len/pad_idx.

Heavy datasets (covertype, yahoo_answers) are prepared lazily and cached under
<repo>/data so repeated runs and Colab sessions don't re-download/re-tokenise.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

_REPO = Path(__file__).resolve().parent.parent
_DATA = _REPO / "data"


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _targets_of(dataset) -> np.ndarray:
    """Best-effort label vector for a dataset (handles tensor/list/ndarray and
    Subset wrappers)."""
    if isinstance(dataset, Subset):
        base = _targets_of(dataset.dataset)
        return base[np.asarray(dataset.indices)]
    t = getattr(dataset, "targets", None)
    if t is None:
        t = getattr(dataset, "labels", None)
    if t is None:
        return np.array([int(dataset[i][1]) for i in range(len(dataset))])
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.asarray(t)


def stratified_val_split(train_dataset, val_fraction: float, seed: int) -> Tuple[Subset, Subset]:
    """Carve a stratified validation Subset out of a training dataset.

    Returns (train_subset, val_subset). Deterministic in `seed`. Stratified so
    every class is represented in val in proportion — important for the
    high-class-count sets (EMNIST-47, CIFAR-100).
    """
    targets = _targets_of(train_dataset)
    rng = np.random.default_rng(seed)
    val_idx, train_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return Subset(train_dataset, train_idx), Subset(train_dataset, val_idx)


def balanced_reference_subset(dataset, num_classes: int, n_per_class: int, seed: int = 2026) -> Subset:
    """Balanced subset (n_per_class per class) for the interference reference set.

    Generalises image_runner.build_balanced_reference_subset to any dataset via
    _targets_of, so tabular/text reference sets work too.
    """
    targets = _targets_of(dataset)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        chosen.extend(idx[:n_per_class].tolist())
    return Subset(dataset, chosen)


# ---------------------------------------------------------------------------
# Image datasets (torchvision)
# ---------------------------------------------------------------------------
def _image_bundle(name: str, val_fraction: float, seed: int) -> DatasetBundle:
    from torchvision import datasets as tvd
    from torchvision import transforms

    _DATA.mkdir(parents=True, exist_ok=True)
    root = str(_DATA)

    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        tr = tvd.MNIST(root, train=True, download=True, transform=tf)
        te = tvd.MNIST(root, train=False, download=True, transform=tf)
        meta = dict(num_classes=10, input_kind="image", model="grayscale_cnn",
                    model_kwargs={}, in_shape=(1, 28, 28), epochs=10, batch_size=128)
    elif name == "emnist_balanced":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1751,), (0.3332,))])
        tr = tvd.EMNIST(root, split="balanced", train=True, download=True, transform=tf)
        te = tvd.EMNIST(root, split="balanced", train=False, download=True, transform=tf)
        meta = dict(num_classes=47, input_kind="image", model="grayscale_cnn",
                    model_kwargs={}, in_shape=(1, 28, 28), epochs=15, batch_size=128)
    elif name == "cifar10":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                      (0.2470, 0.2435, 0.2616))])
        tr = tvd.CIFAR10(root, train=True, download=True, transform=tf)
        te = tvd.CIFAR10(root, train=False, download=True, transform=tf)
        meta = dict(num_classes=10, input_kind="image", model="small_cifar_cnn",
                    model_kwargs={}, in_shape=(3, 32, 32), epochs=30, batch_size=128)
    elif name == "cifar100":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5071, 0.4865, 0.4409),
                                                      (0.2673, 0.2564, 0.2762))])
        tr = tvd.CIFAR100(root, train=True, download=True, transform=tf)
        te = tvd.CIFAR100(root, train=False, download=True, transform=tf)
        meta = dict(num_classes=100, input_kind="image", model="resnet18_cifar",
                    model_kwargs={}, in_shape=(3, 32, 32), epochs=50, batch_size=128)
    else:
        raise ValueError(name)

    train_sub, val_sub = stratified_val_split(tr, val_fraction, seed)
    return DatasetBundle(train_sub, val_sub, te, meta)


# ---------------------------------------------------------------------------
# Tabular — Covertype (sklearn)
# ---------------------------------------------------------------------------
def _covertype_bundle(val_fraction: float, seed: int) -> DatasetBundle:
    from sklearn.datasets import fetch_covtype

    cache = _DATA / "covertype_cache.pkl"
    _DATA.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with open(cache, "rb") as f:
            X, y = pickle.load(f)
    else:
        data = fetch_covtype()
        X = data.data.astype(np.float32)
        y = (data.target.astype(np.int64) - 1)  # labels 1..7 -> 0..6
        # standardise the 10 continuous columns; the rest are binary indicators
        mean = X[:, :10].mean(axis=0); std = X[:, :10].std(axis=0) + 1e-8
        X = X.copy(); X[:, :10] = (X[:, :10] - mean) / std
        with open(cache, "wb") as f:
            pickle.dump((X, y), f)

    rng = np.random.default_rng(seed)
    n = len(y); idx = rng.permutation(n)
    n_test = int(0.2 * n)
    test_idx, rest = idx[:n_test], idx[n_test:]

    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    full_train = TensorDataset(Xt[rest], yt[rest])
    test = TensorDataset(Xt[test_idx], yt[test_idx])
    train_sub, val_sub = stratified_val_split(full_train, val_fraction, seed)

    meta = dict(num_classes=7, input_kind="tabular", model="mlp",
                model_kwargs={"in_features": X.shape[1]}, in_features=X.shape[1],
                epochs=20, batch_size=256)
    return DatasetBundle(train_sub, val_sub, test, meta)


# ---------------------------------------------------------------------------
# Text — Yahoo! Answers (HuggingFace datasets), tokenised to fixed-len id seqs
# ---------------------------------------------------------------------------
def _yahoo_bundle(val_fraction: float, seed: int,
                  vocab_size: int = 30000, seq_len: int = 100,
                  train_subsample: int = 100_000, test_subsample: int = 20_000) -> DatasetBundle:
    """Yahoo! Answers Topics (10 classes). Whitespace tokeniser + frequency vocab,
    cached so we tokenise once. Train is subsampled for tractable Colab sweeps."""
    sig = f"yahoo_v{vocab_size}_s{seq_len}_tr{train_subsample}_te{test_subsample}_seed{seed}"
    cache = _DATA / f"{sig}.pkl"
    _DATA.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        with open(cache, "rb") as f:
            blob = pickle.load(f)
    else:
        from datasets import load_dataset

        ds = load_dataset("yahoo_answers_topics")
        rng = np.random.default_rng(seed)

        def texts_labels(split, k):
            n = len(ds[split])
            sel = rng.choice(n, size=min(k, n), replace=False)
            rows = ds[split].select(sel.tolist())
            txt = [(r.get("question_title", "") + " " + r.get("question_content", "") + " "
                    + r.get("best_answer", "")).lower() for r in rows]
            lab = [int(r["topic"]) for r in rows]
            return txt, lab

        tr_txt, tr_lab = texts_labels("train", train_subsample)
        te_txt, te_lab = texts_labels("test", test_subsample)

        # frequency vocab from train; 0=pad, 1=unk
        from collections import Counter
        cnt = Counter()
        for t in tr_txt:
            cnt.update(t.split())
        itos = ["<pad>", "<unk>"] + [w for w, _ in cnt.most_common(vocab_size - 2)]
        stoi = {w: i for i, w in enumerate(itos)}

        def encode(txt):
            out = np.zeros((len(txt), seq_len), dtype=np.int64)
            for i, t in enumerate(txt):
                toks = t.split()[:seq_len]
                for j, w in enumerate(toks):
                    out[i, j] = stoi.get(w, 1)
            return out

        blob = dict(
            Xtr=encode(tr_txt), ytr=np.asarray(tr_lab, dtype=np.int64),
            Xte=encode(te_txt), yte=np.asarray(te_lab, dtype=np.int64),
            vocab_size=len(itos),
        )
        with open(cache, "wb") as f:
            pickle.dump(blob, f)

    full_train = TensorDataset(torch.from_numpy(blob["Xtr"]), torch.from_numpy(blob["ytr"]))
    test = TensorDataset(torch.from_numpy(blob["Xte"]), torch.from_numpy(blob["yte"]))
    train_sub, val_sub = stratified_val_split(full_train, val_fraction, seed)

    meta = dict(num_classes=10, input_kind="text", model="text_cnn",
                model_kwargs={"vocab_size": blob["vocab_size"], "embed_dim": 128,
                              "pad_idx": 0},
                vocab_size=blob["vocab_size"], seq_len=seq_len, pad_idx=0,
                epochs=15, batch_size=128)
    return DatasetBundle(train_sub, val_sub, test, meta)


# ---------------------------------------------------------------------------
# Synthetic sub-class split of CIFAR-10 (for COSGD class-count scalability 20.07)
# ---------------------------------------------------------------------------
class _RelabeledDataset(Dataset):
    """Wrap a dataset, replacing its label with a deterministic pseudo-label in
    [0, n_subclasses) computed by hashing (original_label, sample_index)."""

    def __init__(self, base, n_subclasses: int, seed: int = 0):
        self.base = base
        self.n_subclasses = int(n_subclasses)
        self.seed = int(seed)
        self._labels = self._compute_labels()

    def _compute_labels(self):
        orig = _targets_of(self.base)
        labels = np.empty(len(orig), dtype=np.int64)
        for i, y in enumerate(orig):
            h = hashlib.md5(f"{self.seed}:{int(y)}:{i}".encode()).digest()
            labels[i] = int.from_bytes(h[:4], "little") % self.n_subclasses
        return labels

    @property
    def targets(self):
        return self._labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, int(self._labels[i])


def cifar10_subclass_bundle(n_subclasses: int, val_fraction: float = 0.1, seed: int = 2026) -> DatasetBundle:
    """CIFAR-10 relabelled into `n_subclasses` pseudo-classes — isolates class
    count from dataset identity for the COSGD O(n^2) scalability experiment."""
    base = _image_bundle("cifar10", val_fraction, seed)
    tr = _RelabeledDataset(base.train, n_subclasses, seed)
    va = _RelabeledDataset(base.val, n_subclasses, seed)
    te = _RelabeledDataset(base.test, n_subclasses, seed)
    meta = dict(base.meta)
    meta.update(num_classes=n_subclasses, model="small_cifar_cnn",
                synthetic_subclasses=n_subclasses)
    return DatasetBundle(tr, va, te, meta)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DATASETS = ("mnist", "emnist_balanced", "cifar10", "cifar100", "covertype", "yahoo_answers")


def get_dataset(name: str, *, val_fraction: float = 0.1, seed: int = 2026,
                **kwargs) -> DatasetBundle:
    """Build a registered dataset bundle (train/val/test + meta)."""
    if name in ("mnist", "emnist_balanced", "cifar10", "cifar100"):
        return _image_bundle(name, val_fraction, seed)
    if name == "covertype":
        return _covertype_bundle(val_fraction, seed)
    if name == "yahoo_answers":
        return _yahoo_bundle(val_fraction, seed, **kwargs)
    raise ValueError(f"Unknown dataset '{name}'. Available: {DATASETS}")


__all__ = [
    "DatasetBundle", "get_dataset", "DATASETS",
    "stratified_val_split", "balanced_reference_subset",
    "cifar10_subclass_bundle",
]
