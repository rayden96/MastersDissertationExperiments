"""
COSGD paper reproduction + investigation on SMALL problems.

The conference paper (Clustered_Orthogonalized_Stochastic_Gradient_Descent.pdf)
showed COSGD's big wins on LOW-DIMENSIONAL problems (Iris: epoch-5 94.2% vs SGD
82.6%; reached at epoch 5 what SGD needed 30 for). The paper's COSGD is the
PLAIN form: classical full Gram-Schmidt, descending-magnitude sort, combine=sum,
NO normalization, NO negative-only, NO gate.

This script:
  1. reproduces the paper's COSGD ("paper" config) on small sklearn datasets,
  2. compares it to the current "improved" defaults (freq+preserve+gate) to test
     whether my CIFAR-tuned changes HURT the low-dim regime,
  3. compares to SGD/Adam/RMSprop baselines,
  4. reports per-epoch accuracy + epochs-to-target speed-up.

Datasets (dim ladder): iris(4), wine(13), breast_cancer(30), digits(64).
Paper protocol: MLP, no momentum, LR-COSGD=0.1 (tabular) / grid for others,
standardised features, multiple seeds.

Run: python paper_repro.py [--datasets iris wine ...] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.optimizers import COSGD  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(name):
    import sklearn.datasets as d
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    loader = {"iris": d.load_iris, "wine": d.load_wine,
              "breast_cancer": d.load_breast_cancer, "digits": d.load_digits}[name]
    ds = loader()
    X, y = ds.data.astype("float32"), ds.target.astype("int64")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr).astype("float32"), sc.transform(Xte).astype("float32")
    n_cls = int(y.max() + 1)
    return (torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte),
            X.shape[1], n_cls)


def make_mlp(in_dim, n_cls, seed):
    torch.manual_seed(seed)
    # paper uses small MLPs, e.g. iris 4->16->3; scale hidden modestly with in_dim
    h = max(16, in_dim)
    return nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, n_cls))


def train_eval(name, opt_kind, lr, epochs, seed, batch=16):
    Xtr, ytr, Xte, yte, in_dim, n_cls = DATA[name]
    Xtr, ytr, Xte, yte = Xtr.to(DEV), ytr.to(DEV), Xte.to(DEV), yte.to(DEV)
    model = make_mlp(in_dim, n_cls, seed).to(DEV)
    crit = nn.CrossEntropyLoss()

    if opt_kind == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr); is_cosgd = False
    elif opt_kind == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr); is_cosgd = False
    elif opt_kind == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=lr); is_cosgd = False
    elif opt_kind == "cosgd_paper":
        # PAPER-FAITHFUL: classical full GS, descending-mag sort, sum, no norm/gate
        opt = COSGD(model.parameters(), model=model, criterion=crit, lr=lr,
                    orthogonalization_method="gram_schmidt_normal",
                    class_order="desc", combine="sum",
                    preserve_magnitude=False, conflict_gate=False); is_cosgd = True
    elif opt_kind == "cosgd_improved":
        # current registry defaults (CIFAR-tuned)
        opt = COSGD(model.parameters(), model=model, criterion=crit, lr=lr,
                    orthogonalization_method="modified_gs_negative",
                    class_order="fixed", combine="freq",
                    preserve_magnitude=True, conflict_gate=True,
                    conflict_threshold=0.1); is_cosgd = True
    else:
        raise ValueError(opt_kind)

    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    per_epoch = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb, yb = Xtr[idx], ytr[idx]
            if is_cosgd:
                opt.step(xb, yb, torch.unique(yb))
            else:
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        per_epoch.append(acc)
    return per_epoch


def epochs_to(curve, target):
    for i, a in enumerate(curve):
        if a >= target:
            return i + 1
    return None


DATA = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "digits"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    LR = {"sgd": 0.1, "adam": 1e-3, "rmsprop": 1e-3,
          "cosgd_paper": 0.1, "cosgd_improved": 0.1}
    METHODS = ["sgd", "adam", "rmsprop", "cosgd_paper", "cosgd_improved"]

    print(f"device={DEV}  COSGD paper-repro on small problems  epochs={args.epochs} seeds={args.seeds}")
    for name in args.datasets:
        DATA[name] = load(name)
        in_dim, n_cls = DATA[name][4], DATA[name][5]
        print(f"\n================ {name}  (dim={in_dim}, classes={n_cls}) ================")
        curves = {}
        for mk in METHODS:
            runs = [train_eval(name, mk, LR[mk], args.epochs, seed=s) for s in args.seeds]
            n = min(len(r) for r in runs)
            mean = np.mean([r[:n] for r in runs], axis=0)
            curves[mk] = mean
        # target = SGD's final-epoch accuracy
        tgt = curves["sgd"][-1]
        sgd_ep = epochs_to(curves["sgd"], tgt)
        print(f"  target = SGD final acc = {tgt:.4f} (SGD reaches at epoch {sgd_ep})")
        print(f"  {'method':<16}{'ep1':>7}{'ep5':>7}{'final':>8}{'ep->SGDfinal':>14}{'speedup':>9}")
        for mk in METHODS:
            c = curves[mk]
            et = epochs_to(c, tgt)
            sp = (sgd_ep / et) if (et and sgd_ep) else None
            print(f"  {mk:<16}{c[0]:>7.3f}{c[4]:>7.3f}{c[-1]:>8.3f}"
                  f"{(str(et) if et else '>'+str(len(c))):>14}{(f'{sp:.2f}x' if sp else 'n/a'):>9}")


if __name__ == "__main__":
    main()
