"""
Reclaim COSGD's low-dim win while fixing high-dim breakage.

Finding from paper_repro.py:
  - cosgd_paper (full GS + sum + desc-sort, NO gate) = 5.6x on iris, 3.0x on
    wine, but BREAKS at digits (dim 64, acc 0.84 stuck) due to sum over-scaling.
  - cosgd_improved (gate+freq+preserve) KILLS the low-dim win (gate skips the
    orthogonalisation that wins on clean low-dim problems).

Goal: a single config that keeps the paper's low-dim speedup AND survives higher
dim. Candidate axes, all WITHOUT the conflict gate (the low-dim killer):
  combine in {sum, mean, freq}  x  preserve_magnitude in {off, on}
  always: full classical GS, descending-magnitude sort (paper).
Compared against cosgd_paper and cosgd_improved as anchors, and SGD baseline.

Run: python reclaim_lowdim.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import importlib.util
spec = importlib.util.spec_from_file_location("pr", _HERE / "paper_repro.py")
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)

from common.optimizers import COSGD  # noqa: E402
DEV = pr.DEV


def train_eval_cfg(name, cfg, lr, epochs, seed, batch=16):
    Xtr, ytr, Xte, yte, in_dim, n_cls = pr.DATA[name]
    Xtr, ytr, Xte, yte = Xtr.to(DEV), ytr.to(DEV), Xte.to(DEV), yte.to(DEV)
    model = pr.make_mlp(in_dim, n_cls, seed).to(DEV)
    crit = nn.CrossEntropyLoss()
    opt = COSGD(model.parameters(), model=model, criterion=crit, lr=lr, **cfg)
    n = Xtr.shape[0]; rng = np.random.default_rng(seed); per = []
    for ep in range(epochs):
        model.train(); perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]; xb, yb = Xtr[idx], ytr[idx]
            opt.step(xb, yb, torch.unique(yb))
        model.eval()
        with torch.no_grad():
            per.append((model(Xte).argmax(1) == yte).float().mean().item())
    return per


CANDIDATES = {
    "paper(sum)":        dict(orthogonalization_method="gram_schmidt_normal", class_order="desc", combine="sum"),
    "full+mean":         dict(orthogonalization_method="gram_schmidt_normal", class_order="desc", combine="mean"),
    # RECLAIMED: paper's sum + full GS + desc, with a norm cap to survive high dim.
    "reclaim cap2":      dict(orthogonalization_method="gram_schmidt_normal", class_order="desc", combine="sum", combine_norm_cap=2.0),
    "reclaim cap3":      dict(orthogonalization_method="gram_schmidt_normal", class_order="desc", combine="sum", combine_norm_cap=3.0),
    "reclaim cap4":      dict(orthogonalization_method="gram_schmidt_normal", class_order="desc", combine="sum", combine_norm_cap=4.0),
    "improved(gate)":    dict(orthogonalization_method="modified_gs_negative", class_order="fixed", combine="freq", preserve_magnitude=True, conflict_gate=True, conflict_threshold=0.1),
}


def epochs_to(curve, t):
    for i, a in enumerate(curve):
        if a >= t:
            return i + 1
    return None


def main():
    datasets = ["iris", "wine", "breast_cancer", "digits"]
    seeds = [0, 1, 2]; epochs = 30
    for name in datasets:
        pr.DATA[name] = pr.load(name)
    # SGD targets (reuse pr.train_eval)
    print(f"device={DEV}  reclaim study  epochs={epochs} seeds={seeds}\n")
    for name in datasets:
        in_dim, n_cls = pr.DATA[name][4], pr.DATA[name][5]
        sgd = np.mean([pr.train_eval(name, "sgd", 0.1, epochs, s) for s in seeds], axis=0)
        tgt = sgd[-1]; sgd_ep = epochs_to(sgd, tgt)
        print(f"================ {name} (dim={in_dim}, cls={n_cls}) — SGD final {tgt:.3f} @ep{sgd_ep} ================")
        print(f"  {'config':<18}{'ep1':>7}{'ep5':>7}{'final':>8}{'ep->tgt':>9}{'speedup':>9}")
        print(f"  {'sgd':<18}{sgd[0]:>7.3f}{sgd[4]:>7.3f}{sgd[-1]:>8.3f}{sgd_ep:>9}{'1.00x':>9}")
        for label, cfg in CANDIDATES.items():
            runs = [train_eval_cfg(name, cfg, 0.1, epochs, s) for s in seeds]
            m = np.mean(runs, axis=0)
            et = epochs_to(m, tgt); sp = (sgd_ep / et) if et else None
            print(f"  {label:<18}{m[0]:>7.3f}{m[4]:>7.3f}{m[-1]:>8.3f}"
                  f"{(str(et) if et else '>'+str(epochs)):>9}{(f'{sp:.2f}x' if sp else 'n/a'):>9}", flush=True)
        print()


if __name__ == "__main__":
    main()
