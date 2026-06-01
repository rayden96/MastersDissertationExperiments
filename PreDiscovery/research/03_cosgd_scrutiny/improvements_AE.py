"""
COSGD improvement study A-E — implement-and-measure, each in isolation.

For each improvement we state what it targets, then measure WITH vs WITHOUT on
three problems of increasing realism:
  - 2D : 3 Gaussian blobs in 2-D, tiny MLP (P>>C, instant, interpretable)
  - 5cls: 5 Gaussian blobs in 40-D with controllable overlap, small MLP
  - cifar10: small CNN (real), short schedule

A preserve_magnitude  -> effective step / LR coupling   (accuracy + LR-shift)
B low_memory GS        -> peak memory (must be ~identical)(equivalence + speed/mem)
C negative+freq        -> best canonical config          (comparative grid)
D cluster_k            -> scalability + granularity       (acc vs K + speed)
E vmap grads           -> compute bottleneck (must match) (equivalence + speed)

Run:
    python improvements_AE.py --which A B C D E   (default all)
    python improvements_AE.py --which D --problem cifar10
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

from common.optimizers import COSGD            # noqa: E402
from common.optimizers.COSGD import (          # noqa: E402
    modified_gram_schmidt_negative, modified_gram_schmidt_negative_inplace,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Problems: each returns (make_model, train_tensors, test_tensors, n_classes)
# ---------------------------------------------------------------------------
def problem_2d(seed=0):
    g = torch.Generator().manual_seed(seed)
    C = 3
    centres = torch.tensor([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])
    def sample(n):
        xs, ys = [], []
        for c in range(C):
            xs.append(centres[c] + 0.6 * torch.randn(n, 2, generator=g))
            ys.append(torch.full((n,), c))
        return torch.cat(xs), torch.cat(ys)
    Xtr, Ytr = sample(400); Xte, Yte = sample(200)
    def make():
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, C))
    return make, (Xtr, Ytr), (Xte, Yte), C


def problem_5cls(seed=0, overlap=6.0):
    """5 Gaussian blobs in 40-D, HEAVILY overlapping (overlap >> centre spread)
    so the classes genuinely conflict and accuracy has real headroom (~50-75%,
    not 100%). This is what lets the behavioural improvements (A/C/D) show signal;
    a separable version saturates at 1.0 and discriminates nothing."""
    g = torch.Generator().manual_seed(seed)
    C, D = 5, 40
    centres = torch.randn(C, D, generator=g) * 2.0
    def sample(n):
        xs, ys = [], []
        for c in range(C):
            xs.append(centres[c] + overlap * torch.randn(n, D, generator=g))
            ys.append(torch.full((n,), c))
        return torch.cat(xs), torch.cat(ys)
    Xtr, Ytr = sample(600); Xte, Yte = sample(300)
    def make():
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, C))
    return make, (Xtr, Ytr), (Xte, Yte), C


def problem_cifar10(seed=0, n_train=8000, n_test=2000):
    sys.path.insert(0, str(_REPO / "PaperReadyExperiments"))
    from common.datasets import get_dataset
    from common.models import get_model
    b = get_dataset("cifar10", seed=2026)
    def load(ds, n):
        idx = list(range(n)); xs = []; ys = []
        for i in idx:
            x, y = ds[i]; xs.append(x); ys.append(y)
        return torch.stack(xs), torch.tensor(ys)
    Xtr, Ytr = load(b.train, n_train); Xte, Yte = load(b.test, n_test)
    def make():
        torch.manual_seed(seed)
        return get_model("small_cifar_cnn", num_classes=10)
    return make, (Xtr, Ytr), (Xte, Yte), 10


PROBLEMS = {"2d": problem_2d, "5cls": problem_5cls, "cifar10": problem_cifar10}


# ---------------------------------------------------------------------------
# Training harness (in-memory tensors, full-pass minibatches)
# ---------------------------------------------------------------------------
def train_eval(make, train, test, opt_factory, epochs, batch=128, seed=0, is_cosgd=True):
    torch.manual_seed(seed)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make().to(DEV)
    crit = nn.CrossEntropyLoss()
    opt = opt_factory(model, crit)
    Xtr, Ytr = train[0].to(DEV), train[1].to(DEV)
    Xte, Yte = test[0].to(DEV), test[1].to(DEV)
    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            if is_cosgd:
                opt.step(xb, yb, torch.unique(yb))
            else:
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6) if DEV.type == "cuda" else float("nan")
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == Yte).float().mean().item()
    return acc, dt, peak


def cosgd_factory(**kw):
    def f(model, crit):
        return COSGD(model.parameters(), model=model, criterion=crit, **kw)
    return f


def sgd_factory(lr, momentum=0.0):
    def f(model, crit):
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return f


EPOCHS = {"2d": 30, "5cls": 30, "cifar10": 4}
LRS = {"2d": [0.02, 0.05, 0.1, 0.2], "5cls": [0.02, 0.05, 0.1, 0.2],
       "cifar10": [0.02, 0.05, 0.1, 0.2]}
SEEDS = [0, 1, 2]


def _sweep_lr(make, tr, te, ep, lrs, factory_for_lr, is_cosgd=True):
    """Return {lr: mean_acc} averaged over SEEDS, and the best (lr, acc)."""
    out = {}
    for lr in lrs:
        accs = [train_eval(make, tr, te, factory_for_lr(lr), ep, seed=s, is_cosgd=is_cosgd)[0]
                for s in SEEDS]
        out[lr] = float(np.mean(accs))
    best_lr = max(out, key=out.get)
    return out, best_lr, out[best_lr]


# ===========================================================================
# A — preserve_magnitude : targets effective-step / LR coupling
# ===========================================================================
def test_A(problem):
    make, tr, te, C = PROBLEMS[problem]()
    ep, lrs = EPOCHS[problem], LRS[problem]
    print(f"\n### A preserve_magnitude on {problem} (C={C}) — targets LR coupling ###")
    base_curve, base_lr, base_acc = _sweep_lr(make, tr, te, ep, lrs, lambda lr: sgd_factory(lr), is_cosgd=False)
    off_curve, off_lr, off_acc = _sweep_lr(make, tr, te, ep, lrs,
        lambda lr: cosgd_factory(lr=lr, orthogonalization_method="modified_gs_negative", combine="freq"))
    on_curve, on_lr, on_acc = _sweep_lr(make, tr, te, ep, lrs,
        lambda lr: cosgd_factory(lr=lr, orthogonalization_method="modified_gs_negative",
                                 combine="freq", preserve_magnitude=True))
    print(f"  baseline SGD       best lr={base_lr:<5} acc={base_acc:.4f}  curve={ {k:round(v,3) for k,v in base_curve.items()} }")
    print(f"  COSGD no-preserve  best lr={off_lr:<5} acc={off_acc:.4f}  curve={ {k:round(v,3) for k,v in off_curve.items()} }")
    print(f"  COSGD +preserve    best lr={on_lr:<5} acc={on_acc:.4f}  curve={ {k:round(v,3) for k,v in on_curve.items()} }")
    print(f"  -> LR-coupling: does +preserve move COSGD's best-lr toward baseline's ({base_lr})? "
          f"off={off_lr} on={on_lr}")


# ===========================================================================
# B — low_memory in-place GS : must be numerically ~identical, save memory
# ===========================================================================
def test_B(problem):
    print(f"\n### B low_memory GS on {problem} — targets peak memory (must match) ###")
    # 1) pure numerical equivalence of the GS kernels
    torch.manual_seed(0)
    V = torch.randn(8, 5000)
    ref = modified_gram_schmidt_negative(V.clone())
    got = modified_gram_schmidt_negative_inplace(V.clone())
    max_diff = (ref - got).abs().max().item()
    print(f"  GS kernel equivalence: max|dense - inplace| = {max_diff:.2e}  "
          f"({'IDENTICAL' if max_diff < 1e-5 else 'MISMATCH!'})")
    # 2) end-to-end accuracy equivalence + peak mem
    make, tr, te, C = PROBLEMS[problem]()
    ep = EPOCHS[problem]
    a_dense, t_dense, m_dense = train_eval(make, tr, te,
        cosgd_factory(lr=0.1, orthogonalization_method="modified_gs_negative", combine="freq", low_memory=False),
        ep, seed=0)
    a_lm, t_lm, m_lm = train_eval(make, tr, te,
        cosgd_factory(lr=0.1, orthogonalization_method="modified_gs_negative", combine="freq", low_memory=True),
        ep, seed=0)
    print(f"  end-to-end: dense acc={a_dense:.4f} peak={m_dense:.0f}MB {t_dense:.1f}s | "
          f"low_mem acc={a_lm:.4f} peak={m_lm:.0f}MB {t_lm:.1f}s")
    print(f"  -> acc match: {abs(a_dense-a_lm) < 0.02}; peak mem delta: {m_dense-m_lm:+.0f}MB")


# ===========================================================================
# C — negative+freq canonical defaults : comparative grid
# ===========================================================================
def test_C(problem):
    make, tr, te, C = PROBLEMS[problem]()
    ep, lrs = EPOCHS[problem], LRS[problem]
    print(f"\n### C canonical defaults on {problem} — GS variant x combine (best over lr) ###")
    variants = ["modified_gs_normal", "modified_gs_negative"]
    combines = ["sum", "mean", "freq"]
    print(f"  {'variant':<22}" + "".join(f"{c:>10}" for c in combines))
    best_overall = (None, -1)
    for v in variants:
        row = f"  {v:<22}"
        for combine in combines:
            _, blr, bacc = _sweep_lr(make, tr, te, ep, lrs,
                lambda lr, v=v, combine=combine: cosgd_factory(
                    lr=lr, orthogonalization_method=v, combine=combine))
            row += f"{bacc:>9.3f}*" if (v, combine) == ("modified_gs_negative", "freq") else f"{bacc:>10.3f}"
            if bacc > best_overall[1]:
                best_overall = ((v, combine), bacc)
        print(row)
    print(f"  -> best config: {best_overall[0]} acc={best_overall[1]:.3f}  (* = negative+freq)")


# ===========================================================================
# D — clustered COSGD : scalability + granularity
# ===========================================================================
def test_D(problem):
    make, tr, te, C = PROBLEMS[problem]()
    ep, lrs = EPOCHS[problem], LRS[problem]
    print(f"\n### D cluster_k on {problem} (C={C}) — targets scalability + granularity ###")
    Ks = sorted({k for k in [2, 3, 5, C] if 0 < k <= C})
    print(f"  {'config':<16}{'best_lr':>9}{'acc':>9}{'sec':>9}")
    # full per-class reference
    _, blr, bacc = _sweep_lr(make, tr, te, ep, lrs,
        lambda lr: cosgd_factory(lr=lr, orthogonalization_method="modified_gs_negative", combine="freq"))
    # time one run at best lr
    _, dt, _ = train_eval(make, tr, te,
        cosgd_factory(lr=blr, orthogonalization_method="modified_gs_negative", combine="freq"), ep, seed=0)
    print(f"  {'full (K=C)':<16}{blr:>9}{bacc:>9.3f}{dt:>9.1f}")
    for K in Ks:
        if K == C:
            continue
        _, klr, kacc = _sweep_lr(make, tr, te, ep, lrs,
            lambda lr, K=K: cosgd_factory(lr=lr, orthogonalization_method="modified_gs_negative",
                                          combine="freq", cluster_k=K))
        _, dtk, _ = train_eval(make, tr, te,
            cosgd_factory(lr=klr, orthogonalization_method="modified_gs_negative", combine="freq", cluster_k=K),
            ep, seed=0)
        print(f"  {'K=' + str(K):<16}{klr:>9}{kacc:>9.3f}{dtk:>9.1f}")
    print(f"  -> does small K retain most of full-C accuracy ({bacc:.3f}) at lower cost?")


# ===========================================================================
# E — vmap per-sample grads : compute (must match loop)
# ===========================================================================
def test_E(problem):
    make, tr, te, C = PROBLEMS[problem]()
    ep = EPOCHS[problem]
    print(f"\n### E vmap grads on {problem} — targets C-backward compute (must match) ###")
    # equivalence: compare the per-class grad matrix from loop vs vmap on one batch
    torch.manual_seed(0)
    model = make().to(DEV); crit = nn.CrossEntropyLoss()
    opt_loop = COSGD(model.parameters(), model=model, criterion=crit, step_method="single_forward")
    xb, yb = tr[0][:128].to(DEV), tr[1][:128].to(DEV)
    valid = torch.unique(yb)
    cg_loop, _, _ = opt_loop._grads_single_forward(xb, yb, valid)
    opt_vmap = COSGD(model.parameters(), model=model, criterion=crit, step_method="vmap")
    cg_vmap, _, _ = opt_vmap._grads_vmap(xb, yb, valid)
    max_diff = (cg_loop - cg_vmap).abs().max().item()
    rel = max_diff / (cg_loop.abs().max().item() + 1e-12)
    print(f"  per-class grad equivalence: max|loop - vmap| = {max_diff:.2e} (rel {rel:.2e}) "
          f"({'MATCH' if rel < 1e-3 else 'MISMATCH!'})")
    # speed: end-to-end
    a_loop, t_loop, _ = train_eval(make, tr, te,
        cosgd_factory(lr=0.1, orthogonalization_method="modified_gs_negative", combine="freq",
                      step_method="single_forward"), ep, seed=0)
    a_vmap, t_vmap, _ = train_eval(make, tr, te,
        cosgd_factory(lr=0.1, orthogonalization_method="modified_gs_negative", combine="freq",
                      step_method="vmap"), ep, seed=0)
    print(f"  end-to-end: loop acc={a_loop:.4f} {t_loop:.1f}s | vmap acc={a_vmap:.4f} {t_vmap:.1f}s "
          f"({t_loop/max(t_vmap,1e-6):.2f}x speed)")


TESTS = {"A": test_A, "B": test_B, "C": test_C, "D": test_D, "E": test_E}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=list(TESTS.keys()))
    ap.add_argument("--problem", nargs="+", default=["2d", "5cls", "cifar10"])
    args = ap.parse_args()
    print(f"device={DEV}  improvements={args.which}  problems={args.problem}")
    for prob in args.problem:
        print(f"\n{'='*70}\nPROBLEM: {prob}\n{'='*70}")
        for w in args.which:
            try:
                TESTS[w](prob)
            except Exception as e:
                import traceback
                print(f"  !! {w} on {prob} failed: {type(e).__name__}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()

