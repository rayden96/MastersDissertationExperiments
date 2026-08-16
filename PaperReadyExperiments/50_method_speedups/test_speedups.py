"""
50 — Fast check on projection/orthogonalisation speed-ups.

NOT a paper experiment. A short, cheap harness for deciding whether an
implementation change is worth adopting before spending compute on the real
ablations. Three questions, in order of how cheaply they can be answered:

  1. CORRECTNESS. Does the variant compute what it claims? Sequential and
     batched projection must agree exactly on an orthogonal buffer and must
     both respect the non-increase property on a correlated one.
  2. SPEED. Per-step cost of each variant, timed back-to-back in one process
     with warmup discarded and the median reported, as in 30.11.
  3. DOES IT STILL TRAIN. A few hundred steps per arm on each dataset, one
     seed, reporting the loss/accuracy curve. Enough to catch a variant that
     diverges or stalls; NOT enough to rank variants on accuracy. Anything
     that survives this gets run through the real ablations.

Run:
    python test_speedups.py                       # all three datasets
    python test_speedups.py --datasets cifar10
    python test_speedups.py --steps 150 --quick
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                    # noqa: E402
from common.datasets import get_dataset                       # noqa: E402
from common.models import get_model                           # noqa: E402
from common.optimizers import BoGrad, COSGD                   # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

K_PER_BASE = {"sgd": 16, "signsgd": 32, "rmsprop": 32, "adam": 128}


# ---------------------------------------------------------------- correctness
def check_correctness(device) -> List[Dict[str, Any]]:
    """Sequential vs batched: identical on an orthogonal buffer (the only case
    where the two operators coincide analytically), and both non-increasing on a
    correlated one (the property Chapter 5's learning-rate argument uses)."""
    out = []
    P, K = 20_000, 8
    dummy = [torch.nn.Parameter(torch.zeros(1, device=device))]

    def mk(guard=True):
        return BoGrad(dummy, base_optimizer_cls=torch.optim.SGD, lr=0.1,
                      buffer_size=K, projection_mode="negative",
                      projection_scope="global", orth_method="batched",
                      batched_norm_guard=guard)

    torch.manual_seed(0)
    # orthogonal buffer -> the two operators must agree
    Q, _ = torch.linalg.qr(torch.randn(P, K, device=device))
    ortho = [Q[:, i] / Q[:, i].norm() for i in range(K)]
    x = torch.randn(P, device=device)
    o = mk()
    diff = (o._project_sequential(x.clone(), ortho)
            - o._project_batched(x.clone(), ortho)).abs().max().item()
    out.append({"check": "orthogonal buffer: sequential == batched",
                "value": diff, "pass": diff < 1e-4})

    # correlated buffer -> both must not lengthen the step
    base = torch.randn(P, device=device)
    corr = []
    for _ in range(K):
        v = base + 0.3 * torch.randn(P, device=device)
        corr.append(v / v.norm())
    x = -base / base.norm() + 0.2 * torch.randn(P, device=device)
    xn = x.norm().item()
    for name, vec in (("sequential", mk()._project_sequential(x.clone(), corr)),
                      ("batched+guard", mk(True)._project_batched(x.clone(), corr))):
        r = vec.norm().item() / xn
        out.append({"check": f"non-increase, {name}", "value": r,
                    "pass": r <= 1.0 + 1e-4})
    # Informational, not a requirement: this is the demonstration that the
    # unguarded batched operator lengthens the step on a correlated buffer,
    # which is why the guard exists. A value above 1 here is the expected
    # result, so it must not be scored as a failure.
    r = mk(False)._project_batched(x.clone(), corr).norm().item() / xn
    out.append({"check": "batched WITHOUT guard lengthens (expected >1)",
                "value": r, "pass": True, "informational": True})
    return out


# ---------------------------------------------------------------------- speed
def time_variant(bundle, base: str, spec_fn, n_steps: int, warmup: int,
                 device) -> Dict[str, float]:
    meta = bundle.meta
    model = get_model(meta["model"], num_classes=meta["num_classes"],
                      **meta.get("model_kwargs", {})).to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt = spec_fn(model)
    loader = DataLoader(bundle.train, batch_size=meta["batch_size"],
                        shuffle=True, num_workers=2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    per_step, seen = [], 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if getattr(opt, "_is_per_class", False) or isinstance(opt, COSGD):
            opt.step(x, y, torch.unique(y))
        else:
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        seen += 1
        if seen > warmup:
            per_step.append(dt)
        if seen >= warmup + n_steps:
            break
    peak = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
    return {"median_step_s": statistics.median(per_step) if per_step else float("nan"),
            "peak_mem_mb": peak}


# ------------------------------------------------------------ does it train
def short_train(bundle, base: str, spec_fn, steps: int, device) -> Dict[str, Any]:
    meta = bundle.meta
    torch.manual_seed(2026)
    model = get_model(meta["model"], num_classes=meta["num_classes"],
                      **meta.get("model_kwargs", {})).to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt = spec_fn(model)
    loader = DataLoader(bundle.train, batch_size=meta["batch_size"],
                        shuffle=True, num_workers=2)
    losses, seen = [], 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(opt, COSGD):
            loss = float(opt.step(x, y, torch.unique(y)))
        else:
            opt.zero_grad(set_to_none=True)
            loss_t = crit(model(x), y); loss_t.backward(); opt.step()
            loss = float(loss_t.item())
        losses.append(loss); seen += 1
        if seen >= steps:
            break
    # held-out accuracy on a slice of test, enough to catch divergence
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(bundle.test, batch_size=512, num_workers=2):
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
            if total >= 5000:
                break
    return {"loss_first": round(statistics.fmean(losses[:10]), 4),
            "loss_last": round(statistics.fmean(losses[-10:]), 4),
            "acc": round(correct / max(total, 1), 4),
            "diverged": not all(l == l for l in losses)}


def variants(base: str, lr: float):
    """(name, factory) for each arm under test."""
    K = K_PER_BASE.get(base, 32)
    BASE_CLS = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam,
                "rmsprop": torch.optim.RMSprop}[base]
    kw = dict(base_optimizer_cls=BASE_CLS, lr=lr, buffer_size=K,
              projection_mode="negative", projection_scope="global",
              collect_stats=False)
    return [
        ("baseline", lambda m: BASE_CLS(m.parameters(), lr=lr)),
        ("bograd_sequential", lambda m: BoGrad(m.parameters(), orth_method="sequential", **kw)),
        ("bograd_batched", lambda m: BoGrad(m.parameters(), orth_method="batched", **kw)),
        ("bograd_batched_bf16", lambda m: BoGrad(
            m.parameters(), orth_method="batched", buffer_dtype=torch.bfloat16, **kw)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["covertype", "cifar10", "cifar100"])
    ap.add_argument("--bases", nargs="+", default=["sgd", "adam"])
    ap.add_argument("--steps", type=int, default=300, help="training steps per arm")
    ap.add_argument("--time_steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="1 dataset, 1 base, 100 steps")
    args = ap.parse_args()
    if args.quick:
        args.datasets = ["cifar10"]; args.bases = ["sgd"]; args.steps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device: {device}"
          f"{' | ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    out: Dict[str, Any] = {"device": str(device), "correctness": [], "cells": []}

    print("\n=== 1. CORRECTNESS ===")
    for c in check_correctness(device):
        out["correctness"].append(c)
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']:<44} {c['value']:.6f}")
    if not all(c["pass"] for c in out["correctness"]):
        print("  !! correctness failed — speed numbers below are not meaningful")

    for ds in args.datasets:
        try:
            bundle = get_dataset(ds, val_fraction=0.1, seed=2026)
        except Exception as e:
            print(f"\n!! {ds}: {type(e).__name__}: {e}"); continue
        nc = bundle.meta["num_classes"]
        print(f"\n=== {ds} (nc={nc}, model={bundle.meta['model']}) ===")
        for base in args.bases:
            lr = args.lr if base == "sgd" else 1e-3
            print(f"\n  -- base {base} (lr={lr}, K={K_PER_BASE.get(base, 32)}) --")
            print(f"     {'variant':<22}{'ms/step':>10}{'vs base':>9}"
                  f"{'peak MB':>10}{'loss':>9}{'acc':>8}")
            ref = None
            for name, fn in variants(base, lr):
                try:
                    t = time_variant(bundle, base, fn, args.time_steps, args.warmup, device)
                    r = short_train(bundle, base, fn, args.steps, device)
                except Exception as e:
                    print(f"     {name:<22} FAILED {type(e).__name__}: {e}"[:100])
                    continue
                ms = t["median_step_s"] * 1000
                if name == "baseline":
                    ref = ms
                ov = (ms / ref) if ref else float("nan")
                out["cells"].append({"dataset": ds, "base": base, "variant": name,
                                     **t, "overhead_x": ov, **r})
                print(f"     {name:<22}{ms:>10.2f}{ov:>8.2f}x{t['peak_mem_mb']:>10.0f}"
                      f"{r['loss_last']:>9.4f}{r['acc']:>8.4f}"
                      + ("  DIVERGED" if r["diverged"] else ""))
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for d in (storage.persistent_dir("50_method_speedups"), _HERE):
        storage.write_json_atomic(Path(d) / "speedup_test.json", out)
    print(f"\nwrote speedup_test.json ({len(out['cells'])} cells)")
    print("\nRead: a variant is worth promoting only if it is FASTER and its "
          "loss/acc are in line with sequential. Ranking on this accuracy is not "
          "valid (one seed, few steps) — that is what the ablations are for.")


if __name__ == "__main__":
    main()
