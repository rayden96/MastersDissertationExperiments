"""
20.07 — Class-count scalability (the O(n^2) wall that motivates BoGrad).

Claim. COSGD's per-class Gram-Schmidt is O(n^2) in dot products and stores n
per-class gradient vectors, so both wall-clock/step and peak memory grow
super-linearly in the number of classes n present per batch. This is the
narrative hinge to BoGrad (which is O(K) in a fixed buffer, independent of n).

Two views (thesis 3.4 / 3.5):
  - synthetic control: CIFAR-10 relabelled into n in {2,5,10,20,50,100} pseudo-
    classes (label sub-hashing) -> isolates class count from dataset identity;
  - real: CIFAR-10 (10), EMNIST-Balanced (47), CIFAR-100 (100).

Measures wall-clock seconds/step and peak GPU memory for COSGD vs the SGD
baseline. Measurement meter is OFF here (we want raw method cost, not diagnostic
overhead); COSGD's own collect_timing is ON for the per-op breakdown.

This is a timing experiment: 1-2 epochs is plenty. Run:
    python run.py                      # synthetic sweep {2..100}
    python run.py --real               # the three real datasets
    python run.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_AXIS = _HERE.parent
_PRE = _AXIS.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                       # noqa: E402
from common.datasets import get_dataset, cifar10_subclass_bundle  # noqa: E402
from common.methods import build_method                          # noqa: E402
from common.models import get_model                              # noqa: E402
from torch.utils.data import DataLoader, Subset                  # noqa: E402


def _time_method(bundle, method, base, lr, n_steps, batch_size, device):
    """Train for n_steps, return (sec_per_step, peak_mem_mb)."""
    meta = bundle.meta
    # Reclaimed COSGD config (the canonical one); GS variant/combine don't change
    # the O(n^2) timing conclusion but we keep it consistent with the chapter.
    spec = build_method(method, base, hp={"lr": lr, "cosgd_method": "gram_schmidt_normal",
                                          "class_order": "desc", "combine": "sum",
                                          "combine_norm_cap": 2.0, "collect_timing": False})
    model = get_model(meta["model"], num_classes=meta["num_classes"], **spec.model_kwargs).to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt = spec.optimizer_factory(model, crit)
    loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True, num_workers=0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter(); steps = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if spec.step_kind == "per_class":
            opt.step(x, y, torch.unique(y))
        else:
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        steps += 1
        if steps >= n_steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / max(steps, 1)
    peak = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
    return sec, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="run the real datasets instead of synthetic")
    ap.add_argument("--counts", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100])
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.counts = [2, 10, 50]; args.n_steps = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = _HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = {"device": str(device), "n_steps": args.n_steps, "batch_size": args.batch_size,
           "points": []}

    if args.real:
        datasets = [("cifar10", get_dataset("cifar10", seed=2026)),
                    ("emnist_balanced", get_dataset("emnist_balanced", seed=2026)),
                    ("cifar100", get_dataset("cifar100", seed=2026))]
        for name, bundle in datasets:
            nc = bundle.meta["num_classes"]
            base_sec, base_mem = _time_method(bundle, "baseline", "sgd", args.lr,
                                              args.n_steps, args.batch_size, device)
            cos_sec, cos_mem = _time_method(bundle, "cosgd", "sgd", args.lr,
                                            args.n_steps, args.batch_size, device)
            pt = {"dataset": name, "n_classes": nc,
                  "baseline_sec_per_step": base_sec, "cosgd_sec_per_step": cos_sec,
                  "baseline_peak_mb": base_mem, "cosgd_peak_mb": cos_mem}
            out["points"].append(pt)
            print(f"  {name:<16} nc={nc:<4} SGD {base_sec*1000:.1f}ms/{base_mem:.0f}MB  "
                  f"COSGD {cos_sec*1000:.1f}ms/{cos_mem:.0f}MB  ({cos_sec/base_sec:.1f}x)", flush=True)
        tag = "real"
    else:
        for n in args.counts:
            bundle = cifar10_subclass_bundle(n_subclasses=n, seed=2026)
            base_sec, base_mem = _time_method(bundle, "baseline", "sgd", args.lr,
                                              args.n_steps, args.batch_size, device)
            cos_sec, cos_mem = _time_method(bundle, "cosgd", "sgd", args.lr,
                                            args.n_steps, args.batch_size, device)
            pt = {"n_classes": n,
                  "baseline_sec_per_step": base_sec, "cosgd_sec_per_step": cos_sec,
                  "baseline_peak_mb": base_mem, "cosgd_peak_mb": cos_mem}
            out["points"].append(pt)
            print(f"  n={n:<4} SGD {base_sec*1000:.1f}ms/{base_mem:.0f}MB  "
                  f"COSGD {cos_sec*1000:.1f}ms/{cos_mem:.0f}MB  ({cos_sec/base_sec:.1f}x)", flush=True)
        tag = "synthetic"

    # Persist to Drive as well as the repo-local folder. The repo copy lives in
    # Colab's ephemeral /content, so a run that only wrote there was lost the
    # moment the session ended.
    persist_dir = storage.persistent_dir("20_cosgd_ablation/07_scalability")
    for d in (persist_dir, res_dir):
        storage.write_json_atomic(d / f"scalability_{tag}.json", out)
    print(f"\nwrote {persist_dir / f'scalability_{tag}.json'} (+ repo-local copy)")


if __name__ == "__main__":
    main()
