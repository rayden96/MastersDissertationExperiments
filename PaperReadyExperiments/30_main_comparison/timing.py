"""
30.11 — Per-step timing, all method arms measured back-to-back in ONE process.

Why this exists. The bakeoff records `mean_step_wall_time_s` per cell, but its
cells are trained across different Colab sessions on whatever GPU was allocated
and under whatever contention existed at the time. Comparing those numbers
across arms produced impossible results (BOGrad appearing FASTER per step than
its own baseline on EMNIST, when the projection can only add work). Wall-clock
speed-up is only meaningful when the arms are timed on the same device, in the
same process, back to back.

This script does exactly that and nothing else: for each (dataset, base, method)
it builds the model, runs `--warmup` steps to settle cuDNN autotuning and
allocator behaviour, then times `--n_steps` training steps and records the
median (robust to a stray slow step) alongside the mean and peak memory. It
trains nothing to convergence and writes no accuracy numbers; the epoch
speed-ups keep coming from the bakeoff curves, and only the per-step cost ratio
is taken from here.

Run:
    python timing.py                                  # default datasets/bases
    python timing.py --datasets cifar10 covertype
    python timing.py --smoke
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                    # noqa: E402
from common.datasets import get_dataset                       # noqa: E402
from common.methods import build_method                       # noqa: E402
from common.models import get_model                           # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

# The knob values the method chapters settled on, so timing reflects the
# configuration actually recommended rather than a library default.
CANONICAL_HP = {
    "bograd": {"projection_mode": "negative", "projection_scope": "global"},
    "cosgd": {"cosgd_method": "gram_schmidt_normal", "class_order": "desc",
              "combine": "sum", "combine_norm_cap": 2.0},
    "graddrop": {"leak": 0.0},
    "dropout": {"dropout_p": 0.1},
}
# Per-base buffer size measured in 10.01.
K_PER_BASE = {"sgd": 16, "signsgd": 32, "rmsprop": 32, "adam": 128}


def time_arm(bundle, base: str, method: str, lr: float, n_steps: int,
             warmup: int, batch_size: int, device,
             K: Optional[int] = None) -> Dict[str, Any]:
    meta = bundle.meta
    hp: Dict[str, Any] = {"lr": lr, **CANONICAL_HP.get(method, {})}
    if method == "bograd":
        hp["K"] = K if K is not None else K_PER_BASE.get(base, 32)
        # The buffer gains one update per step and the projection's cost grows
        # with how many it holds. A fixed 15-step warmup timed a K=128 buffer
        # that was never more than 75 entries full, so warm up until it is.
        warmup = max(warmup, hp["K"] + 1)

    spec = build_method(method, base, hp=hp)
    mk = {**meta.get("model_kwargs", {}), **spec.model_kwargs}
    model = get_model(meta["model"], num_classes=meta["num_classes"], **mk).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if method == "bograd" and device.type == "cuda":
        # A full buffer larger than the device may not raise: on Windows the
        # driver can fall back to system memory, and the step time is then the
        # paging, not the projection. Refuse it up front instead.
        need = hp["K"] * sum(p.numel() * p.element_size() for p in model.parameters())
        have = torch.cuda.get_device_properties(device).total_memory
        if need > have:
            raise MemoryError(f"a full K={hp['K']} buffer needs {need / 1e9:.1f} GB "
                              f"and the device has {have / 1e9:.1f} GB")
    crit = torch.nn.CrossEntropyLoss()
    opt = spec.optimizer_factory(model, crit)
    loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True, num_workers=2)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    per_step: List[float] = []
    seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if spec.step_kind == "per_class":
            opt.step(x, y, torch.unique(y))
        else:
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        seen += 1
        if seen > warmup:                    # discard warmup steps entirely
            per_step.append(dt)
        if seen >= warmup + n_steps:
            break

    peak = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
    return {
        "base": base, "method": method, "hp": hp, "n_timed": len(per_step),
        "n_params": n_params, "warmup": warmup,
        "median_step_s": statistics.median(per_step) if per_step else float("nan"),
        "mean_step_s": statistics.fmean(per_step) if per_step else float("nan"),
        "peak_mem_mb": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["covertype", "cifar10", "cifar100", "emnist_balanced",
                             "mnist", "yahoo_answers"])
    ap.add_argument("--bases", nargs="+", default=["sgd", "signsgd", "rmsprop", "adam"])
    ap.add_argument("--methods", nargs="+",
                    default=["baseline", "cosgd", "bograd", "graddrop", "dropout"])
    ap.add_argument("--n_steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cosgd_max_classes", type=int, default=10,
                    help="skip COSGD above this class count (the 20.07 wall)")
    ap.add_argument("--bograd_K", type=int, nargs="+", default=None,
                    help="time BOGrad at each of these buffer sizes instead of "
                         "the per-base K, for the overhead-against-K table")
    ap.add_argument("--out", default="timing.json",
                    help="file name, so a K sweep does not overwrite timing.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.datasets = ["cifar10"]; args.bases = ["sgd"]
        args.n_steps = 5; args.warmup = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out: Dict[str, Any] = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "n_steps": args.n_steps, "warmup": args.warmup, "rows": [], "failed": [],
    }

    for ds in args.datasets:
        try:
            bundle = get_dataset(ds, val_fraction=0.1, seed=2026, augment=True)
        except Exception as e:
            print(f"!! {ds}: {type(e).__name__}: {e}"); continue
        nc, bs = bundle.meta["num_classes"], bundle.meta["batch_size"]
        print(f"\n=== {ds} (nc={nc}, batch={bs}) ===", flush=True)
        for base in args.bases:
            base_med = None
            for method in args.methods:
                if method == "cosgd" and nc > args.cosgd_max_classes:
                    continue
                Ks = args.bograd_K if (method == "bograd" and args.bograd_K) else [None]
                for K in Ks:
                    tag = method if K is None else f"{method} K={K}"
                    failed = None
                    try:
                        r = time_arm(bundle, base, method, args.lr, args.n_steps,
                                     args.warmup, bs, device, K=K)
                    except Exception as e:
                        failed = f"{type(e).__name__}: {str(e)[:120]}"
                    if failed:
                        # A buffer too large for the device is a result, not a
                        # crash, so it is recorded for the table to report. It
                        # goes in its own list so readers of `rows` are unchanged.
                        print(f"  {base:<8}{tag:<14} FAILED {failed}", flush=True)
                        out["failed"].append({"dataset": ds, "base": base,
                                              "method": method, "K": K,
                                              "error": failed})
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    r["dataset"] = ds
                    if method == "baseline":
                        base_med = r["median_step_s"]
                    r["overhead_x"] = (r["median_step_s"] / base_med) if base_med else None
                    out["rows"].append(r)
                    ov = f"{r['overhead_x']:.2f}x" if r["overhead_x"] else "ref"
                    print(f"  {base:<8}{tag:<14}{r['median_step_s']*1000:8.2f} ms/step"
                          f"{ov:>9}   peak {r['peak_mem_mb']:.0f} MB", flush=True)
                    del r
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    for d in (storage.persistent_dir("30_main_comparison/timing"), _HERE):
        storage.write_json_atomic(Path(d) / args.out, out)
    print(f"\nwrote {args.out} ({len(out['rows'])} rows, {len(out['failed'])} failed)")


if __name__ == "__main__":
    main()
