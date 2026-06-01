"""
30.04-30.06 — sensitivity sweeps (LR, BoGrad-K, batch size).

Focused sweeps that reuse the bakeoff cell runner with tuning OFF (we are
deliberately scanning one axis, not picking a winner). Each writes a
sensitivity_<kind>.json the plot reads.

  --kind lr     30.04  test acc vs learning rate, per method, on CIFAR-10/CIFAR-100
  --kind K      30.05  test acc vs BoGrad buffer K, per dataset
  --kind batch  30.06  test acc vs batch size, baseline vs BoGrad vs COSGD

Usage:
    python sensitivity.py --kind lr --datasets cifar10 cifar100
    python sensitivity.py --kind K  --datasets cifar10 mnist emnist_balanced
    python sensitivity.py --kind batch --datasets cifar10 emnist_balanced
    python sensitivity.py --kind lr --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                       # noqa: E402
from common.datasets import get_dataset, balanced_reference_subset  # noqa: E402
from _bakeoff import run_bakeoff_cell                            # noqa: E402

LR_GRID = {
    "sgd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "rmsprop": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "adam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "signsgd": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}
K_GRID = [1, 2, 4, 8, 16, 32]
BATCH_GRID = [32, 64, 128, 256, 512]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_points(kind, dataset, args, device) -> List[Dict[str, Any]]:
    bundle = get_dataset(dataset, val_fraction=0.1, seed=2026)
    ep = args.epochs if args.epochs else max(5, bundle.meta["epochs"] // 3)
    bs_default = bundle.meta["batch_size"]
    ref_ds = balanced_reference_subset(bundle.train, num_classes=bundle.meta["num_classes"],
                                       n_per_class=50, seed=2026)
    out_dir = _HERE / "_sensitivity" / kind / dataset
    points: List[Dict[str, Any]] = []

    base = args.base
    if kind == "lr":
        for method in ("baseline", "bograd", "cosgd"):
            for lr in (LR_GRID[base] if not args.smoke else [1e-2, 1e-1]):
                hp: Dict[str, Any] = {"lr": lr}
                if method == "bograd":
                    hp.update(K=32, projection_mode="negative")
                if method == "cosgd":
                    hp.update(cosgd_method="modified_gs_negative", combine="mean")
                rows = run_bakeoff_cell(dataset=dataset, base=base, method=method, bundle=bundle,
                                        ref_ds=ref_ds, out_root=out_dir / f"{method}_lr{lr:g}",
                                        seeds=args.seeds, epochs=ep, batch_size=bs_default,
                                        device=device, tune=False, fixed_hp=hp, measure=False)
                points.append({"method": method, "lr": lr,
                               "acc": float(np.nanmean([r["final_test_acc"] for r in rows]))})
    elif kind == "K":
        mid_lr = LR_GRID[base][len(LR_GRID[base]) // 2]
        for K in (K_GRID if not args.smoke else [4, 32]):
            hp = {"lr": mid_lr, "K": K, "projection_mode": "negative"}
            rows = run_bakeoff_cell(dataset=dataset, base=base, method="bograd", bundle=bundle,
                                    ref_ds=ref_ds, out_root=out_dir / f"K{K}",
                                    seeds=args.seeds, epochs=ep, batch_size=bs_default,
                                    device=device, tune=False, fixed_hp=hp, measure=False)
            points.append({"K": K, "acc": float(np.nanmean([r["final_test_acc"] for r in rows]))})
    elif kind == "batch":
        mid_lr = LR_GRID[base][len(LR_GRID[base]) // 2]
        for method in ("baseline", "bograd", "cosgd"):
            for bs in (BATCH_GRID if not args.smoke else [64, 256]):
                hp = {"lr": mid_lr}
                if method == "bograd":
                    hp.update(K=32, projection_mode="negative")
                if method == "cosgd":
                    hp.update(cosgd_method="modified_gs_negative", combine="mean")
                rows = run_bakeoff_cell(dataset=dataset, base=base, method=method, bundle=bundle,
                                        ref_ds=ref_ds, out_root=out_dir / f"{method}_bs{bs}",
                                        seeds=args.seeds, epochs=ep, batch_size=bs_default,
                                        device=device, tune=False, fixed_hp=hp,
                                        batch_size_override=bs, measure=False)
                points.append({"method": method, "batch": bs,
                               "acc": float(np.nanmean([r["final_test_acc"] for r in rows]))})
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["lr", "K", "batch"])
    ap.add_argument("--datasets", nargs="+", default=["cifar10"])
    ap.add_argument("--base", default="sgd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.datasets = ["cifar10"]; args.seeds = [2026]; args.epochs = 1

    device = _device()
    out = {"kind": args.kind, "base": args.base, "datasets": {}}
    for ds in args.datasets:
        t0 = time.time()
        out["datasets"][ds] = _run_points(args.kind, ds, args, device)
        print(f"[sensitivity:{args.kind}] {ds} done in {time.time()-t0:.0f}s", flush=True)

    res_dir = _HERE / "_sensitivity"
    res_dir.mkdir(parents=True, exist_ok=True)
    storage.write_json_atomic(res_dir / f"sensitivity_{args.kind}.json", out)
    print(f"wrote {res_dir / f'sensitivity_{args.kind}.json'}")


if __name__ == "__main__":
    main()
