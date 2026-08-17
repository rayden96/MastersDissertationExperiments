"""
Drive all COSGD-ablation axes from one entry point (Colab or local).

Mirror of 10_bograd_ablation/run_all.py. Loads each axis's run.py main() and
invokes it with a shared budget; resumable via the per-cell JobManager.

    python run_all.py                          # all training axes, proxy budget
    python run_all.py --axes 01 05 06          # subset
    python run_all.py --smoke
    python run_all.py --aggregate              # build 20.08 after runs

20.07 (scalability) and 20.03's synthetic_dim_sweep are timing/synthetic scripts
with their own CLIs; run them directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

AXES = {
    "01": "01_gs_variant",
    "02": "02_class_order",
    "03": "03_prenormalize",
    "04": "04_step_method",
    "05": "05_combine",
    "06": "06_base_optimizer",
    "09": "09_batch_size",
    "10": "10_learning_rate",
}

# 04 is an implementation-equivalence check, not a chapter axis; the three
# per-class strategies compute the same subgradients and differ only in cost,
# so it is excluded from the default set.
DEFAULT_AXES = ["01", "02", "03", "05", "06", "09", "10"]


def _invoke(folder: str, argv: list[str]) -> None:
    path = _HERE / folder / "run.py"
    spec = importlib.util.spec_from_file_location(f"axis_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.argv
    sys.argv = [str(path), *argv]
    try:
        mod.main()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+", default=DEFAULT_AXES)
    ap.add_argument("--bases", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--epochs", type=int, default=None,
                    help="epoch override; None = each dataset's meta default")
    # Default testbed (revised): the two real anchors, low-dim tabular (covertype)
    # and high-dim image (cifar10). The synthetic dim-sweep and scalability remain
    # separate scripts. Cap covertype with --train_subset to stay tractable.
    ap.add_argument("--datasets", nargs="+", default=["covertype", "cifar10"])
    ap.add_argument("--train_subset", type=int, default=None,
                    help="cap train size per cell (e.g. 50000 to keep covertype tractable)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for num in args.axes:
        if num not in AXES:
            print(f"!! unknown axis {num}"); continue
        folder = AXES[num]
        argv = ["--datasets", *args.datasets, "--seeds", *map(str, args.seeds)]
        if args.smoke:
            argv = ["--smoke"]
        else:
            if args.epochs is not None:
                argv += ["--epochs", str(args.epochs)]
            if args.bases is not None:
                argv += ["--bases", *args.bases]
            if args.train_subset is not None:
                argv += ["--train_subset", str(args.train_subset)]
        print(f"\n{'='*70}\n[run_all] AXIS {num} -> {folder}  argv={argv}\n{'='*70}", flush=True)
        try:
            _invoke(folder, argv)
        except SystemExit:
            pass
        except Exception as e:
            print(f"!! axis {num} ({folder}) raised {type(e).__name__}: {e}", flush=True)

    if args.aggregate:
        _invoke("08_cross_summary", [])
    print(f"\n[run_all] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
