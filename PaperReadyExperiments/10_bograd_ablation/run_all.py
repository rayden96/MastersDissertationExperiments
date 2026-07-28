"""
Drive all 9 BoGrad-ablation axes from one entry point (Colab or local).

This is the logic behind the Colab launcher notebook (colab_launcher.ipynb).
It imports each axis's run.py main() and invokes it with a shared budget, so the
whole M1 chapter runs (and resumes) from a single command. The per-cell
JobManager means a re-run skips finished cells — safe to re-launch after a Colab
disconnect.

Usage
-----
    python run_all.py                          # all axes, default proxy budget
    python run_all.py --axes 01 03 07          # subset of axes (by number)
    python run_all.py --bases sgd adam --epochs 10 --seeds 2026 2027 2028
    python run_all.py --smoke                  # tiny end-to-end shake-out

Each axis writes under <axis>/results/...; aggregate with 09_cross_summary.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# axis-number -> (folder, supports_bases_flag)
AXES = {
    "01": "01_buffer_K",
    "02": "02_lr_retune",
    "03": "03_projection_mode",
    "04": "04_orth_method",
    "05": "05_projection_scope",
    "06": "06_magnitude",
    "07": "07_momentum_2x2",
    "08": "08_batch_K",
}
# 09 is the aggregator, run last and separately.


def _load_axis_main(folder: str):
    path = _HERE / folder / "run.py"
    spec = importlib.util.spec_from_file_location(f"axis_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _invoke(folder: str, argv: list[str]) -> None:
    """Run an axis's main() with a synthesized argv."""
    mod = _load_axis_main(folder)
    old = sys.argv
    sys.argv = [str(_HERE / folder / "run.py"), *argv]
    try:
        mod.main()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+", default=list(AXES.keys()),
                    help="axis numbers to run, e.g. 01 03 07 (default: all)")
    ap.add_argument("--bases", nargs="+", default=None,
                    help="base optimizers (default: each axis's own default set)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--epochs", type=int, default=None,
                    help="epoch override; None = each dataset's meta default")
    # Default testbed (revised, shared with 20_cosgd_ablation): low-dim tabular
    # (covertype) + high-dim image (cifar10). The CIFAR-100 rung is a transfer
    # confirmation read from the 30_main_comparison bakeoff (10_scale_transfer),
    # not a full ablation grid. Cap covertype with --train_subset.
    ap.add_argument("--datasets", nargs="+", default=["covertype", "cifar10"])
    ap.add_argument("--train_subset", type=int, default=None,
                    help="cap train size per cell (e.g. 50000 to keep covertype tractable)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true",
                    help="after running, build the 10.09 master table")
    args = ap.parse_args()

    t0 = time.time()
    # Dataset outer, axis inner: the cheap dataset (covertype) completes fully
    # first, so partial Colab sessions still land whole per-dataset result sets.
    for dataset in args.datasets:
        for num in args.axes:
            if num not in AXES:
                print(f"!! unknown axis {num}, skipping"); continue
            folder = AXES[num]
            argv = ["--dataset", dataset, "--seeds", *map(str, args.seeds)]
            if args.smoke:
                argv = ["--smoke"]
            else:
                if args.epochs is not None:
                    argv += ["--epochs", str(args.epochs)]
                if args.bases is not None:
                    argv += ["--bases", *args.bases]
                if args.train_subset is not None:
                    argv += ["--train_subset", str(args.train_subset)]
            print(f"\n{'='*70}\n[run_all] {dataset} AXIS {num} -> {folder}  argv={argv}\n{'='*70}", flush=True)
            try:
                _invoke(folder, argv)
            except SystemExit:
                pass  # axis used argparse exit; continue to next
            except Exception as e:
                print(f"!! axis {num} ({folder}) raised {type(e).__name__}: {e}", flush=True)

    if args.aggregate:
        print(f"\n{'='*70}\n[run_all] AGGREGATE -> 09_cross_summary\n{'='*70}", flush=True)
        _invoke("09_cross_summary", [])

    print(f"\n[run_all] all requested axes done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
