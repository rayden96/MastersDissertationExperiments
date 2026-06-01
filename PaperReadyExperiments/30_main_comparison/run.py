"""
30 — Main comparison (bakeoff) core runner.

Tunes (on val) then runs multi-seed paired-order test trainings for every
(dataset x base optimizer x method) cell, producing the canonical record set that
the view axes (30.01 trajectories, 30.02 budget tables, 30.03 final bars, 30.09
Pareto) read. Resumable.

Datasets (the confirmed 6, three modalities):
  mnist, cifar10, cifar100, emnist_balanced, covertype, yahoo_answers

Run:
    python run.py --smoke                                   # 1 ds, 2 methods, 1 seed, tiny
    python run.py --datasets cifar10 --seeds 2026 2027 2028 # one dataset, 3 seeds
    python run.py                                           # full bakeoff (large!)
    python run.py --no-tune                                 # skip val tuning (mid-grid hp)

Budget warning: the full grid is 6 x 4 x 5 x 5 = 600 tuned runs (+ tuning). Run
per-dataset across Colab sessions; everything resumes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _bakeoff import run_bakeoff  # noqa: E402

ALL_DATASETS = ["mnist", "cifar10", "cifar100", "emnist_balanced", "covertype", "yahoo_answers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--bases", nargs="+", default=["sgd", "signsgd", "rmsprop", "adam"])
    ap.add_argument("--methods", nargs="+",
                    default=["baseline", "cosgd", "bograd", "graddrop", "dropout"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--epochs", type=int, default=None, help="override per-dataset default")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--no-measure", action="store_true",
                    help="skip the interference meter (faster; accuracy-only)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.datasets = ["cifar10"]; args.bases = ["sgd"]
        args.methods = ["baseline", "bograd"]; args.seeds = [2026]
        args.epochs = 1

    run_bakeoff(
        datasets=args.datasets, bases=args.bases, methods=args.methods,
        seeds=args.seeds, epochs=(1 if args.smoke else args.epochs),
        tune=(not args.no_tune and not args.smoke),
        measure=(not args.no_measure),
    )


if __name__ == "__main__":
    main()
