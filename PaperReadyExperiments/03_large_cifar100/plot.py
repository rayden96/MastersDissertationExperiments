"""Plotter for 03_large_cifar100. Reads latest run, produces figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from interference.image_runner import plot_image_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()
    plot_image_experiment(
        results_root=_HERE / "results",
        run_id=args.run_id,
        experiment_name="03_large_cifar100",
    )


if __name__ == "__main__":
    main()
