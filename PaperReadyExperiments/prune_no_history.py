"""
Delete completed runs that carry no per-epoch history, so a re-launch redoes them.

A run resumed from a checkpoint accumulates `history.epoch_test_acc` only from
the resume point, so a run resumed at or past its final epoch finishes with an
empty history. Its final scalars are correct, but it contributes no learning
curve, and the ablation figures of Chapters 4 and 5 are curves. Since the
JobManager skips anything with a completed results.json, the only way to get the
curve back is to remove the directory and let the campaign re-run that cell.

Dry run by default — nothing is deleted without --apply.

    python prune_no_history.py 20_cosgd_ablation/20_03_prenormalize_covertype
    python prune_no_history.py 20_cosgd_ablation --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_PRE = Path(__file__).resolve().parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subpath", nargs="?", default="",
                    help="path under the results root, e.g. 20_cosgd_ablation")
    ap.add_argument("--apply", action="store_true", help="actually delete")
    args = ap.parse_args()

    root = storage.get_results_root() / args.subpath if args.subpath \
        else storage.get_results_root()
    if not root.exists():
        raise SystemExit(f"no such results directory: {root}")
    print(f"scanning {root}")

    victims = []
    for rj in sorted(root.rglob("results.json")):
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        if r.get("status") != "completed":
            continue
        h = r.get("history") or {}
        if h.get("epoch_test_acc") or h.get("epoch_train_loss"):
            continue
        victims.append(rj.parent)

    for d in victims:
        print(f"  {'removing' if args.apply else 'would remove'} {d.relative_to(root)}")
        if args.apply:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(victims)} run(s) without a curve"
          f"{' removed' if args.apply else '; re-run with --apply to delete'}")


if __name__ == "__main__":
    main()
