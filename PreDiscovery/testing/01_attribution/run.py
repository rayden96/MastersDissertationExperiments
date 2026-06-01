"""
Test 01 — Attribution: direction or magnitude?

See README.md for full motivation. In short: compare baseline,
BoGrad (default, magnitude-reducing), BoGrad with magnitude preserved,
and BoGrad with random-projection control, across a small LR sweep,
to decompose BoGrad's effect.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.optimizers import BoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, aggregate, build_cifar10, delta_marker,
    make_test_loader, make_train_loader, train_run,
)


# ---------------------------------------------------------------------------
# Variant factories — return optimizer when given a model
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    factory: Callable[[Any, float], torch.optim.Optimizer]
    note: str


def baseline_factory(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def bograd_factory(model, lr):
    return BoGrad(
        model.parameters(), torch.optim.SGD,
        buffer_size=32, project_stage="update",
        projection_mode="negative", orth_method="sequential",
        lr=lr, momentum=0.9,
    )


def bograd_rescale_factory(model, lr):
    return BoGrad(
        model.parameters(), torch.optim.SGD,
        buffer_size=32, project_stage="update",
        projection_mode="negative", orth_method="sequential",
        preserve_magnitude=True, max_rescale=5.0,
        lr=lr, momentum=0.9,
    )


def bograd_random_factory(model, lr):
    return BoGrad(
        model.parameters(), torch.optim.SGD,
        buffer_size=32, project_stage="update",
        projection_mode="negative", orth_method="sequential",
        random_projection=True,
        lr=lr, momentum=0.9,
    )


VARIANTS: List[Variant] = [
    Variant("baseline", baseline_factory, "SGD+momentum, no projection"),
    Variant("bograd", bograd_factory, "BoGrad K=32 update-neg (current default)"),
    Variant("bograd_rescale", bograd_rescale_factory, "BoGrad + preserve_magnitude (max_rescale=5)"),
    Variant("bograd_random", bograd_random_factory, "BoGrad with random-projection control"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--lrs", type=str, default="0.025,0.05,0.1")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()

    lrs: List[float] = [float(x) for x in args.lrs.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Trials: {args.trials} | Epochs: {args.epochs} | LRs: {lrs}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "testing" / "01_attribution" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial_dir = out_dir / "per_trial"
    per_trial_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}\n")

    # results[variant][lr] = list of trial dicts
    results: Dict[str, Dict[float, List[Dict[str, Any]]]] = {}

    total = len(VARIANTS) * len(lrs) * args.trials
    run_idx = 0

    for trial in range(args.trials):
        trial_seed = args.base_seed + trial * 1000
        print(f"=== Trial {trial + 1}/{args.trials} (seed={trial_seed}) ===")

        for variant in VARIANTS:
            results.setdefault(variant.name, {})
            for lr in lrs:
                train_loader = make_train_loader(
                    train_ds, args.batch_size, args.num_workers,
                    generator_seed=trial_seed,
                    pin_memory=(device.type == "cuda"),
                )
                run_idx += 1
                cfg_label = f"{variant.name} lr={lr}"

                def opt_factory(model, lr=lr, fn=variant.factory):
                    return fn(model, lr)

                try:
                    r = train_run(
                        f"[{run_idx}/{total}] {cfg_label}",
                        SmallCNN, opt_factory,
                        train_loader, test_loader, device,
                        epochs=args.epochs, seed=trial_seed,
                    )
                except Exception as exc:
                    print(f"\n    !! {cfg_label} failed: {exc}")
                    r = {
                        "final_test_acc": float("nan"),
                        "best_test_acc": float("nan"),
                        "epoch_test_acc": [],
                        "wall_clock_s": 0.0,
                        "error": str(exc),
                    }
                r["trial"] = trial
                r["lr"] = lr
                r["variant"] = variant.name
                r["note"] = variant.note
                results[variant.name].setdefault(lr, []).append(r)
                with (per_trial_dir / f"{variant.name}__lr{lr}__t{trial}.json").open("w") as fh:
                    json.dump(r, fh, indent=2)

    # Save summary
    summary = {
        "config": vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "lrs": lrs,
        "variants": [v.name for v in VARIANTS],
        "results": {v: {str(lr): trials for lr, trials in by_lr.items()}
                    for v, by_lr in results.items()},
    }
    with (out_dir / "results.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- Summary table ----
    print("\n" + "=" * 92)
    print("ATTRIBUTION RESULTS — final test accuracy (mean ± std across trials)")
    print("=" * 92)
    print(f"{'variant':18s} {'lr':>6s} {'mean':>7s} {'std':>7s} {'best':>7s}  vs. baseline")
    print("-" * 92)

    # Compute baseline mean per LR for delta computation
    baseline_means: Dict[float, float] = {}
    for lr in lrs:
        finals = [t["final_test_acc"] for t in results.get("baseline", {}).get(lr, [])]
        baseline_means[lr] = aggregate(finals)["mean"]

    for variant in VARIANTS:
        for lr in lrs:
            trials = results[variant.name].get(lr, [])
            finals = [t["final_test_acc"] for t in trials]
            bests = [max(t["epoch_test_acc"]) if t["epoch_test_acc"] else float("nan") for t in trials]
            agg = aggregate(finals)
            best_agg = aggregate(bests)
            base_mean = baseline_means.get(lr, float("nan"))
            delta = agg["mean"] - base_mean if (agg["n"] and base_mean == base_mean) else float("nan")
            marker = "" if variant.name == "baseline" else delta_marker(delta)
            print(
                f"{variant.name:18s} {lr:>6.3f} "
                f"{agg['mean']:>7.4f} {agg['std']:>7.4f} {best_agg['mean']:>7.4f}  "
                f"{delta:>+7.4f}{marker}"
            )
        print("-" * 92)

    print(f"\nDone. Results at {out_dir}")
    print("Read README.md decision criteria with this table to attribute the effect.")


if __name__ == "__main__":
    main()
