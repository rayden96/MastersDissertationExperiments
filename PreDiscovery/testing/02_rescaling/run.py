"""
Test 02 — Rescaling ablation.

If preserve_magnitude=True wins in Test 01, this test characterises the right
form: full rescale, clipped at 2x, clipped at 5x. Run at K=32 and K=128 to
detect whether clipping matters more at large K (where the buffer is more
likely to fully span g and produce blow-up factors).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.optimizers import BoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, aggregate, build_cifar10, delta_marker,
    make_test_loader, make_train_loader, train_run,
)


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    preserve_magnitude: bool
    max_rescale: Optional[float]
    note: str


VARIANTS: List[Variant] = [
    Variant("no_rescale", False, None, "current default"),
    Variant("full_rescale", True, None, "no clipping — risky at large K"),
    Variant("clipped_2x", True, 2.0, "factor capped at 2"),
    Variant("clipped_5x", True, 5.0, "factor capped at 5"),
]


def make_optimizer(model, lr, K, variant: Variant) -> torch.optim.Optimizer:
    return BoGrad(
        model.parameters(), torch.optim.SGD,
        buffer_size=K, project_stage="update",
        projection_mode="negative", orth_method="sequential",
        preserve_magnitude=variant.preserve_magnitude,
        max_rescale=variant.max_rescale,
        lr=lr, momentum=0.9,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--Ks", type=str, default="32,128")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()

    Ks: List[int] = [int(x) for x in args.Ks.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Trials: {args.trials} | Epochs: {args.epochs} | Ks: {Ks} | lr: {args.lr}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "testing" / "02_rescaling" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial_dir = out_dir / "per_trial"
    per_trial_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}\n")

    # results[variant_name][K] = list of trial dicts
    results: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    total = len(VARIANTS) * len(Ks) * args.trials
    run_idx = 0

    for trial in range(args.trials):
        trial_seed = args.base_seed + trial * 1000
        print(f"=== Trial {trial + 1}/{args.trials} (seed={trial_seed}) ===")

        for K in Ks:
            for variant in VARIANTS:
                results.setdefault(variant.name, {})
                train_loader = make_train_loader(
                    train_ds, args.batch_size, args.num_workers,
                    generator_seed=trial_seed,
                    pin_memory=(device.type == "cuda"),
                )
                run_idx += 1
                cfg_label = f"{variant.name} K={K}"

                def opt_factory(model, K=K, v=variant, lr=args.lr):
                    return make_optimizer(model, lr, K, v)

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
                r["K"] = K
                r["variant"] = variant.name
                r["note"] = variant.note
                results[variant.name].setdefault(K, []).append(r)
                with (per_trial_dir / f"{variant.name}__K{K}__t{trial}.json").open("w") as fh:
                    json.dump(r, fh, indent=2)

    summary = {
        "config": vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "Ks": Ks,
        "variants": [v.name for v in VARIANTS],
        "results": {v: {str(K): trials for K, trials in by_K.items()}
                    for v, by_K in results.items()},
    }
    with (out_dir / "results.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- Summary table ----
    print("\n" + "=" * 92)
    print("RESCALING ABLATION — final test accuracy (mean ± std across trials)")
    print("=" * 92)
    print(f"{'variant':18s} {'K':>5s} {'mean':>7s} {'std':>7s} {'best':>7s}  vs. no_rescale")
    print("-" * 92)

    for K in Ks:
        no_rescale_finals = [t["final_test_acc"] for t in results.get("no_rescale", {}).get(K, [])]
        no_rescale_mean = aggregate(no_rescale_finals)["mean"]
        for variant in VARIANTS:
            trials = results[variant.name].get(K, [])
            finals = [t["final_test_acc"] for t in trials]
            bests = [max(t["epoch_test_acc"]) if t["epoch_test_acc"] else float("nan") for t in trials]
            agg = aggregate(finals)
            best_agg = aggregate(bests)
            delta = agg["mean"] - no_rescale_mean if (agg["n"] and no_rescale_mean == no_rescale_mean) else float("nan")
            marker = "" if variant.name == "no_rescale" else delta_marker(delta)
            print(
                f"{variant.name:18s} {K:>5d} "
                f"{agg['mean']:>7.4f} {agg['std']:>7.4f} {best_agg['mean']:>7.4f}  "
                f"{delta:>+7.4f}{marker}"
            )
        print("-" * 92)

    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
