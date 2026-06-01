"""
Test 05 — 2×2 grid: momentum × BoGrad per optimiser family.

The cell configs are populated based on the best (stage, K, mode) discovered
in earlier sweeps. Each cell runs an internal LR sweep (3 points by default);
the cell's reported metric is the best mean across LR.

For Adam, "no momentum" means betas=(0.0, 0.999) — disables the first-moment
EMA while keeping the second-moment preconditioner.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.optimizers import BoGrad  # noqa: E402
from discoveryPhase2.enhanced_variants import SignSGD  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, aggregate, build_cifar10, delta_marker,
    make_test_loader, make_train_loader, train_run,
)


# ---------------------------------------------------------------------------
# Cell configurations
# ---------------------------------------------------------------------------
# Each cell is (factory, stage_label, default_K)
# stage labels are informational; the factory produces the actual optimizer.

@dataclass
class CellSpec:
    factory: Callable[[Any, float], torch.optim.Optimizer]
    note: str


@dataclass
class FamilySpec:
    name: str
    lr_default: float
    cells: Dict[str, CellSpec]   # keys: "no_mom_no_bg", "mom_no_bg", "no_mom_bg", "mom_bg"


# Constructors per family

def sgd_no_mom_no_bg(m, lr):  return torch.optim.SGD(m.parameters(), lr=lr)
def sgd_mom_no_bg(m, lr):     return torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
def sgd_no_mom_bg(m, lr):
    return BoGrad(m.parameters(), torch.optim.SGD,
                  buffer_size=8, project_stage="gradient",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr)
def sgd_mom_bg(m, lr):
    return BoGrad(m.parameters(), torch.optim.SGD,
                  buffer_size=32, project_stage="update",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr, momentum=0.9)


def rms_no_mom_no_bg(m, lr): return torch.optim.RMSprop(m.parameters(), lr=lr, alpha=0.99)
def rms_mom_no_bg(m, lr):    return torch.optim.RMSprop(m.parameters(), lr=lr, alpha=0.99, momentum=0.9)
def rms_no_mom_bg(m, lr):
    return BoGrad(m.parameters(), torch.optim.RMSprop,
                  buffer_size=8, project_stage="update",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr, alpha=0.99)
def rms_mom_bg(m, lr):
    return BoGrad(m.parameters(), torch.optim.RMSprop,
                  buffer_size=16, project_stage="update",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr, alpha=0.99, momentum=0.9)


def adam_no_mom_no_bg(m, lr): return torch.optim.Adam(m.parameters(), lr=lr, betas=(0.0, 0.999))
def adam_mom_no_bg(m, lr):    return torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.999))
def adam_no_mom_bg(m, lr):
    return BoGrad(m.parameters(), torch.optim.Adam,
                  buffer_size=8, project_stage="update",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr, betas=(0.0, 0.999))
def adam_mom_bg(m, lr):
    return BoGrad(m.parameters(), torch.optim.Adam,
                  buffer_size=128, project_stage="update",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr, betas=(0.9, 0.999))


def sign_no_mom_no_bg(m, lr): return SignSGD(m.parameters(), lr=lr, momentum=0.0)
def sign_mom_no_bg(m, lr):    return SignSGD(m.parameters(), lr=lr, momentum=0.9)
def sign_no_mom_bg(m, lr):
    return BoGrad(m.parameters(), SignSGD,
                  buffer_size=8, project_stage="gradient",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr, momentum=0.0)
def sign_mom_bg(m, lr):
    return BoGrad(m.parameters(), SignSGD,
                  buffer_size=64, project_stage="update",
                  projection_mode="negative", orth_method="sequential",
                  lr=lr, momentum=0.9)


def build_families() -> Dict[str, FamilySpec]:
    return {
        "sgd": FamilySpec(
            name="sgd", lr_default=0.05,
            cells={
                "no_mom_no_bg": CellSpec(sgd_no_mom_no_bg, "vanilla SGD"),
                "mom_no_bg":    CellSpec(sgd_mom_no_bg, "SGD+momentum"),
                "no_mom_bg":    CellSpec(sgd_no_mom_bg, "BoGrad grad-neg K=8"),
                "mom_bg":       CellSpec(sgd_mom_bg, "BoGrad update-neg K=32 + momentum"),
            },
        ),
        "rmsprop": FamilySpec(
            name="rmsprop", lr_default=1e-3,
            cells={
                "no_mom_no_bg": CellSpec(rms_no_mom_no_bg, "RMSprop"),
                "mom_no_bg":    CellSpec(rms_mom_no_bg, "RMSprop+momentum"),
                "no_mom_bg":    CellSpec(rms_no_mom_bg, "BoGrad update-neg K=8"),
                "mom_bg":       CellSpec(rms_mom_bg, "BoGrad update-neg K=16 + momentum"),
            },
        ),
        "adam": FamilySpec(
            name="adam", lr_default=1e-3,
            cells={
                "no_mom_no_bg": CellSpec(adam_no_mom_no_bg, "Adam β₁=0"),
                "mom_no_bg":    CellSpec(adam_mom_no_bg, "Adam β₁=0.9"),
                "no_mom_bg":    CellSpec(adam_no_mom_bg, "BoGrad update-neg K=8 (β₁=0)"),
                "mom_bg":       CellSpec(adam_mom_bg, "BoGrad update-neg K=128 (β₁=0.9)"),
            },
        ),
        "signsgd": FamilySpec(
            name="signsgd", lr_default=1e-3,
            cells={
                "no_mom_no_bg": CellSpec(sign_no_mom_no_bg, "SignSGD"),
                "mom_no_bg":    CellSpec(sign_mom_no_bg, "SignSGD+momentum"),
                "no_mom_bg":    CellSpec(sign_no_mom_bg, "BoGrad grad-neg K=8"),
                "mom_bg":       CellSpec(sign_mom_bg, "BoGrad update-neg K=64 + momentum"),
            },
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--lrs", type=str, default=None,
                    help="Comma-separated LRs; defaults to {0.5×, 1×, 2×} of family's lr_default")
    ap.add_argument("--no-lr-sweep", action="store_true",
                    help="Run only the family's default LR (no sweep within cells)")
    ap.add_argument("--families", type=str, default="sgd,rmsprop,adam,signsgd")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()

    family_names = [s.strip() for s in args.families.split(",") if s.strip()]
    families = build_families()
    selected = [families[n] for n in family_names if n in families]
    if not selected:
        print(f"No valid families. Available: {list(families.keys())}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "testing" / "05_2x2_grid" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial_dir = out_dir / "per_trial"
    per_trial_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}\n")

    cell_keys = ["no_mom_no_bg", "mom_no_bg", "no_mom_bg", "mom_bg"]

    # Determine LRs per family
    def lrs_for(family: FamilySpec) -> List[float]:
        if args.lrs:
            return [float(x) for x in args.lrs.split(",") if x.strip()]
        if args.no_lr_sweep:
            return [family.lr_default]
        return [family.lr_default * 0.5, family.lr_default, family.lr_default * 2.0]

    # results[family_name][cell_key][lr] = list of trial dicts
    results: Dict[str, Dict[str, Dict[float, List[Dict[str, Any]]]]] = {}

    total = sum(len(cell_keys) * len(lrs_for(fam)) for fam in selected) * args.trials
    run_idx = 0

    for trial in range(args.trials):
        seed = args.base_seed + trial * 1000
        print(f"=== Trial {trial + 1}/{args.trials} (seed={seed}) ===")

        for fam in selected:
            results.setdefault(fam.name, {})
            lrs = lrs_for(fam)
            print(f"  -- {fam.name}  lrs={lrs} --")
            for cell_key in cell_keys:
                results[fam.name].setdefault(cell_key, {})
                cell = fam.cells[cell_key]
                for lr in lrs:
                    train_loader = make_train_loader(
                        train_ds, args.batch_size, args.num_workers,
                        seed, (device.type == "cuda"),
                    )
                    run_idx += 1
                    label = f"[{run_idx}/{total}] {fam.name}.{cell_key} lr={lr}"

                    def opt_factory(model, lr=lr, fac=cell.factory):
                        return fac(model, lr)

                    try:
                        r = train_run(
                            label, SmallCNN, opt_factory,
                            train_loader, test_loader, device,
                            epochs=args.epochs, seed=seed,
                        )
                    except Exception as exc:
                        print(f"\n    !! {label} failed: {exc}")
                        r = {
                            "final_test_acc": float("nan"),
                            "best_test_acc": float("nan"),
                            "epoch_test_acc": [],
                            "wall_clock_s": 0.0,
                            "error": str(exc),
                        }
                    r["trial"] = trial
                    r["family"] = fam.name
                    r["cell"] = cell_key
                    r["lr"] = lr
                    r["note"] = cell.note
                    results[fam.name][cell_key].setdefault(lr, []).append(r)
                    with (per_trial_dir / f"{fam.name}__{cell_key}__lr{lr}__t{trial}.json").open("w") as fh:
                        json.dump(r, fh, indent=2)

    # Save full results
    summary = {
        "config": vars(args),
        "device": str(device),
        "results": {
            fam: {
                cell: {str(lr): trials for lr, trials in by_lr.items()}
                for cell, by_lr in cells.items()
            }
            for fam, cells in results.items()
        },
    }
    with (out_dir / "results.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- 2×2 tables per family ----
    print("\n" + "=" * 92)
    print("2×2 TABLES — best LR per cell, mean ± std across trials")
    print("=" * 92)

    def best_cell(family_name: str, cell_key: str) -> Dict[str, float]:
        by_lr = results.get(family_name, {}).get(cell_key, {})
        best = {"lr": float("nan"), "mean": float("nan"), "std": float("nan"), "n": 0}
        for lr, trials in by_lr.items():
            finals = [t["final_test_acc"] for t in trials]
            agg = aggregate(finals)
            if agg["n"] and (best["n"] == 0 or agg["mean"] > best["mean"]):
                best = {"lr": lr, "mean": agg["mean"], "std": agg["std"], "n": agg["n"]}
        return best

    for fam in selected:
        a = best_cell(fam.name, "no_mom_no_bg")
        b = best_cell(fam.name, "mom_no_bg")
        c = best_cell(fam.name, "no_mom_bg")
        d = best_cell(fam.name, "mom_bg")

        print(f"\n--- {fam.name} ---")
        print(f"{'':14s} {'no momentum':>22s}   {'momentum':>22s}   Δ(momentum)")
        print("-" * 92)

        def cell_str(c: Dict[str, float]) -> str:
            return f"{c['mean']:.4f}±{c['std']:.4f} (lr={c['lr']:g})"

        delta_b_a = b["mean"] - a["mean"] if a["n"] and b["n"] else float("nan")
        delta_d_c = d["mean"] - c["mean"] if c["n"] and d["n"] else float("nan")
        delta_c_a = c["mean"] - a["mean"] if a["n"] and c["n"] else float("nan")
        delta_d_b = d["mean"] - b["mean"] if b["n"] and d["n"] else float("nan")

        print(f"{'no BoGrad':14s} {cell_str(a):>22s}   {cell_str(b):>22s}   {delta_b_a:>+7.4f}")
        print(f"{'BoGrad':14s} {cell_str(c):>22s}   {cell_str(d):>22s}   {delta_d_c:>+7.4f}")
        print(f"{'Δ(BoGrad)':14s} {delta_c_a:>+22.4f}   {delta_d_b:>+22.4f}")

    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
