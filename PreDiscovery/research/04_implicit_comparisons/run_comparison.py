"""
Implicit-method comparison — extends Stage 1 protocol with implicit
interference reducers.

Methods compared:
  - sgd_vanilla       : SGD, no momentum
  - sgd_momentum      : SGD + momentum (Stage 1 baseline)
  - sgd_mom_dropout01 : SGD + momentum + Dropout(0.1)
  - sgd_mom_dropout03 : SGD + momentum + Dropout(0.3)
  - sgd_mom_clip1     : SGD + momentum + grad clip 1.0
  - sgd_mom_clip5     : SGD + momentum + grad clip 5.0
  - adam              : Adam (lr=1e-3)
  - adam_lr05         : Adam (lr=0.05)
  - bograd            : SGD + momentum + BoGrad upd-K=32 neg (reference)
  - cosgd             : COSGD modified-GS single-fwd (reference)

Captured metrics (same as Stage 1):
  - inter-batch (IB_%neg, IB_⟨cos⟩) via per-class probe gradients
  - between-batch (OOB total, OOB per-step, WW_K) via InterferenceTracker
  - test accuracy per epoch

Output: per-method JSON history + summary table.

Usage
-----
    python research/04_implicit_comparisons/run_comparison.py
    python research/04_implicit_comparisons/run_comparison.py --quick
    python research/04_implicit_comparisons/run_comparison.py --epochs 2
    python research/04_implicit_comparisons/run_comparison.py \
        --only sgd_vanilla,adam,bograd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure stdout can handle unicode on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.diagnostics import ClassProbeSet, InterferenceTracker  # noqa: E402
from common.optimizers import BoGrad, COSGD  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, build_cifar10, evaluate, make_test_loader, make_train_loader,
)
from stage1.inter_batch_metric import measure_inter_batch_interference  # noqa: E402


# ---------------------------------------------------------------------------
# SmallCNN with optional dropout (separate from testing._common.SmallCNN
# so the baseline stays untouched).
# ---------------------------------------------------------------------------
class SmallCNNDropout(nn.Module):
    """Same 3-conv backbone as SmallCNN, with dropout on activations.

    Dropout sits after each ReLU. Probability is configurable; 0 disables
    (matches plain SmallCNN exactly).
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)

        cls_layers: List[nn.Module] = [nn.Flatten()]
        if dropout > 0:
            cls_layers.append(nn.Dropout(dropout))
        cls_layers.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*cls_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    label: str
    kind: str   # "standard" | "cosgd" | "clip"
    model_factory: Callable[[], nn.Module]
    optimizer_factory: Callable[[nn.Module, nn.Module], torch.optim.Optimizer]
    grad_clip: Optional[float] = None  # only used when kind == "clip"


def build_variants(
    *, lr: float, momentum: float, bograd_K: int,
    adam_lr_default: float, adam_lr_matched: float,
    dropout_low: float, dropout_high: float,
    clip_tight: float, clip_loose: float,
) -> List[Variant]:
    """All ten variants. Order matters for the printed table."""
    return [
        Variant(
            "sgd_vanilla", "SGD vanilla (no momentum)",
            kind="standard",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: torch.optim.SGD(m.parameters(), lr=lr),
        ),
        Variant(
            "sgd_momentum", "SGD + momentum (baseline)",
            kind="standard",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum),
        ),
        Variant(
            "sgd_mom_dropout01", f"SGD+mom + Dropout({dropout_low})",
            kind="standard",
            model_factory=lambda d=dropout_low: SmallCNNDropout(dropout=d),
            optimizer_factory=lambda m, c: torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum),
        ),
        Variant(
            "sgd_mom_dropout03", f"SGD+mom + Dropout({dropout_high})",
            kind="standard",
            model_factory=lambda d=dropout_high: SmallCNNDropout(dropout=d),
            optimizer_factory=lambda m, c: torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum),
        ),
        Variant(
            "sgd_mom_clip1", f"SGD+mom + clip {clip_tight}",
            kind="clip",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum),
            grad_clip=clip_tight,
        ),
        Variant(
            "sgd_mom_clip5", f"SGD+mom + clip {clip_loose}",
            kind="clip",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum),
            grad_clip=clip_loose,
        ),
        Variant(
            "adam", f"Adam (lr={adam_lr_default})",
            kind="standard",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: torch.optim.Adam(m.parameters(), lr=adam_lr_default),
        ),
        Variant(
            "adam_lr05", f"Adam (lr={adam_lr_matched})",
            kind="standard",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: torch.optim.Adam(m.parameters(), lr=adam_lr_matched),
        ),
        Variant(
            "bograd", f"SGD+mom + BoGrad upd-K={bograd_K} neg",
            kind="standard",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=bograd_K, project_stage="update",
                projection_mode="negative", orth_method="sequential",
                lr=lr, momentum=momentum,
            ),
        ),
        Variant(
            "cosgd", "COSGD (modified GS, single_forward)",
            kind="cosgd",
            model_factory=lambda: SmallCNN(),
            optimizer_factory=lambda m, c: COSGD(
                m.parameters(), lr=lr,
                model=m, criterion=c,
                orthogonalization_method="modified_gs_normal",
                step_method="single_forward",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Per-variant training run
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    variant: str
    label: str
    final_acc: float
    epoch_acc: List[float]
    history: List[Dict[str, Any]]
    inter_batch_history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    forgetting: Dict[str, Any]
    wasted_work: Dict[str, Any]
    wall_clock_s: float


def run_variant(
    variant: Variant,
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe_set: ClassProbeSet,
    device: torch.device,
    *,
    epochs: int,
    log_every: int,
    probe_every: int,
    inter_batch_every: int,
    pairwise_K: int,
    seed: int,
) -> RunResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = variant.model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = variant.optimizer_factory(model, criterion)

    tracker = InterferenceTracker(
        model, criterion,
        probe_set=probe_set,
        log_every=log_every,
        probe_every=probe_every,
        wasted_work_K=32,
        device=device,
        trajectory_lags=[1, 4, 16],
        pairwise_K=pairwise_K,
    )

    inter_batch_history: List[Dict[str, Any]] = []

    print(f"\n=== {variant.label} ===")
    t0 = time.time()
    epoch_acc: List[float] = []
    step_count = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Inter-batch interference measurement (snapshot — won't
            # affect training because we zero grads after).
            if step_count % inter_batch_every == 0:
                ib_stats = measure_inter_batch_interference(model, criterion, probe_set)
                ib_stats["step"] = step_count
                inter_batch_history.append(ib_stats)

            # Training step
            if variant.kind == "standard":
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                tracker.before_step()
                optimizer.step()
                batch_classes = set(y.unique().cpu().tolist())
                tracker.after_step(loss=loss.item(), batch_classes=batch_classes)

            elif variant.kind == "clip":
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                # Apply the clip BEFORE before_step so the snapshot
                # reflects the (post-clip) gradient that's actually used.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=variant.grad_clip,
                )
                tracker.before_step()
                optimizer.step()
                batch_classes = set(y.unique().cpu().tolist())
                tracker.after_step(loss=loss.item(), batch_classes=batch_classes)

            elif variant.kind == "cosgd":
                # COSGD does its own per-class fwd+bwd inside step().
                # Reconstruct what the tracker needs for displacement-based
                # metrics from a parameter snapshot.
                params_before = [p.detach().clone() for p in model.parameters()
                                 if p.requires_grad]
                unique_labels = y.unique()
                loss_val = optimizer.step(x, y, unique_labels)
                tracker._params_before_step = torch.cat(
                    [p.reshape(-1) for p in params_before]
                )
                tracker.after_step(
                    loss=float(loss_val) if loss_val is not None else float("nan"),
                    batch_classes=set(y.unique().cpu().tolist()),
                )
                del params_before

            else:
                raise ValueError(f"Unknown variant.kind {variant.kind}")

            step_count += 1

        # Eval per epoch
        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{epochs}  test_acc={acc:.4f}  "
              f"steps={step_count}  ({elapsed:.0f}s)")

    return RunResult(
        variant=variant.name,
        label=variant.label,
        final_acc=epoch_acc[-1] if epoch_acc else float("nan"),
        epoch_acc=epoch_acc,
        history=tracker.get_history(),
        inter_batch_history=inter_batch_history,
        summary=tracker.summary(),
        forgetting=tracker.forgetting.summary(),
        wasted_work=tracker.wasted_work.summary(),
        wall_clock_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bograd-K", type=int, default=32)
    ap.add_argument("--adam-lr-default", type=float, default=1e-3)
    ap.add_argument("--adam-lr-matched", type=float, default=0.05)
    ap.add_argument("--dropout-low", type=float, default=0.1)
    ap.add_argument("--dropout-high", type=float, default=0.3)
    ap.add_argument("--clip-tight", type=float, default=1.0)
    ap.add_argument("--clip-loose", type=float, default=5.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--probe-n-per-class", type=int, default=64)
    ap.add_argument("--inter-batch-every", type=int, default=25)
    ap.add_argument("--pairwise-K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated subset of variant names")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"epochs={args.epochs}  lr={args.lr}  mu={args.momentum}  "
          f"batch={args.batch_size}  log_every={args.log_every}  "
          f"inter_batch_every={args.inter_batch_every}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(
        test_ds, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Building probe set...")
    probe = ClassProbeSet(
        test_ds, num_classes=10,
        n_per_class=args.probe_n_per_class,
        seed=args.seed, device=device,
    )

    variants = build_variants(
        lr=args.lr, momentum=args.momentum, bograd_K=args.bograd_K,
        adam_lr_default=args.adam_lr_default,
        adam_lr_matched=args.adam_lr_matched,
        dropout_low=args.dropout_low, dropout_high=args.dropout_high,
        clip_tight=args.clip_tight, clip_loose=args.clip_loose,
    )
    if args.only:
        wanted = {v.strip() for v in args.only.split(",") if v.strip()}
        variants = [v for v in variants if v.name in wanted]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "research" / "04_implicit_comparisons" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    with (out_dir / "config.json").open("w") as fh:
        json.dump(
            {"args": vars(args), "device": str(device), "run_id": run_id},
            fh, indent=2,
        )

    results: Dict[str, RunResult] = {}
    for variant in variants:
        train_loader = make_train_loader(
            train_ds, args.batch_size, args.num_workers,
            generator_seed=args.seed,
            pin_memory=(device.type == "cuda"),
        )
        try:
            r = run_variant(
                variant, train_loader, test_loader, probe, device,
                epochs=args.epochs, log_every=args.log_every,
                probe_every=args.probe_every,
                inter_batch_every=args.inter_batch_every,
                pairwise_K=args.pairwise_K,
                seed=args.seed,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"\n  !! {variant.name} failed: {exc}")
            continue

        results[variant.name] = r

        sub = out_dir / variant.name
        sub.mkdir(exist_ok=True)
        with (sub / "history.json").open("w") as fh:
            json.dump(r.history, fh, indent=2)
        with (sub / "inter_batch_history.json").open("w") as fh:
            json.dump(r.inter_batch_history, fh, indent=2)
        with (sub / "summary.json").open("w") as fh:
            json.dump(r.summary, fh, indent=2, default=str)
        with (sub / "result.json").open("w") as fh:
            json.dump({
                "variant": r.variant,
                "label": r.label,
                "final_acc": r.final_acc,
                "epoch_acc": r.epoch_acc,
                "wall_clock_s": r.wall_clock_s,
                "forgetting": r.forgetting,
                "wasted_work": r.wasted_work,
            }, fh, indent=2, default=str)

    # ---------- Console summary ----------
    print("\n" + "=" * 130)
    print("IMPLICIT-METHOD COMPARISON — interference axes")
    print("=" * 130)
    print(f"{'method':40s} {'final_acc':>10s} {'IB_%neg':>9s} {'IB_<cos>':>10s} "
          f"{'OOB_total':>10s} {'OOB_per_step':>13s} {'WW_K':>7s}")
    print("-" * 130)

    for vname, r in results.items():
        ib_records = [rec for rec in r.inter_batch_history if rec.get("n_pairs", 0) > 0]
        if ib_records:
            ib_frac_neg = sum(rec["frac_negative"] for rec in ib_records) / len(ib_records)
            ib_mean_cos = sum(rec["mean_cos"] for rec in ib_records) / len(ib_records)
        else:
            ib_frac_neg = float("nan")
            ib_mean_cos = float("nan")

        oob_total = r.forgetting.get("out_of_batch_total_magnitude", float("nan"))
        oob_per_step = r.summary.get("forgetting_oob_per_unit_step", float("nan"))
        ww = r.summary.get("ww_wasted_work_ratio_mean", float("nan"))

        print(f"{r.label:40s} {r.final_acc:>10.4f} {ib_frac_neg:>9.3f} {ib_mean_cos:>+10.3f} "
              f"{oob_total:>10.3f} {oob_per_step:>13.4f} {ww:>7.3f}")

    print("\nReading guide:")
    print("  IB_%neg     : inter-batch class-pair conflict (lower = COSGD-like reduction)")
    print("  IB_<cos>    : mean class-pair cosine within batch (positive = aligned)")
    print("  OOB_total   : between-batch out-of-batch forgetting magnitude (BoGrad target;")
    print("                structurally 0 in standard CIFAR-10)")
    print("  OOB_per_step: OOB forgetting per unit parameter movement")
    print("  WW_K        : wasted-work ratio over K=32 steps (1 = perfect efficiency)")
    print()
    print(f"Done. Results at {out_dir}")


if __name__ == "__main__":
    main()
