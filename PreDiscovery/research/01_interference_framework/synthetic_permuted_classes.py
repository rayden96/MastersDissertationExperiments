"""
Synthetic experiment 3 — Sequential class blocks on CIFAR-10.

Implements the experiment described in framework.md §7.2 (third bullet).
This is a continual-learning-style setup *within* a single dataset:
training is divided into blocks where each block contains only one class
in succession (all class-0 batches → all class-1 batches → ... → all
class-9 batches). The framework metrics, especially per-class
out-of-batch forgetting, should detect dramatic per-class regressions at
each class transition, with optional recovery if the schedule cycles.

Two batch regimes
-----------------
- interleaved : standard random-shuffle (every batch contains all classes,
                approximately).
- blocks      : sequential class blocks. Each class gets `n_batches_per_class`
                consecutive batches. Optionally cycle through the class
                order multiple times.

Variants
--------
1. interleaved + SGD+momentum
2. blocks + SGD+momentum
3. blocks + BoGrad (update-stage K=32 negative)
4. blocks + BoGrad (update-stage K=32 full)        — F9-style mode comparison
5. blocks + BoGrad (update-stage K=32 positive)    — same

Predictions
-----------
- P1: blocks regime produces dramatic per-class forgetting at transitions
  (visible as spikes in OOB forgetting magnitude).
- P2: blocks regime has lower final test accuracy than interleaved.
- P3: BoGrad-negative reduces forgetting at transitions and recovers
  some accuracy.
- P4: BoGrad-full and -positive collapse training (consistent with F9).

This is the most direct test of whether per-class forgetting is *causal*
to training slowdown — in continual-learning regimes within a single
dataset, forgetting is undeniable, and we can watch it happen.

Output
------
research/01_interference_framework/results/synthetic_permuted_classes/run_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# Ensure stdout can handle unicode on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, Subset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.diagnostics import ClassProbeSet, InterferenceTracker  # noqa: E402
from common.optimizers import BoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, build_cifar10, evaluate, make_test_loader,
)


# ============================================================================
# Class-block batch sampler
# ============================================================================
class ClassBlockBatchSampler(Sampler[List[int]]):
    """Yields batches in block-of-class order.

    For class_order = [0, 1, 2] and n_batches_per_block = 50, the sampler
    yields:
      batches  0..49   : drawn only from class 0
      batches 50..99   : drawn only from class 1
      batches 100..149 : drawn only from class 2
    Repeated `n_cycles` times.

    This is a strong continual-learning-style schedule. With
    n_batches_per_block = 1 it reduces to the rapid cycling of
    ClassDisjointBatchSampler in synthetic_class_disjoint.py.
    """

    def __init__(
        self,
        indices_by_class: Dict[int, List[int]],
        class_order: List[int],
        batch_size: int,
        n_batches_per_block: int,
        n_cycles: int = 1,
        seed: int = 2026,
    ):
        self.indices_by_class = {c: list(v) for c, v in indices_by_class.items()}
        self.class_order = list(class_order)
        self.batch_size = int(batch_size)
        self.n_batches_per_block = int(n_batches_per_block)
        self.n_cycles = int(n_cycles)
        self.seed = int(seed)

        rng = torch.Generator().manual_seed(seed)
        self._shuffled: Dict[int, List[int]] = {}
        for c, lst in self.indices_by_class.items():
            order = torch.randperm(len(lst), generator=rng).tolist()
            self._shuffled[c] = [lst[i] for i in order]

    def __iter__(self) -> Iterator[List[int]]:
        for cycle in range(self.n_cycles):
            for class_idx, cls in enumerate(self.class_order):
                indices = self._shuffled[cls]
                cursor = 0
                for batch_in_block in range(self.n_batches_per_block):
                    if cursor + self.batch_size > len(indices):
                        # Re-shuffle and wrap.
                        rng = torch.Generator().manual_seed(
                            self.seed + cycle * 100 + class_idx * 10 + batch_in_block
                        )
                        order = torch.randperm(
                            len(self.indices_by_class[cls]), generator=rng
                        ).tolist()
                        indices = [self.indices_by_class[cls][i] for i in order]
                        self._shuffled[cls] = indices
                        cursor = 0
                    yield indices[cursor : cursor + self.batch_size]
                    cursor += self.batch_size

    def __len__(self) -> int:
        return self.n_cycles * len(self.class_order) * self.n_batches_per_block


def build_indices_by_class(dataset, num_classes: int) -> Dict[int, List[int]]:
    """Walk the dataset once to bucket indices by class. Fast path uses
    dataset.targets when available."""
    out: Dict[int, List[int]] = defaultdict(list)
    if hasattr(dataset, "targets"):
        for idx, y in enumerate(dataset.targets):
            out[int(y)].append(idx)
    elif (
        hasattr(dataset, "dataset")
        and hasattr(dataset.dataset, "targets")
        and hasattr(dataset, "indices")
    ):
        underlying = dataset.dataset.targets
        for sub_idx, orig_idx in enumerate(dataset.indices):
            out[int(underlying[orig_idx])].append(sub_idx)
    else:
        for idx in range(len(dataset)):
            _, y = dataset[idx]
            out[int(y)].append(idx)
    return dict(out)


# ============================================================================
# Run helper
# ============================================================================
@dataclass
class VariantResult:
    name: str
    final_acc: float
    epoch_acc: List[float]
    per_class_loss_trace: List[Dict[int, float]]   # one entry per probe step
    transition_steps: List[int]                    # steps at which class blocks change
    summary: Dict[str, Any]
    forgetting: Dict[str, Any]
    wasted_work: Dict[str, Any]
    history: List[Dict[str, Any]]
    wall_clock_s: float


def run_variant(
    name: str,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe_set: ClassProbeSet,
    device: torch.device,
    log_every: int,
    probe_every: int,
    seed: int,
    transition_steps: List[int],
    pairwise_K: int = 32,
    use_batch_classes: bool = False,
) -> VariantResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SmallCNN().to(device)
    optimizer = optimizer_factory(model)
    criterion = nn.CrossEntropyLoss()

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

    print(f"\n=== {name} ===")
    t0 = time.time()
    per_class_loss_trace: List[Dict[int, float]] = []

    model.train()
    step = 0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        tracker.before_step()
        optimizer.step()
        if use_batch_classes:
            batch_classes = set(y.unique().cpu().tolist())
            tracker.after_step(loss=loss.item(), batch_classes=batch_classes)
        else:
            tracker.after_step(loss=loss.item())

        step += 1

    # Final eval
    final_acc = evaluate(model, test_loader, device)

    # Extract per-class loss trace from tracker history
    for h in tracker.history:
        pcl = h.get("per_class_loss")
        if isinstance(pcl, dict):
            per_class_loss_trace.append({int(k): float(v) for k, v in pcl.items()})

    elapsed = time.time() - t0
    print(f"  final test_acc={final_acc:.4f}  steps={step}  ({elapsed:.0f}s)")

    return VariantResult(
        name=name,
        final_acc=final_acc,
        epoch_acc=[final_acc],
        per_class_loss_trace=per_class_loss_trace,
        transition_steps=transition_steps,
        summary=tracker.summary(),
        forgetting=tracker.forgetting.summary(),
        wasted_work=tracker.wasted_work.summary(),
        history=tracker.history,
        wall_clock_s=elapsed,
    )


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches-per-block", type=int, default=40,
                    help="How many consecutive batches per class block")
    ap.add_argument("--n-cycles", type=int, default=1,
                    help="How many times to cycle through the class order")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--bograd-K", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--probe-n-per-class", type=int, default=64)
    ap.add_argument("--pairwise-K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))
    print("Bucketing class indices...")
    indices_by_class = build_indices_by_class(train_ds, num_classes=10)
    print(f"  Class sizes: {[(c, len(v)) for c, v in sorted(indices_by_class.items())]}")

    print("Building probe set...")
    probe = ClassProbeSet(test_ds, num_classes=10,
                          n_per_class=args.probe_n_per_class,
                          seed=args.seed, device=device)

    # Total batches for blocks regime: 10 classes × n_batches_per_block × n_cycles
    n_classes = 10
    total_batches = n_classes * args.n_batches_per_block * args.n_cycles
    print(f"\nTotal batches per run: {total_batches}")
    print(f"Blocks regime: {n_classes} classes × {args.n_batches_per_block} batches/block × {args.n_cycles} cycles")

    # Class transitions occur at end of each block.
    transition_steps = [
        cycle * n_classes * args.n_batches_per_block + (c + 1) * args.n_batches_per_block
        for cycle in range(args.n_cycles)
        for c in range(n_classes)
    ]

    # Output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "research" / "01_interference_framework" / "results" / "synthetic_permuted_classes" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id,
                   "transition_steps": transition_steps}, fh, indent=2)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def make_interleaved_loader():
        # Use a Subset to limit to total_batches × batch_size examples for fairness
        n_examples = total_batches * args.batch_size
        sample_indices = list(range(min(n_examples, len(train_ds))))
        # If we need more than the dataset has, just iterate through.
        # For simplicity, use the whole dataset with random shuffle.
        g = torch.Generator()
        g.manual_seed(args.seed)
        return DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            generator=g,
        )

    def make_blocks_loader():
        sampler = ClassBlockBatchSampler(
            indices_by_class=indices_by_class,
            class_order=list(range(n_classes)),
            batch_size=args.batch_size,
            n_batches_per_block=args.n_batches_per_block,
            n_cycles=args.n_cycles,
            seed=args.seed,
        )
        return DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------
    def baseline_factory(m):
        return torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum)

    def make_bograd_factory(mode: str):
        return lambda m: BoGrad(
            m.parameters(), torch.optim.SGD,
            buffer_size=args.bograd_K, project_stage="update",
            projection_mode=mode, orth_method="sequential",
            lr=args.lr, momentum=args.momentum,
        )

    runs_to_do = [
        ("interleaved_baseline", "Interleaved + SGD+mom",
         make_interleaved_loader, baseline_factory),
        ("blocks_baseline", "Blocks + SGD+mom",
         make_blocks_loader, baseline_factory),
        ("blocks_bograd_neg", f"Blocks + BoGrad upd-K={args.bograd_K} neg",
         make_blocks_loader, make_bograd_factory("negative")),
        ("blocks_bograd_full", f"Blocks + BoGrad upd-K={args.bograd_K} FULL",
         make_blocks_loader, make_bograd_factory("full")),
        ("blocks_bograd_pos", f"Blocks + BoGrad upd-K={args.bograd_K} POSITIVE",
         make_blocks_loader, make_bograd_factory("positive")),
    ]

    results: Dict[str, VariantResult] = {}
    for key, label, loader_factory, opt_factory in runs_to_do:
        try:
            results[key] = run_variant(
                label, opt_factory, loader_factory(), test_loader, probe, device,
                log_every=args.log_every, probe_every=args.probe_every,
                seed=args.seed, transition_steps=transition_steps,
                pairwise_K=args.pairwise_K, use_batch_classes=True,
            )
        except Exception as exc:
            print(f"\n  !! {key} failed: {exc}")
            continue

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    for key, r in results.items():
        sub = out_dir / key
        sub.mkdir(exist_ok=True)
        with (sub / "history.json").open("w") as fh:
            json.dump(r.history, fh, indent=2)
        with (sub / "summary.json").open("w") as fh:
            json.dump(r.summary, fh, indent=2, default=str)
        with (sub / "result.json").open("w") as fh:
            json.dump({
                "name": r.name,
                "final_acc": r.final_acc,
                "wall_clock_s": r.wall_clock_s,
                "transition_steps": r.transition_steps,
                "forgetting": r.forgetting,
                "wasted_work": r.wasted_work,
            }, fh, indent=2, default=str)
        # The per-class loss trace is the most useful artifact for plotting
        # forgetting accumulation.
        with (sub / "per_class_loss_trace.json").open("w") as fh:
            json.dump(r.per_class_loss_trace, fh, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 130)
    print(f"PERMUTED-CLASSES — {args.n_batches_per_block} batches/block × {args.n_cycles} cycle(s)")
    print("=" * 130)

    header = (f"{'variant':28s} {'acc':>7s} {'Δ_total':>9s} {'Δ_OOB':>8s} "
              f"{'WW_K':>7s} {'%pos_g':>7s} {'%neg_g':>7s} {'%pos_u':>7s} {'%neg_u':>7s}")
    print(header)
    print("-" * 130)

    for key, r in results.items():
        s = r.summary
        forget_total = r.forgetting.get("total_magnitude", float("nan"))
        forget_oob = r.forgetting.get("out_of_batch_total_magnitude", float("nan"))
        ww = s.get("ww_wasted_work_ratio_mean", float("nan"))
        pw = s.get("pairwise_summary", {}) or {}
        gp = pw.get("grad_frac_positive_mean", float("nan"))
        gn = pw.get("grad_frac_negative_mean", float("nan"))
        up = pw.get("update_frac_positive_mean", float("nan"))
        un = pw.get("update_frac_negative_mean", float("nan"))
        print(f"{key:28s} {r.final_acc:>7.4f} {forget_total:>9.3f} {forget_oob:>8.3f} "
              f"{ww:>7.3f} {gp:>7.3f} {gn:>7.3f} {up:>7.3f} {un:>7.3f}")

    print("\nForgetting at class transitions can be inspected via:")
    print(f"  per_class_loss_trace.json — per-class loss at every probe step")
    print(f"  Compare baseline blocks vs BoGrad-neg blocks: at each transition")
    print(f"  step, the just-trained class loss is low, but the next class's loss")
    print(f"  has spiked. BoGrad-neg should reduce the spike magnitude.")

    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
