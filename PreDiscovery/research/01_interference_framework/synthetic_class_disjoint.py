"""
Synthetic experiment 2 — Class-disjoint mini-batches on CIFAR-10.

Implements the experiment described in framework.md §7.2 (second bullet)
and provides the test of Hypothesis (W) that synthetic_quadratic could not:
the injected interference here is *biased* (per-class forgetting genuinely
moves loss in a fixed direction, doesn't average to zero).

Setting
-------
Two batch-sampling regimes on CIFAR-10:
  - interleaved : standard iid random shuffle (every batch has all classes).
  - disjoint    : each batch contains only K_classes = 1 class. Classes
                  cycle through the dataset; consecutive batches contain
                  *different* classes. Maximum per-batch class disjointness.

Hypothesis (per framework.md §5): the disjoint regime should produce
higher per-class forgetting (because consecutive batches train on entirely
different classes, each step damages the others) AND should produce a
slower training speed than the interleaved regime. Orthogonalisation
should partly recover the slowdown.

Four configurations
-------------------
1. interleaved baseline (SGD+momentum)
2. disjoint baseline (SGD+momentum)
3. interleaved + BoGrad (update-stage K=32 negative)
4. disjoint + BoGrad (update-stage K=32 negative)

Predictions
-----------
- P1: disjoint baseline has substantially higher cumulative forgetting
  (Δ_total) than interleaved baseline.
- P2: disjoint baseline has lower final accuracy / slower convergence than
  interleaved baseline.
- P3: BoGrad reduces forgetting magnitude in the disjoint case more than
  in the interleaved case (BoGrad helps where there's actual interference
  to remove).
- P4: BoGrad recovers some accuracy in the disjoint case (>= disjoint
  baseline; possibly approaching interleaved+BoGrad).

If P1 + P2 + P3 + P4 all pass, this experiment validates Hypothesis (W)
in a setting where the framework's earlier validation (synthetic_quadratic)
failed P4 because the injected interference was zero-mean.

Output
------
research/01_interference_framework/results/synthetic_class_disjoint/run_<timestamp>/
    config.json
    <variant>/history.json, summary.json, result.json
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

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, Subset

# Ensure stdout can handle unicode on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.diagnostics import ClassProbeSet, InterferenceTracker  # noqa: E402
from common.optimizers import BoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, build_cifar10, evaluate, make_test_loader,
)


# ============================================================================
# Class-restricted batch sampler
# ============================================================================
class ClassDisjointBatchSampler(Sampler[List[int]]):
    """A batch sampler that restricts each batch to a single class.

    The class for batch i is chosen by cycling through ``class_order``.
    Within a batch, examples of that class are sampled (with replacement
    if the batch_size exceeds the number of available examples for the
    class, but typically with-shuffle no-replacement within an "epoch
    pass" of that class).

    Parameters
    ----------
    indices_by_class : dict[int, list[int]]
        Mapping from class id to list of dataset indices for that class.
    class_order : list[int]
        Order in which classes appear in successive batches.
    batch_size : int
    n_batches : int
        How many batches to yield.
    seed : int
        Sampling seed (controls within-class shuffle).
    """

    def __init__(
        self,
        indices_by_class: Dict[int, List[int]],
        class_order: List[int],
        batch_size: int,
        n_batches: int,
        seed: int = 2026,
    ):
        self.indices_by_class = {c: list(v) for c, v in indices_by_class.items()}
        self.class_order = list(class_order)
        self.batch_size = int(batch_size)
        self.n_batches = int(n_batches)
        self.seed = int(seed)

        # Pre-shuffle each class's indices (will cycle through repeatedly).
        rng = torch.Generator().manual_seed(seed)
        self._shuffled: Dict[int, List[int]] = {}
        for c, lst in self.indices_by_class.items():
            order = torch.randperm(len(lst), generator=rng).tolist()
            self._shuffled[c] = [lst[i] for i in order]
        self._cursors: Dict[int, int] = {c: 0 for c in self.indices_by_class}

    def __iter__(self) -> Iterator[List[int]]:
        for batch_idx in range(self.n_batches):
            cls = self.class_order[batch_idx % len(self.class_order)]
            indices = self._shuffled[cls]
            cursor = self._cursors[cls]
            # Wrap if we exhaust the class.
            if cursor + self.batch_size > len(indices):
                # Shuffle and reset (but with a different seed each pass to vary).
                rng = torch.Generator().manual_seed(self.seed + batch_idx)
                order = torch.randperm(len(self.indices_by_class[cls]), generator=rng).tolist()
                self._shuffled[cls] = [self.indices_by_class[cls][i] for i in order]
                indices = self._shuffled[cls]
                cursor = 0
            batch = indices[cursor : cursor + self.batch_size]
            self._cursors[cls] = cursor + self.batch_size
            yield batch

    def __len__(self) -> int:
        return self.n_batches


def build_indices_by_class(dataset, num_classes: int) -> Dict[int, List[int]]:
    """Walk the dataset once to bucket indices by class.

    Fast path: torchvision's CIFAR-10 exposes labels as ``dataset.targets``;
    we use that directly to avoid decoding each image. For Subset wrappings
    we look one level into the underlying dataset.
    """
    out: Dict[int, List[int]] = defaultdict(list)

    if hasattr(dataset, "targets"):
        targets = dataset.targets
        for idx, y in enumerate(targets):
            out[int(y)].append(idx)
    elif (
        hasattr(dataset, "dataset")
        and hasattr(dataset.dataset, "targets")
        and hasattr(dataset, "indices")
    ):
        # Subset wrapping — labels live on the underlying dataset.
        underlying_targets = dataset.dataset.targets
        for sub_idx, orig_idx in enumerate(dataset.indices):
            out[int(underlying_targets[orig_idx])].append(sub_idx)
    else:
        # Slow fallback: walk the dataset and decode each item.
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
    history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    forgetting: Dict[str, Any]
    wasted_work: Dict[str, Any]
    wall_clock_s: float
    total_steps: int


def run_variant(
    name: str,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe_set: ClassProbeSet,
    device: torch.device,
    epochs: int,
    log_every: int,
    probe_every: int,
    seed: int,
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
    )

    print(f"\n=== {name} ===")
    t0 = time.time()
    epoch_acc: List[float] = []
    step_count = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            tracker.before_step()
            optimizer.step()
            # Capture batch class membership so the tracker can compute
            # in-batch / out-of-batch forgetting (framework.md §8 F6).
            batch_classes = set(y.unique().cpu().tolist())
            tracker.after_step(loss=loss.item(), batch_classes=batch_classes)
            step_count += 1

        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        print(f"  epoch {epoch + 1}/{epochs}  test_acc={acc:.4f}  steps={step_count}  ({time.time() - t0:.0f}s)")

    return VariantResult(
        name=name,
        final_acc=epoch_acc[-1] if epoch_acc else float("nan"),
        epoch_acc=epoch_acc,
        history=tracker.get_history(),
        summary=tracker.summary(),
        forgetting=tracker.forgetting.summary(),
        wasted_work=tracker.wasted_work.summary(),
        wall_clock_s=time.time() - t0,
        total_steps=step_count,
    )


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-batches-per-epoch", type=int, default=391,
                    help="Number of batches per epoch (391 for full CIFAR-10 at bs=128)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--bograd-K", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="1/4 of CIFAR-10 train set for fast iteration")
    ap.add_argument("--log-every", type=int, default=10,
                    help="Step cadence for diagnostic logging (1 = every step, expensive)")
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--probe-n-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")
    print(f"Epochs: {args.epochs}  batches/epoch: {args.n_batches_per_epoch}  "
          f"batch_size: {args.batch_size}  lr: {args.lr}")

    # Data
    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )
    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    print("Bucketing training indices by class...")
    indices_by_class = build_indices_by_class(train_ds, num_classes=10)
    sizes = ", ".join(f"{c}:{len(v)}" for c, v in sorted(indices_by_class.items()))
    print(f"  Class sizes — {sizes}")

    print("Building per-class probe set (held-out from test)...")
    probe = ClassProbeSet(
        test_ds, num_classes=10, n_per_class=args.probe_n_per_class,
        seed=args.seed, device=device,
    )

    # Output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "research" / "01_interference_framework" / "results" / "synthetic_class_disjoint"
    out_dir = out_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    with (out_dir / "config.json").open("w") as fh:
        json.dump({"args": vars(args), "device": str(device), "run_id": run_id}, fh, indent=2)

    # ------------------------------------------------------------------
    # Build train loaders — one for each batch regime
    # ------------------------------------------------------------------
    n_batches_total = args.epochs * args.n_batches_per_epoch

    # Interleaved: normal random shuffle.
    def make_interleaved_loader():
        g = torch.Generator()
        g.manual_seed(args.seed)
        return DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            generator=g,
        )

    # Disjoint: each batch is one class; classes cycle 0,1,2,...,9,0,1,...
    def make_disjoint_loader():
        sampler = ClassDisjointBatchSampler(
            indices_by_class=indices_by_class,
            class_order=list(range(10)),
            batch_size=args.batch_size,
            n_batches=n_batches_total,
            seed=args.seed,
        )
        return DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------
    results: Dict[str, VariantResult] = {}

    def baseline_factory(m):
        return torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum)

    def make_bograd_factory(mode: str):
        def factory(m):
            return BoGrad(
                m.parameters(), torch.optim.SGD,
                buffer_size=args.bograd_K, project_stage="update",
                projection_mode=mode, orth_method="sequential",
                lr=args.lr, momentum=args.momentum,
            )
        return factory

    runs_to_do = [
        ("interleaved_baseline", "Interleaved + SGD+momentum",
         make_interleaved_loader, baseline_factory),
        ("disjoint_baseline", "Class-disjoint + SGD+momentum",
         make_disjoint_loader, baseline_factory),
        ("interleaved_bograd_neg", f"Interleaved + BoGrad update-K={args.bograd_K} negative",
         make_interleaved_loader, make_bograd_factory("negative")),
        ("disjoint_bograd_neg", f"Class-disjoint + BoGrad update-K={args.bograd_K} negative",
         make_disjoint_loader, make_bograd_factory("negative")),
        # Mode comparison on the disjoint regime (where the interference is real).
        ("disjoint_bograd_full", f"Class-disjoint + BoGrad update-K={args.bograd_K} FULL",
         make_disjoint_loader, make_bograd_factory("full")),
        ("disjoint_bograd_pos", f"Class-disjoint + BoGrad update-K={args.bograd_K} POSITIVE",
         make_disjoint_loader, make_bograd_factory("positive")),
    ]

    for key, display, loader_factory, opt_factory in runs_to_do:
        results[key] = run_variant(
            display, opt_factory, loader_factory(), test_loader, probe, device,
            epochs=args.epochs, log_every=args.log_every, probe_every=args.probe_every,
            seed=args.seed,
        )

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
                "epoch_acc": r.epoch_acc,
                "wall_clock_s": r.wall_clock_s,
                "total_steps": r.total_steps,
                "forgetting": r.forgetting,
                "wasted_work": r.wasted_work,
            }, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Console summary + prediction check
    # ------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("CLASS-DISJOINT — Hypothesis (W) test, with BoGrad-mode comparison")
    print("=" * 130)
    header = (f"{'variant':30s} {'acc':>6s} {'Δ_OOB':>8s} {'WW_K':>7s} "
              f"{'%pos_g':>7s} {'%neg_g':>7s} {'⟨cos_g+⟩':>9s} {'⟨cos_g-⟩':>9s} "
              f"{'%pos_u':>7s} {'%neg_u':>7s} {'⟨cos_u+⟩':>9s} {'⟨cos_u-⟩':>9s}")
    print(header)
    print("-" * 130)

    for key, r in results.items():
        s = r.summary
        forget_out = r.forgetting.get("out_of_batch_total_magnitude", float("nan"))
        ww = s.get("ww_wasted_work_ratio_mean", float("nan"))

        pw = s.get("pairwise_summary", {}) or {}
        gp = pw.get("grad_frac_positive_mean", float("nan"))
        gn = pw.get("grad_frac_negative_mean", float("nan"))
        gp_cos = pw.get("grad_mean_positive_cos_mean", float("nan"))
        gn_cos = pw.get("grad_mean_negative_cos_mean", float("nan"))
        up_frac = pw.get("update_frac_positive_mean", float("nan"))
        un_frac = pw.get("update_frac_negative_mean", float("nan"))
        up_cos = pw.get("update_mean_positive_cos_mean", float("nan"))
        un_cos = pw.get("update_mean_negative_cos_mean", float("nan"))

        print(f"{key:30s} {r.final_acc:>6.4f} {forget_out:>8.3f} {ww:>7.3f} "
              f"{gp:>7.3f} {gn:>7.3f} {gp_cos:>+9.3f} {gn_cos:>+9.3f} "
              f"{up_frac:>7.3f} {un_frac:>7.3f} {up_cos:>+9.3f} {un_cos:>+9.3f}")

    # Predictions
    print("\nPREDICTION CHECK:")
    interleaved_baseline = results["interleaved_baseline"]
    disjoint_baseline = results["disjoint_baseline"]
    interleaved_bograd = results["interleaved_bograd_neg"]
    disjoint_bograd = results["disjoint_bograd_neg"]

    inter_forget = interleaved_baseline.forgetting.get("total_magnitude", 0.0)
    disjoint_forget = disjoint_baseline.forgetting.get("total_magnitude", 0.0)

    # Use OUT-OF-BATCH forgetting for the cleaner P1 test (per F6 — total
    # forgetting magnitude is too noisy because in-batch fluctuations
    # dominate). Out-of-batch is the "interference proper" channel.
    inter_oob = interleaved_baseline.forgetting.get("out_of_batch_total_magnitude", 0.0)
    disjoint_oob = disjoint_baseline.forgetting.get("out_of_batch_total_magnitude", 0.0)
    p1 = disjoint_oob > inter_oob * 1.2
    print(f"  P1 (disjoint > interleaved OOB-forgetting): {'PASS' if p1 else 'FAIL'}  "
          f"(out-of-batch Δ: {inter_oob:.3f} vs {disjoint_oob:.3f})")

    p2 = disjoint_baseline.final_acc < interleaved_baseline.final_acc - 0.02
    print(f"  P2 (disjoint slows convergence):       {'PASS' if p2 else 'FAIL'}  "
          f"(acc: {interleaved_baseline.final_acc:.4f} vs {disjoint_baseline.final_acc:.4f})")

    # P3: BoGrad's reduction in OOB forgetting should be bigger in disjoint
    # case (because there's more genuine OOB interference there).
    inter_bg_oob = interleaved_bograd.forgetting.get("out_of_batch_total_magnitude", 0.0)
    disjoint_bg_oob = disjoint_bograd.forgetting.get("out_of_batch_total_magnitude", 0.0)
    inter_oob_reduction = inter_oob - inter_bg_oob
    disjoint_oob_reduction = disjoint_oob - disjoint_bg_oob
    p3 = disjoint_oob_reduction > inter_oob_reduction
    print(f"  P3 (BoGrad reduces OOB-forget more in disjoint): {'PASS' if p3 else 'FAIL'}  "
          f"(reductions: interleaved {inter_oob_reduction:+.3f}, disjoint {disjoint_oob_reduction:+.3f})")

    p4 = disjoint_bograd.final_acc > disjoint_baseline.final_acc + 0.005
    print(f"  P4 (BoGrad recovers acc in disjoint):  {'PASS' if p4 else 'FAIL'}  "
          f"(acc: {disjoint_baseline.final_acc:.4f} -> {disjoint_bograd.final_acc:.4f})")

    overall = all([p1, p2, p3, p4])
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'} — "
          f"{'Hypothesis (W) supported on biased per-class interference (with OOB-refined metric).' if overall else 'one or more predictions failed; iterate.'}")

    # ---- BoGrad mode comparison on the disjoint regime ----
    # All three projection modes (full / negative / positive) on the same
    # disjoint setup, judged by (a) final accuracy and (b) OOB forgetting.
    print("\n" + "=" * 80)
    print("BoGrad MODE COMPARISON — disjoint regime only")
    print("=" * 80)
    print(f"{'mode':25s} {'final_acc':>10s} {'Δacc vs base':>14s} {'Δ_OOB':>8s}")
    print("-" * 80)
    base_acc = disjoint_baseline.final_acc
    base_oob = disjoint_baseline.forgetting.get("out_of_batch_total_magnitude", 0.0)
    for mode_key, label in [
        ("disjoint_baseline", "no BoGrad"),
        ("disjoint_bograd_full", "BoGrad full"),
        ("disjoint_bograd_neg", "BoGrad negative"),
        ("disjoint_bograd_pos", "BoGrad positive"),
    ]:
        if mode_key not in results:
            continue
        r = results[mode_key]
        oob = r.forgetting.get("out_of_batch_total_magnitude", float("nan"))
        d_acc = r.final_acc - base_acc
        print(f"{label:25s} {r.final_acc:>10.4f} {d_acc:>+14.4f} {oob:>8.3f}")

    print("\nInterpretation:")
    print("  - 'full' subtracts BOTH redundant (cos>0) and destructive (cos<0) overlap.")
    print("  - 'negative' subtracts only destructive overlap (preserves cooperative signal).")
    print("  - 'positive' subtracts only redundant overlap (preserves new orthogonal signal).")
    print("  Use the table above + the Δ_OOB and pairwise stats to decide which type")
    print("  of overlap actually carries the wasted-work in this regime.")

    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
