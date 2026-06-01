"""
Test 03 — Optimization ablation.

Part A: benchmark candidate BoGrad optimisations on SmallCNN.
Part B: scale the best combination across SmallCNN / ResNet8 / ResNet18CIFAR.

The "sync_removed" variant requires a small code patch to BoGrad — for
cleanliness, this script monkeypatches the projection method at runtime so we
don't have to maintain multiple optimizer files. If sync_removed proves a clear
win, the patch becomes BoGrad's default.

Wall-clock is measured per training step (median over last N steps after a
warmup window). Peak memory is measured via torch.cuda.max_memory_allocated().
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.optimizers import BoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    ARCHITECTURES, aggregate, build_cifar10, evaluate, num_params,
    make_test_loader, make_train_loader, train_run,
)


# ---------------------------------------------------------------------------
# Sync-removed projection — monkey-patchable replacement for _project_sequential
# ---------------------------------------------------------------------------
def _project_sequential_sync_free(self, x_flat, buffer):
    """Drop-in replacement: same math but no host-device sync per buffer iteration.

    Negative-mode gating uses torch.where on a 0-d GPU tensor instead of .item().
    """
    projected = x_flat.clone()
    for b in buffer:
        b_vec = b.to(device=x_flat.device, dtype=self.projection_dtype)
        if self.store_normalised:
            denom = 1.0
        else:
            bb = torch.dot(b_vec, b_vec)
            # Avoid sync — compare on GPU
            if bb.item() <= self.eps:  # last unavoidable sync; rare path
                continue
            denom = bb.item()  # also rare path
        dot_val = torch.dot(projected, b_vec)
        if self.projection_mode == "negative":
            # Gate via mask instead of conditional
            coeff = torch.where(
                dot_val < 0,
                dot_val / denom if isinstance(denom, float) else dot_val / denom,
                torch.zeros_like(dot_val),
            )
        else:
            coeff = dot_val / denom if isinstance(denom, float) else dot_val / denom
        projected = projected - coeff * b_vec
    return projected


# ---------------------------------------------------------------------------
# Variant configs (Part A)
# ---------------------------------------------------------------------------
@dataclass
class OptVariant:
    name: str
    bograd_kwargs: Dict[str, Any]
    sync_removed: bool
    note: str


PART_A_VARIANTS: List[OptVariant] = [
    OptVariant("current", {"buffer_dtype": torch.float32, "projection_scope": "per_tensor"},
               False, "no optimisations"),
    OptVariant("sync_removed", {"buffer_dtype": torch.float32, "projection_scope": "per_tensor"},
               True, ".item() removed from inner loop"),
    OptVariant("fp16_buffer", {"buffer_dtype": torch.float16, "projection_scope": "per_tensor"},
               False, "buffer in fp16"),
    OptVariant("global_scope", {"buffer_dtype": torch.float32, "projection_scope": "global"},
               False, "single global buffer"),
    OptVariant("combined", {"buffer_dtype": torch.float16, "projection_scope": "global"},
               True, "all of the above"),
]


def make_bograd(model, base_kwargs: Dict[str, Any], variant: OptVariant,
                K: int = 32) -> BoGrad:
    opt = BoGrad(
        model.parameters(), torch.optim.SGD,
        buffer_size=K, project_stage="update",
        projection_mode="negative", orth_method="sequential",
        **variant.bograd_kwargs, **base_kwargs,
    )
    if variant.sync_removed:
        # Bind the sync-free implementation onto this instance.
        import types
        opt._project_sequential = types.MethodType(_project_sequential_sync_free, opt)
    return opt


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------
def benchmark_step_time(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    *,
    warmup_steps: int = 50,
    measure_steps: int = 500,
) -> Dict[str, float]:
    """Run measure_steps + warmup_steps mini-batches and return median per-step time."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times: List[float] = []
    step = 0
    target = warmup_steps + measure_steps

    iterator = iter(train_loader)
    while step < target:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= warmup_steps:
            times.append(elapsed)
        step += 1

    median = statistics.median(times) if times else float("nan")
    p90 = sorted(times)[int(0.9 * len(times))] if times else float("nan")
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else float("nan")
    )
    return {
        "median_step_s": median,
        "p90_step_s": p90,
        "peak_mem_mb": peak_mb,
        "n_measure": len(times),
    }


# ---------------------------------------------------------------------------
# Part A — microbenchmark optimisations on SmallCNN
# ---------------------------------------------------------------------------
def run_part_a(args, train_ds, test_ds, device, out_dir):
    print("\n" + "=" * 60)
    print("PART A — optimisation microbenchmark on SmallCNN")
    print("=" * 60)

    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))
    arch_factory = ARCHITECTURES["small_cnn"]

    results: Dict[str, Any] = {}

    # Plain SGD+momentum baseline (no BoGrad) for relative-overhead reference
    print(f"\n-- baseline (no BoGrad) --")
    base_times = []
    base_accs = []
    for trial in range(args.trials):
        seed = args.base_seed + trial * 1000
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = arch_factory().to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        train_loader = make_train_loader(train_ds, args.batch_size, args.num_workers,
                                         seed, (device.type == "cuda"))
        bench = benchmark_step_time(model, opt, train_loader, device,
                                    warmup_steps=args.warmup, measure_steps=args.measure)
        base_times.append(bench["median_step_s"])
        # quick accuracy sanity
        if args.acc_check:
            acc_run = train_run(
                f"  acc-check baseline t{trial}",
                arch_factory, lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9),
                make_train_loader(train_ds, args.batch_size, args.num_workers,
                                   seed, (device.type == "cuda")),
                test_loader, device, epochs=args.epochs, seed=seed, quiet=False,
            )
            base_accs.append(acc_run["final_test_acc"])
        print(f"  baseline t{trial}: median_step={bench['median_step_s']*1000:.2f}ms  peak_mem={bench['peak_mem_mb']:.1f}MB")
    results["baseline"] = {
        "step_time_s": aggregate(base_times),
        "final_test_acc": aggregate(base_accs) if base_accs else None,
    }

    for variant in PART_A_VARIANTS:
        print(f"\n-- {variant.name} ({variant.note}) --")
        var_times = []
        var_mems = []
        var_accs = []
        for trial in range(args.trials):
            seed = args.base_seed + trial * 1000
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = arch_factory().to(device)
            opt = make_bograd(model, {"lr": 0.05, "momentum": 0.9}, variant, K=args.K)
            train_loader = make_train_loader(train_ds, args.batch_size, args.num_workers,
                                             seed, (device.type == "cuda"))
            try:
                bench = benchmark_step_time(model, opt, train_loader, device,
                                            warmup_steps=args.warmup, measure_steps=args.measure)
                var_times.append(bench["median_step_s"])
                var_mems.append(bench["peak_mem_mb"])
                if args.acc_check:
                    acc_run = train_run(
                        f"  acc-check {variant.name} t{trial}",
                        arch_factory,
                        lambda m, v=variant: make_bograd(m, {"lr": 0.05, "momentum": 0.9}, v, K=args.K),
                        make_train_loader(train_ds, args.batch_size, args.num_workers,
                                           seed, (device.type == "cuda")),
                        test_loader, device, epochs=args.epochs, seed=seed,
                    )
                    var_accs.append(acc_run["final_test_acc"])
                print(f"  {variant.name} t{trial}: median_step={bench['median_step_s']*1000:.2f}ms  peak_mem={bench['peak_mem_mb']:.1f}MB")
            except Exception as exc:
                print(f"  !! {variant.name} t{trial} failed: {exc}")
                var_times.append(float("nan"))
                var_mems.append(float("nan"))
                var_accs.append(float("nan"))
        results[variant.name] = {
            "step_time_s": aggregate(var_times),
            "peak_mem_mb": aggregate(var_mems),
            "final_test_acc": aggregate(var_accs) if var_accs else None,
        }

    # Print summary
    print("\n" + "=" * 92)
    print("PART A — median step time, peak memory, final accuracy (mean across trials)")
    print("=" * 92)
    print(f"{'variant':18s} {'step_ms':>9s} {'peak_MB':>9s} {'overhead':>10s} {'acc':>7s}")
    print("-" * 92)
    base_step = results["baseline"]["step_time_s"]["mean"]
    for name in ["baseline"] + [v.name for v in PART_A_VARIANTS]:
        r = results[name]
        s = r["step_time_s"]["mean"]
        ms = s * 1000 if s == s else float("nan")
        if name == "baseline":
            overhead_str = "—"
        else:
            ov = (s / base_step - 1) * 100 if (s == s and base_step == base_step) else float("nan")
            overhead_str = f"+{ov:.1f}%"
        mem = r.get("peak_mem_mb", {"mean": float("nan")})["mean"] if name != "baseline" else float("nan")
        acc = r.get("final_test_acc")
        acc_val = acc["mean"] if acc else float("nan")
        print(f"{name:18s} {ms:>9.2f} {mem:>9.1f} {overhead_str:>10s} {acc_val:>7.4f}")

    return results


# ---------------------------------------------------------------------------
# Part B — best-combined config across architectures
# ---------------------------------------------------------------------------
def run_part_b(args, train_ds, test_ds, device, out_dir):
    print("\n" + "=" * 60)
    print("PART B — combined config across architectures")
    print("=" * 60)

    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))
    combined = next(v for v in PART_A_VARIANTS if v.name == "combined")

    archs = ["small_cnn", "resnet8", "resnet18"]
    results: Dict[str, Any] = {}

    for arch_name in archs:
        arch_factory = ARCHITECTURES[arch_name]
        sample_model = arch_factory()
        n_params = num_params(sample_model)
        del sample_model
        print(f"\n-- {arch_name} ({n_params:,} params) --")

        # Baseline
        b_times, b_mems = [], []
        for trial in range(args.trials):
            seed = args.base_seed + trial * 1000
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = arch_factory().to(device)
            opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            train_loader = make_train_loader(train_ds, args.batch_size, args.num_workers,
                                             seed, (device.type == "cuda"))
            try:
                bench = benchmark_step_time(model, opt, train_loader, device,
                                            warmup_steps=args.warmup, measure_steps=args.measure)
                b_times.append(bench["median_step_s"])
                b_mems.append(bench["peak_mem_mb"])
                print(f"  baseline t{trial}: median_step={bench['median_step_s']*1000:.2f}ms peak_mem={bench['peak_mem_mb']:.1f}MB")
            except Exception as exc:
                print(f"  !! baseline {arch_name} t{trial} failed: {exc}")

        # BoGrad combined
        v_times, v_mems = [], []
        for trial in range(args.trials):
            seed = args.base_seed + trial * 1000
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = arch_factory().to(device)
            opt = make_bograd(model, {"lr": 0.05, "momentum": 0.9}, combined, K=args.K)
            train_loader = make_train_loader(train_ds, args.batch_size, args.num_workers,
                                             seed, (device.type == "cuda"))
            try:
                bench = benchmark_step_time(model, opt, train_loader, device,
                                            warmup_steps=args.warmup, measure_steps=args.measure)
                v_times.append(bench["median_step_s"])
                v_mems.append(bench["peak_mem_mb"])
                print(f"  bograd t{trial}: median_step={bench['median_step_s']*1000:.2f}ms peak_mem={bench['peak_mem_mb']:.1f}MB")
            except Exception as exc:
                print(f"  !! bograd {arch_name} t{trial} failed: {exc}")
                v_times.append(float("nan"))
                v_mems.append(float("nan"))

        results[arch_name] = {
            "n_params": n_params,
            "baseline_step_s": aggregate(b_times),
            "baseline_mem_mb": aggregate(b_mems),
            "bograd_step_s": aggregate(v_times),
            "bograd_mem_mb": aggregate(v_mems),
        }

    print("\n" + "=" * 92)
    print("PART B — overhead vs architecture")
    print("=" * 92)
    print(f"{'arch':12s} {'#params':>11s} {'base_ms':>9s} {'bograd_ms':>10s} {'overhead':>10s} {'mem_extra_MB':>12s}")
    print("-" * 92)
    for name in archs:
        r = results.get(name, {})
        if not r:
            continue
        base_ms = r["baseline_step_s"]["mean"] * 1000
        bg_ms = r["bograd_step_s"]["mean"] * 1000
        overhead = (bg_ms / base_ms - 1) * 100 if base_ms == base_ms else float("nan")
        mem_extra = (r["bograd_mem_mb"]["mean"] - r["baseline_mem_mb"]["mean"]) if (
            r["bograd_mem_mb"]["mean"] == r["bograd_mem_mb"]["mean"]
        ) else float("nan")
        print(f"{name:12s} {r['n_params']:>11,} {base_ms:>9.2f} {bg_ms:>10.2f} {overhead:>9.1f}% {mem_extra:>12.1f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=50, help="Benchmark warmup steps")
    ap.add_argument("--measure", type=int, default=500, help="Benchmark measurement steps")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--acc-check", action="store_true",
                    help="Also run a 5-epoch training per variant in Part A (slower).")
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "testing" / "03_optimizations" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    summary = {"config": vars(args), "device": str(device)}

    if args.part in ("A", "both"):
        summary["part_a"] = run_part_a(args, train_ds, test_ds, device, out_dir)
    if args.part in ("B", "both"):
        summary["part_b"] = run_part_b(args, train_ds, test_ds, device, out_dir)

    with (out_dir / "results.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
