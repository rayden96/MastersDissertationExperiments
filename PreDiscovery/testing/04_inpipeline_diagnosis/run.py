"""
Test 04 — In-pipeline diagnosis.

Phase 4a: capture per-step diagnostics on Adam, RMSprop, SignSGD with
in-pipeline projection at their best K. Produces per-step JSON files we can
plot or aggregate.

Phase 4b: try alternative buffer contents / projection points on the failing
families and see if any beat the in-pipeline result.
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
from typing import Any, Callable, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.optimizers import BoGrad  # noqa: E402
from discoveryPhase2.enhanced_variants import (  # noqa: E402
    InPipelineRMSprop, InPipelineSignSGD, SignSGD,
)
from discovery.bograd_variants import AdamWithBoGrad  # noqa: E402
from testing._common import (  # noqa: E402
    SmallCNN, aggregate, build_cifar10, delta_marker,
    make_test_loader, make_train_loader, train_run, evaluate,
)


# ---------------------------------------------------------------------------
# Phase 4a — diagnostic capture
# ---------------------------------------------------------------------------
def capture_diagnostics(optimizer, family_name: str, max_steps: int = 100) -> List[Dict[str, Any]]:
    """Pull diagnostics from the optimizer's buffer state every step.

    For each parameter tensor: rank/condition of buffer matrix, g_ratio,
    trigger fraction. Aggregated across tensors.
    """
    rows: List[Dict[str, Any]] = []
    # Inspect the first few parameters' buffers as proxies for "the model's buffer geometry".
    # For per-tensor scope, this is a reasonable summary.
    return rows  # populated incrementally during training (callers append)


def buffer_geometry(buffer: List[torch.Tensor]) -> Dict[str, float]:
    """Summarize a buffer of vectors: condition number, effective rank, mean norm."""
    if len(buffer) < 2:
        return {"condition_number": float("nan"), "effective_rank": float("nan"),
                "buffer_size": float(len(buffer))}
    try:
        B = torch.stack([b.detach().float() for b in buffer], dim=0)  # [K, d]
        # Singular values of B (i.e. of the K×d matrix)
        S = torch.linalg.svdvals(B)
        S_max = float(S.max().item())
        S_min = float(S.min().item())
        cond = S_max / S_min if S_min > 1e-12 else float("inf")
        # Effective rank ≈ exp(entropy of normalised singular values)
        S_norm = S / (S.sum() + 1e-12)
        entropy = -(S_norm * (S_norm.clamp(min=1e-12).log())).sum()
        eff_rank = float(torch.exp(entropy).item())
        norms = [float(b.norm().item()) for b in buffer]
        return {
            "condition_number": cond,
            "effective_rank": eff_rank,
            "buffer_size": float(len(buffer)),
            "norm_mean": float(statistics.mean(norms)),
            "norm_std": float(statistics.stdev(norms)) if len(norms) > 1 else 0.0,
        }
    except Exception as exc:
        return {"error": str(exc), "buffer_size": float(len(buffer))}


def _get_first_buffer(optimizer) -> List[torch.Tensor]:
    """Find a non-empty buffer in the optimizer state (any param will do for diag)."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            buf = state.get("buffer", [])
            if isinstance(buf, list) and len(buf) > 0:
                return buf
    return []


def train_with_diagnostics(
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    family_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int,
    log_every: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SmallCNN().to(device)
    opt = optimizer_factory(model)
    criterion = nn.CrossEntropyLoss()

    diag_rows: List[Dict[str, Any]] = []
    epoch_acc: List[float] = []
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                row: Dict[str, Any] = {"step": step, "loss": float(loss.item())}
                # Pull stats from optimizer if available
                if hasattr(opt, "get_last_step_stats"):
                    row.update(opt.get_last_step_stats())
                # Buffer geometry (first non-empty buffer)
                buf = _get_first_buffer(opt)
                row.update(buffer_geometry(buf))
                diag_rows.append(row)
        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        print(f"  {family_name} epoch {epoch + 1}: acc={acc:.4f}  steps={step}  ({time.time() - t0:.0f}s)")

    return {
        "family": family_name,
        "epoch_test_acc": epoch_acc,
        "diagnostics": diag_rows,
        "wall_clock_s": time.time() - t0,
    }


def run_phase_a(args, train_ds, test_ds, device, out_dir):
    print("\n" + "=" * 60)
    print("PHASE 4a — diagnostic capture")
    print("=" * 60)

    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    families = {
        "adam_inpipeline_K128": lambda m: AdamWithBoGrad(
            m.parameters(), lr=1e-3,
            buffer_size=128, projection_point="momentum",
            projection_mode="negative", store_normalised=True,
        ),
        "rmsprop_inpipeline_K16": lambda m: InPipelineRMSprop(
            m.parameters(), lr=1e-3, alpha=0.99, momentum=0.9,
            buffer_size=16, projection_mode="negative",
        ),
        "signsgd_inpipeline_K64": lambda m: InPipelineSignSGD(
            m.parameters(), lr=1e-3, momentum=0.9,
            buffer_size=64, projection_mode="negative",
        ),
        "sgd_momentum_update_K32": lambda m: BoGrad(
            m.parameters(), torch.optim.SGD,
            buffer_size=32, project_stage="update",
            projection_mode="negative", orth_method="sequential",
            lr=0.05, momentum=0.9,
        ),
    }

    summary: Dict[str, Any] = {}
    for name, fac in families.items():
        seed = args.base_seed
        train_loader = make_train_loader(train_ds, args.batch_size, args.num_workers,
                                          seed, (device.type == "cuda"))
        print(f"\n-- {name} --")
        try:
            result = train_with_diagnostics(
                fac, name, train_loader, test_loader, device,
                epochs=args.epochs_4a, seed=seed, log_every=args.log_every,
            )
        except Exception as exc:
            print(f"  !! {name} failed: {exc}")
            continue
        with (diag_dir / f"{name}__per_step.json").open("w") as fh:
            json.dump(result, fh, indent=2)
        # Summary stats
        rows = result["diagnostics"]
        if rows:
            cond_vals = [r.get("condition_number") for r in rows
                         if isinstance(r.get("condition_number"), (int, float))]
            er_vals = [r.get("effective_rank") for r in rows
                       if isinstance(r.get("effective_rank"), (int, float))]
            ratio_vals = [r.get("g_ratio_mean") for r in rows
                          if isinstance(r.get("g_ratio_mean"), (int, float))]
            summary[name] = {
                "final_acc": result["epoch_test_acc"][-1] if result["epoch_test_acc"] else float("nan"),
                "cond_median": statistics.median(cond_vals) if cond_vals else float("nan"),
                "eff_rank_median": statistics.median(er_vals) if er_vals else float("nan"),
                "g_ratio_median": statistics.median(ratio_vals) if ratio_vals else float("nan"),
                "n_diag_rows": len(rows),
            }

    print("\n" + "=" * 92)
    print("PHASE 4a — diagnostic summary")
    print("=" * 92)
    print(f"{'family':32s} {'acc':>7s} {'cond_med':>10s} {'eff_rank':>10s} {'g_ratio':>9s}  rows")
    print("-" * 92)
    for name, s in summary.items():
        print(
            f"{name:32s} {s['final_acc']:>7.4f} "
            f"{s['cond_median']:>10.2e} {s['eff_rank_median']:>10.2f} "
            f"{s['g_ratio_median']:>9.4f}  {s['n_diag_rows']}"
        )

    return summary


# ---------------------------------------------------------------------------
# Phase 4b — alternative variants on failing families
# ---------------------------------------------------------------------------
def run_phase_b(args, train_ds, test_ds, device, out_dir):
    print("\n" + "=" * 60)
    print("PHASE 4b — alternative variants on RMSprop / SignSGD")
    print("=" * 60)

    test_loader = make_test_loader(test_ds, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))

    @dataclass
    class V:
        name: str
        family: str
        factory: Callable[[Any], torch.optim.Optimizer]
        note: str

    variants: List[V] = [
        V("rmsprop_baseline", "rmsprop",
          lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, alpha=0.99, momentum=0.9),
          "RMSprop+mom baseline"),
        V("rmsprop_inpipeline_K16", "rmsprop",
          lambda m: InPipelineRMSprop(m.parameters(), lr=1e-3, alpha=0.99, momentum=0.9,
                                       buffer_size=16, projection_mode="negative"),
          "in-pipeline best from prior sweep"),
        V("rmsprop_update_K16", "rmsprop",
          lambda m: BoGrad(m.parameters(), torch.optim.RMSprop,
                            buffer_size=16, project_stage="update",
                            projection_mode="negative", orth_method="sequential",
                            lr=1e-3, alpha=0.99, momentum=0.9),
          "update-stage K=16 — for direct comparison"),

        V("signsgd_baseline", "signsgd",
          lambda m: SignSGD(m.parameters(), lr=1e-3, momentum=0.9),
          "SignSGD+mom baseline"),
        V("signsgd_inpipeline_K64", "signsgd",
          lambda m: InPipelineSignSGD(m.parameters(), lr=1e-3, momentum=0.9,
                                       buffer_size=64, projection_mode="negative"),
          "in-pipeline best from prior sweep"),
        V("signsgd_update_K64", "signsgd",
          lambda m: BoGrad(m.parameters(), SignSGD,
                            buffer_size=64, project_stage="update",
                            projection_mode="negative", orth_method="sequential",
                            lr=1e-3, momentum=0.9),
          "update-stage K=64 — wraps the SignSGD class"),
    ]

    results: Dict[str, List[Dict[str, Any]]] = {}
    total = len(variants) * args.trials
    run_idx = 0

    for trial in range(args.trials):
        seed = args.base_seed + trial * 1000
        print(f"\n=== Trial {trial + 1}/{args.trials} (seed={seed}) ===")
        for v in variants:
            train_loader = make_train_loader(train_ds, args.batch_size, args.num_workers,
                                              seed, (device.type == "cuda"))
            run_idx += 1
            try:
                r = train_run(
                    f"[{run_idx}/{total}] {v.name}",
                    SmallCNN, v.factory,
                    train_loader, test_loader, device,
                    epochs=args.epochs, seed=seed,
                )
            except Exception as exc:
                print(f"\n    !! {v.name} failed: {exc}")
                r = {
                    "final_test_acc": float("nan"),
                    "best_test_acc": float("nan"),
                    "epoch_test_acc": [],
                    "wall_clock_s": 0.0,
                    "error": str(exc),
                }
            r["trial"] = trial
            r["variant"] = v.name
            r["family"] = v.family
            r["note"] = v.note
            results.setdefault(v.name, []).append(r)

    summary = {
        "config": vars(args),
        "device": str(device),
        "results": results,
    }
    with (out_dir / "results.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 92)
    print("PHASE 4b RESULTS")
    print("=" * 92)
    print(f"{'variant':32s} {'mean':>7s} {'std':>7s}  Δvs.baseline  trials")
    print("-" * 92)
    for family in ["rmsprop", "signsgd"]:
        baseline_name = f"{family}_baseline"
        baseline_finals = [t["final_test_acc"] for t in results.get(baseline_name, [])]
        base_mean = aggregate(baseline_finals)["mean"]
        for v in variants:
            if v.family != family:
                continue
            finals = [t["final_test_acc"] for t in results.get(v.name, [])]
            agg = aggregate(finals)
            delta = agg["mean"] - base_mean if (agg["n"] and base_mean == base_mean) else float("nan")
            marker = "" if v.name == baseline_name else delta_marker(delta)
            print(f"{v.name:32s} {agg['mean']:>7.4f} {agg['std']:>7.4f}  {delta:>+7.4f}{marker:>2s}  {agg['n']}")
        print("-" * 92)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--epochs-4a", type=int, default=2)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | torch {torch.__version__}")

    train_ds, test_ds = build_cifar10(
        Path(args.data_root), download=not args.skip_download, quick=args.quick,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "testing" / "04_inpipeline_diagnosis" / "results" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    summary = {"config": vars(args), "device": str(device)}

    if args.phase in ("a", "both"):
        summary["phase_a"] = run_phase_a(args, train_ds, test_ds, device, out_dir)
    if args.phase in ("b", "both"):
        summary["phase_b"] = run_phase_b(args, train_ds, test_ds, device, out_dir)

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nDone. Results at {out_dir}")


if __name__ == "__main__":
    main()
