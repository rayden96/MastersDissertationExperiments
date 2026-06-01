"""
Shared training + plotting utility for image-classification interference
experiments (CIFAR-10 medium, CIFAR-100 large, and any future siblings).

The experiment-specific runners (`02_medium_cifar10/run.py`,
`03_large_cifar100/run.py`) build a model factory and dataset loaders, then
call `run_image_experiment(...)` here. This keeps experiment scripts tiny
and ensures consistent measurement logic.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from interference.meter import InterferenceMeter
from interference.summary import summarize_run
from interference.torch_classification import TorchClassificationProblem


def build_balanced_reference_subset(
    dataset, num_classes: int, n_per_class: int, seed: int = 2026,
) -> Subset:
    rng = np.random.default_rng(seed)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [int(dataset[i][1]) for i in range(len(dataset))]
    elif isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    elif isinstance(targets, np.ndarray):
        targets = targets.tolist()
    by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for i, y in enumerate(targets):
        if 0 <= int(y) < num_classes:
            by_class[int(y)].append(i)
    chosen: List[int] = []
    for c in range(num_classes):
        pool = by_class[c]
        rng.shuffle(pool)
        chosen.extend(pool[:n_per_class])
    return Subset(dataset, chosen)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    was = model.training
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    if was:
        model.train()
    return correct / max(total, 1)


def run_one_seed(
    *,
    seed: int,
    model_factory: Callable[[], nn.Module],
    train_dataset,
    test_dataset,
    ref_dataset,
    num_classes: int,
    device: torch.device,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    lr: float,
    epochs: int,
    batch_size: int,
    log_every: int,
    ref_refresh_every: int,
    K_values: List[int],
    num_workers: int = 2,
    is_cosgd: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(model)

    g = torch.Generator()
    g.manual_seed(seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0), generator=g,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    ref_loader = DataLoader(
        ref_dataset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    problem = TorchClassificationProblem(
        model=model, criterion=criterion, ref_loader=ref_loader, device=device,
    )
    meter = InterferenceMeter(
        problem=problem, lr=lr, K_values=K_values,
        log_every=log_every, ref_refresh_every=ref_refresh_every,
    )
    meter.initialize()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [seed {seed}] params={n_params} epochs={epochs} batch={batch_size} lr={lr}",
          flush=True)

    epoch_acc: List[float] = []
    epoch_train_loss: List[float] = []
    t0 = time.time()
    step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        et0 = time.time()
        dot_every = max(1, len(train_loader) // 10)
        print(f"    e{epoch + 1:>2}: ", end="", flush=True)

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_cosgd:
                # COSGD does its own forward+backward inside .step()
                meter.before_step()
                cosgd_loss = optimizer.step(x, y, torch.unique(y))
                loss_val = float(cosgd_loss) if cosgd_loss is not None else float("nan")
                meter.after_step(step, (x, y), applied_loss=loss_val)
            else:
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                meter.before_step()
                optimizer.step()
                loss_val = float(loss.item())
                meter.after_step(step, (x, y), applied_loss=loss_val)

            step += 1
            epoch_loss_sum += loss_val
            n_batches += 1
            if (batch_idx + 1) % dot_every == 0:
                print(".", end="", flush=True)

        train_loss = epoch_loss_sum / max(n_batches, 1)
        epoch_train_loss.append(train_loss)
        acc = evaluate(model, test_loader, device)
        epoch_acc.append(acc)
        print(f" train_loss={train_loss:.4f}  test_acc={acc:.4f}  "
              f"({time.time() - et0:.1f}s)")

    elapsed = time.time() - t0
    summary = summarize_run(
        logs=meter.logs,
        calibration_logs=meter.calibration_logs,
        K_values=K_values,
        cum_deficit=meter.cum_deficit,
        cum_deficit_count=meter.cum_deficit_count,
    )
    summary.update({
        "seed": seed,
        "n_params": n_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "final_test_acc": epoch_acc[-1] if epoch_acc else float("nan"),
        "best_test_acc": max(epoch_acc) if epoch_acc else float("nan"),
        "epoch_test_acc": epoch_acc,
        "epoch_train_loss": epoch_train_loss,
        "wall_clock_s": elapsed,
    })
    return {
        "summary": summary,
        "logs": meter.logs,
        "calibration_logs": meter.calibration_logs,
    }


def run_image_experiment(
    *,
    experiment_name: str,
    model_factory: Callable[[], nn.Module],
    train_dataset,
    test_dataset,
    num_classes: int,
    out_root: Path,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    lr: float,
    epochs: int,
    batch_size: int,
    seeds: List[int],
    log_every: int = 50,
    ref_refresh_every: int = 100,
    K_values: Optional[List[int]] = None,
    ref_n_per_class: int = 200,
    ref_seed: int = 2026,
    num_workers: int = 2,
    is_cosgd: bool = False,
    method_label: str = "baseline",
) -> Path:
    if K_values is None:
        K_values = [4, 32, 128]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"[{experiment_name}] device={device}, method={method_label}, lr={lr}, epochs={epochs}, "
          f"batch={batch_size}, K_values={K_values}, seeds={seeds}")

    ref_ds = build_balanced_reference_subset(
        train_dataset, num_classes=num_classes,
        n_per_class=ref_n_per_class, seed=ref_seed,
    )
    print(f"[{experiment_name}] reference set: {len(ref_ds)} samples balanced "
          f"({ref_n_per_class}/class × {num_classes} classes)")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    # Tag the run dir with method label so baseline/bograd/cosgd runs are
    # never overwritten and easy to identify.
    out_dir = out_root / f"run_{run_id}_{method_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries: List[Dict] = []
    for seed in seeds:
        print(f"\n[{experiment_name}] seed {seed} starting", flush=True)
        result = run_one_seed(
            seed=seed,
            model_factory=model_factory,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            ref_dataset=ref_ds,
            num_classes=num_classes,
            device=device,
            optimizer_factory=optimizer_factory,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            log_every=log_every,
            ref_refresh_every=ref_refresh_every,
            K_values=K_values,
            num_workers=num_workers,
            is_cosgd=is_cosgd,
        )
        with open(out_dir / f"logs_seed{seed}.json", "w") as f:
            json.dump({
                "summary": result["summary"],
                "logs": result["logs"],
                "calibration_logs": result["calibration_logs"],
            }, f, indent=2, default=str)
        with open(out_dir / f"summary_seed{seed}.json", "w") as f:
            json.dump(result["summary"], f, indent=2, default=str)
        seed_summaries.append(result["summary"])

    # Aggregate
    def agg(key):
        vals = [s.get(key, float("nan")) for s in seed_summaries
                if isinstance(s.get(key), (int, float))]
        vals = [v for v in vals if v == v]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    headline_keys = [
        "final_test_acc", "best_test_acc", "wall_clock_s",
        "I_inter_mean", "I_inter_std",
        "inter_frac_neg_mean", "inter_mean_cos_mean",
        "inter_useful_descent_frac_mean", "inter_max_min_ratio_mean",
        "I_between_K32_mean", "between_K32_mean_cos_mean",
        "between_K32_useful_path_frac_mean",
        "cum_deficit", "mean_deficit_per_step",
        "corr_I_inter_vs_Dt", "corr_inter_mean_cos_vs_Dt",
        "corr_inter_max_min_ratio_vs_Dt",
        "corr_I_between_K32_vs_Dt", "corr_between_K32_mean_cos_vs_Dt",
        "ref_loss_drop",
    ]
    headline = {k: agg(k) for k in headline_keys}

    with open(out_dir / "headline.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({
            "experiment_name": experiment_name,
            "run_id": run_id,
            "seeds": list(seeds),
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "K_values": K_values,
            "log_every": log_every, "ref_refresh_every": ref_refresh_every,
            "ref_n_per_class": ref_n_per_class,
        }, f, indent=2, default=str)

    print(f"\n=== [{experiment_name}] headline (mean ± std across seeds) ===")
    for k in headline_keys:
        v = headline[k]
        print(f"  {k:<40} {v['mean']:+.4f} ± {v['std']:.4f}  (n={v['n']})")
    print(f"\nResults written to {out_dir}")
    print(f"Run id: {run_id}")
    return out_dir


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _resolve_latest_run(results_root: Path) -> Path:
    runs = sorted(results_root.glob("run_*"))
    if not runs:
        raise SystemExit(f"No runs found in {results_root}")
    return runs[-1]


def _seeds_in_run(run_dir: Path) -> List[int]:
    seeds = []
    for f in run_dir.glob("summary_seed*.json"):
        seeds.append(int(f.stem.replace("summary_seed", "")))
    return sorted(seeds)


def _series_mean_std(seed_logs: List[List[Dict]], key: str):
    per_seed = []
    for logs in seed_logs:
        steps = [l["step"] for l in logs if key in l and l[key] == l[key]]
        vals = [l[key] for l in logs if key in l and l[key] == l[key]]
        per_seed.append((np.array(steps), np.array(vals, dtype=np.float64)))
    if not per_seed:
        return np.array([]), np.array([]), np.array([])
    common = set(per_seed[0][0].tolist())
    for s, _ in per_seed[1:]:
        common &= set(s.tolist())
    common = sorted(common)
    if not common:
        return np.array([]), np.array([]), np.array([])
    aligned = []
    for s, v in per_seed:
        idx = np.array([np.where(s == cs)[0][0] for cs in common])
        aligned.append(v[idx])
    M = np.stack(aligned, axis=0)
    return np.array(common), M.mean(axis=0), M.std(axis=0)


def plot_image_experiment(results_root: Path, run_id: Optional[str] = None,
                          experiment_name: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = (results_root / f"run_{run_id}") if run_id else _resolve_latest_run(results_root)
    print(f"Plotting from {run_dir}")
    seeds = _seeds_in_run(run_dir)
    if not seeds:
        raise SystemExit(f"No seed summaries in {run_dir}")
    seed_logs = []
    summaries = []
    for s in seeds:
        with open(run_dir / f"logs_seed{s}.json") as f:
            j = json.load(f)
            seed_logs.append(j["logs"])
        with open(run_dir / f"summary_seed{s}.json") as f:
            summaries.append(json.load(f))

    keys = [
        ("I_inter",                   "Inter-batch cancellation index"),
        ("inter_mean_cos",            "Mean pairwise cosine (inter)"),
        ("inter_useful_descent_frac", "Useful descent frac of batch grad"),
        ("I_between_K32",             "Between-batch cancellation (K=32)"),
        ("between_K32_mean_cos",      "Mean pairwise cosine within K=32"),
        ("D_t",                       "Per-step first-order deficit $D_t$"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for ax, (key, title) in zip(axs.flatten(), keys):
        steps, mean, std = _series_mean_std(seed_logs, key)
        if steps.size == 0:
            ax.set_title(f"{title}\n(no data)")
            continue
        ax.plot(steps, mean, color="tab:blue", lw=1.5)
        ax.fill_between(steps, mean - std, mean + std, color="tab:blue", alpha=0.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{experiment_name} — interference metrics over training "
                 f"(mean ± std across seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(run_dir / "metrics_timeseries.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {run_dir / 'metrics_timeseries.png'}")

    # Correlations bar
    corr_keys = [
        ("corr_I_inter_vs_Dt",             "$I_{\\rm inter}$ vs $D_t$"),
        ("corr_inter_mean_cos_vs_Dt",      "mean inter cos vs $D_t$"),
        ("corr_inter_max_min_ratio_vs_Dt", "max/min ratio vs $D_t$"),
        ("corr_I_between_K32_vs_Dt",       "$I_{\\rm between, 32}$ vs $D_t$"),
        ("corr_between_K32_mean_cos_vs_Dt", "K=32 mean cos vs $D_t$"),
    ]
    means: List[float] = []
    stds: List[float] = []
    for key, _ in corr_keys:
        vals = [s.get(key, float("nan")) for s in summaries
                if isinstance(s.get(key), (int, float)) and not math.isnan(s.get(key))]
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(float("nan"))
            stds.append(0.0)
    fig2, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(corr_keys))
    ax.bar(x, means, yerr=stds, color="tab:blue", alpha=0.85, capsize=4)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in corr_keys], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"{experiment_name} — correlation: geometric metric vs $D_t$")
    ax.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(run_dir / "correlations.png", dpi=140)
    plt.close(fig2)
    print(f"  wrote {run_dir / 'correlations.png'}")
