"""
Shared utilities for BoGradExperimentsLive runners.

Provides:
- Models: SmallCIFARCNN (3-channel, no BN), SmallMNISTCNN (1-channel, no BN),
  ResNet18CIFAR re-exported from testing._common (with BN).
- DEFAULT_LRS: per-method family-appropriate default learning rates.
- ALL_METHODS: ordered list of base optimisers tested in every runner.
- get_base_optimizer / make_optimizer: optimiser factory (with optional BoGrad
  wrap, project_stage="update", projection_mode="negative").
- evaluate / train_one_run: training and evaluation primitives.
- run_sweep: top-level "run all methods × {baseline, +BoGrad}" sweep, with
  JSON output, comparison tables, and matplotlib plots.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repo root to path so we can import common.optimizers etc.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.optimizers import BoGrad  # noqa: E402
from discoveryPhase2.enhanced_variants import SignSGD  # noqa: E402
from testing._common import ResNet18CIFAR  # noqa: E402

# Local — sibling module
from windowed_mean_bograd import WindowedMeanBoGrad  # noqa: E402


# Wrapper kinds: "none" = baseline, "bograd" = K-buffer subspace projection,
# "windowed_mean" = rank-1 mean-of-buffer projection.
WRAPPERS = ("none", "bograd", "windowed_mean")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SmallCIFARCNN(nn.Module):
    """3-conv CNN, ~93k params, no BN/dropout."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallMNISTCNN(nn.Module):
    """2-conv CNN for MNIST (1-channel), ~5k params, no BN/dropout."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
ALL_METHODS: List[str] = [
    "sgd", "sgd_momentum", "adam", "rmsprop", "signsgd", "signsgd_momentum",
]

DEFAULT_LRS: Dict[str, float] = {
    "sgd":              0.05,
    "sgd_momentum":     0.05,
    "adam":             0.001,
    "rmsprop":          0.001,
    "signsgd":          0.001,
    "signsgd_momentum": 0.001,
}

# Stable colour per BASE method; linestyle distinguishes BoGrad vs baseline.
METHOD_COLORS: Dict[str, str] = {
    "sgd":              "tab:blue",
    "sgd_momentum":     "tab:orange",
    "adam":             "tab:green",
    "rmsprop":          "tab:red",
    "signsgd":          "tab:purple",
    "signsgd_momentum": "tab:brown",
}

# Plot families: which methods share a chart.
PLOT_FAMILIES: Dict[str, List[str]] = {
    "sgd":     ["sgd", "sgd_momentum"],
    "adam":    ["adam"],
    "rmsprop": ["rmsprop"],
    "signsgd": ["signsgd", "signsgd_momentum"],
}


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------
def get_base_optimizer(method: str, lr: float):
    if method == "sgd":
        return torch.optim.SGD, {"lr": lr, "momentum": 0.0}
    if method == "sgd_momentum":
        return torch.optim.SGD, {"lr": lr, "momentum": 0.5}
    if method == "adam":
        return torch.optim.Adam, {"lr": lr}
    if method == "rmsprop":
        return torch.optim.RMSprop, {"lr": lr}
    if method == "signsgd":
        return SignSGD, {"lr": lr, "momentum": 0.0}
    if method == "signsgd_momentum":
        return SignSGD, {"lr": lr, "momentum": 0.5}
    raise ValueError(f"unknown method {method!r}")


def make_optimizer(method: str, model: nn.Module, lr: float, wrapper: str, K: int):
    """wrapper ∈ {"none", "bograd", "windowed_mean"}."""
    base_cls, base_kwargs = get_base_optimizer(method, lr)
    if wrapper == "none":
        return base_cls(model.parameters(), **base_kwargs)
    if wrapper == "bograd":
        return BoGrad(
            model.parameters(),
            base_optimizer_cls=base_cls,
            buffer_size=K,
            project_stage="update",
            projection_mode="negative",
            orth_method="sequential",
            collect_stats=False,
            **base_kwargs,
        )
    if wrapper == "windowed_mean":
        return WindowedMeanBoGrad(
            model.parameters(),
            base_optimizer_cls=base_cls,
            buffer_size=K,
            projection_mode="negative",
            **base_kwargs,
        )
    raise ValueError(f"unknown wrapper {wrapper!r}; must be one of {WRAPPERS}")


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


_WRAPPER_LABEL = {
    "none": "",
    "bograd": "+bograd",
    "windowed_mean": "+winmean",
}


def train_one_run(
    *,
    method: str,
    wrapper: str,
    dataset_name: str,
    train_dataset,
    test_dataset,
    model_factory: Callable[[], nn.Module],
    device: torch.device,
    lr: float,
    K: int,
    epochs: int,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> Dict[str, Any]:
    if wrapper not in WRAPPERS:
        raise ValueError(f"unknown wrapper {wrapper!r}; must be one of {WRAPPERS}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(method, model, lr, wrapper, K)

    g = torch.Generator()
    g.manual_seed(seed)
    pin = (device.type == "cuda")
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

    label = f"{method}{_WRAPPER_LABEL[wrapper]}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    K_disp = K if wrapper != "none" else "-"
    print(f"\n[{label} | {dataset_name}] lr={lr:g}  K={K_disp}  "
          f"batch={batch_size}  epochs={epochs}  params={n_params}  seed={seed}")
    t0 = time.time()

    epoch_test_acc: List[float] = []
    epoch_train_loss: List[float] = []
    diverged = False
    diverged_at_step = -1
    n_batches_per_epoch = max(1, len(train_loader))
    dot_every = max(1, n_batches_per_epoch // 10)

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        running_recent = 0.0
        ema_alpha = 0.05
        epoch_t0 = time.time()
        print(f"  e{epoch + 1:>2}: ", end="", flush=True)

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if not (loss_val == loss_val):  # NaN
                diverged = True
                diverged_at_step = epoch * n_batches_per_epoch + batch_idx
                break
            epoch_loss_sum += loss_val
            n_batches += 1
            running_recent = loss_val if batch_idx == 0 else (
                (1 - ema_alpha) * running_recent + ema_alpha * loss_val
            )
            if (batch_idx + 1) % dot_every == 0:
                print(".", end="", flush=True)

        train_loss = epoch_loss_sum / max(n_batches, 1)
        epoch_train_loss.append(train_loss)
        epoch_dt = time.time() - epoch_t0

        if diverged:
            epoch_test_acc.append(float("nan"))
            print(f" DIVERGED @ step {diverged_at_step}, last_loss≈{running_recent:.4g} "
                  f"({epoch_dt:.1f}s)")
            break

        acc = evaluate(model, test_loader, device)
        epoch_test_acc.append(acc)
        print(f" train_loss={train_loss:.4f}  test_acc={acc:.4f}  ({epoch_dt:.1f}s)")

    elapsed = time.time() - t0
    valid_acc = [a for a in epoch_test_acc if a == a]
    if valid_acc:
        best = max(valid_acc)
        final = epoch_test_acc[-1] if (epoch_test_acc[-1] == epoch_test_acc[-1]) else valid_acc[-1]
        print(f"  done: final_acc={final:.4f}  best_acc={best:.4f}  total={elapsed:.1f}s")
    else:
        print(f"  done: NO VALID EPOCHS  total={elapsed:.1f}s")

    return {
        "method": method,
        "wrapper": wrapper,
        "with_bograd": (wrapper == "bograd"),  # legacy compat
        "dataset": dataset_name,
        "lr": lr,
        "K": K if wrapper != "none" else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "diverged": diverged,
        "diverged_at_step": diverged_at_step if diverged else None,
        "final_test_acc": epoch_test_acc[-1] if epoch_test_acc else float("nan"),
        "best_test_acc": max(valid_acc) if valid_acc else float("nan"),
        "epoch_test_acc": epoch_test_acc,
        "epoch_train_loss": epoch_train_loss,
        "wall_clock_s": elapsed,
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_acc(r: Optional[Dict]) -> str:
    if not r:
        return "    -    "
    if r.get("error"):
        return "  ERROR  "
    if r.get("diverged"):
        return " diverged"
    v = r.get("final_test_acc", float("nan"))
    return f"  {v:.4f} " if v == v else "    NaN  "


def _fmt_delta(base: Optional[Dict], bg: Optional[Dict]) -> str:
    if not base or not bg or base.get("error") or bg.get("error"):
        return "    -    "
    if base.get("diverged") and bg.get("diverged"):
        return "  both div"
    if base.get("diverged"):
        return " base div "
    if bg.get("diverged"):
        return "  BG div  "
    a = base.get("final_test_acc", float("nan"))
    b = bg.get("final_test_acc", float("nan"))
    if not (a == a) or not (b == b):
        return "    -    "
    d = b - a
    marker = (
        "++" if d > 0.02 else
        "+" if d > 0.005 else
        "~" if d > -0.005 else
        "-" if d > -0.02 else
        "--"
    )
    return f"{d:+.4f} {marker}"


def print_comparison_tables(
    all_results: List[Dict],
    methods: List[str],
    dataset_name: str,
    lrs: Dict[str, float],
    K: int,
    wrappers: List[str],
) -> None:
    by_key = {(r.get("method"), r.get("wrapper")): r for r in all_results
              if r.get("dataset") == dataset_name}

    print(f"\n=== {dataset_name.upper()} — final test accuracy ===")

    # Build dynamic header based on which wrappers are present.
    wrapper_titles = {
        "none":          "baseline",
        "bograd":        f"+BoGrad K={K}",
        "windowed_mean": f"+WinMean K={K}",
    }
    cells = [f"{'method':<22}", f"{'lr':>7}"]
    for w in wrappers:
        cells.append(f"{wrapper_titles[w]:^14}")
    if "none" in wrappers:
        for w in wrappers:
            if w != "none":
                cells.append(f"{'Δ (' + w + ' − base)':^17}")
    header = "  ".join(cells)
    print(header)
    print("-" * len(header))

    base_present = "none" in wrappers
    for method in methods:
        row_cells = [f"{method:<22}", f"{lrs[method]:>7g}"]
        for w in wrappers:
            r = by_key.get((method, w))
            row_cells.append(f"{_fmt_acc(r):^14}")
        if base_present:
            base = by_key.get((method, "none"))
            for w in wrappers:
                if w == "none":
                    continue
                bg = by_key.get((method, w))
                row_cells.append(f"{_fmt_delta(base, bg):^17}")
        print("  ".join(row_cells))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_WRAPPER_STYLE = {
    "none":          {"linestyle": "-",  "marker": "o", "label_suffix": ""},
    "bograd":        {"linestyle": "--", "marker": "x", "label_suffix": " + BoGrad"},
    "windowed_mean": {"linestyle": ":",  "marker": "s", "label_suffix": " + WinMean"},
}


def plot_results(
    all_results: List[Dict],
    methods: List[str],
    dataset_name: str,
    K: int,
    out_dir: Path,
    wrappers: List[str],
) -> None:
    """One figure per family, with two subplots (train_loss + test_acc).
    Color = base method, linestyle/marker = wrapper kind."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(r.get("method"), r.get("wrapper")): r for r in all_results
              if r.get("dataset") == dataset_name and not r.get("error")}

    methods_set = set(methods)
    for family_name, family_methods in PLOT_FAMILIES.items():
        family_methods_present = [m for m in family_methods if m in methods_set]
        if not family_methods_present:
            continue

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
        any_line = False

        for method in family_methods_present:
            color = METHOD_COLORS.get(method, "tab:gray")
            for w in wrappers:
                r = by_key.get((method, w))
                if not r:
                    continue
                loss = r.get("epoch_train_loss", [])
                acc = r.get("epoch_test_acc", [])
                if not loss and not acc:
                    continue
                style = _WRAPPER_STYLE[w]
                label = method + style["label_suffix"]
                if loss:
                    xs = list(range(1, len(loss) + 1))
                    ax_loss.plot(xs, loss, color=color, linestyle=style["linestyle"],
                                 marker=style["marker"], markersize=5, label=label)
                if acc:
                    xs = list(range(1, len(acc) + 1))
                    ax_acc.plot(xs, acc, color=color, linestyle=style["linestyle"],
                                marker=style["marker"], markersize=5, label=label)
                any_line = True

        if not any_line:
            plt.close(fig)
            continue

        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("train loss")
        ax_loss.set_title(f"{dataset_name} {family_name}: train loss")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="best", fontsize=9)

        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("test accuracy")
        ax_acc.set_title(f"{dataset_name} {family_name}: test accuracy")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc="best", fontsize=9)

        wrappers_label = ",".join(wrappers)
        fig.suptitle(
            f"{dataset_name.upper()} — {family_name} (K={K}, mode=negative, stage=update; wrappers: {wrappers_label})",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fname = out_dir / f"{dataset_name}_{family_name}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  plot: {fname.name}")


# ---------------------------------------------------------------------------
# End-to-end sweep runner
# ---------------------------------------------------------------------------
def run_sweep(
    *,
    dataset_name: str,
    train_dataset,
    test_dataset,
    model_factory: Callable[[], nn.Module],
    methods: List[str],
    lrs: Dict[str, float],
    K: int,
    epochs: int,
    batch_size: int,
    seed: int,
    out_root: Path,
    wrappers: Optional[List[str]] = None,
    num_workers: int = 2,
) -> Path:
    """Run all methods × wrappers on one dataset.
    Default wrappers = ("none", "bograd", "windowed_mean").
    Saves JSON results, prints comparison table, generates plots.
    Returns the per-run output directory.
    """
    if wrappers is None:
        wrappers = list(WRAPPERS)
    for w in wrappers:
        if w not in WRAPPERS:
            raise ValueError(f"unknown wrapper {w!r}; must be one of {WRAPPERS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perf knobs (safe across configurations).
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"device: {device}, dataset={dataset_name}, K={K}, epochs={epochs}, "
          f"batch={batch_size}, seed={seed}")
    print(f"wrappers: {wrappers}")
    print("LRs: " + ", ".join(f"{m}={lrs[m]}" for m in methods))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for method in methods:
        for w in wrappers:
            try:
                res = train_one_run(
                    method=method,
                    wrapper=w,
                    dataset_name=dataset_name,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    model_factory=model_factory,
                    device=device,
                    lr=lrs[method],
                    K=K,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=seed,
                    num_workers=num_workers,
                )
                all_results.append(res)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"  [{method}{_WRAPPER_LABEL[w]} | {dataset_name}] ERROR: {e}")
                print(tb)
                all_results.append({
                    "method": method, "wrapper": w, "with_bograd": (w == "bograd"),
                    "dataset": dataset_name, "error": str(e), "traceback": tb,
                })

    # Save raw results
    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Comparison table
    print_comparison_tables(all_results, methods, dataset_name, lrs, K, wrappers)

    # Plots
    print(f"\nGenerating plots in {out_dir}")
    plot_results(all_results, methods, dataset_name, K, out_dir, wrappers)

    print(f"\nResults written to {out_dir}")
    return out_dir
