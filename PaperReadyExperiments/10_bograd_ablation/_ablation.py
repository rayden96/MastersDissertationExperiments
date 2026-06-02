"""
Shared BoGrad-ablation harness.

Each axis under 10_bograd_ablation/ is a thin run.py that declares a sweep grid
and calls `run_bograd_sweep(...)`. The harness:

  - builds the workhorse (or any) dataset + model from the registries,
  - for each (base optimizer x HP-cell x seed) trains via common.training.Trainer
    with an InterferenceMeter attached (so every accuracy point comes with the
    interference metric that explains it),
  - is resumable: a completed cell's results.json is skipped (JobManager),
  - aggregates mean +/- std across seeds into an axis-level summary.json.

Why a harness and not per-axis copies: CLAUDE.md forbids pasting helpers between
experiments. Every axis shares this exact training + measurement path, so the
only thing that differs between 10.01..10.09 is the grid each run.py declares.

Fixed-by-design across the whole BoGrad chapter: project_stage="update"
(FocusedWork S04 sums applied updates -> the applied update is the projection
target, and it is the correct stage for momentum/preconditioned bases). This is
not swept.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent                      # PaperReadyExperiments/
_REPO = _PRE.parent                      # repo root
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                    # noqa: E402
from common.datasets import get_dataset, balanced_reference_subset  # noqa: E402
from common.methods import build_method                       # noqa: E402
from common.models import get_model                           # noqa: E402
from common.training import Trainer, TrainConfig, JobManager  # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

from interference.meter import InterferenceMeter              # noqa: E402
from interference.summary import summarize_run                # noqa: E402
from interference.torch_classification import TorchClassificationProblem  # noqa: E402


# Headline interference fields pulled into every axis summary (the "why").
INTERFERENCE_HEADLINE = [
    "I_inter_mean", "inter_mean_cos_mean",
    "I_between_K4_mean", "I_between_K32_mean", "I_between_K128_mean",
    "between_K32_mean_cos_mean", "between_K32_useful_path_frac_mean",
    "cum_deficit", "mean_deficit_per_step",
    "cum_deficit_precond", "mean_deficit_precond_per_step",
    "corr_I_between_K32_vs_Dt",
]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cell_key(base: str, cell_label: str, seed: int) -> str:
    return f"{base}__{cell_label}__seed{seed}"


def run_bograd_sweep(
    *,
    axis_name: str,
    cells: Sequence[Dict[str, Any]],
    bases: Sequence[str] = ("sgd", "signsgd", "rmsprop", "adam"),
    dataset: str = "cifar10",
    seeds: Sequence[int] = (2026, 2027, 2028),
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    out_root: Optional[Path] = None,
    log_every: int = 50,
    ref_refresh_every: int = 100,
    K_values: Sequence[int] = (4, 32, 128),
    ref_n_per_class: int = 100,
    train_subset: Optional[int] = None,
    measure: bool = True,
    num_workers: int = 2,
    extra_hp: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run one ablation axis.

    Parameters
    ----------
    axis_name : folder tag, e.g. "10_01_buffer_K".
    cells : list of dicts, each {"label": str, "method": "baseline"|"bograd",
            "hp": {...}}. The method is usually "bograd"; include one "baseline"
            cell per base where the axis wants the no-BoGrad reference.
    bases : base optimizers to cross with every cell.
    dataset : registry name (workhorse default cifar10).
    epochs/batch_size : override the dataset meta defaults (ablations run short).
    train_subset : cap train size for fast sweeps (None = full train split).
    measure : attach the InterferenceMeter (set False for pure timing axes).
    extra_hp : merged into every cell's hp (e.g. a fixed lr for an axis that
               isn't sweeping lr).

    Returns the axis output directory. Writes per-cell run dirs + summary.json.
    """
    device = _device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    bundle = get_dataset(dataset, val_fraction=0.1, seed=2026)
    meta = bundle.meta
    model_name = meta["model"]
    num_classes = meta["num_classes"]
    epochs = epochs if epochs is not None else meta["epochs"]
    batch_size = batch_size if batch_size is not None else meta["batch_size"]

    train_ds = bundle.train
    if train_subset is not None:
        from torch.utils.data import Subset
        idx = list(range(min(train_subset, len(train_ds))))
        train_ds = Subset(train_ds, idx)

    ref_ds = balanced_reference_subset(train_ds, num_classes=num_classes,
                                       n_per_class=ref_n_per_class, seed=2026)

    out_root = out_root or (_HERE / axis_name.split("_", 2)[-1] / "results")
    run_id = storage.new_run_id()
    axis_dir = Path(out_root) / f"run_{run_id}"
    jm = JobManager(axis_dir / "cells")
    axis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{axis_name}] dataset={dataset} model={model_name} nc={num_classes} "
          f"epochs={epochs} batch={batch_size} device={device}")
    print(f"[{axis_name}] bases={list(bases)} cells={[c['label'] for c in cells]} "
          f"seeds={list(seeds)} -> {len(bases)*len(cells)*len(seeds)} runs")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for base in bases:
        for cell in cells:
            label = cell["label"]
            method = cell.get("method", "bograd")
            for seed in seeds:
                key = _cell_key(base, label, seed)
                run_dir = jm.run_dir_for(key)
                if jm.is_done(key):
                    res = storage.read_json(run_dir / "results.json")
                    rows.append(_row_from_result(base, label, seed, res))
                    print(f"  skip (done) {key}")
                    continue

                hp = dict(cell.get("hp", {}))
                if extra_hp:
                    for k, v in extra_hp.items():
                        hp.setdefault(k, v)
                lr = hp.get("lr", None)

                spec = build_method(method, base, hp=hp)
                # Merge the DATASET's model kwargs (e.g. in_features for MLPs,
                # vocab_size for text) with the METHOD's (e.g. dropout_p); method
                # overrides dataset. Without this, tabular/text models build with
                # the wrong input dim.
                mk = {**meta.get("model_kwargs", {}), **spec.model_kwargs}
                model = get_model(model_name, num_classes=num_classes, **mk)
                crit = torch.nn.CrossEntropyLoss()

                meter = summarize = None
                if measure:
                    ref_loader = DataLoader(ref_ds, batch_size=256, shuffle=False,
                                            num_workers=0)
                    prob = TorchClassificationProblem(model, crit, ref_loader, device)
                    eff_lr = lr if lr is not None else _fallback_lr(base)
                    meter = InterferenceMeter(prob, lr=eff_lr, K_values=list(K_values),
                                              log_every=log_every,
                                              ref_refresh_every=ref_refresh_every)

                    def summarize(m):
                        return summarize_run(
                            m.logs, m.calibration_logs, K_values=list(K_values),
                            cum_deficit=m.cum_deficit, cum_deficit_count=m.cum_deficit_count,
                            cum_deficit_precond=m.cum_deficit_precond,
                            cum_deficit_precond_count=m.cum_deficit_precond_count)

                cfg = TrainConfig(
                    experiment=f"10_bograd_ablation/{axis_name}",
                    dataset=dataset, model=model_name, method=method,
                    base_optimizer=base, num_classes=num_classes, epochs=epochs,
                    batch_size=batch_size, seed=seed, trial_index=0, hp=hp,
                    model_kwargs=mk, log_every_n_steps=log_every,
                    checkpoint_every_n_steps=1000, num_workers=num_workers,
                )
                t = Trainer(cfg, spec, model, train_ds, bundle.val, bundle.test,
                            run_dir, device, criterion=crit, meter=meter,
                            summarize=summarize)
                if measure:
                    base_opt = getattr(t.optimizer, "base_optimizer", t.optimizer)
                    meter.problem.set_precond_optimizer(base_opt)

                print(f"  run {key}  hp={hp}", flush=True)
                res = t.run()
                rows.append(_row_from_result(base, label, seed, res))

    summary = _aggregate(axis_name, dataset, rows)
    storage.write_json_atomic(axis_dir / "summary.json", summary)
    storage.write_json_atomic(axis_dir / "rows.json", rows)
    print(f"\n[{axis_name}] done in {time.time()-t0:.1f}s -> {axis_dir}")
    _print_summary(summary)
    return axis_dir


def _fallback_lr(base: str) -> float:
    return {"sgd": 0.05, "signsgd": 1e-3, "rmsprop": 1e-3, "adam": 1e-3}[base]


def _row_from_result(base, label, seed, res) -> Dict[str, Any]:
    sc = res.get("scalars", {})
    row = {
        "base": base, "cell": label, "seed": seed,
        "final_test_acc": sc.get("final_test_acc"),
        "best_test_acc": sc.get("best_test_acc"),
        "final_val_acc": sc.get("final_val_acc"),
        "mean_step_wall_time_s": sc.get("mean_step_wall_time_s"),
    }
    interf = res.get("interference", {})
    for k in INTERFERENCE_HEADLINE:
        if k in interf:
            row[k] = interf[k]
    return row


def _aggregate(axis_name, dataset, rows) -> Dict[str, Any]:
    by_cell: Dict[tuple, List[Dict]] = {}
    for r in rows:
        by_cell.setdefault((r["base"], r["cell"]), []).append(r)

    def agg(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    cells_out = []
    metric_keys = ["final_test_acc", "best_test_acc", "mean_step_wall_time_s",
                   *INTERFERENCE_HEADLINE]
    for (base, cell), rs in sorted(by_cell.items()):
        entry = {"base": base, "cell": cell,
                 "metrics": {k: agg([r.get(k) for r in rs]) for k in metric_keys}}
        cells_out.append(entry)

    return {"axis": axis_name, "dataset": dataset, "n_rows": len(rows),
            "cells": cells_out}


def _print_summary(summary) -> None:
    print(f"\n=== {summary['axis']} summary (mean test_acc +/- std) ===")
    for e in summary["cells"]:
        acc = e["metrics"]["final_test_acc"]
        Ib = e["metrics"].get("I_between_K32_mean", {})
        print(f"  {e['base']:<8} {e['cell']:<22} acc={acc['mean']:.4f}+/-{acc['std']:.4f} "
              f"(n={acc['n']})  I_between_K32={Ib.get('mean', float('nan')):.3f}")


__all__ = ["run_bograd_sweep", "INTERFERENCE_HEADLINE"]
