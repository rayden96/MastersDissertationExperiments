"""
Shared COSGD-ablation harness.

Sibling of 10_bograd_ablation/_ablation.py, specialised for the per-class
methods (COSGD, GradDrop). Each axis under 20_cosgd_ablation/ is a thin run.py
that declares a sweep grid of COSGD cells and calls `run_cosgd_sweep(...)`.

Differences from the BoGrad harness:
  - method defaults to "cosgd" (step_kind="per_class"); the Trainer already
    dispatches per-class .step(x, y, unique(y)).
  - the headline interference metric is the *inter-batch* I_inter (within-batch
    per-class cancellation) — COSGD's target — plus IB pairwise cosine; the
    between-batch fields are still logged for the COSGD<->BoGrad contrast (20.08).
  - exposes COSGD knobs in each cell's hp: cosgd_method, class_order,
    prenormalize, combine, step_method, collect_timing.

Reuses common.training.Trainer + InterferenceMeter unchanged.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                    # noqa: E402
from common.datasets import get_dataset, balanced_reference_subset, cifar10_subclass_bundle  # noqa: E402
from common.methods import build_method                       # noqa: E402
from common.models import get_model                           # noqa: E402
from common.training import Trainer, TrainConfig, JobManager  # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

from interference.meter import InterferenceMeter              # noqa: E402
from interference.summary import summarize_run                # noqa: E402
from interference.torch_classification import TorchClassificationProblem  # noqa: E402

# Inter-batch is COSGD's axis; keep a couple of between-batch fields for 20.08.
INTERFERENCE_HEADLINE = [
    "I_inter_mean", "inter_frac_neg_mean", "inter_mean_cos_mean",
    "inter_useful_descent_frac_mean", "inter_useful_mass_mean", "inter_wasted_mass_mean",
    "inter_max_min_ratio_mean",
    "I_between_K32_mean", "between_K32_mean_cos_mean",
    "cum_deficit", "mean_deficit_per_step",
    "cum_deficit_precond", "mean_deficit_precond_per_step",
    "corr_I_inter_vs_Dt",
]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fallback_lr(base: str) -> float:
    return {"sgd": 0.05, "signsgd": 1e-3, "rmsprop": 1e-3, "adam": 1e-3}[base]


def run_cosgd_sweep(
    *,
    axis_name: str,
    cells: Sequence[Dict[str, Any]],
    bases: Sequence[str] = ("sgd",),
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
    subclasses: Optional[int] = None,
    extra_hp: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run one COSGD ablation axis.

    cells: each {"label", "method" (default "cosgd"), "hp": {...}}; include one
           "baseline" cell where the axis wants the no-COSGD reference.
    subclasses: if set, use the synthetic CIFAR-10 N-subclass relabelling
           (20.07 scalability); overrides `dataset`.
    """
    device = _device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if subclasses is not None:
        bundle = cifar10_subclass_bundle(n_subclasses=subclasses, seed=2026)
        ds_label = f"cifar10_sub{subclasses}"
    else:
        bundle = get_dataset(dataset, val_fraction=0.1, seed=2026)
        ds_label = dataset
    meta = bundle.meta
    model_name = meta["model"]
    num_classes = meta["num_classes"]
    epochs = epochs if epochs is not None else meta["epochs"]
    batch_size = batch_size if batch_size is not None else meta["batch_size"]

    train_ds = bundle.train
    if train_subset is not None:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))

    # Adapt the reference-set size to small datasets: cap per-class samples at
    # ~1/3 of the smallest class so tiny sets (iris ~34/class) still build a
    # sensible balanced reference instead of silently using everything.
    import numpy as _np
    from common.datasets import _targets_of as _tof
    _counts = _np.bincount(_tof(train_ds), minlength=num_classes)
    _min_cls = int(_counts[_counts > 0].min()) if (_counts > 0).any() else 1
    eff_ref_npc = max(1, min(ref_n_per_class, _min_cls // 3 if _min_cls >= 6 else _min_cls))
    ref_ds = balanced_reference_subset(train_ds, num_classes=num_classes,
                                       n_per_class=eff_ref_npc, seed=2026)

    # Persist under get_results_root() (Drive on Colab) keyed by axis_name, so
    # outputs survive a session end without relying on notebook symlinks. The
    # caller's `out_root` (a local _HERE/results/<ds> path) is used only for its
    # trailing dataset/base sub-parts; the ROOT is always the persistent one.
    if out_root is not None:
        # take sub-parts after the axis 'results' dir (e.g. '<ds>' or '<ds>/<base>')
        op = Path(out_root)
        sub = op.parts[op.parts.index("results") + 1:] if "results" in op.parts else ()
        out_root = storage.persistent_dir(f"20_cosgd_ablation/{axis_name}", *sub)
    else:
        out_root = storage.persistent_dir(f"20_cosgd_ablation/{axis_name}")
    # STABLE run dir keyed by the sweep config (NOT a fresh timestamp), so a
    # re-run reuses the same cells/ and the JobManager SKIPS completed cells
    # instead of starting over.
    cfg_sig = storage.config_hash({
        "axis": axis_name, "dataset": ds_label, "bases": list(bases),
        "seeds": list(seeds), "epochs": epochs, "batch_size": batch_size,
        "cells": [c.get("label") for c in cells],
    })
    axis_dir = Path(out_root) / f"run_{cfg_sig}"
    jm = JobManager(axis_dir / "cells")
    axis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{axis_name}] dataset={ds_label} model={model_name} nc={num_classes} "
          f"epochs={epochs} batch={batch_size} device={device}")
    print(f"[{axis_name}] bases={list(bases)} cells={[c['label'] for c in cells]} "
          f"seeds={list(seeds)} -> {len(bases)*len(cells)*len(seeds)} runs")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for base in bases:
        for cell in cells:
            label = cell["label"]
            method = cell.get("method", "cosgd")
            hp_cell = dict(cell.get("hp", {}))
            if extra_hp:
                for k, v in extra_hp.items():
                    hp_cell.setdefault(k, v)
            # batch size is a training hyperparameter, not a method knob, so a
            # cell may override the sweep default without it reaching
            # build_method(). Recorded on TrainConfig for provenance.
            cell_bs = int(hp_cell.pop("batch_size", batch_size))
            lr = hp_cell.get("lr", None)

            for seed in seeds:
                key = f"{base}__{label}__seed{seed}"
                run_dir = jm.run_dir_for(key)
                # hp is part of the identity check: a cell whose definition has
                # changed since the last launch must re-run, not be skipped
                # because its label is unchanged.
                if jm.is_done(key, hp=hp_cell, batch_size=cell_bs, epochs=epochs):
                    res = storage.read_json(run_dir / "results.json")
                    rows.append(_row(base, label, seed, res)); print(f"  skip (done) {key}")
                    continue

                hp = dict(hp_cell)
                spec = build_method(method, base, hp=hp)
                # Merge the DATASET's model kwargs (e.g. in_features for MLPs)
                # with the METHOD's (e.g. dropout_p); method overrides dataset.
                mk = {**meta.get("model_kwargs", {}), **spec.model_kwargs}
                model = get_model(model_name, num_classes=num_classes, **mk)
                crit = torch.nn.CrossEntropyLoss()

                meter = summarize = None
                if measure:
                    ref_loader = DataLoader(ref_ds, batch_size=256, shuffle=False, num_workers=0)
                    prob = TorchClassificationProblem(model, crit, ref_loader, device)
                    eff_lr = lr if lr is not None else _fallback_lr(base)
                    meter = InterferenceMeter(prob, lr=eff_lr, K_values=list(K_values),
                                              log_every=log_every, ref_refresh_every=ref_refresh_every)

                    def summarize(m):
                        return summarize_run(
                            m.logs, m.calibration_logs, K_values=list(K_values),
                            cum_deficit=m.cum_deficit, cum_deficit_count=m.cum_deficit_count,
                            cum_deficit_precond=m.cum_deficit_precond,
                            cum_deficit_precond_count=m.cum_deficit_precond_count)

                cfg = TrainConfig(
                    experiment=f"20_cosgd_ablation/{axis_name}",
                    dataset=ds_label, model=model_name, method=method,
                    base_optimizer=base, num_classes=num_classes, epochs=epochs,
                    batch_size=cell_bs, seed=seed, trial_index=0, hp=hp,
                    model_kwargs=mk, log_every_n_steps=log_every,
                    checkpoint_every_n_steps=1000, num_workers=num_workers,
                )
                t = Trainer(cfg, spec, model, train_ds, bundle.val, bundle.test,
                            run_dir, device, criterion=crit, meter=meter, summarize=summarize)
                if measure:
                    base_opt = getattr(t.optimizer, "base_optimizer", t.optimizer)
                    meter.problem.set_precond_optimizer(base_opt)

                print(f"  run {key}  hp={hp}", flush=True)
                res = t.run()
                rows.append(_row(base, label, seed, res))

    summary = _aggregate(axis_name, ds_label, rows)
    storage.write_json_atomic(axis_dir / "summary.json", summary)
    storage.write_json_atomic(axis_dir / "rows.json", rows)
    print(f"\n[{axis_name}] done in {time.time()-t0:.1f}s -> {axis_dir}")
    _print_summary(summary)
    return axis_dir


def _row(base, label, seed, res) -> Dict[str, Any]:
    sc = res.get("scalars", {})
    row = {"base": base, "cell": label, "seed": seed,
           "final_test_acc": sc.get("final_test_acc"),
           "best_test_acc": sc.get("best_test_acc"),
           "final_val_acc": sc.get("final_val_acc"),
           "mean_step_wall_time_s": sc.get("mean_step_wall_time_s")}
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

    metric_keys = ["final_test_acc", "best_test_acc", "mean_step_wall_time_s", *INTERFERENCE_HEADLINE]
    cells_out = [{"base": b, "cell": c,
                  "metrics": {k: agg([r.get(k) for r in rs]) for k in metric_keys}}
                 for (b, c), rs in sorted(by_cell.items())]
    return {"axis": axis_name, "dataset": dataset, "n_rows": len(rows), "cells": cells_out}


def _print_summary(summary) -> None:
    print(f"\n=== {summary['axis']} summary (mean test_acc +/- std) ===")
    for e in summary["cells"]:
        acc = e["metrics"]["final_test_acc"]
        Ii = e["metrics"].get("I_inter_mean", {})
        print(f"  {e['base']:<8} {e['cell']:<24} acc={acc['mean']:.4f}+/-{acc['std']:.4f} "
              f"(n={acc['n']})  I_inter={Ii.get('mean', float('nan')):.3f}")


__all__ = ["run_cosgd_sweep", "INTERFERENCE_HEADLINE"]
