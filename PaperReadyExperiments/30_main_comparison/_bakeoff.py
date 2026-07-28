"""
Main-comparison (bakeoff) harness — the Gap-3 core.

The bakeoff differs from the ablations: it is **tune-then-run**. For each
(dataset x base optimizer x method) cell:

  1. TUNE on the validation split (common.tuning.tune_cell) over a small HP grid
     bounded by the M1/M2 ablation winners (lr always; K for bograd; dropout_p
     for dropout; GS variant/combine for cosgd; leak for graddrop). Cached.
  2. RUN the tuned config for `n_seeds` seeds with PAIRED data order, evaluating
     on the held-out TEST split, with the InterferenceMeter attached.

The five method arms {baseline, cosgd, bograd, graddrop, dropout} x four base
optimizers x six datasets, identical architecture + schedule per dataset.

This module produces ONE canonical per-cell record set under
30_main_comparison/_core/results/<dataset>/. The thesis output axes (30.01
trajectories, 30.02 budget tables, 30.03 final bars, 30.09 Pareto) are all VIEWS
over these records — they read, never retrain. The sensitivity / scale / grad-stat
axes (30.04-30.08) are their own focused sweeps.

Rigor controls (docs/experiment_design.md §7): paired_shuffle order is identical
across all methods within a (dataset, seed); val-only tuning; test-only reporting;
git_sha + resolved hp recorded per run; resumable via JobManager.
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

from common import storage                                       # noqa: E402
from common.datasets import get_dataset, balanced_reference_subset  # noqa: E402
from common.methods import build_method, METHODS, BASE_OPTIMIZERS  # noqa: E402
from common.models import get_model                              # noqa: E402
from common.training import Trainer, TrainConfig, JobManager     # noqa: E402
from common.tuning import tune_cell, grid                        # noqa: E402
from torch.utils.data import DataLoader                          # noqa: E402

from interference.meter import InterferenceMeter                 # noqa: E402
from interference.summary import summarize_run                   # noqa: E402
from interference.torch_classification import TorchClassificationProblem  # noqa: E402


# COSGD is excluded on many-class datasets (EMNIST-47, CIFAR-100): its per-step
# cost grows super-linearly in the class count (measured by 20.07 scalability),
# so running it there would be an impractical bill for a foregone conclusion.
# The exclusion is a stated protocol rule in Chapter 6, justified by Chapter 4's
# scalability section — not a silent omission.
COSGD_MAX_CLASSES = 10


# ---------------------------------------------------------------------------
# Per-(base, method) HP search space — bounded by the ablation winners.
# Keep small: the bakeoff tunes MANY cells. lr is the universal axis; method
# knobs default to the ablation-recommended values and get a tiny local grid.
# ---------------------------------------------------------------------------
def hp_axes_for(base: str, method: str) -> Dict[str, List[Any]]:
    lr_grid = {
        "sgd":     [0.02, 0.05, 0.1],
        "signsgd": [3e-4, 1e-3, 3e-3],
        "rmsprop": [3e-4, 1e-3, 3e-3],
        "adam":    [3e-4, 1e-3, 3e-3],
    }[base]
    axes: Dict[str, List[Any]] = {"lr": lr_grid}
    if method == "bograd":
        # K bracketed around the MEASURED 10.01 optima (sgd 16, signsgd 32,
        # rmsprop 32, adam 128), not the older D3 priors. The rmsprop bracket
        # used to be [8,16], which excluded its own measured optimum of 32 and
        # so could never reproduce the ablation's recommended setting.
        axes["K"] = {"sgd": [8, 16, 32], "signsgd": [16, 32, 64],
                     "rmsprop": [16, 32, 64], "adam": [64, 128]}[base]
        axes["projection_mode"] = ["negative"]
    elif method == "cosgd":
        # RECLAIMED COSGD (03_cosgd_scrutiny/FINDINGS.md): the conference-paper
        # algorithm (full GS + desc sort + combine="sum") + a norm cap that
        # preserves the low-dim speedup and prevents high-dim divergence. Tune
        # only the cap (2.0 best across the dim ladder; 3.0 for safety margin).
        axes["cosgd_method"] = ["gram_schmidt_normal"]
        axes["class_order"] = ["desc"]
        axes["combine"] = ["sum"]
        axes["combine_norm_cap"] = [2.0, 3.0]
    elif method == "graddrop":
        axes["leak"] = [0.0]
    elif method == "dropout":
        axes["dropout_p"] = [0.1, 0.3]
    return axes


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_meter(model, crit, ref_ds, lr, K_values, log_every, ref_refresh_every, device):
    ref_loader = DataLoader(ref_ds, batch_size=256, shuffle=False, num_workers=0)
    prob = TorchClassificationProblem(model, crit, ref_loader, device)
    return InterferenceMeter(prob, lr=lr, K_values=list(K_values),
                             log_every=log_every, ref_refresh_every=ref_refresh_every)


def run_bakeoff_cell(
    *,
    dataset: str,
    base: str,
    method: str,
    bundle,
    ref_ds,
    out_root: Path,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    device,
    tune: bool = True,
    tune_epochs: Optional[int] = None,
    measure: bool = True,
    fixed_hp: Optional[Dict[str, Any]] = None,
    batch_size_override: Optional[int] = None,
    K_values: Sequence[int] = (4, 32, 128),
    log_every: int = 50,
    ref_refresh_every: int = 100,
) -> List[Dict[str, Any]]:
    """Tune (on val) then run multi-seed (on test, paired order) for one cell.
    Returns the per-seed result rows; writes per-seed run dirs + a cell.json.

    `fixed_hp` (when given, with tune=False) is the EXACT hp dict to run — used by
    the sensitivity sweeps that pin a specific (lr, K, ...) rather than picking a
    winner. `batch_size_override` lets the batch-size sweep vary it per point."""
    meta = bundle.meta
    model_name = meta["model"]
    num_classes = meta["num_classes"]
    if batch_size_override is not None:
        batch_size = batch_size_override

    jm = JobManager(out_root / "cells")
    cell_tag = f"{base}__{method}"

    # The dataset's recipe weight decay is part of the benchmark, not a tuned
    # knob: pin it as a single-value axis so tuning and the final runs agree.
    axes = hp_axes_for(base, method)
    wd = meta.get("weight_decay")
    if wd:
        axes["weight_decay"] = [wd]

    # 1) tune on val (cached) ------------------------------------------------
    best_hp: Dict[str, Any] = {}
    if fixed_hp is not None:
        best_hp = dict(fixed_hp)
    elif tune:
        tr = tune_cell(
            dataset=dataset, method=method, base=base, model_name=model_name,
            num_classes=num_classes, train_dataset=bundle.train, val_dataset=bundle.val,
            hp_axes=axes,
            out_dir=out_root / "tune" / cell_tag, device=device,
            epochs=(tune_epochs or max(3, epochs // 3)), batch_size=batch_size,
            tune_seeds=[seeds[0]], search="grid",
            model_kwargs=meta.get("model_kwargs", {}),
        )
        best_hp = tr.best_hp
        print(f"  [{dataset}/{cell_tag}] tuned -> {best_hp} (val {tr.best_val_acc:.4f})", flush=True)
    else:
        best_hp = {k: v[len(v) // 2] for k, v in axes.items()}  # middle of each grid
    if wd:
        best_hp.setdefault("weight_decay", wd)

    # 2) multi-seed final runs on test, paired order -------------------------
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        key = f"{cell_tag}__seed{seed}"
        run_dir = jm.run_dir_for(key)
        if jm.is_done(key):
            rows.append(_row(dataset, base, method, seed, storage.read_json(run_dir / "results.json")))
            print(f"    skip (done) {key}"); continue

        spec = build_method(method, base, hp=best_hp)
        # Merge dataset model kwargs (in_features / vocab_size / ...) with the
        # method's; method overrides. Without this, tabular/text models build
        # with the wrong input dim.
        mk = {**meta.get("model_kwargs", {}), **spec.model_kwargs}
        model = get_model(model_name, num_classes=num_classes, **mk)
        crit = torch.nn.CrossEntropyLoss()
        meter = summarize = None
        if measure:
            meter = _make_meter(model, crit, ref_ds, best_hp.get("lr", 0.05),
                                K_values, log_every, ref_refresh_every, device)

            def summarize(m):
                return summarize_run(m.logs, m.calibration_logs, K_values=list(K_values),
                                     cum_deficit=m.cum_deficit, cum_deficit_count=m.cum_deficit_count,
                                     cum_deficit_precond=m.cum_deficit_precond,
                                     cum_deficit_precond_count=m.cum_deficit_precond_count)

        cfg = TrainConfig(
            experiment=f"30_main_comparison/{dataset}", dataset=dataset, model=model_name,
            method=method, base_optimizer=base, num_classes=num_classes, epochs=epochs,
            batch_size=batch_size, seed=seed, trial_index=0, hp=best_hp,
            model_kwargs=mk, log_every_n_steps=log_every,
            checkpoint_every_n_steps=1000, num_workers=2, paired=True,
        )
        t = Trainer(cfg, spec, model, bundle.train, bundle.val, bundle.test,
                    run_dir, device, criterion=crit, meter=meter, summarize=summarize)
        if measure:
            base_opt = getattr(t.optimizer, "base_optimizer", t.optimizer)
            meter.problem.set_precond_optimizer(base_opt)
        print(f"    run {key}  hp={best_hp}", flush=True)
        rows.append(_row(dataset, base, method, seed, t.run()))

    storage.write_json_atomic(out_root / f"cell_{cell_tag}.json",
                              {"dataset": dataset, "base": base, "method": method,
                               "best_hp": best_hp, "rows": rows})
    return rows


def run_bakeoff(
    *,
    datasets: Sequence[str],
    bases: Sequence[str] = BASE_OPTIMIZERS,
    methods: Sequence[str] = METHODS,
    seeds: Sequence[int] = (2026, 2027, 2028, 2029, 2030),
    out_root: Optional[Path] = None,
    campaign: Optional[str] = None,
    epochs: Optional[int] = None,
    tune: bool = True,
    measure: bool = True,
    ref_n_per_class: int = 100,
) -> Path:
    """Full bakeoff over datasets x bases x methods. Writes one canonical record
    set the view-axes (30.01/02/03/09) read."""
    device = _device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    out_root = out_root or (_HERE / "_core" / "results")
    # A fixed campaign name lets per-dataset runs in CONCURRENT Colab sessions
    # accumulate into ONE record set: each dataset writes its own subfolder, so
    # there is no cross-session collision. Default "main"; pass None for a
    # timestamped one-off.
    campaign_dir = Path(out_root) / (campaign or f"run_{storage.new_run_id()}")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bakeoff] {len(datasets)}ds x {len(bases)}opt x {len(methods)}method x {len(seeds)}seed "
          f"= {len(datasets)*len(bases)*len(methods)*len(seeds)} runs -> {campaign_dir}")

    all_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        # The bakeoff is the head-to-head evaluation, so the image benchmarks
        # run the STANDARD recipe (crop + flip + weight decay). Without it a
        # ResNet-18 memorises CIFAR-100 by epoch ~15 and every arm saturates at
        # the same overfitted ceiling, which measures the recipe rather than the
        # optimiser. The ablation chapters deliberately keep augment=False.
        bundle = get_dataset(dataset, val_fraction=0.1, seed=2026, augment=True)
        ep = epochs if epochs is not None else bundle.meta["epochs"]
        bs = bundle.meta["batch_size"]
        # Reference gradients must be measured on a deterministic view of the
        # training data, never on randomly cropped samples.
        ref_ds = balanced_reference_subset(bundle.train_eval or bundle.train,
                                           num_classes=bundle.meta["num_classes"],
                                           n_per_class=ref_n_per_class, seed=2026)
        ds_dir = campaign_dir / dataset
        for base in bases:
            for method in methods:
                if method == "cosgd" and bundle.meta["num_classes"] > COSGD_MAX_CLASSES:
                    print(f"  [{dataset}] skip cosgd: {bundle.meta['num_classes']} classes "
                          f"> {COSGD_MAX_CLASSES} (excluded by the 20.07 scalability wall)", flush=True)
                    continue
                t0 = time.time()
                rows = run_bakeoff_cell(
                    dataset=dataset, base=base, method=method, bundle=bundle, ref_ds=ref_ds,
                    out_root=ds_dir, seeds=seeds, epochs=ep, batch_size=bs, device=device,
                    tune=tune, measure=measure,
                )
                all_rows.extend(rows)
                print(f"  [{dataset}/{base}+{method}] {len(rows)} seeds in {time.time()-t0:.0f}s", flush=True)

    # Per-session convenience snapshot. NOT authoritative across concurrent
    # sessions (each would overwrite it); views.py aggregates the per-cell
    # cell_*.json files instead, which never collide.
    storage.write_json_atomic(campaign_dir / "all_rows.json", all_rows)
    print(f"\n[bakeoff] done -> {campaign_dir}")
    return campaign_dir


def _row(dataset, base, method, seed, res) -> Dict[str, Any]:
    sc = res.get("scalars", {})
    row = {"dataset": dataset, "base": base, "method": method, "seed": seed,
           "label": res.get("label"),
           "final_test_acc": sc.get("final_test_acc"),
           "best_test_acc": sc.get("best_test_acc"),
           "final_val_acc": sc.get("final_val_acc"),
           "final_train_loss": sc.get("final_train_loss"),
           # FOGO-style cost columns (memory / timing / size); most table fields
           # are derived post-hoc in views.py from these + the accuracy curves.
           "mean_step_wall_time_s": sc.get("mean_step_wall_time_s"),
           "mean_train_step_s": sc.get("mean_train_step_s"),
           "total_wall_time_s": sc.get("total_wall_time_s"),
           "peak_mem_mb": sc.get("peak_mem_mb"),
           "n_params": sc.get("n_params"),
           "total_steps": res.get("total_steps"),
           "total_epochs": res.get("total_epochs"),
           "epoch_test_acc": res.get("history", {}).get("epoch_test_acc", []),
           "epoch_train_loss": res.get("history", {}).get("epoch_train_loss", []),
           "hp": res.get("config", {}).get("hp", {})}
    interf = res.get("interference", {})
    for k in ("I_inter_mean", "inter_mean_cos_mean", "I_between_K32_mean",
              "between_K32_mean_cos_mean", "cum_deficit", "mean_deficit_per_step",
              "mean_deficit_precond_per_step"):
        if k in interf:
            row[k] = interf[k]
    return row


__all__ = ["run_bakeoff", "run_bakeoff_cell", "hp_axes_for"]
