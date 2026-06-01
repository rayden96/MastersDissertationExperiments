"""
common.tuning — hyperparameter search over the VALIDATION split.

The bakeoff (Gap 3) requires *independent* HP tuning per (dataset × base × method),
selected on val and reported on test. This module runs that search with the same
Trainer/registry machinery so tuned configs are produced consistently, caches the
winner per cell, and is resumable (skips trials whose results.json exists).

Entry points
------------
  grid(**axes)                      -> list[dict]   cartesian product of HP axes
  random_grid(axes, n, seed)        -> list[dict]   n random draws from axes
  tune_cell(...)                    -> TuneResult   search one cell, cache best

`tune_cell` returns the best HP dict (by mean val accuracy over `tune_seeds`)
and writes a `best.json` so re-runs are instant. The caller then trains the
final multi-seed run at the returned HP on the test split.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from common import storage
from common.methods import build_method
from common.training import Trainer, TrainConfig


def grid(**axes) -> List[Dict[str, Any]]:
    """Cartesian product of HP axes. grid(lr=[...], K=[...]) -> list of dicts."""
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*[axes[k] for k in keys])]


def random_grid(axes: Dict[str, List[Any]], n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """`n` random draws (with replacement across axes, deduped) from HP axes."""
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        cand = {k: v[int(rng.integers(len(v)))] for k, v in axes.items()}
        key = tuple(sorted(cand.items(), key=lambda kv: str(kv)))
        if key not in seen:
            seen.add(key); out.append(cand)
        attempts += 1
    return out


@dataclass
class TuneResult:
    best_hp: Dict[str, Any]
    best_val_acc: float
    table: List[Dict[str, Any]] = field(default_factory=list)


def tune_cell(
    *,
    dataset: str,
    method: str,
    base: str,
    model_name: str,
    num_classes: int,
    train_dataset,
    val_dataset,
    hp_axes: Dict[str, List[Any]],
    out_dir: Path,
    device,
    epochs: int,
    batch_size: int,
    tune_seeds: List[int] = (2026,),
    search: str = "grid",
    n_random: int = 12,
    random_seed: int = 0,
    get_model: Optional[Callable] = None,
) -> TuneResult:
    """Search `hp_axes` for one cell; select by mean val accuracy over tune_seeds.

    Tuning trains on `train_dataset`, evaluates on `val_dataset` (passed as the
    Trainer's test slot so its eval/early-stop act on val). The held-out test set
    is never touched here. Results are cached in `out_dir/best.json`.
    """
    if get_model is None:
        from common.models import get_model as _gm
        get_model = _gm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.json"
    if best_path.exists():
        cached = storage.read_json(best_path)
        return TuneResult(cached["best_hp"], cached["best_val_acc"], cached.get("table", []))

    candidates = (grid(**hp_axes) if search == "grid"
                  else random_grid(hp_axes, n_random, random_seed))

    table: List[Dict[str, Any]] = []
    for ci, hp in enumerate(candidates):
        accs = []
        for s in tune_seeds:
            spec = build_method(method, base, hp=hp)
            model = get_model(model_name, num_classes=num_classes, **spec.model_kwargs)
            cfg = TrainConfig(
                experiment=f"_tune/{dataset}/{base}/{method}",
                dataset=dataset, model=model_name, method=method, base_optimizer=base,
                num_classes=num_classes, epochs=epochs, batch_size=batch_size,
                seed=s, trial_index=0, hp=hp, model_kwargs=spec.model_kwargs,
                eval_every_epoch=True, num_workers=0,
            )
            run_dir = out_dir / f"cand{ci}_seed{s}"
            # use val as the Trainer's test slot; a tiny val subset stands in for both
            t = Trainer(cfg, spec, model, train_dataset, val_dataset, val_dataset,
                        run_dir, device)
            res = t.run()
            accs.append(res["scalars"]["final_val_acc"])
        mean_acc = float(np.mean(accs))
        table.append({"hp": hp, "mean_val_acc": mean_acc, "seeds": list(tune_seeds)})
        print(f"    tune[{ci+1}/{len(candidates)}] {hp} -> val_acc {mean_acc:.4f}", flush=True)

    best = max(table, key=lambda r: r["mean_val_acc"])
    result = TuneResult(best["hp"], best["mean_val_acc"], table)
    storage.write_json_atomic(best_path, {
        "best_hp": result.best_hp, "best_val_acc": result.best_val_acc,
        "table": table, "dataset": dataset, "base": base, "method": method,
    })
    return result


__all__ = ["grid", "random_grid", "tune_cell", "TuneResult"]
