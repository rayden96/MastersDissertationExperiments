"""25 — COSGD against the standard optimisers, on the conference study's ladder.

Claim under test
----------------
Section 4.4 predicts that COSGD's scope for acting shrinks as dimension and
class count grow, because the per-class subgradients approach mutual
orthogonality and there is less conflict left to remove. The ablation of
Section 4.6 tests the knobs on CIFAR-10 alone and finds parity. This study
tests the prediction's other end: whether the advantage is present on the
low-dimensional problems where the geometry says it should be largest.

Design
------
Five optimiser arms per dataset: COSGD (wrapping SGD) against plain SGD, Adam,
RMSProp and SignSGD. Datasets, architectures, batch sizes and per-optimiser
learning rates follow the COSGD conference study; the rates there came from a
grid search over {0.1, 0.01, 0.001} per optimiser, so each arm runs at its own
tuned rate rather than a shared one.

Deviations from the conference study, both stated in the chapter:
  - three seeds rather than 10 to 30, because COSGD costs 8 to 10 backward
    passes per step and the budget does not stretch to the paper's trial count
  - 15 epochs on the image sets rather than 7, so that epochs-to-target has
    enough resolution to rank the arms on speed rather than on endpoint

    python run.py                       # everything
    python run.py --datasets iris titanic
    python run.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

from common import storage
from common.datasets import get_dataset
from common.methods import build_method
from common.models import get_model
from common.seeding import seed_everything
from common.training import Trainer, TrainConfig, JobManager

# Per-dataset configuration, from Table I of the conference study. `lr` holds
# the tuned rate for each arm; "other" covers Adam, RMSProp and SignSGD, which
# the paper's grid search settled on the same value for.
LADDER = {
    "iris":          dict(batch=12,  epochs=30, lr={"cosgd": 0.1,  "sgd": 0.1,  "other": 0.001}),
    "titanic":       dict(batch=8,   epochs=15, lr={"cosgd": 0.01, "sgd": 0.01, "other": 0.001}),
    "mnist":         dict(batch=128, epochs=15, lr={"cosgd": 0.01, "sgd": 0.1,  "other": 0.001}),
    "fashion_mnist": dict(batch=128, epochs=15, lr={"cosgd": 0.01, "sgd": 0.1,  "other": 0.001}),
    "cifar10":       dict(batch=128, epochs=15, lr={"cosgd": 0.01, "sgd": 0.1,  "other": 0.001}),
}

# Canonical COSGD, as fixed by the ablation: classical Gram-Schmidt, descending
# magnitude order, summed, no norm cap.
COSGD_HP = dict(cosgd_method="gram_schmidt_normal", class_order="desc",
                combine="sum", combine_norm_cap=0.0)

ARMS = [("cosgd", "sgd"), ("baseline", "sgd"), ("baseline", "adam"),
        ("baseline", "rmsprop"), ("baseline", "signsgd")]


def arm_label(method: str, base: str) -> str:
    return "cosgd" if method == "cosgd" else base


def lr_for(cfg: dict, method: str, base: str) -> float:
    if method == "cosgd":
        return cfg["lr"]["cosgd"]
    return cfg["lr"]["sgd"] if base == "sgd" else cfg["lr"]["other"]


def run_dataset(name: str, seeds, epochs=None, batch=None, measure=False):
    cfg = LADDER[name]
    epochs = epochs or cfg["epochs"]
    batch = batch or cfg["batch"]

    bundle = get_dataset(name)
    meta = bundle.meta
    num_classes = int(meta["num_classes"])
    model_name = meta["model"]
    mk = dict(meta.get("model_kwargs") or {})

    out_root = storage.persistent_dir(f"25_cosgd_bakeoff/{name}")
    sig = storage.config_hash({"exp": "25_cosgd_bakeoff", "dataset": name,
                               "arms": [arm_label(m, b) for m, b in ARMS],
                               "seeds": list(seeds), "epochs": epochs,
                               "batch_size": batch})
    run_dir = Path(out_root) / f"run_{sig}"
    jm = JobManager(run_dir / "cells")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.CrossEntropyLoss()
    print(f"\n[{name}] model={model_name} classes={num_classes} "
          f"epochs={epochs} batch={batch} device={device}", flush=True)

    rows = []
    for method, base in ARMS:
        label = arm_label(method, base)
        lr = lr_for(cfg, method, base)
        hp = {"lr": lr}
        if method == "cosgd":
            hp.update(COSGD_HP)

        for seed in seeds:
            # _curves parses "<base>__<cell>__seed<n>", so the arm name has to
            # sit in the middle field for the shared loader to pick it up.
            key = f"{base}__{label}__seed{seed}"
            cell = jm.run_dir_for(key)
            if jm.is_done(key, hp=hp, batch_size=batch, epochs=epochs):
                print(f"  skip (done) {key}", flush=True)
                continue

            seed_everything(seed)
            spec = build_method(method, base, hp=dict(hp))
            model = get_model(model_name, num_classes=num_classes,
                              **{**mk, **spec.model_kwargs}).to(device)

            tcfg = TrainConfig(
                experiment=f"25_cosgd_bakeoff/{name}",
                dataset=name, model=model_name, method=method,
                base_optimizer=base, num_classes=num_classes, epochs=epochs,
                batch_size=batch, seed=seed, trial_index=0, hp=hp,
                model_kwargs=mk, log_every_n_steps=50,
                checkpoint_every_n_steps=0, num_workers=0,
            )
            t = Trainer(tcfg, spec, model, bundle.train, bundle.val, bundle.test,
                        cell, device, criterion=crit, meter=None, summarize=False)
            print(f"  run {key}  lr={lr}", flush=True)
            res = t.run()
            rows.append({"dataset": name, "arm": label, "seed": seed,
                         "final_test_acc": res["scalars"]["final_test_acc"]})

    storage.write_json_atomic(run_dir / "rows.json", rows)
    print(f"[{name}] -> {run_dir}", flush=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LADDER))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.datasets = ["iris"]; args.seeds = [2026]; args.epochs = 3

    t0 = time.time()
    failed = []
    for name in args.datasets:
        if name not in LADDER:
            print(f"!! unknown dataset {name}"); continue
        try:
            run_dataset(name, args.seeds, args.epochs, args.batch)
        except Exception as e:
            import traceback
            print(f"!! {name} raised {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed.append(name)
    print(f"\n[25_cosgd_bakeoff] done in {(time.time()-t0)/60:.1f} min")
    if failed:
        print(f"[25_cosgd_bakeoff] FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
