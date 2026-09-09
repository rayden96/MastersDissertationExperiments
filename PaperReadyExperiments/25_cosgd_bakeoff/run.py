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
RMSProp and SignSGD, each at its own tuned learning rate.

Three things differ from the first version of this study, and all three were
distorting it:

  1. No momentum. `common.methods` gives SGD momentum 0.9 by default, so the
     "SGD" arm was really SGD with momentum, and so was the base step COSGD
     wraps. The conference study used none. Every arm here sets momentum to
     zero explicitly. Adam keeps its own betas, since its first moment is part
     of the method rather than an addition to it.

  2. Real tuning. The conference study searched {0.1, 0.01, 0.001}, three
     points, and the adaptive arms were pinned at the bottom of it. This runs a
     five-point grid at factor-3 spacing per arm, on one seed, and picks each
     arm's rate before the multi-seed comparison starts. When an arm's winner
     lands on an edge of its grid the search extends one point past that edge
     and re-selects, up to twice, so no arm is reported at a rate that is only
     best because the grid stopped there.

  3. Selection on validation, ranked by speed. The rate chosen for an arm is
     the one that reaches the common target in the fewest epochs on the
     validation split, not the one with the highest endpoint, and not anything
     measured on test. The target is 99% of the best final validation accuracy
     any arm reaches at any rate on that dataset, so all five are tuned against
     one bar rather than each against its own ceiling. An arm that never
     reaches it at a given rate is charged the tuning budget plus one, and ties
     are broken by the higher endpoint.

Ladder
------
Iris is dropped: 150 samples and a 30-epoch budget put its epochs-to-target
inside the seed noise. Titanic is kept, since it is the conference study's
tabular problem, but it cannot show a training curve: two classes, an 80%
ceiling that every arm reaches inside two epochs, and one pair of per-class
subgradients for COSGD to act on, which is the smallest intervention the method
can make. Pendigits replaces Iris at the bottom of the ladder and is what
actually carries it: 16 features, so still low-dimensional, but ten classes and
eleven thousand samples, which gives both a graded curve and ten subgradients.

    python run.py --stage tune                        # grid search, 1 seed
    python run.py --stage final                       # 3 seeds at tuned rates
    python run.py --stage both --datasets pendigits
    python run.py --smoke
"""
from __future__ import annotations

import argparse
import json
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

EXP = "25_cosgd_bakeoff_tuned"
TARGET_FRAC = 0.99

# Per-dataset budget. `tune_epochs` is the shortened budget the grid search
# runs at; the multi-seed comparison runs at `epochs`. Ranking rates on speed
# does not need the full budget, and the grid is 21 runs per dataset.
LADDER = {
    "titanic":       dict(batch=8,   epochs=20, tune_epochs=12),
    "pendigits":     dict(batch=32,  epochs=30, tune_epochs=18),
    "mnist":         dict(batch=128, epochs=15, tune_epochs=9),
    "fashion_mnist": dict(batch=128, epochs=20, tune_epochs=12),
    "cifar10":       dict(batch=128, epochs=20, tune_epochs=12),
}

# Learning-rate grids, factor-3 spacing. The image sets get a lower COSGD grid
# because the ablation of Section 4.6 established that COSGD's usable band sits
# about a decade below the baseline's on CIFAR-10; searching {0.3, 0.1} for it
# there would spend two of five runs on rates already known to diverge.
GRID_TABULAR = {
    "cosgd":   [0.3, 0.1, 0.03, 0.01, 0.003],
    "sgd":     [0.3, 0.1, 0.03, 0.01, 0.003],
    "adam":    [0.03, 0.01, 0.003, 0.001, 0.0003],
    "rmsprop": [0.01, 0.003, 0.001, 0.0003, 0.0001],
    "signsgd": [0.003, 0.001, 0.0003, 0.0001, 0.00003],
}
GRID_IMAGE = {
    "cosgd":   [0.1, 0.03, 0.01, 0.003, 0.001],
    "sgd":     [0.3, 0.1, 0.03, 0.01, 0.003],
    "adam":    [0.01, 0.003, 0.001, 0.0003, 0.0001],
    "rmsprop": [0.01, 0.003, 0.001, 0.0003, 0.0001],
    "signsgd": [0.003, 0.001, 0.0003, 0.0001, 0.00003],
}
GRIDS = {"titanic": GRID_TABULAR, "pendigits": GRID_TABULAR,
         "mnist": GRID_IMAGE, "fashion_mnist": GRID_IMAGE, "cifar10": GRID_IMAGE}

# Canonical COSGD, as fixed by the ablation: classical Gram-Schmidt, descending
# magnitude order, summed, no norm cap.
COSGD_HP = dict(cosgd_method="gram_schmidt_normal", class_order="desc",
                combine="sum", combine_norm_cap=0.0)

ARMS = [("cosgd", "sgd"), ("baseline", "sgd"), ("baseline", "adam"),
        ("baseline", "rmsprop"), ("baseline", "signsgd")]

# Bases whose torch constructor takes a `momentum` argument. Adam is absent on
# purpose: its first-moment coefficient is part of the method, not an optional
# accelerator bolted onto it, and zeroing beta_1 would not be Adam.
TAKES_MOMENTUM = {"sgd", "signsgd", "rmsprop"}


def arm_label(method: str, base: str) -> str:
    return "cosgd" if method == "cosgd" else base


def hp_for(method: str, base: str, lr: float) -> dict:
    hp = {"lr": float(lr)}
    if base in TAKES_MOMENTUM:
        hp["momentum"] = 0.0
    if method == "cosgd":
        hp.update(COSGD_HP)
    return hp


def epochs_to(curve, target):
    """First 1-indexed epoch at which `curve` reaches `target`, else None."""
    for i, v in enumerate(curve):
        if v >= target:
            return i + 1
    return None


# Bounds and step for the edge extension. A winner sitting on the edge of its
# grid is not a tuned rate, it is the grid running out, so the search walks one
# factor-3 point outwards and re-selects.
LR_MIN, LR_MAX, MAX_EXTENSIONS = 1e-5, 3.0, 2


def _round_lr(x):
    """Two significant figures, so repeated division by three stays readable."""
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + 1)


def _select(curves, target, budget):
    """Pick each arm's rate: fewest epochs to `target`, endpoint breaks ties."""
    picked, report = {}, {}
    for arm, by_lr in curves.items():
        scored = []
        for lr, (vc, fv) in by_lr.items():
            hit = epochs_to(vc, target)
            # censored rates are charged the budget plus one, so an arm that
            # never reaches the bar still resolves to its closest rate
            scored.append((hit if hit is not None else budget + 1, -fv, lr,
                           hit, fv))
        scored.sort()
        picked[arm] = scored[0][2]
        report[arm] = [{"lr": lr, "epochs": hit, "final_val": fv}
                       for _, _, lr, hit, fv in scored]
    return picked, report


def _train_one(name, bundle, method, base, lr, seed, epochs, batch, cell_dir,
               jm, key, device, crit):
    """Run one cell unless it is already on disk, and return its results dict."""
    meta = bundle.meta
    num_classes = int(meta["num_classes"])
    model_name = meta["model"]
    mk = dict(meta.get("model_kwargs") or {})
    hp = hp_for(method, base, lr)

    if jm.is_done(key, hp=hp, batch_size=batch, epochs=epochs):
        res = json.loads((cell_dir / "results.json").read_text())
        print(f"  skip (done) {key} lr={lr}", flush=True)
        return res

    seed_everything(seed)
    spec = build_method(method, base, hp=dict(hp))
    model = get_model(model_name, num_classes=num_classes,
                      **{**mk, **spec.model_kwargs}).to(device)
    tcfg = TrainConfig(
        experiment=f"{EXP}/{name}", dataset=name, model=model_name,
        method=method, base_optimizer=base, num_classes=num_classes,
        epochs=epochs, batch_size=batch, seed=seed, trial_index=0, hp=hp,
        model_kwargs=mk, log_every_n_steps=50, checkpoint_every_n_steps=0,
        num_workers=0,
    )
    t = Trainer(tcfg, spec, model, bundle.train, bundle.val, bundle.test,
                cell_dir, device, criterion=crit, meter=None, summarize=False)
    print(f"  run {key}  lr={lr}", flush=True)
    return t.run()


# ---------------------------------------------------------------------------
# Stage 1 — the grid search
# ---------------------------------------------------------------------------
def tune_dataset(name, bundle, seed, device, crit, epochs=None, batch=None):
    cfg = LADDER[name]
    epochs = epochs or cfg["tune_epochs"]
    batch = batch or cfg["batch"]
    grid = GRIDS[name]

    out_root = storage.persistent_dir(f"{EXP}/{name}")
    sig = storage.config_hash({"exp": EXP, "stage": "tune", "dataset": name,
                               "grid": {k: list(v) for k, v in grid.items()},
                               "seed": seed, "epochs": epochs,
                               "batch_size": batch})
    tune_dir = Path(out_root) / f"tune_{sig}"
    tune_dir.mkdir(parents=True, exist_ok=True)
    jm = JobManager(tune_dir / "cells")

    print(f"\n[{name}] tuning: {sum(len(v) for v in grid.values())} runs, "
          f"{epochs} epochs, batch {batch}, seed {seed}", flush=True)

    todo = {arm_label(m, b): list(grid[arm_label(m, b)]) for m, b in ARMS}
    curves = {arm: {} for arm in todo}   # curves[arm][lr] = (val curve, final val)

    for rnd in range(MAX_EXTENSIONS + 1):
        for method, base in ARMS:
            arm = arm_label(method, base)
            for lr in sorted(todo[arm], reverse=True):
                if lr in curves[arm]:
                    continue
                key = f"{base}__{arm}_lr{lr:g}__seed{seed}"
                res = _train_one(name, bundle, method, base, lr, seed, epochs,
                                 batch, jm.run_dir_for(key), jm, key, device,
                                 crit)
                vc = res.get("history", {}).get("epoch_val_acc") or []
                fv = res["scalars"].get("final_val_acc", float("nan"))
                curves[arm][lr] = (list(vc), float(fv))

        # One bar for all five arms: 99% of the best endpoint anything reached.
        # It can only rise as the grid widens, so re-selecting after an
        # extension stays consistent with the rounds before it.
        best = max(fv for arm in curves.values() for _, fv in arm.values())
        target = best * TARGET_FRAC
        picked, report = _select(curves, target, epochs)

        if rnd == MAX_EXTENSIONS:
            break
        widened = False
        for arm, by_lr in curves.items():
            lo, hi = min(by_lr), max(by_lr)
            if picked[arm] == hi and hi < LR_MAX:
                nxt = _round_lr(min(hi * 3, LR_MAX))
            elif picked[arm] == lo and lo > LR_MIN:
                nxt = _round_lr(max(lo / 3, LR_MIN))
            else:
                continue
            if nxt not in by_lr:
                todo[arm].append(nxt)
                widened = True
                print(f"  [{arm}] winner {picked[arm]:g} sits on the grid "
                      f"edge; extending to {nxt:g}", flush=True)
        if not widened:
            break

    out = {"dataset": name, "seed": seed, "tune_epochs": epochs,
           "batch_size": batch, "target": target, "best_final_val": best,
           "picked": picked, "detail": report}
    storage.write_json_atomic(tune_dir / "tuning.json", out)
    storage.write_json_atomic(Path(out_root) / "tuned_lr.json", out)

    print(f"\n[{name}] target {target:.4f} (99% of {best:.4f})")
    for arm in ("cosgd", "sgd", "adam", "rmsprop", "signsgd"):
        rows = "  ".join(
            f"{r['lr']:g}:{'--' if r['epochs'] is None else r['epochs']}"
            f"/{r['final_val']:.3f}"
            for r in sorted(report[arm], key=lambda r: -r["lr"]))
        # a winner still on an edge after both extensions is a rate the search
        # could not bracket, and the reader should see that
        flag = " (EDGE)" if picked[arm] in (min(curves[arm]), max(curves[arm])) else ""
        print(f"  {arm:8s} -> lr {picked[arm]:g}{flag}   [{rows}]", flush=True)
    return picked


# ---------------------------------------------------------------------------
# Stage 2 — the multi-seed comparison
# ---------------------------------------------------------------------------
def final_dataset(name, bundle, seeds, device, crit, picked,
                  epochs=None, batch=None):
    cfg = LADDER[name]
    epochs = epochs or cfg["epochs"]
    batch = batch or cfg["batch"]

    out_root = storage.persistent_dir(f"{EXP}/{name}")
    # Deliberately not keyed on `picked`: correcting one arm's rate must
    # re-run that arm inside the existing directory (the JobManager compares
    # hp, so it does) rather than opening a new directory holding only the
    # corrected arm.
    sig = storage.config_hash({"exp": EXP, "stage": "final", "dataset": name,
                               "arms": [arm_label(m, b) for m, b in ARMS],
                               "seeds": list(seeds), "epochs": epochs,
                               "batch_size": batch})
    run_dir = Path(out_root) / f"run_{sig}"
    run_dir.mkdir(parents=True, exist_ok=True)
    jm = JobManager(run_dir / "cells")

    print(f"\n[{name}] final: {epochs} epochs, batch {batch}, "
          f"seeds {list(seeds)}, rates {picked}", flush=True)

    rows = []
    for method, base in ARMS:
        arm = arm_label(method, base)
        lr = picked[arm]
        for seed in seeds:
            # _curves parses "<base>__<cell>__seed<n>", so the arm name has to
            # sit in the middle field for the shared loader to pick it up.
            key = f"{base}__{arm}__seed{seed}"
            res = _train_one(name, bundle, method, base, lr, seed, epochs,
                             batch, jm.run_dir_for(key), jm, key, device, crit)
            rows.append({"dataset": name, "arm": arm, "seed": seed, "lr": lr,
                         "final_test_acc": res["scalars"]["final_test_acc"]})

    storage.write_json_atomic(run_dir / "rows.json", rows)
    storage.write_json_atomic(run_dir / "rates.json", picked)
    print(f"[{name}] -> {run_dir}", flush=True)
    return run_dir


def run_dataset(name, seeds, stage, epochs=None, batch=None, tune_epochs=None,
                tune_seed=2026, override_lr=None):
    bundle = get_dataset(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.CrossEntropyLoss()
    print(f"\n{'=' * 66}\n[{name}] model={bundle.meta['model']} "
          f"classes={bundle.meta['num_classes']} device={device}", flush=True)

    picked = None
    if stage in ("tune", "both"):
        picked = tune_dataset(name, bundle, tune_seed, device, crit,
                              epochs=tune_epochs, batch=batch)
    if stage in ("final", "both"):
        if picked is None:
            f = Path(storage.persistent_dir(f"{EXP}/{name}")) / "tuned_lr.json"
            if not f.exists():
                raise FileNotFoundError(
                    f"no tuned_lr.json for {name}; run --stage tune first")
            picked = json.loads(f.read_text())["picked"]
        picked = dict(picked)
        for arm, lr in (override_lr or {}).items():
            if arm not in picked:
                raise ValueError(f"--override_lr names unknown arm '{arm}'")
            print(f"  [{name}] override: {arm} {picked[arm]:g} -> {lr:g}",
                  flush=True)
            picked[arm] = lr
        final_dataset(name, bundle, seeds, device, crit, picked,
                      epochs=epochs, batch=batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LADDER))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--stage", choices=["tune", "final", "both"], default="both")
    ap.add_argument("--tune_seed", type=int, default=2026)
    ap.add_argument("--tune_epochs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--override_lr", nargs="+", default=None,
                    metavar="ARM=LR",
                    help="replace a tuned rate, e.g. sgd=0.3. For an arm whose "
                         "tuned rate diverges on some seeds: single-seed "
                         "tuning cannot see that, and the next rate down is "
                         "the honest substitute")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    override = {}
    for spec in (args.override_lr or []):
        arm, _, val = spec.partition("=")
        if not val:
            raise SystemExit(f"--override_lr wants ARM=LR, got '{spec}'")
        override[arm] = float(val)
    if args.smoke:
        args.datasets = ["titanic"]; args.seeds = [2026]
        args.epochs = 3; args.tune_epochs = 3

    t0 = time.time()
    failed = []
    for name in args.datasets:
        if name not in LADDER:
            print(f"!! unknown dataset {name}"); continue
        try:
            run_dataset(name, args.seeds, args.stage, args.epochs, args.batch,
                        args.tune_epochs, args.tune_seed, override)
        except Exception as e:
            import traceback
            print(f"!! {name} raised {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed.append(name)
    print(f"\n[{EXP}] done in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"[{EXP}] FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
