"""
30.07-30.08 — model-scale study + gradient statistics during training.

  --kind scale     30.07  test acc + sec/epoch across model capacity:
                   (a) width-scaled CIFAR-10 small CNN, width in {0.25,0.5,1,2};
                   (b) real ResNet-18 / ResNet-34 on CIFAR-100.
                   Methods: baseline, bograd, cosgd (per base). Shows whether the
                   method ranking + relative benefit hold as capacity grows.

  --kind gradstats 30.08  per-step gradient geometry on CIFAR-10, baseline vs
                   bograd vs cosgd: cos(u_t,u_{t-1}) (cos_prev via meter logs),
                   ||g|| (u_norm), fraction removed (1 - g_ratio for bograd/cosgd),
                   effective step size. Written from the InterferenceMeter logs.

Usage:
    python scale_gradstats.py --kind scale
    python scale_gradstats.py --kind gradstats
    python scale_gradstats.py --kind scale --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PRE = _HERE.parent
_REPO = _PRE.parent
for p in (str(_REPO), str(_PRE), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import storage                                       # noqa: E402
from common.datasets import get_dataset, balanced_reference_subset  # noqa: E402
from common.methods import build_method                          # noqa: E402
from common.models import get_model, count_params               # noqa: E402
from common.training import Trainer, TrainConfig                 # noqa: E402
from torch.utils.data import DataLoader                          # noqa: E402

from interference.meter import InterferenceMeter                 # noqa: E402
from interference.summary import summarize_run                   # noqa: E402
from interference.torch_classification import TorchClassificationProblem  # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one(dataset, model_name, model_kwargs, base, method, hp, epochs,
               device, out_dir, seed, measure=False):
    bundle = get_dataset(dataset, val_fraction=0.1, seed=2026)
    nc = bundle.meta["num_classes"]; bs = bundle.meta["batch_size"]
    spec = build_method(method, base, hp=hp)
    mk = {**model_kwargs, **spec.model_kwargs}
    model = get_model(model_name, num_classes=nc, **mk)
    crit = torch.nn.CrossEntropyLoss()
    meter = summarize = None
    if measure:
        ref_ds = balanced_reference_subset(bundle.train, num_classes=nc, n_per_class=50, seed=2026)
        ref_loader = DataLoader(ref_ds, batch_size=256, shuffle=False, num_workers=0)
        prob = TorchClassificationProblem(model, crit, ref_loader, device)
        meter = InterferenceMeter(prob, lr=hp.get("lr", 0.05), K_values=[4, 32, 128],
                                  log_every=25, ref_refresh_every=100)
        summarize = lambda m: summarize_run(m.logs, m.calibration_logs, K_values=[4, 32, 128],
                                            cum_deficit=m.cum_deficit, cum_deficit_count=m.cum_deficit_count)
    cfg = TrainConfig(experiment=f"30_main_comparison/_scale/{dataset}", dataset=dataset,
                      model=model_name, method=method, base_optimizer=base, num_classes=nc,
                      epochs=epochs, batch_size=bs, seed=seed, hp=hp, model_kwargs=mk,
                      log_every_n_steps=25, num_workers=2)
    t = Trainer(cfg, spec, model, bundle.train, bundle.val, bundle.test, out_dir, device,
                criterion=crit, meter=meter, summarize=summarize)
    if measure:
        base_opt = getattr(t.optimizer, "base_optimizer", t.optimizer)
        meter.problem.set_precond_optimizer(base_opt)
    res = t.run()
    return res, count_params(model), (meter.logs if meter else [])


# COSGD is excluded above this class count, the same protocol rule the bakeoff
# applies (30_main_comparison/_bakeoff.COSGD_MAX_CLASSES). Its per-class stack is
# O(C x p), so on CIFAR-100 with a ResNet it exhausts the device: a measured
# 125.9x step cost and 14.6 GB peak at 100 classes (20.07). Running it here does
# not produce a scale data point, it produces an OutOfMemoryError.
COSGD_MAX_CLASSES = 10

_DATASET_CLASSES = {"cifar10": 10, "cifar100": 100}


def run_scale(args, device):
    scale_dir = _HERE / "_scale"
    out_path = scale_dir / "scale_30_07.json"
    # Resume: keep whatever a previous (possibly crashed) invocation produced.
    out = {"params": {}, "points": []}
    if out_path.exists():
        try:
            out = storage.read_json(out_path)
            out.setdefault("params", {}); out.setdefault("points", [])
        except Exception:
            pass
    done = {(p["scale"], p["method"]) for p in out["points"]}

    configs = []
    if not args.smoke:
        for w in (0.25, 0.5, 1.0, 2.0):
            configs.append(("cifar10", "small_cifar_cnn", {"width_mult": w}, f"cnn_w{w}"))
        configs.append(("cifar100", "resnet18_cifar", {}, "resnet18"))
        configs.append(("cifar100", "resnet34_cifar", {}, "resnet34"))
    else:
        configs = [("cifar10", "small_cifar_cnn", {"width_mult": 0.25}, "cnn_w0.25")]

    for dataset, model_name, mk, tag in configs:
        for method in (("baseline", "bograd", "cosgd") if not args.smoke else ("baseline", "bograd")):
            if method == "cosgd" and _DATASET_CLASSES.get(dataset, 10) > COSGD_MAX_CLASSES:
                print(f"  {tag:<12} {method:<9} skipped: {_DATASET_CLASSES[dataset]} classes "
                      f"> {COSGD_MAX_CLASSES} (20.07 scalability wall)", flush=True)
                continue
            if (tag, method) in done:
                print(f"  {tag:<12} {method:<9} skip (already recorded)", flush=True)
                continue
            hp = {"lr": 0.05 if args.base == "sgd" else 1e-3}
            if method == "bograd":
                hp.update(K=32, projection_mode="negative")
            if method == "cosgd":
                hp.update(cosgd_method="modified_gs_negative", combine="mean")
            t0 = time.time()
            try:
                res, nparams, _ = _train_one(dataset, model_name, mk, args.base, method, hp,
                                             args.epochs or 5, device,
                                             scale_dir / tag / f"{args.base}_{method}",
                                             args.seeds[0], measure=False)
            except torch.OutOfMemoryError as e:
                # Record the failure as data rather than losing the whole sweep:
                # "this configuration does not fit" is itself a scale result.
                print(f"  {tag:<12} {method:<9} OOM: {e}"[:200], flush=True)
                out["points"].append({"scale": tag, "model": model_name, "dataset": dataset,
                                      "method": method, "oom": True})
                storage.write_json_atomic(out_path, out)
                torch.cuda.empty_cache()
                continue
            out["params"][tag] = nparams
            out["points"].append({"scale": tag, "model": model_name, "dataset": dataset,
                                  "params": nparams, "method": method,
                                  "final_test_acc": res["scalars"]["final_test_acc"],
                                  "sec_per_step": res["scalars"]["mean_step_wall_time_s"]})
            # Write after EVERY point: this sweep trains for hours and a crash in
            # the last cell must not discard the ones already paid for.
            storage.write_json_atomic(out_path, out)
            print(f"  {tag:<12} {method:<9} params={nparams:>9,} "
                  f"acc={res['scalars']['final_test_acc']:.3f} "
                  f"{res['scalars']['mean_step_wall_time_s']*1000:.1f}ms/step "
                  f"({time.time()-t0:.0f}s)", flush=True)
    storage.write_json_atomic(out_path, out)
    print(f"wrote {out_path} ({len(out['points'])} points)")


def run_gradstats(args, device):
    out = {"methods": {}}
    for method in ("baseline", "bograd", "cosgd"):
        hp = {"lr": 0.05}
        if method == "bograd":
            hp.update(K=32, projection_mode="negative")
        if method == "cosgd":
            hp.update(cosgd_method="modified_gs_negative", combine="mean")
        res, _, logs = _train_one("cifar10", "small_cifar_cnn", {}, "sgd", method, hp,
                                  args.epochs or 5, device,
                                  _HERE / "_gradstats" / method, args.seeds[0], measure=True)
        series = {k: [(l["step"], l[k]) for l in logs if k in l and l[k] == l[k]]
                  for k in ("u_norm", "I_between_K32", "between_K32_mean_cos", "D_t")}
        out["methods"][method] = {"final_test_acc": res["scalars"]["final_test_acc"], "series": series}
        print(f"  {method:<9} acc={res['scalars']['final_test_acc']:.3f} logged {len(logs)} steps", flush=True)
    storage.write_json_atomic(_HERE / "_gradstats" / "gradstats_30_08.json", out)
    print(f"wrote {_HERE/'_gradstats'/'gradstats_30_08.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["scale", "gradstats"])
    ap.add_argument("--base", default="sgd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [2026]; args.epochs = 1
    device = _device()
    (run_scale if args.kind == "scale" else run_gradstats)(args, device)


if __name__ == "__main__":
    main()
