"""
Phase-0 consolidated smoke test.

Proves the whole shared layer composes: dataset registry -> model registry ->
method registry -> Trainer -> interference meter -> schema-valid results.json,
for representative (method × base) cells across the modalities the bakeoff uses.

    python tests/test_phase0_smoke.py            # CIFAR-10 cells (fast, GPU)
    python tests/test_phase0_smoke.py --full     # also tabular (covertype) cell

Kept tiny (subset + 1 epoch) so it runs in well under a minute on a laptop GPU.
Heavier datasets (EMNIST/CIFAR-100/Yahoo) are exercised by the real experiments;
this asserts the *plumbing*, not convergence.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parent.parent
_PRE = _REPO / "PaperReadyExperiments"
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.datasets import get_dataset, balanced_reference_subset  # noqa: E402
from common.models import get_model  # noqa: E402
from common.methods import build_method, METHODS  # noqa: E402
from common.training import Trainer, TrainConfig  # noqa: E402
from interference.meter import InterferenceMeter  # noqa: E402
from interference.summary import summarize_run  # noqa: E402
from interference.torch_classification import TorchClassificationProblem  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUIRED_RESULT_KEYS = {"config", "label", "status", "total_steps", "scalars",
                        "history", "schema_version"}
REQUIRED_SCALAR_KEYS = {"final_test_acc", "best_test_acc", "final_val_acc",
                        "final_train_loss", "total_wall_time_s"}


def _run_cell(method, base, ds_name, model_name, num_classes, run_dir,
              train, val, test, ref_ds, hp):
    spec = build_method(method, base, hp=hp)
    model = get_model(model_name, num_classes=num_classes, **spec.model_kwargs)
    crit = torch.nn.CrossEntropyLoss()
    ref_loader = DataLoader(ref_ds, batch_size=128, shuffle=False)
    prob = TorchClassificationProblem(model, crit, ref_loader, DEVICE)
    meter = InterferenceMeter(prob, lr=hp.get("lr", 0.05), K_values=[4, 32],
                              log_every=25, ref_refresh_every=60)
    cfg = TrainConfig(
        experiment=f"_smoke/{ds_name}", dataset=ds_name, model=model_name,
        method=method, base_optimizer=base, num_classes=num_classes,
        epochs=1, batch_size=128, seed=2026, log_every_n_steps=10,
        checkpoint_every_n_steps=1000, num_workers=0, hp=hp,
        model_kwargs=spec.model_kwargs,
    )
    t = Trainer(cfg, spec, model, train, val, test, run_dir, DEVICE,
                criterion=crit, meter=meter,
                summarize=lambda m: summarize_run(
                    m.logs, m.calibration_logs, K_values=[4, 32],
                    cum_deficit=m.cum_deficit, cum_deficit_count=m.cum_deficit_count,
                    cum_deficit_precond=m.cum_deficit_precond,
                    cum_deficit_precond_count=m.cum_deficit_precond_count))
    base_opt = getattr(t.optimizer, "base_optimizer", t.optimizer)
    meter.problem.set_precond_optimizer(base_opt)
    res = t.run()
    # schema validation
    assert REQUIRED_RESULT_KEYS <= set(res), f"missing result keys: {REQUIRED_RESULT_KEYS - set(res)}"
    assert REQUIRED_SCALAR_KEYS <= set(res["scalars"]), \
        f"missing scalar keys: {REQUIRED_SCALAR_KEYS - set(res['scalars'])}"
    assert res["status"] == "completed" and res["schema_version"] == 2
    assert res["scalars"]["final_test_acc"] == res["scalars"]["final_test_acc"]  # not NaN
    assert (run_dir / "config.json").exists() and (run_dir / "metrics.jsonl").exists()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the tabular cell")
    args = ap.parse_args()

    print(f"Phase-0 smoke (device={DEVICE}):")

    # --- CIFAR-10 cells: all 5 methods x {sgd, adam} -------------------
    b = get_dataset("cifar10", val_fraction=0.1, seed=2026)
    tr = Subset(b.train, list(range(2000)))
    va = Subset(b.val, list(range(400)))
    te = Subset(b.test, list(range(800)))
    ref = balanced_reference_subset(tr, num_classes=10, n_per_class=15, seed=1)

    n_ok = 0
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for base in ("sgd", "adam"):
            lr = 0.05 if base == "sgd" else 1e-3
            for method in METHODS:
                hp = {"lr": lr}
                if method == "bograd":
                    hp["K"] = 16
                if method == "dropout":
                    hp["dropout_p"] = 0.2
                if method == "cosgd":
                    hp["cosgd_method"] = "modified_gs_negative"
                    hp["combine"] = "mean"   # avoid the sum-scale blowup on tiny subsets
                res = _run_cell(method, base, "cifar10", "small_cifar_cnn", 10,
                                d / f"{base}_{method}", tr, va, te, ref, hp)
                n_ok += 1
                print(f"  OK  {res['label']:<18} acc={res['scalars']['final_test_acc']:.3f} "
                      f"steps={res['total_steps']} deficit={res['interference']['cum_deficit']:+.3f}")
    print(f"  -> {n_ok}/{2*len(METHODS)} CIFAR-10 cells trained + schema-valid")

    # --- optional tabular cell (covertype downloads on first use) ------
    if args.full:
        cb = get_dataset("covertype", val_fraction=0.1, seed=2026)
        tr2 = Subset(cb.train, list(range(4000)))
        va2 = Subset(cb.val, list(range(800)))
        te2 = Subset(cb.test, list(range(1600)))
        ref2 = balanced_reference_subset(tr2, num_classes=7, n_per_class=20, seed=1)
        with tempfile.TemporaryDirectory() as d:
            res = _run_cell("bograd", "adam", "covertype", "mlp", 7,
                            Path(d) / "cov", tr2, va2, te2, ref2,
                            {"lr": 1e-3, "K": 16})
        print(f"  OK  tabular {res['label']:<12} acc={res['scalars']['final_test_acc']:.3f}")

    print("\nPhase-0 smoke PASSED.")


if __name__ == "__main__":
    main()
