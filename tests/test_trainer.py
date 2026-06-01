"""
End-to-end Trainer verification on a tiny CIFAR-10 subset.

    python tests/test_trainer.py

Checks:
  1. A standard (SGD baseline) run completes, writes config/results/metrics, and
     interference instrumentation populates logs + both deficits.
  2. A per-class run (COSGD) completes through the same Trainer path.
  3. Paired data order: two different methods in the same trial see the SAME
     batch order (order_hash matches) — the bakeoff rigor control.
  4. Checkpoint/resume: a run stopped after 1 epoch resumes and finishes epoch 2,
     reaching the same global_step as an uninterrupted 2-epoch run.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Subset

_REPO = Path(__file__).resolve().parent.parent
_PRE = _REPO / "PaperReadyExperiments"
for p in (str(_REPO), str(_PRE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.datasets import get_dataset, balanced_reference_subset  # noqa: E402
from common.models import get_model  # noqa: E402
from common.methods import build_method  # noqa: E402
from common.training import Trainer, TrainConfig  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from interference.meter import InterferenceMeter  # noqa: E402
from interference.summary import summarize_run  # noqa: E402
from interference.torch_classification import TorchClassificationProblem  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_cifar(n_train=2000, n_test=1000):
    b = get_dataset("cifar10", val_fraction=0.1, seed=2026)
    tr = Subset(b.train, list(range(n_train)))
    va = Subset(b.val, list(range(500)))
    te = Subset(b.test, list(range(n_test)))
    return tr, va, te, b.meta


def _make_meter(model, criterion, ref_ds, lr, opt_for_precond=None):
    ref_loader = DataLoader(ref_ds, batch_size=128, shuffle=False)
    prob = TorchClassificationProblem(model, criterion, ref_loader, DEVICE,
                                      precond_optimizer=opt_for_precond)
    return InterferenceMeter(prob, lr=lr, K_values=[4, 32], log_every=20, ref_refresh_every=50)


def _cfg(method, base, epochs, run_name, **kw):
    return TrainConfig(
        experiment=f"_test/{run_name}", dataset="cifar10", model="small_cifar_cnn",
        method=method, base_optimizer=base, num_classes=10, epochs=epochs,
        batch_size=128, seed=2026, trial_index=0, log_every_n_steps=10,
        checkpoint_every_n_steps=8, num_workers=0, **kw)


def _build(method, base, epochs, run_dir, ref_ds, attach_meter=True, hp=None):
    spec = build_method(method, base, hp=hp or {"lr": 0.05})
    model = get_model("small_cifar_cnn", num_classes=10, **spec.model_kwargs)
    crit = torch.nn.CrossEntropyLoss()
    meter = None
    if attach_meter:
        opt_probe = None  # precond optimiser wired below after optimizer exists
        meter = _make_meter(model, crit, ref_ds, lr=0.05)
    tr, va, te, _ = _tiny_cifar()
    cfg = _cfg(method, base, epochs, run_dir.name)
    t = Trainer(cfg, spec, model, tr, va, te, run_dir, DEVICE,
                criterion=crit, meter=meter,
                summarize=(lambda m: summarize_run(
                    m.logs, m.calibration_logs, K_values=[4, 32],
                    cum_deficit=m.cum_deficit, cum_deficit_count=m.cum_deficit_count,
                    cum_deficit_precond=m.cum_deficit_precond,
                    cum_deficit_precond_count=m.cum_deficit_precond_count)) if attach_meter else None)
    # wire precond optimiser (base) into the problem so the precond deficit fires
    if attach_meter and hasattr(t.optimizer, "base_optimizer"):
        meter.problem.set_precond_optimizer(t.optimizer.base_optimizer)
    elif attach_meter:
        meter.problem.set_precond_optimizer(t.optimizer)
    return t


def main():
    print(f"Trainer verification (device={DEVICE}):")
    tr, va, te, meta = _tiny_cifar()
    ref_ds = balanced_reference_subset(tr, num_classes=10, n_per_class=20, seed=1)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # [1] standard baseline run
        t = _build("baseline", "sgd", 2, d / "base", ref_ds)
        res = t.run()
        assert res["status"] == "completed"
        assert (d / "base" / "config.json").exists()
        assert (d / "base" / "results.json").exists()
        assert (d / "base" / "metrics.jsonl").exists()
        assert res["interference"]["n_logs"] > 0, "meter produced no logs"
        assert res["interference"]["n_steps_total"] > 0
        assert res["interference"]["cum_deficit"] == res["interference"]["cum_deficit"], "cum_deficit is NaN"
        # precond deficit should also have fired (SGD => equals yardstick, but counted)
        assert res["interference"]["mean_deficit_precond_per_step"] == res["interference"]["mean_deficit_precond_per_step"], \
            "precond deficit did not fire (NaN)"
        print(f"  [1] baseline+SGD completed; logs={res['interference']['n_logs']} "
              f"final_test_acc={res['scalars']['final_test_acc']:.3f}  OK")

        # [2] per-class COSGD run through the same Trainer
        t2 = _build("cosgd", "sgd", 1, d / "cosgd", ref_ds,
                    hp={"lr": 0.05, "cosgd_method": "modified_gs_negative"})
        res2 = t2.run()
        assert res2["status"] == "completed" and res2["total_steps"] > 0
        print(f"  [2] cosgd+SGD (per_class path) completed; "
              f"final_test_acc={res2['scalars']['final_test_acc']:.3f}  OK")

        # [3] paired order identical across two methods in same trial
        ta = _build("baseline", "sgd", 1, d / "pa", ref_ds, attach_meter=False)
        tb = _build("bograd", "sgd", 1, d / "pb", ref_ds, attach_meter=False)
        ra, rb = ta.run(), tb.run()
        assert ra["order_hash"] is not None and ra["order_hash"] == rb["order_hash"], \
            "paired order mismatch across methods!"
        print(f"  [3] paired data order identical across baseline/bograd "
              f"(hash={ra['order_hash']})  OK")

        # [4] checkpoint / resume reaches same global_step as uninterrupted.
        # Both runs use the SAME epochs=2 config (so config_hash matches); the
        # first is interrupted at the start of epoch 2 to simulate a dead session.
        full = _build("baseline", "sgd", 2, d / "full", ref_ds, attach_meter=False)
        full_steps = full.run()["total_steps"]

        part = _build("baseline", "sgd", 2, d / "resume", ref_ds, attach_meter=False)
        _orig_loader = part._train_loader

        def _interrupt_on_epoch2(epoch, _orig=_orig_loader):
            if epoch >= 1:                      # epoch 0 done + checkpointed; die now
                raise KeyboardInterrupt
            return _orig(epoch)
        part._train_loader = _interrupt_on_epoch2
        part_res = part.run()
        assert part_res["status"] == "interrupted"
        part_ckpt = torch.load(d / "resume" / "checkpoints" / "last.pt",
                               map_location="cpu", weights_only=False)
        assert part_ckpt["epoch"] == 1, f"expected checkpoint at epoch boundary 1, got {part_ckpt['epoch']}"

        cont = _build("baseline", "sgd", 2, d / "resume", ref_ds, attach_meter=False)
        cont_res = cont.run()
        assert cont_res["total_steps"] == full_steps, \
            f"resume steps {cont_res['total_steps']} != full {full_steps}"
        print(f"  [4] resume: interrupted@epoch2 (step {part_ckpt['global_step']}) -> "
              f"finished at step {cont_res['total_steps']} == uninterrupted ({full_steps})  OK")

    print("\nAll Trainer assertions passed.")


if __name__ == "__main__":
    main()
