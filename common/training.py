"""
common.training — the one Trainer that knows how to train a model for the
paper-ready experiments, plus a JobManager for resumable multi-cell campaigns.

Design contract (docs/experiment_design.md §4/§5):
  - seed everything from config["seed"] (via common.seeding);
  - paired data order across methods within a trial (PairedBatchSampler);
  - step loop dispatching on MethodSpec.step_kind:
        "standard"  -> zero_grad / forward / backward / optimizer.step()
        "per_class" -> optimizer.step(x, y, unique(y))  (COSGD / GradDrop)
  - optional interference instrumentation via an injected `meter` (the Trainer
    itself imports nothing from PaperReadyExperiments — the experiment layer
    passes a meter built against common.methods' optimizer);
  - eval cadence (every N steps and/or every epoch) on val and/or test;
  - checkpoint/resume (Colab-safe, atomic) keyed by config_hash;
  - writes config.json (at start), metrics.jsonl (streaming), results.json (end).

Kept deliberately framework-agnostic so it serves images, tabular, and text
without special-casing — the dataset bundle + model registry + method registry
carry all problem specifics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common import storage
from common.methods import MethodSpec
from common.seeding import PairedBatchSampler, paired_shuffle, seed_everything, order_hash


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    experiment: str
    dataset: str
    model: str
    method: str
    base_optimizer: str
    num_classes: int
    epochs: int
    batch_size: int
    seed: int = 2026
    trial_index: int = 0
    hp: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_every_epoch: bool = True
    eval_every_n_steps: Optional[int] = None
    checkpoint_every_n_steps: int = 1000   # 0 disables mid-run checkpointing
    log_every_n_steps: int = 50
    test_target: Optional[float] = None
    train_target: Optional[float] = None
    num_workers: int = 2
    paired: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    was = model.training
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += crit(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    if was:
        model.train()
    return loss_sum / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Train one (dataset × model × method × base × seed) cell to completion.

    Parameters
    ----------
    config : TrainConfig
    spec : MethodSpec               (from common.methods.build_method)
    model : nn.Module               built with spec.model_kwargs applied
    train_dataset, val_dataset, test_dataset
    run_dir : Path                  results/<experiment>/run_<id>/...
    device : torch.device
    meter : optional                an interference meter exposing
                                    before_step/after_step/state_dict/load_state_dict
                                    + .logs/.calibration_logs/.cum_deficit*. The
                                    Trainer never imports the meter class.
    summarize : optional callable   (meter) -> dict, folded into results["interference"].
    """

    def __init__(
        self,
        config: TrainConfig,
        spec: MethodSpec,
        model: nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
        run_dir: Path,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        meter: Any = None,
        summarize: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ):
        self.config = config
        self.spec = spec
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.meter = meter
        self.summarize = summarize

        self.optimizer = spec.optimizer_factory(self.model, self.criterion)

        self._train_targets = self._targets(train_dataset)
        self._n_train = len(self._train_targets)

        # state
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = -math.inf
        self._t_offset = 0.0
        self._cfg_hash = storage.config_hash(config.to_dict())
        # cost measurement (FOGO-style results): param count + clean training-step
        # time + peak training memory, measured around the train op only so the
        # meter's diagnostics and the eval passes do not contaminate them.
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._train_step_time_sum = 0.0
        self._measured_steps = 0
        self._peak_train_bytes = 0

    # ---- data ---------------------------------------------------------
    @staticmethod
    def _targets(ds):
        from common.datasets import _targets_of
        return _targets_of(ds)

    def _eff_workers(self, ds) -> int:
        """Use 0 workers for in-memory tensor datasets / very small sets — DataLoader
        multiprocessing adds no benefit there and is a source of flaky worker
        crashes (e.g. on tiny tabular sets like iris/wine). Also 0 on Windows,
        where DataLoader workers use the spawn start-method and hang when the
        entry module was loaded dynamically (the ablation run_all importlib
        loader). No-op on Linux/Colab where fork is used."""
        import os
        from torch.utils.data import TensorDataset
        if os.name == "nt":
            return 0
        base = ds.dataset if isinstance(ds, Subset) else ds
        if isinstance(base, TensorDataset) or len(ds) < 2000:
            return 0
        return self.config.num_workers

    def _train_loader(self, epoch: int) -> DataLoader:
        pin = self.device.type == "cuda"
        nw = self._eff_workers(self.train_dataset)
        if self.config.paired:
            order = paired_shuffle(self._n_train, self.config.epochs,
                                   self.config.seed, self.config.trial_index)
            sampler = PairedBatchSampler(order, self.config.batch_size, drop_last=False)
            sampler.set_epoch(epoch)
            self._last_order_hash = order_hash(order)
            return DataLoader(self.train_dataset, batch_sampler=sampler,
                              num_workers=nw, pin_memory=pin)
        g = torch.Generator(); g.manual_seed(self.config.seed + epoch)
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                          shuffle=True, num_workers=nw,
                          pin_memory=pin, generator=g)

    def _eval_loader(self, ds) -> DataLoader:
        pin = self.device.type == "cuda"
        return DataLoader(ds, batch_size=512, shuffle=False,
                          num_workers=self._eff_workers(ds), pin_memory=pin)

    # ---- checkpoint / resume -----------------------------------------
    @staticmethod
    def _strip_buffers(obj):
        """Drop BOGrad's FIFO buffer from an optimiser state dict before saving.

        The buffer holds K vectors the size of the full parameter vector, so for
        a ResNet-18 at K=64 it is roughly 3 GB while the model itself is 45 MB.
        Checkpointing it every `checkpoint_every_n_steps` writes gigabytes per
        cell, which exhausted both the Colab disk and Drive mid-campaign.

        It is also the one piece of state not worth persisting: on resume the
        buffer refills within K steps, a fraction of a percent of a multi-epoch
        run, so the cost of rebuilding it is negligible against the cost of
        storing it. Both projection scopes name the entry `buffer` (global scope
        under global_state, per-tensor under the per-parameter state), so scrub
        recursively rather than by fixed path.
        """
        if isinstance(obj, dict):
            return {k: ([] if k == "buffer" else Trainer._strip_buffers(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [Trainer._strip_buffers(v) for v in obj]
        return obj

    def _save_checkpoint(self, tag: str = "last") -> None:
        state = {
            "global_step": self.global_step,
            "epoch": self.start_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self._strip_buffers(self.optimizer.state_dict()),
            "best_metric": self.best_metric,
            "config_hash": self._cfg_hash,
            "t_offset": self._t_offset + (time.time() - self._t0),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            },
            "meter_state": self.meter.state_dict() if self.meter is not None else None,
        }
        # A checkpoint is a convenience for resuming, so failing to write one
        # must not lose a run that is otherwise fine. Windows in particular
        # denies the atomic replace when a scanner holds the temporary file.
        try:
            storage.save_checkpoint(self.ckpt_dir / f"{tag}.pt", state)
        except Exception as e:
            print(f"    [warn] checkpoint '{tag}' not written "
                  f"({type(e).__name__}: {e}); training continues", flush=True)

    def _maybe_resume(self) -> bool:
        ckpt = storage.load_checkpoint(self.ckpt_dir / "last.pt", map_location=self.device)
        if ckpt is None:
            return False
        if ckpt.get("config_hash") != self._cfg_hash:
            raise RuntimeError(
                f"Checkpoint config_hash {ckpt.get('config_hash')} != current {self._cfg_hash}. "
                "Refusing to resume into a different config. Delete the run dir or change run id."
            )
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt["global_step"]
        self.start_epoch = ckpt["epoch"]
        self.best_metric = ckpt.get("best_metric", -math.inf)
        self._t_offset = ckpt.get("t_offset", 0.0)
        rng = ckpt.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception:
                pass
        if self.meter is not None and ckpt.get("meter_state") is not None:
            self.meter.load_state_dict(ckpt["meter_state"])
        return True

    # ---- the loop -----------------------------------------------------
    def run(self) -> Dict[str, Any]:
        seed_everything(self.config.seed + self.config.trial_index, deterministic=False)

        # config.json written at start (crashed runs keep their config)
        cfg = self.config.to_dict()
        cfg.update({"run_id": self.run_dir.name, "config_hash": self._cfg_hash,
                    "git_sha": storage.git_sha(), "label": self.spec.label,
                    "step_kind": self.spec.step_kind})
        storage.write_json_atomic(self.run_dir / "config.json", cfg)

        resumed = self._maybe_resume()
        if self.meter is not None and not resumed:
            self.meter.initialize()
        elif self.meter is not None and resumed:
            self.meter.initialize(append_calibration=False, at_step=self.global_step)

        logger = storage.JsonlLogger(self.run_dir / "metrics.jsonl")
        self._t0 = time.time()
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        status = "completed"
        epoch_test_acc: List[float] = []
        epoch_train_loss: List[float] = []
        is_per_class = self.spec.step_kind == "per_class"

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                self.model.train()
                loader = self._train_loader(epoch)
                loss_sum, n_batches = 0.0, 0

                for x, y in loader:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    # Time + peak-memory the training op in isolation (forward /
                    # backward / optimizer.step), excluding the meter's heavy
                    # after_step diagnostics and the eval passes, so the recorded
                    # cost reflects the method itself.
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                    _t_step = time.perf_counter()
                    if is_per_class:
                        if self.meter is not None:
                            self.meter.before_step()
                        loss_val = float(self.optimizer.step(x, y, torch.unique(y)))
                    else:
                        self.optimizer.zero_grad(set_to_none=True)
                        out = self.model(x)
                        loss = self.criterion(out, y)
                        loss.backward()
                        if self.meter is not None:
                            self.meter.before_step()
                        self.optimizer.step()
                        loss_val = float(loss.item())
                    if cuda:
                        torch.cuda.synchronize()
                        self._peak_train_bytes = max(
                            self._peak_train_bytes, torch.cuda.max_memory_allocated())
                    self._train_step_time_sum += time.perf_counter() - _t_step
                    self._measured_steps += 1

                    if self.meter is not None:
                        self.meter.after_step(self.global_step, (x, y), loss_val)

                    loss_sum += loss_val
                    n_batches += 1
                    self.global_step += 1

                    if self.global_step % self.config.log_every_n_steps == 0:
                        logger.log({"step": self.global_step, "epoch": epoch,
                                    "phase": "train", "train_loss": loss_val,
                                    "t": time.time()})
                    if (self.config.eval_every_n_steps and
                            self.global_step % self.config.eval_every_n_steps == 0):
                        self._eval_and_log(logger, epoch, which="val")
                    # 0 disables mid-run checkpointing. A short ablation cell
                    # is cheaper to repeat than to checkpoint, and on Colab the
                    # write goes to Drive, which makes it the one step in the
                    # loop that can fail for reasons having nothing to do with
                    # the run.
                    if (self.config.checkpoint_every_n_steps and
                            self.global_step % self.config.checkpoint_every_n_steps == 0):
                        self._save_checkpoint("last")

                train_loss = loss_sum / max(n_batches, 1)
                epoch_train_loss.append(train_loss)
                self.start_epoch = epoch + 1

                if self.config.eval_every_epoch:
                    vloss, vacc = self._eval_and_log(logger, epoch, which="val")
                    tloss, tacc = self._eval_and_log(logger, epoch, which="test")
                    epoch_test_acc.append(tacc)
                    if vacc > self.best_metric:
                        self.best_metric = vacc
                        self._save_checkpoint("best")
                    print(f"    [{self.spec.label}] e{epoch + 1}/{self.config.epochs} "
                          f"train_loss={train_loss:.4f} val_acc={vacc:.4f} test_acc={tacc:.4f}",
                          flush=True)
                self._save_checkpoint("last")

        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            logger.close()

        return self._write_results(status, epoch_test_acc, epoch_train_loss)

    def _eval_and_log(self, logger, epoch, which: str):
        ds = self.val_dataset if which == "val" else self.test_dataset
        loss, acc = evaluate(self.model, self._eval_loader(ds), self.device)
        logger.log({"step": self.global_step, "epoch": epoch, "phase": "eval",
                    "split": which, f"{which}_loss": loss, f"{which}_acc": acc,
                    "t": time.time()})
        return loss, acc

    def _write_results(self, status, epoch_test_acc, epoch_train_loss) -> Dict[str, Any]:
        total_wall = self._t_offset + (time.time() - self._t0)
        final_test_loss, final_test_acc = evaluate(
            self.model, self._eval_loader(self.test_dataset), self.device)
        final_val_loss, final_val_acc = evaluate(
            self.model, self._eval_loader(self.val_dataset), self.device)

        scalars = {
            "final_test_acc": final_test_acc,
            "best_test_acc": max(epoch_test_acc) if epoch_test_acc else final_test_acc,
            "final_val_acc": final_val_acc,
            "final_train_loss": epoch_train_loss[-1] if epoch_train_loss else float("nan"),
            "total_wall_time_s": total_wall,
            "mean_step_wall_time_s": total_wall / max(self.global_step, 1),
            "n_params": int(self.n_params),
            "peak_mem_mb": (self._peak_train_bytes / 1e6) if self._peak_train_bytes else float("nan"),
            "mean_train_step_s": (self._train_step_time_sum / self._measured_steps)
                                 if self._measured_steps else float("nan"),
        }

        results: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "label": self.spec.label,
            "status": status,
            "total_steps": self.global_step,
            "total_epochs": self.config.epochs,
            "scalars": scalars,
            "history": {
                "epoch_test_acc": epoch_test_acc,
                "epoch_train_loss": epoch_train_loss,
            },
            "order_hash": getattr(self, "_last_order_hash", None),
            "schema_version": 2,
        }

        if self.meter is not None:
            results["interference_logs"] = self.meter.logs
            results["calibration_logs"] = self.meter.calibration_logs
            if self.summarize is not None:
                results["interference"] = self.summarize(self.meter)

        storage.write_json_atomic(self.run_dir / "results.json", results)
        return results


# ---------------------------------------------------------------------------
# Job manager — skip completed cells so a campaign resumes across sessions
# ---------------------------------------------------------------------------
class JobManager:
    """Track which (cell-key) runs are done by the presence of a completed
    results.json, so a re-launched campaign skips finished work."""

    def __init__(self, campaign_root: Path):
        self.root = Path(campaign_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir_for(self, cell_key: str) -> Path:
        return self.root / cell_key

    def is_done(self, cell_key: str, hp: Optional[Dict[str, Any]] = None,
                **cfg_must_match: Any) -> bool:
        """True when this cell already has a completed run.

        Pass `hp` (and any other config fields) to guard against reusing a
        result produced under different settings: a campaign whose cell
        definitions have changed since the last launch must re-run those
        cells, not skip them because the label is unchanged.
        """
        rj = self.run_dir_for(cell_key) / "results.json"
        if not rj.exists():
            return False
        try:
            res = storage.read_json(rj)
        except Exception:
            return False
        if res.get("status") != "completed":
            return False
        cfg = res.get("config", {})
        if hp is not None and dict(cfg.get("hp") or {}) != dict(hp):
            return False
        return all(cfg.get(k) == v for k, v in cfg_must_match.items())


__all__ = ["TrainConfig", "Trainer", "JobManager", "evaluate"]
