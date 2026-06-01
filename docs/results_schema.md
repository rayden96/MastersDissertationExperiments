# Results Schema

> **Status: ACTIVE (schema_version 2).** Implemented by `common/training.Trainer`
> and `common/storage.py`. `results.json` carries `"schema_version": 2`; plotting
> helpers in `common/plotting.py` read this shape. Earlier (implicit v1) IJCNN-era
> files predate the Trainer and are not produced by paper-ready experiments.

Every paper-ready run writes three files under `results/<experiment>/run_<id>/`:
`config.json` (at start), `metrics.jsonl` (streaming), and `results.json` (at end).

If you need to add a field, add it here first, then update `common/training.Trainer`
and any plotting helpers.

## What v2 added over the v1 sketch

- **`config.json`**: `method`, `base_optimizer`, `hp` (resolved HP dict),
  `model_kwargs`, `step_kind`, `trial_index`, `label`.
- **`results.json`**: `label`, `order_hash` (paired-data-order audit),
  `scalars.final_val_acc` (val/test separation), and two interference blocks —
  `interference` (run-level summary from `summarize_run`) and `interference_logs`
  / `calibration_logs` (raw per-step), present only when a meter was attached.
- **Two deficit totals** in the interference summary:
  - `cum_deficit` / `mean_deficit_per_step` — **SGD-yardstick** ideal
    (`u_ideal = −lr·g̃`); the cross-method common scale.
  - `cum_deficit_precond` / `mean_deficit_precond_per_step` — **per-optimiser**
    ideal (`u_ideal = Opt(g̃)` via a state-preserving shadow step); isolates
    mini-batch/trajectory cancellation from preconditioning. NaN/absent when the
    problem does not implement `preconditioned_ideal` (e.g. the 2D toy). For
    plain SGD the two coincide by construction. This is the documented extension
    that lets the framework span RMSprop/Adam, which FocusedWork §01/§02 defer.

---

## `config.json`

Written at the **start** of the run so crashed runs still have their config preserved. One JSON object, all hyperparameters resolved (no implicit defaults).

```json
{
  "experiment": "ch5/01_main_trajectories",
  "run_id": "20260421_163045__a3f1c8",
  "config_hash": "a3f1c8",
  "git_sha": "6bdedd0",
  "created_at": "2026-04-21T16:30:45Z",

  "dataset": "cifar10",
  "model": "small_cnn",
  "optimizer": "sgd",
  "wrapper": "bograd",
  "wrapper_kwargs": {"K": 8},

  "lr": 0.05,
  "batch_size": 128,
  "num_epochs": 30,
  "seed": 2026,
  "trial_index": 0,

  "log_every_n_steps": 50,
  "eval_every_n_steps": 200,
  "checkpoint_every_n_steps": 500,

  "device": "cuda:0",
  "hardware": "Tesla T4"
}
```

- `wrapper` is `null`, `"bograd"`, or `"cosgd"`.
- Keys in snake_case throughout.
- `hardware` is best-effort from `torch.cuda.get_device_name()` or `platform.processor()`.

---

## `metrics.jsonl`

Append-only. One JSON object per line. Written every `log_every_n_steps` and on every evaluation.

```json
{"step": 1200, "epoch": 3, "t": 1714839105.3, "phase": "train", "train_loss": 0.821, "g_norm": 1.42, "g_tilde_norm": 1.05}
{"step": 1200, "epoch": 3, "t": 1714839107.1, "phase": "eval",  "test_loss": 0.94, "test_acc": 0.612}
```

- `step` is the global training step (monotonic across epochs).
- `t` is `time.time()` (seconds since epoch). Use deltas, not absolutes, for "time to target" metrics — wall-clock resumes after a checkpoint restore.
- `phase` is `"train"` or `"eval"`. Not every field appears in every row — plotting code filters by `phase`.
- Experiment-specific diagnostics (e.g. `fraction_removed`, `cosine_prev_update`, `peak_mem_mb`) are added as additional keys. Document them in the experiment's `README.md` if they're not one of the common fields below.

### Common fields

| Field | Type | Phase | Meaning |
|---|---|---|---|
| `step` | int | both | global step |
| `epoch` | int | both | epoch index |
| `t` | float | both | wall-clock time |
| `phase` | str | — | `"train"` or `"eval"` |
| `train_loss` | float | train | per-step loss |
| `test_loss` | float | eval | full test-set loss |
| `test_acc` | float | eval | full test-set top-1 accuracy ∈ [0, 1] |
| `g_norm` | float | train | mean ‖g_t‖ across parameter tensors |
| `g_tilde_norm` | float | train | mean ‖g̃_t‖ after projection (if wrapper ≠ null) |
| `g_ratio` | float | train | `g_tilde_norm / g_norm` |
| `cosine_prev_update` | float | train | cos(update_t, update_{t−1}) |
| `fraction_removed` | float | train | `1 − g_ratio`, for 5.8 |
| `applied_step_norm` | float | train | ‖actual parameter delta‖ |
| `step_wall_time` | float | train | seconds per training step (smoothed) |
| `peak_mem_mb` | float | train | `torch.cuda.max_memory_allocated()` / 1e6 |

---

## `results.json`

Written at the **end** of the run (or on `KeyboardInterrupt`). This is what plotting code loads. Designed to be self-sufficient — a plotting notebook should not need `metrics.jsonl` under normal circumstances.

```json
{
  "config": { ... },                  // embedded copy of config.json
  "status": "completed",              // "completed" | "interrupted" | "errored"
  "total_steps": 11720,
  "total_epochs": 30,
  "total_wall_time_s": 1843.2,

  "scalars": {
    "final_test_acc": 0.7682,
    "max_test_acc": 0.7826,
    "final_train_loss": 0.214,
    "steps_to_test_target": 1794,
    "time_to_test_target": 91.5,
    "steps_to_train_target": 1580,
    "time_to_train_target": 79.9,
    "mean_step_wall_time_s": 0.157,
    "peak_mem_mb": 824.3
  },

  "history": {
    "train": {
      "step":        [50, 100, 150, ...],
      "t":           [0.8, 1.6, 2.4, ...],
      "train_loss":  [2.30, 2.01, 1.72, ...],
      "g_norm":      [...],
      "g_tilde_norm":[...]
    },
    "eval": {
      "step":        [200, 400, 600, ...],
      "t":           [3.1, 6.2, 9.3, ...],
      "test_loss":   [...],
      "test_acc":    [0.31, 0.47, 0.55, ...]
    }
  },

  "sentinels": {
    "test_target": 0.70,
    "train_target": 0.01
  },

  "raw_log_path": "results/ch5/01_main_trajectories/runs/20260421_163045__a3f1c8/metrics.jsonl"
}
```

### Field rules

- `status` is `"completed"` if the run finished its scheduled budget, `"interrupted"` if stopped by `KeyboardInterrupt`, `"errored"` otherwise.
- `scalars` — all derived scalar metrics, pre-computed so plotting doesn't need to re-derive them. Any `steps_to_*` or `time_to_*` that was never reached stores `null`, not `-1` or `inf` (the IJCNN notebook used `-1`; we're explicitly using `null` here to avoid ambiguity).
- `history` is **down-sampled** if the run logged more than ~5000 points per series. Down-sampling rule: keep every `k`-th point such that the resulting series is ≤ 5000 long. Document the down-sample factor in `scalars` under `history_downsample_factor` if applied.
- `history.train` and `history.eval` are separate so plotting doesn't have to mask by `phase`.
- `sentinels` records the accuracy thresholds used to compute `*_to_target` metrics, so plots can label the target line without hard-coding it.
- `raw_log_path` is a repo-relative path (not absolute) so the pointer survives moving the repo.

---

## Aggregation across trials

For experiments with multiple trials (most of Chapter 5), each trial writes its own run directory. Plotting code aggregates across trials using `common.plotting.load_runs(runs_dir)`, which returns a structure indexed by the config-varying fields (`optimizer`, `wrapper`, `trial_index`) and exposes `curve_mean_std(...)` like the IJCNN notebook.

**There is no "summary" file that spans trials.** Aggregation is always done on-the-fly in plotting notebooks. Rationale: adding a trial should not require re-running an aggregation step, and an out-of-date summary is worse than no summary.

---

## Versioning

If this schema changes incompatibly, bump the top-level field `schema_version` in `results.json` (currently implicit `1`). Plotting helpers will warn on unknown versions. Bumping is better than silent drift.
