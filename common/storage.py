"""
common.storage — run identity, results root resolution, atomic I/O, checkpoints.

Implements the contract in docs/experiment_design.md §5/§6 so every paper-ready
experiment writes the same shapes to the same place in both local and Colab
environments.

Key entry points
----------------
  get_results_root()            -> Path     (Colab Drive mount, else local ./results)
  new_run_id(config)            -> str       "YYYYMMDD_HHMMSS__<hash6>"
  config_hash(config)           -> str       first 6 hex of sha256(canonical json)
  git_sha()                     -> str|None
  write_json_atomic(path, obj)              tmp-write + os.replace
  read_json(path)               -> obj
  append_jsonl(path, record)                one JSON object per line
  save_checkpoint(path, state)              atomic torch.save
  load_checkpoint(path)         -> dict|None
  JsonlLogger                   small append-only metrics.jsonl writer
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

IS_COLAB = "google.colab" in sys.modules

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Results root
# ---------------------------------------------------------------------------
def get_results_root() -> Path:
    """Root directory for all results.

    - Colab: mount Drive once and return `<Drive>/dissertation/results`
      (override the Drive subpath with env var DISSERTATION_RESULTS_SUBPATH).
    - Local: env var DISSERTATION_RESULTS_ROOT if set, else `<repo>/results`.

    The directory is created if missing. Experiments then write to
    `get_results_root() / <experiment> / run_<id> / ...`.
    """
    override = os.environ.get("DISSERTATION_RESULTS_ROOT")
    if override:
        root = Path(override)
    elif IS_COLAB:
        try:
            from google.colab import drive  # type: ignore

            mount_point = Path("/content/drive")
            if not (mount_point / "MyDrive").exists():
                drive.mount(str(mount_point))
            subpath = os.environ.get(
                "DISSERTATION_RESULTS_SUBPATH", "MyDrive/dissertation/results"
            )
            root = mount_point / subpath
        except Exception:
            # Fall back to local if the mount fails (e.g. headless CI).
            root = _REPO_ROOT / "results"
    else:
        root = _REPO_ROOT / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def persistent_dir(experiment: str, *parts: str) -> Path:
    """A durable output directory under get_results_root(), keyed by experiment.

    This is THE way experiment scripts should choose where to write, so outputs
    survive a Colab session end whenever DISSERTATION_RESULTS_ROOT (or the Drive
    mount) is set — without relying on notebook-side symlinks. Falls back to the
    local ./results root otherwise.

    e.g. persistent_dir("20_cosgd_ablation/05_combine", "iris")
         -> <drive>/dissertation/results/20_cosgd_ablation/05_combine/iris
    """
    d = get_results_root().joinpath(experiment, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
def _canonical(obj: Any) -> Any:
    """Make a config JSON-canonical & hashable: sort keys, stringify the
    non-JSON bits (types, dtypes, callables) deterministically."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    # types, torch.dtype, callables, etc. -> stable string
    name = getattr(obj, "__name__", None)
    return name if name else str(obj)


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:6]


def new_run_id(config: Optional[Dict[str, Any]] = None) -> str:
    """`YYYYMMDD_HHMMSS__<hash6>` (hash omitted if no config given)."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    if config is None:
        return ts
    return f"{ts}__{config_hash(config)}"


def git_sha(short: bool = True) -> Optional[str]:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(
            args, cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Atomic JSON / JSONL
# ---------------------------------------------------------------------------
def write_json_atomic(path: os.PathLike | str, obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
    os.replace(tmp, path)


def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: os.PathLike | str, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


class JsonlLogger:
    """Buffered append-only writer for metrics.jsonl. Flushes every `flush_every`
    records and on close, so a dead session keeps everything up to the last flush."""

    def __init__(self, path: os.PathLike | str, flush_every: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: list[str] = []
        self._flush_every = int(flush_every)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        self._buf.append(json.dumps(record, default=str))
        if len(self._buf) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self._fh.write("\n".join(self._buf) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: os.PathLike | str, state: Dict[str, Any]) -> None:
    """Atomic torch.save (tmp + os.replace), so a crash mid-save never corrupts
    the one checkpoint you have."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: os.PathLike | str, map_location: Any = "cpu") -> Optional[Dict[str, Any]]:
    import torch

    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


__all__ = [
    "IS_COLAB",
    "get_results_root",
    "persistent_dir",
    "new_run_id",
    "config_hash",
    "git_sha",
    "write_json_atomic",
    "read_json",
    "append_jsonl",
    "JsonlLogger",
    "save_checkpoint",
    "load_checkpoint",
]
