#!/usr/bin/env python3
"""Shared utilities for activation-oracle experiment scripts."""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import string
from pathlib import Path
from typing import Dict, Iterable, Optional


def utc_now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def make_run_id() -> str:
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    token = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{stamp}-{token}"


def normalize_segment(name: str) -> str:
    s = name.strip().lower().replace("/", "_")
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "default"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_run_root(
    output_root: Path,
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    probe_set: str,
    variant: str,
    run_id: str,
) -> Path:
    return (
        output_root
        / "runs"
        / normalize_segment(experiment_name)
        / normalize_segment(model_name)
        / normalize_segment(dataset_name)
        / normalize_segment(probe_set)
        / normalize_segment(variant)
        / run_id
    )


def init_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def init_status(meta_dir: Path, run_id: str, state: str, current_step: Optional[str]) -> None:
    write_json(
        meta_dir / "status.json",
        {
            "run_id": run_id,
            "state": state,
            "current_step": current_step,
            "last_updated_utc": utc_now_iso(),
        },
    )


def update_status(meta_dir: Path, state: str, current_step: Optional[str]) -> None:
    status_path = meta_dir / "status.json"
    status: Dict[str, object] = {}
    if status_path.exists():
        status = read_json(status_path)
    status["state"] = state
    status["current_step"] = current_step
    status["last_updated_utc"] = utc_now_iso()
    write_json(status_path, status)


def load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {
            "completed_steps": [],
            "completed_units": {},
            "updated_at_utc": utc_now_iso(),
        }
    payload = read_json(progress_path)
    payload.setdefault("completed_steps", [])
    payload.setdefault("completed_units", {})
    return payload


def save_progress(progress_path: Path, progress: dict) -> None:
    progress["updated_at_utc"] = utc_now_iso()
    write_json(progress_path, progress)


def mark_step_completed(progress: dict, step: str) -> None:
    if step not in progress["completed_steps"]:
        progress["completed_steps"].append(step)


def get_completed_set(progress: dict, bucket: str) -> set:
    units = progress.setdefault("completed_units", {})
    units.setdefault(bucket, [])
    return set(units[bucket])


def save_completed_set(progress: dict, bucket: str, values: set) -> None:
    progress.setdefault("completed_units", {})[bucket] = sorted(values)
