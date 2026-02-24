#!/usr/bin/env python3
"""
Analyze whether probe AUC is a bottleneck by fitting LDA directly on denoised x' vectors.

For each GLP run in a sweep:
1) Load pooled denoised vectors x' from saved unit checkpoints (seed-mean).
2) Fit LDA on x' only and compute:
   - in-sample AUC (upper-bound style)
   - out-of-fold CV AUC (less optimistic)
3) Compare LDA AUC vs probe AUC already logged by the GLP pipeline.

Artifacts are saved under:
  <analysis_root>/<model>/<dataset>/<split>/layer_<L>/<analysis_id>/
    meta/run_manifest.json
    meta/status.json
    checkpoints/progress.json
    results/rows.jsonl
    results/summary.json
    results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def make_run_id() -> str:
    token = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{token}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def set_status(meta_dir: Path, run_id: str, state: str, current_step: Optional[str]) -> None:
    write_json(
        meta_dir / "status.json",
        {
            "run_id": run_id,
            "state": state,
            "current_step": current_step,
            "last_updated_utc": utc_now_iso(),
        },
    )


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return 0.5
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def resolve_glp_run_dir(output_root: Path, target_dir: Path, layer: int, run_id: str) -> Path:
    split = target_dir.name
    dataset = target_dir.parent.name
    model_dir = target_dir.parent.parent.name
    return output_root / "glp_target_denoise" / model_dir / dataset / split / f"layer_{layer}" / run_id


def resolve_glp_layer_root(output_root: Path, target_dir: Path, layer: int) -> Path:
    return resolve_glp_run_dir(output_root, target_dir, layer, run_id="dummy").parent


def load_manifest_labels(target_dir: Path) -> Dict[str, int]:
    manifest_path = target_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {target_dir}")
    out: Dict[str, int] = {}
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("id", ""))
            if sid:
                out[sid] = int(row.get("label", -1))
    return out


def load_labels_for_run(results_dir: Path, target_dir: Path) -> np.ndarray:
    sample_ids_path = results_dir / "sample_ids.json"
    if not sample_ids_path.exists():
        raise FileNotFoundError(f"Missing {sample_ids_path}")
    payload = read_json(sample_ids_path)
    sample_ids = [str(v) for v in payload.get("sample_ids", [])]
    if not sample_ids:
        raise ValueError(f"No sample_ids found in {sample_ids_path}")

    label_map = load_manifest_labels(target_dir)
    labels = []
    for sid in sample_ids:
        if sid not in label_map:
            raise ValueError(f"Sample id {sid} not found in target manifest {target_dir / 'manifest.jsonl'}")
        label = int(label_map[sid])
        if label not in (0, 1):
            raise ValueError(f"Sample id {sid} has unsupported label {label}; expected 0/1.")
        labels.append(label)
    return np.asarray(labels, dtype=np.int64)


def load_xprime_seed_mean(checkpoints_dir: Path, timestep: int, num_seeds: int) -> np.ndarray:
    xs: List[np.ndarray] = []
    for seed in range(num_seeds):
        ckpt = checkpoints_dir / f"t{timestep}_s{seed}.npz"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        blob = np.load(ckpt, allow_pickle=False)
        if "pooled_x_prime" not in blob:
            raise ValueError(f"Checkpoint missing pooled_x_prime: {ckpt}")
        xs.append(blob["pooled_x_prime"].astype(np.float32))
    stack = np.stack(xs, axis=0)  # (S,N,D)
    return np.mean(stack, axis=0)


def cv_auc_lda(x: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> Optional[float]:
    class_counts = np.bincount(y, minlength=2)
    min_class = int(np.min(class_counts[class_counts > 0])) if np.any(class_counts > 0) else 0
    usable_splits = min(int(n_splits), int(min_class))
    if usable_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=usable_splits, shuffle=True, random_state=seed)
    scores = np.zeros(y.shape[0], dtype=np.float64)
    for train_idx, test_idx in skf.split(x, y):
        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(x[train_idx], y[train_idx])
        scores[test_idx] = lda.decision_function(x[test_idx])
    return safe_auc(y, scores)


def analyze_unit(
    *,
    run_dir: Path,
    results_payload: dict,
    timestep: int,
    labels: np.ndarray,
    cv_folds: int,
    cv_seed: int,
) -> dict:
    checkpoints_dir = run_dir / "checkpoints"
    by_t = results_payload.get("results_by_timestep", {})
    payload = by_t.get(f"t{timestep}", {})
    num_seeds = int(payload.get("num_seeds", results_payload.get("num_seeds", 0)))
    if num_seeds <= 0:
        raise ValueError(f"Invalid num_seeds for run {run_dir.name}, t={timestep}")

    xprime = load_xprime_seed_mean(checkpoints_dir, timestep=timestep, num_seeds=num_seeds)
    if xprime.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Sample mismatch for {run_dir.name} t={timestep}: xprime has {xprime.shape[0]}, labels has {labels.shape[0]}"
        )

    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(xprime, labels)
    scores_train = lda.decision_function(xprime)
    auc_lda_train = safe_auc(labels, scores_train)
    auc_lda_cv = cv_auc_lda(xprime, labels, n_splits=cv_folds, seed=cv_seed)

    probe_auc_xprime = payload.get("auc_xprime_mean")
    probe_auc_x = payload.get("auc_x")
    probe_auc_xprime_f = float(probe_auc_xprime) if probe_auc_xprime is not None else None
    probe_auc_x_f = float(probe_auc_x) if probe_auc_x is not None else None

    w = lda.coef_.reshape(-1).astype(np.float64)
    b = float(lda.intercept_.reshape(-1)[0]) if hasattr(lda, "intercept_") else 0.0

    row = {
        "timestamp_utc": utc_now_iso(),
        "run_id": str(results_payload.get("run_id", run_dir.name)),
        "run_dir": str(run_dir),
        "target_dir": str(results_payload.get("target_dir", "")),
        "num_timesteps": int(timestep),
        "start_timestep_mode": payload.get("start_timestep_mode"),
        "noise_scale": float(payload.get("noise_scale")) if payload.get("noise_scale") is not None else None,
        "num_seeds": int(num_seeds),
        "num_samples": int(xprime.shape[0]),
        "feature_dim": int(xprime.shape[1]),
        "auc_probe_x": probe_auc_x_f,
        "auc_probe_xprime": probe_auc_xprime_f,
        "auc_lda_xprime_train": float(auc_lda_train),
        "auc_lda_xprime_cv": None if auc_lda_cv is None else float(auc_lda_cv),
        "delta_lda_train_minus_probe": (
            None if probe_auc_xprime_f is None else float(auc_lda_train - probe_auc_xprime_f)
        ),
        "delta_lda_cv_minus_probe": (
            None if (probe_auc_xprime_f is None or auc_lda_cv is None) else float(auc_lda_cv - probe_auc_xprime_f)
        ),
        "lda_w_norm": float(np.linalg.norm(w)),
        "lda_bias": b,
    }
    return row


def build_summary(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"num_rows": 0}

    best_train = max(rows, key=lambda r: float(r.get("auc_lda_xprime_train", -1.0)))
    valid_cv = [r for r in rows if r.get("auc_lda_xprime_cv") is not None]
    best_cv = max(valid_cv, key=lambda r: float(r.get("auc_lda_xprime_cv", -1.0))) if valid_cv else None
    best_probe = max(rows, key=lambda r: float(r.get("auc_probe_xprime", -1.0) if r.get("auc_probe_xprime") is not None else -1.0))

    return {
        "num_rows": len(rows),
        "best_auc_probe_xprime": {
            "run_id": best_probe.get("run_id"),
            "num_timesteps": best_probe.get("num_timesteps"),
            "value": best_probe.get("auc_probe_xprime"),
        },
        "best_auc_lda_xprime_train": {
            "run_id": best_train.get("run_id"),
            "num_timesteps": best_train.get("num_timesteps"),
            "value": best_train.get("auc_lda_xprime_train"),
            "probe_auc_xprime": best_train.get("auc_probe_xprime"),
            "delta_vs_probe": best_train.get("delta_lda_train_minus_probe"),
        },
        "best_auc_lda_xprime_cv": None
        if best_cv is None
        else {
            "run_id": best_cv.get("run_id"),
            "num_timesteps": best_cv.get("num_timesteps"),
            "value": best_cv.get("auc_lda_xprime_cv"),
            "probe_auc_xprime": best_cv.get("auc_probe_xprime"),
            "delta_vs_probe": best_cv.get("delta_lda_cv_minus_probe"),
        },
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LDA bottleneck on GLP-denoised x' vectors")
    parser.add_argument("--target_dir", type=str, required=True, help="Target activations split dir")
    parser.add_argument("--output_root", type=str, default="results/GLP_Experiments")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--run_id_prefix", type=str, required=True, help="Sweep prefix, e.g. pilot_sweep")
    parser.add_argument(
        "--analysis_root",
        type=str,
        default=None,
        help="Root for analysis artifacts. Defaults to <output_root>/glp_target_denoise/_analysis/lda_probe_bottleneck",
    )
    parser.add_argument("--analysis_id", type=str, default=None)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    output_root = Path(args.output_root).resolve()

    layer_root = resolve_glp_layer_root(output_root, target_dir, int(args.layer))
    if not layer_root.exists():
        raise FileNotFoundError(f"No layer root found: {layer_root}")

    pattern = str(layer_root / f"{args.run_id_prefix}_ns*_st*")
    run_dirs = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs matched pattern: {pattern}")

    model_dir = target_dir.parent.parent.name
    dataset = target_dir.parent.name
    split = target_dir.name
    analysis_root = (
        Path(args.analysis_root).resolve()
        if args.analysis_root
        else output_root / "glp_target_denoise" / "_analysis" / "lda_probe_bottleneck"
    )
    analysis_id = args.analysis_id or f"{args.run_id_prefix}-{make_run_id()}"
    analysis_dir = analysis_root / model_dir / dataset / split / f"layer_{args.layer}" / analysis_id
    meta_dir = analysis_dir / "meta"
    checkpoints_dir = analysis_dir / "checkpoints"
    results_dir = analysis_dir / "results"
    for d in [meta_dir, checkpoints_dir, results_dir]:
        ensure_dir(d)

    progress_path = checkpoints_dir / "progress.json"
    rows_jsonl = results_dir / "rows.jsonl"

    manifest = {
        "run_id": analysis_id,
        "created_at_utc": utc_now_iso(),
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "target_dir": str(target_dir),
            "output_root": str(output_root),
            "layer": int(args.layer),
            "run_id_prefix": args.run_id_prefix,
            "matched_run_dirs": [str(p) for p in run_dirs],
        },
        "config": {
            "cv_folds": int(args.cv_folds),
            "cv_seed": int(args.cv_seed),
            "resume": bool(args.resume),
        },
    }
    write_json(meta_dir / "run_manifest.json", manifest)
    set_status(meta_dir, analysis_id, "running", "init")

    if args.resume and progress_path.exists():
        progress = read_json(progress_path)
    else:
        progress = {
            "run_id": analysis_id,
            "completed_units": [],
            "updated_at_utc": utc_now_iso(),
        }
        write_json(progress_path, progress)
    completed = set(str(v) for v in progress.get("completed_units", []))

    existing_rows = read_jsonl(rows_jsonl)
    existing_keys = {f"{r.get('run_id')}|t{r.get('num_timesteps')}" for r in existing_rows}

    try:
        for run_dir in run_dirs:
            set_status(meta_dir, analysis_id, "running", f"run_{run_dir.name}")
            results_payload = read_json(run_dir / "results" / "results.json")
            run_target_dir = Path(str(results_payload.get("target_dir", target_dir))).resolve()
            labels = load_labels_for_run(run_dir / "results", run_target_dir)
            timesteps = [int(t) for t in results_payload.get("timesteps", [])]
            if not timesteps:
                by_t = results_payload.get("results_by_timestep", {})
                timesteps = sorted(int(str(k).replace("t", "")) for k in by_t.keys())

            for t in timesteps:
                unit_key = f"{run_dir.name}|t{t}"
                if args.resume and (unit_key in completed or unit_key in existing_keys):
                    if args.verbose:
                        print(f"Skipping completed unit: {unit_key}")
                    continue

                row = analyze_unit(
                    run_dir=run_dir,
                    results_payload=results_payload,
                    timestep=t,
                    labels=labels,
                    cv_folds=int(args.cv_folds),
                    cv_seed=int(args.cv_seed),
                )
                append_jsonl(rows_jsonl, row)
                existing_rows.append(row)
                existing_keys.add(unit_key)
                completed.add(unit_key)
                progress["completed_units"] = sorted(completed)
                progress["updated_at_utc"] = utc_now_iso()
                write_json(progress_path, progress)

        summary = build_summary(existing_rows)
        write_json(results_dir / "summary.json", summary)
        write_csv(results_dir / "summary.csv", existing_rows)
        set_status(meta_dir, analysis_id, "completed", None)
        print(f"Wrote analysis: {analysis_dir}")
        print(f"Rows: {len(existing_rows)}")
        if summary.get("best_auc_lda_xprime_cv") is not None:
            print("Best CV LDA row:", json.dumps(summary["best_auc_lda_xprime_cv"], indent=2))
        print("Best probe row:", json.dumps(summary.get("best_auc_probe_xprime", {}), indent=2))
        return 0
    except Exception:
        set_status(meta_dir, analysis_id, "failed", None)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

