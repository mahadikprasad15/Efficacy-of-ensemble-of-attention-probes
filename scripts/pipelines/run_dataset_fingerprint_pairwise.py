#!/usr/bin/env python3
"""Run dataset-fingerprint residualization and pairwise probe evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from safetensors.torch import load_file, save_file

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))

from actprobe.features.dataset_fingerprint import (  # noqa: E402
    dataset_name_to_index,
    fingerprint_basis_from_weights,
    fit_multinomial_logistic_regression,
    load_pooled_layer_features,
    residualize_activation_tensor,
    sample_balanced_binary_ids,
    select_equal_count_indices,
)


DEFAULT_DATASETS = [
    "Deception-ConvincingGame-completion",
    "Deception-HarmPressureChoice-completion",
    "Deception-InstructedDeception-completion",
    "Deception-Mask-completion",
    "Deception-AILiar-completion",
    "Deception-Roleplaying-completion",
    "Deception-InsiderTrading-SallyConcat-completion",
]
SPLITS = ["train", "validation", "test"]
DEFAULT_VARIANT = "sevenway-completion-sally"
DEFAULT_PROBE_SET = "dataset-fingerprint"
DEFAULT_MODEL_VARIANT = "l15-mean-multinomial"
INSIDER_SALLY_DATASET = "Deception-InsiderTrading-SallyConcat-completion"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def update_status(status_path: Path, state: str, message: str) -> None:
    current = read_json(status_path, default={})
    current["state"] = state
    current["message"] = message
    current["updated_at"] = utc_now()
    write_json(status_path, current)


def parse_dataset_list(value: str | None) -> List[str]:
    if not value:
        return list(DEFAULT_DATASETS)
    return [item.strip() for item in value.split(",") if item.strip()]


def dataset_base(dataset_name: str) -> str:
    if dataset_name.endswith("-completion"):
        return dataset_name[: -len("-completion")]
    if dataset_name.endswith("-full"):
        return dataset_name[: -len("-full")]
    return dataset_name


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dataset_fingerprint_pairwise")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def ensure_dataset_dirs(
    activations_root: Path,
    model_dir: str,
    datasets: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    resolved: Dict[str, Dict[str, str]] = {}
    for dataset in datasets:
        resolved[dataset] = {}
        for split in SPLITS:
            split_dir = activations_root / model_dir / dataset / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Missing activation split dir: {split_dir}")
            manifest_path = split_dir / "manifest.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest in {split_dir}")
            resolved[dataset][split] = str(split_dir)
    return resolved


def fit_fingerprint_basis(
    *,
    datasets: Sequence[str],
    activations_root: Path,
    model_dir: str,
    layer_idx: int,
    pooling: str,
    count_per_dataset: int | None,
    fingerprint_seed: int,
    regularization_c: float,
    max_iter: int,
) -> tuple[np.ndarray, Dict[str, Any], Dict[str, List[str]]]:
    feature_blocks: List[np.ndarray] = []
    dataset_labels: List[str] = []
    sample_ids: List[str] = []
    train_counts: Dict[str, int] = {}

    for dataset in datasets:
        train_dir = activations_root / model_dir / dataset / "train"
        X, ids, _ = load_pooled_layer_features(train_dir, layer_idx=layer_idx, pooling=pooling)
        feature_blocks.append(X)
        dataset_labels.extend([dataset] * len(ids))
        sample_ids.extend(ids)
        train_counts[dataset] = len(ids)

    X_all = np.concatenate(feature_blocks, axis=0)
    selected_idx = select_equal_count_indices(
        dataset_labels,
        count_per_group=count_per_dataset,
        seed=fingerprint_seed,
    )

    dataset_to_index = dataset_name_to_index(datasets)
    selected_labels = [dataset_labels[idx] for idx in selected_idx.tolist()]
    y_selected = np.asarray([dataset_to_index[label] for label in selected_labels], dtype=np.int64)
    X_selected = X_all[selected_idx]

    clf = fit_multinomial_logistic_regression(
        X_selected,
        y_selected,
        regularization_c=regularization_c,
        max_iter=max_iter,
        seed=fingerprint_seed,
    )
    predictions = clf.predict(X_selected)
    train_accuracy = float(np.mean(predictions == y_selected))
    basis, centered_weights, rank = fingerprint_basis_from_weights(clf.coef_)

    sample_index: Dict[str, List[str]] = {dataset: [] for dataset in datasets}
    for raw_idx in selected_idx.tolist():
        sample_index[dataset_labels[raw_idx]].append(sample_ids[raw_idx])

    metrics = {
        "dataset_to_index": dataset_to_index,
        "train_counts": train_counts,
        "selected_count_per_dataset": int(len(selected_idx) // len(datasets)),
        "num_selected_total": int(len(selected_idx)),
        "layer": int(layer_idx),
        "pooling": pooling,
        "regularization_c": float(regularization_c),
        "max_iter": int(max_iter),
        "seed": int(fingerprint_seed),
        "train_accuracy": train_accuracy,
        "classes": list(clf.classes_.tolist()),
        "weights_shape": list(clf.coef_.shape),
        "basis_shape": list(basis.shape),
        "basis_rank": int(rank),
    }
    numeric_payload = {
        "basis": basis.astype(np.float32),
        "weights": clf.coef_.astype(np.float32),
        "centered_weights": centered_weights.astype(np.float32),
        "intercept": clf.intercept_.astype(np.float32),
    }
    return basis, {"metrics": metrics, "numeric_payload": numeric_payload}, sample_index


def residualize_split_dir(
    *,
    input_dir: Path,
    output_dir: Path,
    basis: np.ndarray,
    layer_idx: int,
    resume: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_in = input_dir / "manifest.jsonl"
    manifest_out = output_dir / "manifest.jsonl"
    if not manifest_out.exists():
        shutil.copy2(manifest_in, manifest_out)

    shard_paths = sorted(input_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {input_dir}")

    processed = 0
    skipped = 0
    for shard_path in shard_paths:
        out_path = output_dir / shard_path.name
        if resume and out_path.exists():
            skipped += 1
            continue
        tensors = load_file(str(shard_path))
        residualized = {}
        for sample_id, tensor in tensors.items():
            residualized[sample_id] = residualize_activation_tensor(tensor, basis=basis, layer_idx=layer_idx).contiguous()
        save_file(residualized, str(out_path))
        processed += 1
        logger.info(f"Residualized {input_dir.name}/{shard_path.name} -> {out_path}")

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed_shards": processed,
        "skipped_shards": skipped,
    }


def build_insider_subset(
    *,
    residualized_root: Path,
    model_dir: str,
    layer_idx: int,
    pooling: str,
    per_class_count: int,
    seed: int,
) -> Dict[str, Any]:
    train_dir = residualized_root / model_dir / INSIDER_SALLY_DATASET / "train"
    _, sample_ids, labels = load_pooled_layer_features(train_dir, layer_idx=layer_idx, pooling=pooling)
    chosen_ids = sample_balanced_binary_ids(
        sample_ids,
        labels,
        per_class_count=per_class_count,
        seed=seed,
    )
    chosen_labels = {
        sample_id: int(label)
        for sample_id, label in zip(sample_ids, labels)
        if sample_id in set(chosen_ids)
    }
    return {
        "dataset": INSIDER_SALLY_DATASET,
        "split": "train",
        "per_class_count": int(per_class_count),
        "total_count": int(len(chosen_ids)),
        "seed": int(seed),
        "layer": int(layer_idx),
        "pooling": pooling,
        "ids": chosen_ids,
        "label_counts": {
            "0": sum(1 for sample_id in chosen_ids if chosen_labels[sample_id] == 0),
            "1": sum(1 for sample_id in chosen_ids if chosen_labels[sample_id] == 1),
        },
    }


def run_subprocess(cmd: List[str], logger: logging.Logger) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset-fingerprint residualized pairwise evaluation.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "last", "none"])
    parser.add_argument("--balance_strategy", type=str, default="equal_count", choices=["equal_count"])
    parser.add_argument("--fingerprint_count_per_dataset", type=int, default=None)
    parser.add_argument("--fingerprint_seed", type=int, default=42)
    parser.add_argument("--fingerprint_max_iter", type=int, default=1000)
    parser.add_argument("--fingerprint_c", type=float, default=1.0)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_patience", type=int, default=5)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--insider_probe_train_cap", type=int, default=500)
    parser.add_argument("--insider_probe_seed", type=int, default=42)
    parser.add_argument("--pairwise_run_id", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_dataset_list(args.datasets)
    if INSIDER_SALLY_DATASET not in datasets:
        raise ValueError(f"{INSIDER_SALLY_DATASET} must be included in --datasets")
    if args.pooling != "mean":
        raise ValueError("This pipeline currently supports only mean pooling.")
    if args.layer < 0:
        raise ValueError("--layer must be non-negative.")
    if args.insider_probe_train_cap % 2 != 0:
        raise ValueError("--insider_probe_train_cap must be even for balanced sampling.")

    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()
    pairwise_run_id = args.pairwise_run_id or run_id

    run_root = (
        Path(args.artifact_root)
        / "runs"
        / "dataset_fingerprint_pairwise"
        / model_dir
        / DEFAULT_VARIANT
        / DEFAULT_PROBE_SET
        / DEFAULT_MODEL_VARIANT
        / run_id
    )
    inputs_dir = run_root / "inputs"
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    logs_dir = run_root / "logs"
    meta_dir = run_root / "meta"
    for directory in [inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "run.log")
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    progress = read_json(
        progress_path,
        default={
            "completed_steps": [],
            "completed_residualized_splits": [],
            "completed_probe_datasets": [],
            "pairwise_completed": False,
        },
    )
    write_json(progress_path, progress)

    activations_root = Path(args.activations_root)
    resolved_inputs = ensure_dataset_dirs(activations_root, model_dir, datasets)
    write_json(inputs_dir / "resolved_activation_dirs.json", resolved_inputs)
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "git_commit": git_commit(),
            "datasets": datasets,
            "paths": {
                "activations_root": str(activations_root),
                "run_root": str(run_root),
                "residualized_root": str(results_dir / "residualized_activations"),
                "probes_root": str(results_dir / "probes"),
                "pairwise_results_root": str(results_dir / "pairwise_matrix"),
            },
            "config": {
                "layer": int(args.layer),
                "pooling": args.pooling,
                "balance_strategy": args.balance_strategy,
                "fingerprint_count_per_dataset": args.fingerprint_count_per_dataset,
                "fingerprint_seed": int(args.fingerprint_seed),
                "fingerprint_max_iter": int(args.fingerprint_max_iter),
                "fingerprint_c": float(args.fingerprint_c),
                "probe_batch_size": int(args.probe_batch_size),
                "probe_epochs": int(args.probe_epochs),
                "probe_patience": int(args.probe_patience),
                "probe_lr": float(args.probe_lr),
                "probe_weight_decay": float(args.probe_weight_decay),
                "eval_batch_size": int(args.eval_batch_size),
                "insider_probe_train_cap": int(args.insider_probe_train_cap),
                "insider_probe_seed": int(args.insider_probe_seed),
                "pairwise_run_id": pairwise_run_id,
            },
        },
    )
    update_status(status_path, "running", "starting")

    fingerprint_dir = results_dir / "fingerprint"
    residualized_root = results_dir / "residualized_activations"
    probes_root = results_dir / "probes"
    pairwise_results_root = results_dir / "pairwise_matrix"
    pairwise_artifact_root = results_dir / "pairwise_artifacts"
    fingerprint_basis_path = fingerprint_dir / "fingerprint_basis.npz"
    fingerprint_metrics_path = fingerprint_dir / "fingerprint_metrics.json"
    fingerprint_index_path = fingerprint_dir / "fingerprint_sample_index.json"
    residualization_manifest_path = results_dir / "residualization_manifest.jsonl"

    try:
        if args.resume and fingerprint_basis_path.exists() and fingerprint_metrics_path.exists():
            logger.info("Resume: loading existing fingerprint basis from %s", fingerprint_basis_path)
            basis = np.load(fingerprint_basis_path, allow_pickle=False)["basis"].astype(np.float32)
        else:
            logger.info("Fitting dataset fingerprint basis on %d datasets", len(datasets))
            basis, fit_payload, sample_index = fit_fingerprint_basis(
                datasets=datasets,
                activations_root=activations_root,
                model_dir=model_dir,
                layer_idx=args.layer,
                pooling=args.pooling,
                count_per_dataset=args.fingerprint_count_per_dataset,
                fingerprint_seed=args.fingerprint_seed,
                regularization_c=args.fingerprint_c,
                max_iter=args.fingerprint_max_iter,
            )
            fingerprint_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                fingerprint_basis_path,
                basis=fit_payload["numeric_payload"]["basis"],
                weights=fit_payload["numeric_payload"]["weights"],
                centered_weights=fit_payload["numeric_payload"]["centered_weights"],
                intercept=fit_payload["numeric_payload"]["intercept"],
            )
            write_json(fingerprint_metrics_path, fit_payload["metrics"])
            write_json(fingerprint_index_path, sample_index)
            progress["completed_steps"] = sorted(set(progress.get("completed_steps", [])) | {"fingerprint_fit"})
            write_json(progress_path, progress)
            logger.info("Saved fingerprint basis to %s", fingerprint_basis_path)

        for dataset in datasets:
            for split in SPLITS:
                split_key = f"{dataset}::{split}"
                input_dir = activations_root / model_dir / dataset / split
                output_dir = residualized_root / model_dir / dataset / split
                if args.resume and split_key in set(progress.get("completed_residualized_splits", [])):
                    logger.info("Resume: skipping residualized split %s", split_key)
                    continue
                rec = residualize_split_dir(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    basis=basis,
                    layer_idx=args.layer,
                    resume=args.resume,
                    logger=logger,
                )
                append_jsonl(
                    residualization_manifest_path,
                    {
                        "dataset": dataset,
                        "split": split,
                        "completed_at": utc_now(),
                        **rec,
                    },
                )
                completed_splits = set(progress.get("completed_residualized_splits", []))
                completed_splits.add(split_key)
                progress["completed_residualized_splits"] = sorted(completed_splits)
                progress["completed_steps"] = sorted(set(progress.get("completed_steps", [])) | {"residualize_activations"})
                write_json(progress_path, progress)

        insider_subset = build_insider_subset(
            residualized_root=residualized_root,
            model_dir=model_dir,
            layer_idx=args.layer,
            pooling=args.pooling,
            per_class_count=args.insider_probe_train_cap // 2,
            seed=args.insider_probe_seed,
        )
        insider_probe_dir = (
            probes_root
            / model_dir
            / f"{dataset_base(INSIDER_SALLY_DATASET)}_slices"
            / INSIDER_SALLY_DATASET
            / args.pooling
        )
        insider_probe_dir.mkdir(parents=True, exist_ok=True)
        insider_subset_path = insider_probe_dir / "insider_train_subset.json"
        write_json(insider_subset_path, insider_subset)
        write_json(inputs_dir / "probe_train_subset_insider.json", insider_subset)

        for dataset in datasets:
            base = dataset_base(dataset)
            probe_dir = probes_root / model_dir / f"{base}_slices" / dataset / args.pooling
            probe_path = probe_dir / f"probe_layer_{args.layer}.pt"
            if args.resume and probe_path.exists():
                logger.info("Resume: skipping existing probe %s", probe_path)
                completed = set(progress.get("completed_probe_datasets", []))
                completed.add(dataset)
                progress["completed_probe_datasets"] = sorted(completed)
                write_json(progress_path, progress)
                continue

            cmd = [
                sys.executable,
                "scripts/training/train_deception_probes.py",
                "--model",
                args.model,
                "--dataset",
                dataset,
                "--activations_dir",
                str(residualized_root),
                "--pooling",
                args.pooling,
                "--layer",
                str(args.layer),
                "--output_dir",
                str(probes_root),
                "--output_subdir",
                f"{base}_slices",
                "--output_dataset_name",
                dataset,
                "--batch_size",
                str(args.probe_batch_size),
                "--epochs",
                str(args.probe_epochs),
                "--patience",
                str(args.probe_patience),
                "--lr",
                str(args.probe_lr),
                "--weight_decay",
                str(args.probe_weight_decay),
            ]
            if dataset == INSIDER_SALLY_DATASET:
                cmd.extend(["--train_id_allowlist_json", str(insider_subset_path)])
            run_subprocess(cmd, logger)

            completed = set(progress.get("completed_probe_datasets", []))
            completed.add(dataset)
            progress["completed_probe_datasets"] = sorted(completed)
            progress["completed_steps"] = sorted(set(progress.get("completed_steps", [])) | {"train_residualized_probes"})
            write_json(progress_path, progress)

        pairwise_summary_path = pairwise_results_root / pairwise_run_id / "results" / "summary.json"
        if args.resume and progress.get("pairwise_completed") and pairwise_summary_path.exists():
            logger.info("Resume: skipping pairwise matrix stage; summary exists at %s", pairwise_summary_path)
        else:
            pairwise_cmd = [
                sys.executable,
                "scripts/pipelines/run_pairwise_eval_matrix.py",
                "--activations_root",
                str(residualized_root),
                "--probes_root",
                str(probes_root),
                "--results_root",
                str(pairwise_results_root),
                "--pipeline_results_root",
                str(pairwise_results_root),
                "--artifact_root",
                str(pairwise_artifact_root),
                "--model",
                args.model,
                "--run_id",
                pairwise_run_id,
                "--resume",
                "--skip_training",
                "--poolings",
                args.pooling,
                "--eval_batch_size",
                str(args.eval_batch_size),
                "--only_segments",
                "completion",
                "--completion_rows",
                ",".join(datasets),
                "--completion_cols",
                ",".join(datasets),
            ]
            run_subprocess(pairwise_cmd, logger)
            progress["pairwise_completed"] = True
            progress["completed_steps"] = sorted(set(progress.get("completed_steps", [])) | {"pairwise_eval"})
            write_json(progress_path, progress)

        final_summary = {
            "run_id": run_id,
            "completed_at": utc_now(),
            "model": args.model,
            "datasets": datasets,
            "layer": int(args.layer),
            "pooling": args.pooling,
            "fingerprint_basis_path": str(fingerprint_basis_path),
            "fingerprint_metrics_path": str(fingerprint_metrics_path),
            "fingerprint_sample_index_path": str(fingerprint_index_path),
            "residualized_root": str(residualized_root / model_dir),
            "probes_root": str(probes_root / model_dir),
            "insider_subset_path": str(insider_subset_path),
            "pairwise_summary_path": str(pairwise_summary_path),
            "progress": progress,
        }
        write_json(results_dir / "results.json", final_summary)
        update_status(status_path, "completed", "finished")
        logger.info("Run complete. Summary written to %s", results_dir / "results.json")
        return 0
    except Exception as exc:
        update_status(status_path, "failed", str(exc))
        logger.exception("Dataset fingerprint pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
