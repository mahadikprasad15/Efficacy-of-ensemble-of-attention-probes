#!/usr/bin/env python3
"""Prepare persisted PCA-removal vectors and AO job manifests from saved PCA artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from common import (
    append_jsonl,
    build_run_root,
    ensure_dir,
    get_completed_set,
    init_logger,
    init_status,
    load_progress,
    make_run_id,
    mark_step_completed,
    read_json,
    save_completed_set,
    save_progress,
    update_status,
    utc_now_iso,
    write_json,
)

STEP_VALIDATE = "validate_inputs"
STEP_PERSIST_VECTORS = "persist_vectors"
STEP_BUILD_JOBS = "build_jobs"
STEP_FINALIZE = "finalize"


def parse_key_value_list(raw: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid KEY=VALUE item: {item}")
        out[k] = v
    return out


def parse_csv_values(raw: str) -> List[str]:
    out = []
    for token in raw.split(","):
        t = token.strip()
        if t:
            out.append(t)
    return out


def default_matrix_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "activation_oracle"
        / "locked_matrix_v1.json"
    )


def default_questions_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "activation_oracle"
        / "questions_default_v1.json"
    )


def load_matrix(matrix_preset: str, matrix_json: Optional[str]) -> dict:
    if matrix_json:
        cfg = read_json(Path(matrix_json))
    else:
        cfg = read_json(default_matrix_config_path())

    name = cfg.get("name", "")
    if matrix_preset and name and matrix_preset != name:
        raise ValueError(f"Requested matrix_preset={matrix_preset}, but config name={name}")

    poolings = cfg.get("poolings", {})
    if not poolings:
        raise ValueError("Matrix config missing 'poolings'")

    for pooling, item in poolings.items():
        layers = item.get("layers", [])
        ks = item.get("k_values", [])
        if not layers or not ks:
            raise ValueError(f"Matrix config requires non-empty layers and k_values for pooling={pooling}")
        if any(int(k) <= 0 for k in ks):
            raise ValueError(f"k_values must be positive for pooling={pooling}")
    return cfg


def load_questions(questions_json: Optional[str]) -> dict:
    path = Path(questions_json) if questions_json else default_questions_config_path()
    cfg = read_json(path)
    for key in ("exp1_combined", "exp3_per_pc"):
        if key not in cfg or not cfg[key]:
            raise ValueError(f"Questions config missing non-empty key: {key}")
    return cfg


def load_label_map(activations_dir: str) -> Dict[str, int]:
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest.jsonl in {activations_dir}")

    label_map: Dict[str, int] = {}
    with open(manifest_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            label_map[entry["id"]] = int(entry.get("label", -1))
    return label_map


def pool_tokens(x_layer: torch.Tensor, pooling: str) -> np.ndarray:
    if pooling == "mean":
        pooled = x_layer.mean(dim=0)
    elif pooling == "last":
        pooled = x_layer[-1, :]
    else:
        raise ValueError(f"Unsupported pooling for this pipeline: {pooling}")
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def load_pooled_split(
    activations_dir: str,
    layers: List[int],
    pooling: str,
    desc: str,
    limit: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[str]]:
    label_map = load_label_map(activations_dir)
    shard_paths = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

    feature_buckets: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    labels: List[int] = []
    sample_ids: List[str] = []

    for shard_path in shard_paths:
        shard = load_file(shard_path)
        for sid, tensor in shard.items():
            if sid not in label_map:
                continue
            label = label_map[sid]
            if label == -1:
                continue
            if tensor.dim() != 3:
                continue
            if max(layers) >= tensor.shape[0]:
                continue

            for layer in layers:
                feature_buckets[layer].append(pool_tokens(tensor[layer, :, :], pooling))
            labels.append(label)
            sample_ids.append(sid)

            if limit > 0 and len(labels) >= limit:
                break
        if limit > 0 and len(labels) >= limit:
            break

    if not labels:
        raise ValueError(f"No labeled examples loaded for {desc}: {activations_dir}")

    x_by_layer = {layer: np.stack(rows).astype(np.float32) for layer, rows in feature_buckets.items()}
    y = np.asarray(labels, dtype=np.int64)
    return x_by_layer, y, sample_ids


def find_probe_path(probes_root: Path, pooling: str, layer: int) -> Optional[Path]:
    candidates = [
        probes_root / pooling / f"probe_layer_{layer}.pt",
        probes_root / f"probe_layer_{layer}.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_classifier_params(state_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, float]:
    weight_key = None
    bias_key = None
    for key in state_dict:
        if key.endswith("classifier.weight"):
            weight_key = key
        if key.endswith("classifier.bias"):
            bias_key = key
    if weight_key is None or bias_key is None:
        raise KeyError("Could not find classifier.weight/classifier.bias in probe state dict")

    w = state_dict[weight_key].detach().cpu().numpy().reshape(-1).astype(np.float32)
    b = float(state_dict[bias_key].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sanitize_sample_id(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return digest


def build_job_id(parts: List[str]) -> str:
    payload = "|".join(parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return digest


def load_existing_job_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                jid = row.get("job_id")
                if jid:
                    ids.add(jid)
            except Exception:
                continue
    return ids


def ensure_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    eval_splits = parse_key_value_list(args.eval_split)
    if not eval_splits:
        raise ValueError("At least one --eval_split KEY=PATH is required")

    matrix_cfg = load_matrix(args.matrix_preset, args.matrix_json)
    questions_cfg = load_questions(args.questions_config)

    model_name = args.model_name
    run_id = args.run_id or make_run_id()
    run_root = build_run_root(
        output_root=Path(args.output_root).resolve(),
        experiment_name=args.experiment_name,
        model_name=model_name,
        dataset_name=args.dataset_name,
        probe_set=args.probe_set,
        variant=args.variant,
        run_id=run_id,
    )

    inputs_dir = run_root / "inputs"
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    logs_dir = run_root / "logs"
    meta_dir = run_root / "meta"

    for p in (inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir):
        ensure_dir(p)

    logger = init_logger(logs_dir / "run.log")
    progress_path = checkpoints_dir / "progress.json"
    progress = load_progress(progress_path)
    completed_steps = set(progress.get("completed_steps", []))

    status_path = meta_dir / "status.json"
    if status_path.exists() and not args.force_rebuild:
        status = read_json(status_path)
        if status.get("state") == "completed":
            logger.info("Status already completed: %s", status_path)
            return 0

    manifest = {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "script": "scripts/activation_oracle/prepare_oracle_vectors_from_saved_pca.py",
        "model_name": model_name,
        "dataset_name": args.dataset_name,
        "probe_set": args.probe_set,
        "variant": args.variant,
        "saved_pca_root": str(Path(args.saved_pca_root).resolve()),
        "eval_splits": eval_splits,
        "probes_root": str(Path(args.probes_root).resolve()),
        "matrix": matrix_cfg,
        "questions_config": questions_cfg,
        "job_splits": parse_csv_values(args.job_splits),
        "experiments": parse_csv_values(args.experiments),
        "max_samples_per_split": int(args.max_samples_per_split),
        "max_samples_for_jobs": int(args.max_samples_for_jobs),
        "force_rebuild": bool(args.force_rebuild),
    }
    write_json(meta_dir / "run_manifest.json", manifest)
    write_json(inputs_dir / "matrix_locked.json", matrix_cfg)
    write_json(inputs_dir / "questions_default.json", questions_cfg)

    init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_VALIDATE)

    try:
        poolings = matrix_cfg["poolings"]
        used_poolings = sorted(poolings.keys())
        if any(p not in ("mean", "last") for p in used_poolings):
            raise ValueError("This pipeline currently supports only mean and last pooling")

        pca_root = Path(args.saved_pca_root).resolve()
        probes_root = Path(args.probes_root).resolve()

        if args.force_rebuild or STEP_VALIDATE not in completed_steps:
            for pooling, spec in poolings.items():
                for layer in spec["layers"]:
                    pca_path = pca_root / pooling / "pca_artifacts" / f"layer_{int(layer)}.npz"
                    if not pca_path.exists():
                        raise FileNotFoundError(f"Missing PCA artifact: {pca_path}")
                    with np.load(pca_path, allow_pickle=False) as data:
                        comps = data["components"]
                        max_k = max(int(k) for k in spec["k_values"])
                        if comps.shape[0] < max_k:
                            raise ValueError(
                                f"PCA artifact {pca_path} has {comps.shape[0]} components, needs >= {max_k}"
                            )
                    probe_path = find_probe_path(probes_root, pooling, int(layer))
                    if probe_path is None and args.strict:
                        raise FileNotFoundError(
                            f"Missing probe for pooling={pooling} layer={layer} under {probes_root}"
                        )
            mark_step_completed(progress, STEP_VALIDATE)
            save_progress(progress_path, progress)
            completed_steps.add(STEP_VALIDATE)

        update_status(meta_dir, state="running", current_step=STEP_PERSIST_VECTORS)

        vectors_root = results_dir / "vectors"
        tables_root = results_dir / "tables"
        jobs_root = results_dir / "jobs"
        payloads_root = jobs_root / "payloads"
        ensure_dir(vectors_root)
        ensure_dir(tables_root)
        ensure_dir(jobs_root)
        ensure_dir(payloads_root)

        # Load split activations once per split/pooling.
        features_cache: Dict[Tuple[str, str], Tuple[Dict[int, np.ndarray], np.ndarray, List[str]]] = {}
        for split_name, split_dir in eval_splits.items():
            for pooling, spec in poolings.items():
                layers = sorted(int(x) for x in spec["layers"])
                logger.info("Loading split=%s pooling=%s from %s", split_name, pooling, split_dir)
                features_cache[(split_name, pooling)] = load_pooled_split(
                    activations_dir=split_dir,
                    layers=layers,
                    pooling=pooling,
                    desc=f"{split_name}/{pooling}",
                    limit=int(args.max_samples_per_split),
                )

        bundle_done = get_completed_set(progress, "bundle_units")
        bundle_rows: List[dict] = []
        prediction_rows: List[dict] = []

        for split_name, split_dir in eval_splits.items():
            for pooling, spec in poolings.items():
                x_by_layer, y, sample_ids = features_cache[(split_name, pooling)]

                for layer in sorted(int(v) for v in spec["layers"]):
                    pca_path = pca_root / pooling / "pca_artifacts" / f"layer_{layer}.npz"
                    with np.load(pca_path, allow_pickle=False) as pca_data:
                        mean = pca_data["mean"].astype(np.float32)
                        components = pca_data["components"].astype(np.float32)
                        evr = pca_data["explained_variance_ratio"].astype(np.float32)

                    probe_path = find_probe_path(probes_root, pooling, layer)
                    w: Optional[np.ndarray] = None
                    b: Optional[float] = None
                    if probe_path is not None:
                        state = torch.load(probe_path, map_location="cpu")
                        w, b = extract_classifier_params(state)

                    x = x_by_layer[layer].astype(np.float32)
                    dim = x.shape[1]
                    if mean.shape[0] != dim:
                        raise ValueError(
                            f"Dim mismatch for split={split_name} pooling={pooling} layer={layer}: "
                            f"activations dim={dim}, pca mean dim={mean.shape[0]}"
                        )
                    centered = x - mean[None, :]

                    for k in sorted(int(v) for v in spec["k_values"]):
                        unit_id = f"{split_name}|{pooling}|{layer}|{k}"
                        bundle_dir = vectors_root / split_name / pooling / f"layer_{layer}" / f"k_{k}"
                        bundle_path = bundle_dir / "bundle.npz"
                        meta_path = bundle_dir / "bundle_meta.json"

                        if (unit_id in bundle_done) and bundle_path.exists() and meta_path.exists() and not args.force_rebuild:
                            bundle_rows.append(
                                {
                                    "unit_id": unit_id,
                                    "split": split_name,
                                    "pooling": pooling,
                                    "layer": layer,
                                    "k": k,
                                    "n_samples": int(len(sample_ids)),
                                    "status": "skipped_existing",
                                    "bundle_path": str(bundle_path),
                                }
                            )
                            continue

                        comps_k = components[:k, :]
                        coeff = centered @ comps_k.T
                        removed_sum = coeff @ comps_k
                        clean = x - removed_sum

                        ensure_dir(bundle_dir)
                        np.savez_compressed(
                            bundle_path,
                            sample_ids=np.asarray(sample_ids),
                            labels=y,
                            orig=x,
                            removed_sum=removed_sum.astype(np.float32),
                            clean=clean.astype(np.float32),
                            coeff_topk=coeff.astype(np.float32),
                            pc_indices=np.arange(k, dtype=np.int32),
                        )

                        write_json(
                            meta_path,
                            {
                                "generated_at_utc": utc_now_iso(),
                                "model_name": model_name,
                                "split": split_name,
                                "source_activation_dir": split_dir,
                                "pooling": pooling,
                                "layer": layer,
                                "k": k,
                                "n_samples": int(len(sample_ids)),
                                "dim": int(dim),
                                "pca_path": str(pca_path),
                                "probe_path": str(probe_path) if probe_path else None,
                                "explained_variance_ratio_topk": [float(v) for v in evr[:k]],
                            },
                        )

                        bundle_rows.append(
                            {
                                "unit_id": unit_id,
                                "split": split_name,
                                "pooling": pooling,
                                "layer": layer,
                                "k": k,
                                "n_samples": int(len(sample_ids)),
                                "status": "written",
                                "bundle_path": str(bundle_path),
                            }
                        )

                        if w is not None and b is not None:
                            if w.shape[0] != dim:
                                raise ValueError(
                                    f"Probe dim mismatch at pooling={pooling} layer={layer}: "
                                    f"probe dim={w.shape[0]}, activation dim={dim}"
                                )
                            baseline_logits = x @ w + b
                            clean_logits = clean @ w + b
                            baseline_probs = sigmoid(baseline_logits)
                            clean_probs = sigmoid(clean_logits)
                            baseline_pred = (baseline_probs >= 0.5).astype(np.int64)
                            clean_pred = (clean_probs >= 0.5).astype(np.int64)

                            for i, sid in enumerate(sample_ids):
                                was_correct = int(baseline_pred[i] == y[i])
                                is_correct = int(clean_pred[i] == y[i])
                                prediction_rows.append(
                                    {
                                        "split": split_name,
                                        "pooling": pooling,
                                        "layer": layer,
                                        "k": k,
                                        "sample_id": sid,
                                        "label": int(y[i]),
                                        "baseline_logit": float(baseline_logits[i]),
                                        "baseline_prob": float(baseline_probs[i]),
                                        "baseline_pred": int(baseline_pred[i]),
                                        "clean_logit": float(clean_logits[i]),
                                        "clean_prob": float(clean_probs[i]),
                                        "clean_pred": int(clean_pred[i]),
                                        "was_correct_baseline": was_correct,
                                        "is_correct_clean": is_correct,
                                        "flip_wrong_to_right": int((was_correct == 0) and (is_correct == 1)),
                                    }
                                )

                        bundle_done.add(unit_id)
                        save_completed_set(progress, "bundle_units", bundle_done)
                        save_progress(progress_path, progress)

        ensure_csv(
            tables_root / "vector_bundles.csv",
            bundle_rows,
            ["unit_id", "split", "pooling", "layer", "k", "n_samples", "status", "bundle_path"],
        )

        if prediction_rows:
            key_cols = ["split", "pooling", "layer", "k", "sample_id"]
            pred_df_new = pd.DataFrame(prediction_rows)

            parquet_path = tables_root / "predictions_by_sample.parquet"
            csv_path = tables_root / "predictions_by_sample.csv"

            pred_df_all = pred_df_new
            if parquet_path.exists():
                try:
                    pred_df_existing = pd.read_parquet(parquet_path)
                    pred_df_all = pd.concat([pred_df_existing, pred_df_new], ignore_index=True)
                except Exception:
                    pred_df_all = pred_df_new
            elif csv_path.exists():
                try:
                    pred_df_existing = pd.read_csv(csv_path)
                    pred_df_all = pd.concat([pred_df_existing, pred_df_new], ignore_index=True)
                except Exception:
                    pred_df_all = pred_df_new

            pred_df_all = pred_df_all.drop_duplicates(subset=key_cols, keep="last")
            try:
                pred_df_all.to_parquet(parquet_path, index=False)
            except Exception:
                pred_df_all.to_csv(csv_path, index=False)

        mark_step_completed(progress, STEP_PERSIST_VECTORS)
        save_progress(progress_path, progress)
        completed_steps.add(STEP_PERSIST_VECTORS)

        update_status(meta_dir, state="running", current_step=STEP_BUILD_JOBS)

        selected_experiments = set(parse_csv_values(args.experiments))
        selected_splits = set(parse_csv_values(args.job_splits))
        if not selected_splits:
            selected_splits = set(eval_splits.keys())

        exp1_jobs_path = jobs_root / "exp1_combined_jobs.jsonl"
        exp3_jobs_path = jobs_root / "exp3_per_pc_jobs.jsonl"

        if args.force_rebuild:
            if exp1_jobs_path.exists():
                exp1_jobs_path.unlink()
            if exp3_jobs_path.exists():
                exp3_jobs_path.unlink()
            for stale_payload in payloads_root.glob("*.npz"):
                stale_payload.unlink()
            existing_exp1 = set()
            existing_exp3 = set()
            done_jobs = set()
        else:
            existing_exp1 = load_existing_job_ids(exp1_jobs_path)
            existing_exp3 = load_existing_job_ids(exp3_jobs_path)
            done_jobs = get_completed_set(progress, "job_units")

        created_counts = {"exp1_combined": 0, "exp3_per_pc": 0}

        for pooling, spec in poolings.items():
            for layer in sorted(int(v) for v in spec["layers"]):
                pca_path = pca_root / pooling / "pca_artifacts" / f"layer_{layer}.npz"
                with np.load(pca_path, allow_pickle=False) as pca_data:
                    components = pca_data["components"].astype(np.float32)

                for split_name in sorted(selected_splits):
                    if split_name not in eval_splits:
                        continue
                    for k in sorted(int(v) for v in spec["k_values"]):
                        bundle_path = vectors_root / split_name / pooling / f"layer_{layer}" / f"k_{k}" / "bundle.npz"
                        if not bundle_path.exists():
                            continue

                        with np.load(bundle_path, allow_pickle=False) as bundle:
                            sample_ids = bundle["sample_ids"].tolist()
                            labels = bundle["labels"].astype(np.int64)
                            removed_sum = bundle["removed_sum"].astype(np.float32)
                            coeff_topk = bundle["coeff_topk"].astype(np.float32)

                        total_n = len(sample_ids)
                        job_n = total_n if args.max_samples_for_jobs <= 0 else min(total_n, int(args.max_samples_for_jobs))

                        for i in range(job_n):
                            sid = str(sample_ids[i])
                            sid_hash = sanitize_sample_id(sid)
                            label = int(labels[i])

                            if "exp1_combined" in selected_experiments:
                                vec = removed_sum[i : i + 1, :]
                                for q in questions_cfg["exp1_combined"]:
                                    job_id = build_job_id(
                                        [
                                            "exp1",
                                            split_name,
                                            pooling,
                                            str(layer),
                                            str(k),
                                            sid,
                                            q["id"],
                                        ]
                                    )
                                    if (job_id in existing_exp1) or (job_id in done_jobs):
                                        continue

                                    payload_path = payloads_root / f"{job_id}.npz"
                                    np.savez_compressed(payload_path, vectors=vec)

                                    row = {
                                        "job_id": job_id,
                                        "experiment": "exp1_combined",
                                        "split": split_name,
                                        "pooling": pooling,
                                        "layer": layer,
                                        "k": k,
                                        "sample_id": sid,
                                        "sample_id_hash": sid_hash,
                                        "label": label,
                                        "target_layer": layer,
                                        "question_id": q["id"],
                                        "question_text": q["text"],
                                        "num_vectors": 1,
                                        "vector_payload_path": str(payload_path.relative_to(jobs_root)),
                                        "prompt_template_version": "layer_placeholder_v1",
                                    }
                                    append_jsonl(exp1_jobs_path, [row])
                                    existing_exp1.add(job_id)
                                    done_jobs.add(job_id)
                                    created_counts["exp1_combined"] += 1

                            if "exp3_per_pc" in selected_experiments:
                                for pc_idx in range(k):
                                    comp_vec = (coeff_topk[i, pc_idx] * components[pc_idx, :]).astype(np.float32)
                                    vec = comp_vec.reshape(1, -1)
                                    for q in questions_cfg["exp3_per_pc"]:
                                        job_id = build_job_id(
                                            [
                                                "exp3",
                                                split_name,
                                                pooling,
                                                str(layer),
                                                str(k),
                                                sid,
                                                str(pc_idx),
                                                q["id"],
                                            ]
                                        )
                                        if (job_id in existing_exp3) or (job_id in done_jobs):
                                            continue

                                        payload_path = payloads_root / f"{job_id}.npz"
                                        np.savez_compressed(payload_path, vectors=vec)

                                        row = {
                                            "job_id": job_id,
                                            "experiment": "exp3_per_pc",
                                            "split": split_name,
                                            "pooling": pooling,
                                            "layer": layer,
                                            "k": k,
                                            "pc_index": pc_idx,
                                            "pc_rank": pc_idx + 1,
                                            "coefficient": float(coeff_topk[i, pc_idx]),
                                            "sample_id": sid,
                                            "sample_id_hash": sid_hash,
                                            "label": label,
                                            "target_layer": layer,
                                            "question_id": q["id"],
                                            "question_text": q["text"],
                                            "num_vectors": 1,
                                            "vector_payload_path": str(payload_path.relative_to(jobs_root)),
                                            "prompt_template_version": "layer_placeholder_v1",
                                        }
                                        append_jsonl(exp3_jobs_path, [row])
                                        existing_exp3.add(job_id)
                                        done_jobs.add(job_id)
                                        created_counts["exp3_per_pc"] += 1

                            save_completed_set(progress, "job_units", done_jobs)
                            save_progress(progress_path, progress)

        write_json(jobs_root / "questions_default_v1.json", questions_cfg)

        mark_step_completed(progress, STEP_BUILD_JOBS)
        save_progress(progress_path, progress)
        completed_steps.add(STEP_BUILD_JOBS)

        update_status(meta_dir, state="running", current_step=STEP_FINALIZE)

        summary = {
            "generated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "run_root": str(run_root),
            "counts": {
                "bundle_rows": len(bundle_rows),
                "prediction_rows": len(prediction_rows),
                "exp1_jobs_created": created_counts["exp1_combined"],
                "exp3_jobs_created": created_counts["exp3_per_pc"],
                "exp1_jobs_total": len(existing_exp1),
                "exp3_jobs_total": len(existing_exp3),
            },
            "paths": {
                "vectors_root": str(vectors_root),
                "tables_root": str(tables_root),
                "jobs_root": str(jobs_root),
                "exp1_jobs": str(exp1_jobs_path),
                "exp3_jobs": str(exp3_jobs_path),
            },
        }
        write_json(results_dir / "results.json", summary)

        mark_step_completed(progress, STEP_FINALIZE)
        save_progress(progress_path, progress)

        update_status(meta_dir, state="completed", current_step=None)
        logger.info("Completed run. Summary: %s", results_dir / "results.json")
        return 0

    except Exception:
        update_status(meta_dir, state="failed", current_step=None)
        write_json(meta_dir / "last_error.json", {"generated_at_utc": utc_now_iso(), "traceback": traceback.format_exc()})
        logger.exception("Run failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare AO vectors/jobs using saved PCA artifacts")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--saved_pca_root", type=str, required=True)
    p.add_argument(
        "--eval_split",
        type=str,
        action="append",
        required=True,
        help="Repeatable KEY=PATH, e.g. id_val=/.../validation",
    )
    p.add_argument("--probes_root", type=str, required=True)
    p.add_argument("--matrix_preset", type=str, default="locked_v1")
    p.add_argument("--matrix_json", type=str, default=None)
    p.add_argument("--questions_config", type=str, default=None)
    p.add_argument("--job_splits", type=str, default="ood_test")
    p.add_argument("--experiments", type=str, default="exp1_combined,exp3_per_pc")
    p.add_argument("--output_root", type=str, default="artifacts")
    p.add_argument("--experiment_name", type=str, default="activation_oracle_pca")
    p.add_argument("--dataset_name", type=str, default="Deception-Roleplaying")
    p.add_argument("--probe_set", type=str, default="roleplaying_probes")
    p.add_argument("--variant", type=str, default="locked_v1")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--max_samples_per_split", type=int, default=0)
    p.add_argument("--max_samples_for_jobs", type=int, default=0)
    p.add_argument("--force_rebuild", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
