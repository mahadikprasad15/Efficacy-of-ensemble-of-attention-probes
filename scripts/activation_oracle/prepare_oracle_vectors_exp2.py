#!/usr/bin/env python3
"""Prepare Exp2 AO jobs by mining wrong-to-right probe flips (OOD only).

This script scans cached OOD activations, applies probes to original and
PCA-removed activations per (pooling, layer, k), selects top-N wrong->right
flips, and creates AO jobs that interpret orig and clean activations
separately using a shared single-activation prompt.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import heapq
import json
import os
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

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
STEP_SCAN = "scan_activations"
STEP_BUILD_JOBS = "build_jobs"
STEP_FINALIZE = "finalize"


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
        / "questions_exp2_v1.json"
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
    if "exp2_single" not in cfg or not cfg["exp2_single"]:
        raise ValueError("Questions config missing non-empty key: exp2_single")
    return cfg


def load_label_map(activations_dir: str) -> Dict[str, int]:
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest.jsonl in {activations_dir}")

    label_map: Dict[str, int] = {}
    with open(manifest_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            label_map[str(entry["id"])] = int(entry.get("label", -1))
    return label_map


def pool_tokens(x_layer: torch.Tensor, pooling: str) -> np.ndarray:
    if pooling == "mean":
        pooled = x_layer.mean(dim=0)
    elif pooling == "last":
        pooled = x_layer[-1, :]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


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


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def sanitize_id(raw: str) -> str:
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
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


def combo_key(pooling: str, layer: int, k: int) -> str:
    return f"{pooling}|{layer}|{k}"


def update_topk(heap: list, item: dict, top_n: int, counter: int) -> int:
    entry = (float(item["delta_logit"]), counter, item)
    if len(heap) < top_n:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)
    return counter + 1


def save_topk_state(path: Path, topk: Dict[str, list]) -> None:
    payload: Dict[str, np.ndarray] = {}
    payload["combo_keys"] = np.asarray(list(topk.keys()), dtype="U")

    for combo, heap in topk.items():
        items = [entry[2] for entry in heap]
        n = len(items)
        payload[f"{combo}__sample_ids"] = np.asarray([it["sample_id"] for it in items], dtype="U")
        payload[f"{combo}__labels"] = np.asarray([it["label"] for it in items], dtype=np.int64)
        payload[f"{combo}__orig_logit"] = np.asarray([it["orig_logit"] for it in items], dtype=np.float32)
        payload[f"{combo}__clean_logit"] = np.asarray([it["clean_logit"] for it in items], dtype=np.float32)
        payload[f"{combo}__delta_logit"] = np.asarray([it["delta_logit"] for it in items], dtype=np.float32)
        payload[f"{combo}__orig_prob"] = np.asarray([it["orig_prob"] for it in items], dtype=np.float32)
        payload[f"{combo}__clean_prob"] = np.asarray([it["clean_prob"] for it in items], dtype=np.float32)
        payload[f"{combo}__delta_prob"] = np.asarray([it["delta_prob"] for it in items], dtype=np.float32)
        if n:
            payload[f"{combo}__orig_vecs"] = np.stack([it["orig_vec"] for it in items]).astype(np.float32)
            payload[f"{combo}__clean_vecs"] = np.stack([it["clean_vec"] for it in items]).astype(np.float32)
        else:
            payload[f"{combo}__orig_vecs"] = np.empty((0, 0), dtype=np.float32)
            payload[f"{combo}__clean_vecs"] = np.empty((0, 0), dtype=np.float32)

    np.savez_compressed(path, **payload)


def load_topk_state(path: Path, combos: List[str]) -> Tuple[Dict[str, list], int]:
    topk: Dict[str, list] = {combo: [] for combo in combos}
    if not path.exists():
        return topk, 0

    data = np.load(path, allow_pickle=True)
    counter = 0
    for combo in combos:
        key = f"{combo}__delta_logit"
        if key not in data:
            continue
        deltas = data[key]
        if len(deltas) == 0:
            continue
        sample_ids = data[f"{combo}__sample_ids"]
        labels = data[f"{combo}__labels"]
        orig_logit = data[f"{combo}__orig_logit"]
        clean_logit = data[f"{combo}__clean_logit"]
        orig_prob = data[f"{combo}__orig_prob"]
        clean_prob = data[f"{combo}__clean_prob"]
        delta_prob = data[f"{combo}__delta_prob"]
        orig_vecs = data[f"{combo}__orig_vecs"]
        clean_vecs = data[f"{combo}__clean_vecs"]

        for i in range(len(deltas)):
            item = {
                "sample_id": str(sample_ids[i]),
                "label": int(labels[i]),
                "orig_logit": float(orig_logit[i]),
                "clean_logit": float(clean_logit[i]),
                "delta_logit": float(deltas[i]),
                "orig_prob": float(orig_prob[i]),
                "clean_prob": float(clean_prob[i]),
                "delta_prob": float(delta_prob[i]),
                "orig_vec": orig_vecs[i].astype(np.float32),
                "clean_vec": clean_vecs[i].astype(np.float32),
            }
            heapq.heappush(topk[combo], (float(deltas[i]), counter, item))
            counter += 1

    return topk, counter


def run(args: argparse.Namespace) -> int:
    progress_every = max(1, int(args.progress_every))

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
        "script": "scripts/activation_oracle/prepare_oracle_vectors_exp2.py",
        "model_name": model_name,
        "dataset_name": args.dataset_name,
        "probe_set": args.probe_set,
        "variant": args.variant,
        "saved_pca_root": str(Path(args.saved_pca_root).resolve()),
        "activations_dir": str(Path(args.activations_dir).resolve()),
        "probes_root": str(Path(args.probes_root).resolve()),
        "matrix": matrix_cfg,
        "questions_config": questions_cfg,
        "split_name": args.split_name,
        "top_n": int(args.top_n),
        "force_rebuild": bool(args.force_rebuild),
    }
    write_json(meta_dir / "run_manifest.json", manifest)
    write_json(inputs_dir / "matrix_locked.json", matrix_cfg)
    write_json(inputs_dir / "questions_exp2.json", questions_cfg)

    init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_VALIDATE)

    try:
        poolings = matrix_cfg["poolings"]
        used_poolings = sorted(poolings.keys())
        if any(p not in ("mean", "last") for p in used_poolings):
            raise ValueError("This pipeline currently supports only mean and last pooling")

        pca_root = Path(args.saved_pca_root).resolve()
        probes_root = Path(args.probes_root).resolve()
        activations_dir = Path(args.activations_dir).resolve()

        if args.force_rebuild or STEP_VALIDATE not in completed_steps:
            validation_units: List[Tuple[str, int, int]] = []
            for pooling, spec in poolings.items():
                max_k = max(int(k) for k in spec["k_values"])
                for layer in spec["layers"]:
                    validation_units.append((pooling, int(layer), max_k))

            for pooling, layer, max_k in tqdm(
                validation_units,
                desc="Validate PCA/probe inputs",
                unit="layer",
                leave=True,
            ):
                pca_path = pca_root / pooling / "pca_artifacts" / f"layer_{layer}.npz"
                if not pca_path.exists():
                    raise FileNotFoundError(f"Missing PCA artifact: {pca_path}")
                with np.load(pca_path, allow_pickle=False) as data:
                    comps = data["components"]
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

        update_status(meta_dir, state="running", current_step=STEP_SCAN)

        label_map = load_label_map(str(activations_dir))
        shard_paths = sorted(glob.glob(os.path.join(str(activations_dir), "shard_*.safetensors")))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

        combos: List[str] = []
        for pooling, spec in poolings.items():
            for layer in spec["layers"]:
                for k in spec["k_values"]:
                    combos.append(combo_key(pooling, int(layer), int(k)))

        state_path = checkpoints_dir / "topk_state.npz"
        topk, counter = load_topk_state(state_path, combos)

        completed_shards = get_completed_set(progress, "shards")
        processed_samples = int(progress.get("processed_samples", 0))

        pca_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        probe_cache: Dict[Tuple[str, int], Tuple[np.ndarray, float]] = {}

        for pooling, spec in poolings.items():
            for layer in spec["layers"]:
                pca_path = pca_root / pooling / "pca_artifacts" / f"layer_{layer}.npz"
                with np.load(pca_path, allow_pickle=False) as pca_data:
                    pca_cache[(pooling, int(layer))] = {
                        "mean": pca_data["mean"].astype(np.float32),
                        "components": pca_data["components"].astype(np.float32),
                    }
                probe_path = find_probe_path(probes_root, pooling, int(layer))
                if probe_path is not None:
                    state = torch.load(probe_path, map_location="cpu")
                    probe_cache[(pooling, int(layer))] = extract_classifier_params(state)

        max_layer = max(int(l) for spec in poolings.values() for l in spec["layers"])

        shard_bar = tqdm(shard_paths, desc="Scan shards", unit="shard", leave=True)
        for shard_path in shard_bar:
            shard_name = os.path.basename(shard_path)
            if shard_name in completed_shards and not args.force_rebuild:
                continue

            shard = load_file(shard_path)
            for sid, tensor in shard.items():
                sid = str(sid)
                if sid not in label_map:
                    continue
                label = int(label_map[sid])
                if label == -1:
                    continue
                if tensor.dim() != 3:
                    continue
                if max_layer >= tensor.shape[0]:
                    continue

                processed_samples += 1

                for pooling, spec in poolings.items():
                    layers = [int(v) for v in spec["layers"]]
                    k_values = [int(v) for v in spec["k_values"]]

                    pooled_by_layer: Dict[int, np.ndarray] = {}
                    for layer in layers:
                        pooled_by_layer[layer] = pool_tokens(tensor[layer, :, :], pooling)

                    for layer in layers:
                        if (pooling, layer) not in probe_cache:
                            continue
                        w, b = probe_cache[(pooling, layer)]
                        x = pooled_by_layer[layer]

                        pca = pca_cache[(pooling, layer)]
                        mean = pca["mean"]
                        comps = pca["components"]

                        if mean.shape[0] != x.shape[0] or w.shape[0] != x.shape[0]:
                            raise ValueError(
                                f"Dim mismatch at pooling={pooling} layer={layer}: "
                                f"x={x.shape[0]}, mean={mean.shape[0]}, w={w.shape[0]}"
                            )

                        centered = x - mean
                        orig_logit = float(x @ w + b)
                        orig_prob = sigmoid(orig_logit)
                        orig_pred = int(orig_prob >= 0.5)
                        was_correct = int(orig_pred == label)

                        for k in k_values:
                            comps_k = comps[:k, :]
                            coeff = centered @ comps_k.T
                            removed = coeff @ comps_k
                            clean = x - removed

                            clean_logit = float(clean @ w + b)
                            clean_prob = sigmoid(clean_logit)
                            clean_pred = int(clean_prob >= 0.5)
                            is_correct = int(clean_pred == label)

                            if not (was_correct == 0 and is_correct == 1):
                                continue

                            delta_logit = float(clean_logit - orig_logit)
                            delta_prob = float(clean_prob - orig_prob)

                            item = {
                                "sample_id": sid,
                                "label": label,
                                "orig_logit": orig_logit,
                                "clean_logit": clean_logit,
                                "delta_logit": delta_logit,
                                "orig_prob": orig_prob,
                                "clean_prob": clean_prob,
                                "delta_prob": delta_prob,
                                "orig_vec": x.astype(np.float32),
                                "clean_vec": clean.astype(np.float32),
                            }
                            key = combo_key(pooling, layer, k)
                            counter = update_topk(topk[key], item, int(args.top_n), counter)

                if processed_samples % progress_every == 0:
                    logger.info("Scan progress: processed_samples=%d", processed_samples)

            completed_shards.add(shard_name)
            save_completed_set(progress, "shards", completed_shards)
            progress["processed_samples"] = processed_samples
            save_progress(progress_path, progress)
            save_topk_state(state_path, topk)

        mark_step_completed(progress, STEP_SCAN)
        save_progress(progress_path, progress)
        completed_steps.add(STEP_SCAN)

        update_status(meta_dir, state="running", current_step=STEP_BUILD_JOBS)

        results_dir = run_root / "results"
        tables_root = results_dir / "tables"
        jobs_root = results_dir / "jobs"
        payloads_root = jobs_root / "payloads"
        for p in (tables_root, jobs_root, payloads_root):
            ensure_dir(p)

        exp2_jobs_path = jobs_root / "exp2_single_jobs.jsonl"
        if args.force_rebuild:
            if exp2_jobs_path.exists():
                exp2_jobs_path.unlink()
            for stale_payload in payloads_root.glob("*.npz"):
                stale_payload.unlink()
            existing_exp2 = set()
            done_jobs = set()
        else:
            existing_exp2 = load_existing_job_ids(exp2_jobs_path)
            done_jobs = get_completed_set(progress, "job_units")

        rows: List[dict] = []
        created_jobs = 0

        for combo in combos:
            pooling, layer_str, k_str = combo.split("|")
            layer = int(layer_str)
            k = int(k_str)

            items = [entry[2] for entry in topk[combo]]
            items_sorted = sorted(items, key=lambda it: it["delta_logit"], reverse=True)

            for rank, item in enumerate(items_sorted, start=1):
                rows.append(
                    {
                        "pooling": pooling,
                        "layer": layer,
                        "k": k,
                        "rank": rank,
                        "sample_id": item["sample_id"],
                        "label": item["label"],
                        "orig_logit": item["orig_logit"],
                        "clean_logit": item["clean_logit"],
                        "delta_logit": item["delta_logit"],
                        "orig_prob": item["orig_prob"],
                        "clean_prob": item["clean_prob"],
                        "delta_prob": item["delta_prob"],
                    }
                )

                pair_id = f"{pooling}|{layer}|{k}|{item['sample_id']}"
                sid_hash = sanitize_id(item["sample_id"])

                role_payloads = [
                    ("orig", item["orig_vec"]),
                    ("clean", item["clean_vec"]),
                ]

                for role, vec in role_payloads:
                    vectors = vec.reshape(1, -1).astype(np.float32)
                    for q in questions_cfg["exp2_single"]:
                        job_id = build_job_id(
                            [
                                "exp2",
                                pooling,
                                str(layer),
                                str(k),
                                item["sample_id"],
                                role,
                                q["id"],
                            ]
                        )
                        if (job_id in existing_exp2) or (job_id in done_jobs):
                            continue

                        payload_path = payloads_root / f"{job_id}.npz"
                        np.savez_compressed(payload_path, vectors=vectors)

                        row = {
                            "job_id": job_id,
                            "experiment": "exp2_single",
                            "split": args.split_name,
                            "pooling": pooling,
                            "layer": layer,
                            "k": k,
                            "sample_id": item["sample_id"],
                            "sample_id_hash": sid_hash,
                            "pair_id": pair_id,
                            "role": role,
                            "label": item["label"],
                            "target_layer": layer,
                            "question_id": q["id"],
                            "question_text": q["text"],
                            "num_vectors": 1,
                            "vector_payload_path": str(payload_path.relative_to(jobs_root)),
                            "prompt_template_version": "layer_placeholder_v1",
                            "vector_strategy": role,
                            "orig_logit": item["orig_logit"],
                            "clean_logit": item["clean_logit"],
                            "delta_logit": item["delta_logit"],
                            "orig_prob": item["orig_prob"],
                            "clean_prob": item["clean_prob"],
                            "delta_prob": item["delta_prob"],
                        }
                        append_jsonl(exp2_jobs_path, [row])
                        existing_exp2.add(job_id)
                        done_jobs.add(job_id)
                        created_jobs += 1

            save_completed_set(progress, "job_units", done_jobs)
            save_progress(progress_path, progress)

        ensure_csv(
            tables_root / "exp2_top_flips.csv",
            rows,
            [
                "pooling",
                "layer",
                "k",
                "rank",
                "sample_id",
                "label",
                "orig_logit",
                "clean_logit",
                "delta_logit",
                "orig_prob",
                "clean_prob",
                "delta_prob",
            ],
        )

        write_json(jobs_root / "questions_exp2_v1.json", questions_cfg)

        mark_step_completed(progress, STEP_BUILD_JOBS)
        save_progress(progress_path, progress)
        completed_steps.add(STEP_BUILD_JOBS)

        update_status(meta_dir, state="running", current_step=STEP_FINALIZE)

        summary = {
            "generated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "run_root": str(run_root),
            "counts": {
                "processed_samples": int(processed_samples),
                "selected_rows": len(rows),
                "exp2_jobs_created": int(created_jobs),
                "exp2_jobs_total": len(existing_exp2),
            },
            "paths": {
                "tables_root": str(tables_root),
                "jobs_root": str(jobs_root),
                "exp2_jobs": str(exp2_jobs_path),
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
    p = argparse.ArgumentParser(description="Prepare Exp2 AO vectors/jobs using saved PCA artifacts")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--saved_pca_root", type=str, required=True)
    p.add_argument("--activations_dir", type=str, required=True)
    p.add_argument("--probes_root", type=str, required=True)
    p.add_argument("--matrix_preset", type=str, default="locked_v1")
    p.add_argument("--matrix_json", type=str, default=None)
    p.add_argument("--questions_config", type=str, default=None)
    p.add_argument("--output_root", type=str, default="artifacts")
    p.add_argument("--experiment_name", type=str, default="activation_oracle_exp2")
    p.add_argument("--dataset_name", type=str, default="Deception-InsiderTrading")
    p.add_argument("--probe_set", type=str, default="roleplaying_probes")
    p.add_argument("--variant", type=str, default="locked_v1")
    p.add_argument("--split_name", type=str, default="ood_test")
    p.add_argument("--top_n", type=int, default=5)
    p.add_argument("--progress_every", type=int, default=100, help="Emit log progress every N samples")
    p.add_argument("--force_rebuild", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--run_id", type=str, default=None)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
