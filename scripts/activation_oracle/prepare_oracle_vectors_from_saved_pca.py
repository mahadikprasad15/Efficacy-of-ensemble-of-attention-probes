#!/usr/bin/env python3
"""Prepare global PCA vectors and AO job manifests from saved PCA artifacts.

This script does not use per-sample activations. It builds:
- Exp1: combined top-k PC sums per pooling/layer/k (global, raw direction sums)
- Exp3: individual PC directions per pooling/layer up to max k (global)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
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
STEP_PERSIST_VECTORS = "persist_vectors"
STEP_BUILD_JOBS = "build_jobs"
STEP_FINALIZE = "finalize"


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


def _relpath(child: Path, base: Path) -> str:
    return os.path.relpath(child, base)


def run(args: argparse.Namespace) -> int:
    progress_every = max(1, int(args.progress_every))

    matrix_cfg = load_matrix(args.matrix_preset, args.matrix_json)
    questions_cfg = load_questions(args.questions_config)
    selected_experiments = set(parse_csv_values(args.experiments))
    if not selected_experiments:
        raise ValueError("--experiments must include exp1_combined and/or exp3_per_pc")

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
        "mode": "global",
        "model_name": model_name,
        "dataset_name": args.dataset_name,
        "probe_set": args.probe_set,
        "variant": args.variant,
        "saved_pca_root": str(Path(args.saved_pca_root).resolve()),
        "matrix": matrix_cfg,
        "questions_config": questions_cfg,
        "experiments": sorted(selected_experiments),
        "force_rebuild": bool(args.force_rebuild),
    }
    write_json(meta_dir / "run_manifest.json", manifest)
    write_json(inputs_dir / "matrix_locked.json", matrix_cfg)
    write_json(inputs_dir / "questions_default.json", questions_cfg)

    logger.info("Run root: %s", run_root)
    logger.info("Poolings: %s", ", ".join(sorted(matrix_cfg["poolings"].keys())))
    logger.info("Experiments: %s", ", ".join(sorted(selected_experiments)))

    init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_VALIDATE)

    try:
        poolings = matrix_cfg["poolings"]
        used_poolings = sorted(poolings.keys())
        if any(p not in ("mean", "last") for p in used_poolings):
            raise ValueError("This pipeline currently supports only mean and last pooling")

        pca_root = Path(args.saved_pca_root).resolve()

        if args.force_rebuild or STEP_VALIDATE not in completed_steps:
            validation_units: List[Tuple[str, int, int]] = []
            for pooling, spec in poolings.items():
                max_k = max(int(k) for k in spec["k_values"])
                for layer in spec["layers"]:
                    validation_units.append((pooling, int(layer), max_k))

            for pooling, layer, max_k in tqdm(
                validation_units,
                desc="Validate PCA inputs",
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
            mark_step_completed(progress, STEP_VALIDATE)
            save_progress(progress_path, progress)
            completed_steps.add(STEP_VALIDATE)

        update_status(meta_dir, state="running", current_step=STEP_PERSIST_VECTORS)

        vectors_root = results_dir / "vectors"
        jobs_root = results_dir / "jobs"
        ensure_dir(vectors_root)
        ensure_dir(jobs_root)

        if args.force_rebuild:
            for stale in vectors_root.rglob("*.npz"):
                stale.unlink()
            for stale in vectors_root.rglob("*.json"):
                stale.unlink()
            if jobs_root.exists():
                for stale in jobs_root.rglob("*.jsonl"):
                    stale.unlink()
                for stale in jobs_root.rglob("*.json"):
                    stale.unlink()

        vector_done = get_completed_set(progress, "vector_units")
        vector_rows: List[dict] = []
        written_vectors = 0
        skipped_vectors = 0

        vector_units: List[Tuple[str, str, int, Optional[int], str]] = []
        for pooling, spec in poolings.items():
            layers = sorted(int(v) for v in spec["layers"])
            k_values = sorted(int(v) for v in spec["k_values"])
            max_k = max(k_values)
            for layer in layers:
                if "exp1_combined" in selected_experiments:
                    for k in k_values:
                        vector_units.append(("exp1_combined", pooling, layer, k, f"k_{k}"))
                if "exp3_per_pc" in selected_experiments:
                    for pc_idx in range(max_k):
                        vector_units.append(("exp3_per_pc", pooling, layer, pc_idx, f"pc_{pc_idx+1}"))

        vector_bar = tqdm(vector_units, desc="Persist PCA vectors", unit="vector", leave=True)

        components_cache: Dict[Tuple[str, int], np.ndarray] = {}
        dims_cache: Dict[Tuple[str, int], int] = {}

        for exp_name, pooling, layer, idx, idx_tag in vector_bar:
            cache_key = (pooling, layer)
            if cache_key not in components_cache:
                pca_path = pca_root / pooling / "pca_artifacts" / f"layer_{layer}.npz"
                with np.load(pca_path, allow_pickle=False) as pca_data:
                    components_cache[cache_key] = pca_data["components"].astype(np.float32)
                dims_cache[cache_key] = int(components_cache[cache_key].shape[1])
            components = components_cache[cache_key]
            dim = dims_cache[cache_key]

            unit_id = f"{exp_name}|{pooling}|{layer}|{idx_tag}"
            if (unit_id in vector_done) and not args.force_rebuild:
                skipped_vectors += 1
                vector_bar.set_postfix({"written": written_vectors, "skipped": skipped_vectors})
                continue

            if exp_name == "exp1_combined":
                k = int(idx) if idx is not None else 0
                vec = components[:k, :].sum(axis=0)
                vector_dir = vectors_root / "exp1_combined" / pooling / f"layer_{layer}" / f"k_{k}"
                meta = {
                    "generated_at_utc": utc_now_iso(),
                    "experiment": "exp1_combined",
                    "pooling": pooling,
                    "layer": layer,
                    "k": k,
                    "pc_indices": list(range(k)),
                    "vector_dim": dim,
                    "pca_path": str(pca_root / pooling / "pca_artifacts" / f"layer_{layer}.npz"),
                    "vector_strategy": "raw_pc_sum",
                }
            elif exp_name == "exp3_per_pc":
                pc_idx = int(idx) if idx is not None else 0
                vec = components[pc_idx, :]
                vector_dir = vectors_root / "exp3_per_pc" / pooling / f"layer_{layer}" / f"pc_{pc_idx+1}"
                meta = {
                    "generated_at_utc": utc_now_iso(),
                    "experiment": "exp3_per_pc",
                    "pooling": pooling,
                    "layer": layer,
                    "pc_index": pc_idx,
                    "pc_rank": pc_idx + 1,
                    "vector_dim": dim,
                    "pca_path": str(pca_root / pooling / "pca_artifacts" / f"layer_{layer}.npz"),
                    "vector_strategy": "raw_pc_direction",
                }
            else:
                raise ValueError(f"Unsupported experiment: {exp_name}")

            ensure_dir(vector_dir)
            vector_path = vector_dir / "vector.npz"
            meta_path = vector_dir / "vector_meta.json"
            np.savez_compressed(vector_path, vectors=vec.reshape(1, -1).astype(np.float32))
            write_json(meta_path, meta)

            vector_rows.append(
                {
                    "unit_id": unit_id,
                    "experiment": exp_name,
                    "pooling": pooling,
                    "layer": layer,
                    "index_tag": idx_tag,
                    "vector_path": str(vector_path),
                    "meta_path": str(meta_path),
                }
            )

            vector_done.add(unit_id)
            written_vectors += 1
            vector_bar.set_postfix({"written": written_vectors, "skipped": skipped_vectors})
            if written_vectors % progress_every == 0:
                logger.info("Vector progress: written=%d skipped=%d", written_vectors, skipped_vectors)
            save_completed_set(progress, "vector_units", vector_done)
            save_progress(progress_path, progress)

        vector_bar.close()

        ensure_csv(
            vectors_root / "vectors_index.csv",
            vector_rows,
            ["unit_id", "experiment", "pooling", "layer", "index_tag", "vector_path", "meta_path"],
        )

        mark_step_completed(progress, STEP_PERSIST_VECTORS)
        save_progress(progress_path, progress)
        completed_steps.add(STEP_PERSIST_VECTORS)

        update_status(meta_dir, state="running", current_step=STEP_BUILD_JOBS)

        exp1_jobs_path = jobs_root / "exp1_combined_jobs.jsonl"
        exp3_jobs_path = jobs_root / "exp3_per_pc_jobs.jsonl"

        if args.force_rebuild:
            if exp1_jobs_path.exists():
                exp1_jobs_path.unlink()
            if exp3_jobs_path.exists():
                exp3_jobs_path.unlink()
            existing_exp1 = set()
            existing_exp3 = set()
            done_jobs = set()
        else:
            existing_exp1 = load_existing_job_ids(exp1_jobs_path)
            existing_exp3 = load_existing_job_ids(exp3_jobs_path)
            done_jobs = get_completed_set(progress, "job_units")

        created_counts = {"exp1_combined": 0, "exp3_per_pc": 0}
        jobs_bar = tqdm(desc="Build AO jobs", unit="job", leave=True)

        for pooling, spec in poolings.items():
            layers = sorted(int(v) for v in spec["layers"])
            k_values = sorted(int(v) for v in spec["k_values"])
            max_k = max(k_values)
            for layer in layers:
                if "exp1_combined" in selected_experiments:
                    for k in k_values:
                        vector_dir = vectors_root / "exp1_combined" / pooling / f"layer_{layer}" / f"k_{k}"
                        vector_path = vector_dir / "vector.npz"
                        if not vector_path.exists():
                            raise FileNotFoundError(f"Missing vector payload: {vector_path}")
                        payload_rel = _relpath(vector_path, jobs_root)
                        for q in questions_cfg["exp1_combined"]:
                            job_id = build_job_id([
                                "exp1",
                                pooling,
                                str(layer),
                                str(k),
                                q["id"],
                            ])
                            if (job_id in existing_exp1) or (job_id in done_jobs):
                                continue

                            row = {
                                "job_id": job_id,
                                "experiment": "exp1_combined",
                                "scope": "global",
                                "split": "global",
                                "pooling": pooling,
                                "layer": layer,
                                "k": k,
                                "target_layer": layer,
                                "question_id": q["id"],
                                "question_text": q["text"],
                                "num_vectors": 1,
                                "vector_payload_path": payload_rel,
                                "prompt_template_version": "layer_placeholder_v1",
                                "vector_strategy": "raw_pc_sum",
                            }
                            append_jsonl(exp1_jobs_path, [row])
                            existing_exp1.add(job_id)
                            done_jobs.add(job_id)
                            created_counts["exp1_combined"] += 1
                            jobs_bar.update(1)

                if "exp3_per_pc" in selected_experiments:
                    for pc_idx in range(max_k):
                        vector_dir = vectors_root / "exp3_per_pc" / pooling / f"layer_{layer}" / f"pc_{pc_idx+1}"
                        vector_path = vector_dir / "vector.npz"
                        if not vector_path.exists():
                            raise FileNotFoundError(f"Missing vector payload: {vector_path}")
                        payload_rel = _relpath(vector_path, jobs_root)
                        for q in questions_cfg["exp3_per_pc"]:
                            job_id = build_job_id([
                                "exp3",
                                pooling,
                                str(layer),
                                str(max_k),
                                str(pc_idx),
                                q["id"],
                            ])
                            if (job_id in existing_exp3) or (job_id in done_jobs):
                                continue

                            row = {
                                "job_id": job_id,
                                "experiment": "exp3_per_pc",
                                "scope": "global",
                                "split": "global",
                                "pooling": pooling,
                                "layer": layer,
                                "k": max_k,
                                "pc_index": pc_idx,
                                "pc_rank": pc_idx + 1,
                                "pc_limit": max_k,
                                "target_layer": layer,
                                "question_id": q["id"],
                                "question_text": q["text"],
                                "num_vectors": 1,
                                "vector_payload_path": payload_rel,
                                "prompt_template_version": "layer_placeholder_v1",
                                "vector_strategy": "raw_pc_direction",
                            }
                            append_jsonl(exp3_jobs_path, [row])
                            existing_exp3.add(job_id)
                            done_jobs.add(job_id)
                            created_counts["exp3_per_pc"] += 1
                            jobs_bar.update(1)

                if (created_counts["exp1_combined"] + created_counts["exp3_per_pc"]) % progress_every == 0:
                    logger.info(
                        "Job build progress: exp1_new=%d exp3_new=%d",
                        created_counts["exp1_combined"],
                        created_counts["exp3_per_pc"],
                    )
                save_completed_set(progress, "job_units", done_jobs)
                save_progress(progress_path, progress)

        jobs_bar.close()

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
                "vectors_written": written_vectors,
                "vectors_skipped": skipped_vectors,
                "exp1_jobs_created": created_counts["exp1_combined"],
                "exp3_jobs_created": created_counts["exp3_per_pc"],
                "exp1_jobs_total": len(existing_exp1),
                "exp3_jobs_total": len(existing_exp3),
            },
            "paths": {
                "vectors_root": str(vectors_root),
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
    p = argparse.ArgumentParser(description="Prepare global AO vectors/jobs using saved PCA artifacts")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--saved_pca_root", type=str, required=True)
    p.add_argument("--matrix_preset", type=str, default="locked_v1")
    p.add_argument("--matrix_json", type=str, default=None)
    p.add_argument("--questions_config", type=str, default=None)
    p.add_argument("--experiments", type=str, default="exp1_combined,exp3_per_pc")
    p.add_argument("--output_root", type=str, default="artifacts")
    p.add_argument("--experiment_name", type=str, default="activation_oracle_pca_global")
    p.add_argument("--dataset_name", type=str, default="Deception-Roleplaying")
    p.add_argument("--probe_set", type=str, default="roleplaying_probes")
    p.add_argument("--variant", type=str, default="locked_v1")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--progress_every", type=int, default=25, help="Emit log progress every N units")
    p.add_argument("--force_rebuild", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
