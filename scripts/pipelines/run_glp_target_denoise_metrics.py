#!/usr/bin/env python3
"""
Run GLP denoising metrics on target activations only.

This script:
1) Loads labeled activations from target split directory
2) Selects a deterministic sample subset (default 100)
3) Applies GLP denoising on layer activations across timesteps/seeds
4) Computes class-conditional drift/separation metrics
5) Saves resumable artifacts under results/GLP_Experiments
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import logging
import os
import random
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
from glp_runner import denoise_on_manifold, ensure_bsd, load_glp_artifacts  # noqa: E402


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


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def init_logger(log_path: Path, verbose: bool) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("glp_target_denoise")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)
    return logger


def set_status(meta_dir: Path, run_id: str, state: str, current_step: Optional[str]) -> None:
    payload = {
        "run_id": run_id,
        "state": state,
        "current_step": current_step,
        "last_updated_utc": utc_now_iso(),
    }
    write_json(meta_dir / "status.json", payload)


def parse_timesteps(spec: str) -> List[int]:
    vals: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("No timesteps provided.")
    return vals


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def describe(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def resolve_run_dir(output_root: Path, target_dir: Path, layer: int, run_id: str) -> Path:
    split = target_dir.name
    dataset = target_dir.parent.name
    model_dir = target_dir.parent.parent.name
    return (
        output_root
        / "glp_target_denoise"
        / model_dir
        / dataset
        / split
        / f"layer_{layer}"
        / run_id
    )


def load_manifest_labels(target_dir: Path) -> Dict[str, int]:
    manifest_path = target_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {target_dir}")

    out: Dict[str, int] = {}
    with manifest_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("id", ""))
            if not sid:
                continue
            out[sid] = int(row.get("label", -1))
    return out


def select_sample_ids(
    label_map: Dict[str, int],
    max_samples: int,
    seed: int,
) -> List[str]:
    labeled = [sid for sid, label in label_map.items() if label in (0, 1)]
    if not labeled:
        raise RuntimeError("No labeled samples found (labels 0/1).")

    if max_samples <= 0 or max_samples >= len(labeled):
        return sorted(labeled)

    rng = random.Random(int(seed))
    sampled = rng.sample(labeled, int(max_samples))
    return sorted(sampled)


def load_selected_layer_activations(
    target_dir: Path,
    sample_ids: Sequence[str],
    label_map: Dict[str, int],
    layer: int,
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    sample_set = set(sample_ids)
    found: Dict[str, torch.Tensor] = {}

    shard_paths = sorted(glob.glob(str(target_dir / "shard_*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors found in {target_dir}")

    for shard_path in shard_paths:
        shard = load_file(shard_path)
        for sid, tensor in shard.items():
            sid = str(sid)
            if sid not in sample_set:
                continue
            if tensor.ndim == 3:
                if layer >= int(tensor.shape[0]):
                    raise ValueError(
                        f"Layer {layer} out of range for sample {sid}, tensor has {int(tensor.shape[0])} layers."
                    )
                x = tensor[layer, :, :]  # (S,D)
            elif tensor.ndim == 2:
                if layer >= int(tensor.shape[0]):
                    raise ValueError(
                        f"Layer {layer} out of range for final-token sample {sid}, tensor has {int(tensor.shape[0])} layers."
                    )
                x = tensor[layer, :].unsqueeze(0)  # (1,D)
            else:
                raise ValueError(f"Unexpected tensor shape {tuple(tensor.shape)} for sample {sid}")
            found[sid] = x.detach().cpu().float()

    missing = [sid for sid in sample_ids if sid not in found]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected samples in shards.")

    ordered_ids = list(sample_ids)
    xs = [found[sid] for sid in ordered_ids]
    first_shape = tuple(xs[0].shape)
    for sid, x in zip(ordered_ids, xs):
        if tuple(x.shape) != first_shape:
            raise ValueError(f"Variable token shapes not supported. {sid} has {tuple(x.shape)}, expected {first_shape}")

    x_bsd = torch.stack(xs, dim=0)  # (B,S,D)
    x_bsd = ensure_bsd(x_bsd)
    labels = np.array([int(label_map[sid]) for sid in ordered_ids], dtype=np.int64)
    return x_bsd, labels, ordered_ids


def resolve_probe_weight(probe_dir: Path, probe_layer: int) -> Tuple[np.ndarray, float, str]:
    probe_path = probe_dir / f"probe_layer_{probe_layer}.pt"
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing probe file: {probe_path}")

    state = torch.load(probe_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected probe checkpoint type for {probe_path}")

    if "classifier.weight" in state:
        w = state["classifier.weight"].reshape(-1).detach().cpu().numpy().astype(np.float32)
        b = float(state.get("classifier.bias", torch.tensor([0.0])).reshape(-1)[0].item())
        return w, b, str(probe_path)

    if "linear.weight" in state:
        w = state["linear.weight"].reshape(-1).detach().cpu().numpy().astype(np.float32)
        b = float(state.get("linear.bias", torch.tensor([0.0])).reshape(-1)[0].item())
        return w, b, str(probe_path)

    # Fallback: first 2D weight matrix in state dict.
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            w = value.reshape(-1).detach().cpu().numpy().astype(np.float32)
            return w, 0.0, str(probe_path)

    raise ValueError(f"Could not extract probe direction from {probe_path}")


def class_cov_trace(x: np.ndarray) -> float:
    if x.shape[0] <= 1:
        return 0.0
    cov = np.cov(x, rowvar=False)
    return float(np.trace(cov))


def run_target_experiment(args: argparse.Namespace, target_dir: Path, logger: logging.Logger) -> None:
    target_dir = target_dir.resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Target dir does not exist: {target_dir}")

    run_id = args.run_id if args.run_id else make_run_id()
    run_dir = resolve_run_dir(Path(args.output_root), target_dir, args.layer, run_id)
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    for d in [meta_dir, checkpoints_dir, results_dir, logs_dir]:
        ensure_dir(d)

    set_status(meta_dir, run_id, "running", "init")

    progress_path = checkpoints_dir / "progress.json"
    if progress_path.exists() and args.resume:
        progress = read_json(progress_path)
    else:
        progress = {
            "run_id": run_id,
            "completed_units": [],
            "sample_ids": [],
            "updated_at_utc": utc_now_iso(),
        }

    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "target_dir": str(target_dir),
            "probe_dir": str(Path(args.probe_dir).resolve()),
            "probe_layer": int(args.probe_layer),
            "glp_model": args.glp_model,
        },
        "config": {
            "layer": int(args.layer),
            "timesteps": parse_timesteps(args.num_timesteps),
            "start_timestep_mode": args.start_timestep_mode,
            "noise_scale": float(args.noise_scale),
            "num_seeds": int(args.num_seeds),
            "max_samples": int(args.max_samples),
            "sample_seed": int(args.sample_seed),
            "batch_size": int(args.batch_size),
            "device": args.device,
            "resume": bool(args.resume),
            "note": "Metrics computed on token-mean pooled vectors for direction/separation; delta_norm from token-level tensors.",
        },
    }
    write_json(meta_dir / "run_manifest.json", manifest)

    set_status(meta_dir, run_id, "running", "load_inputs")
    label_map = load_manifest_labels(target_dir)
    if progress.get("sample_ids") and args.resume:
        sample_ids = list(progress["sample_ids"])
        logger.info("Using %d sample IDs from progress.", len(sample_ids))
    else:
        sample_ids = select_sample_ids(label_map, max_samples=int(args.max_samples), seed=int(args.sample_seed))
        progress["sample_ids"] = sample_ids
        progress["updated_at_utc"] = utc_now_iso()
        write_json(progress_path, progress)
    write_json(results_dir / "sample_ids.json", {"sample_ids": sample_ids, "count": len(sample_ids)})

    x_bsd, labels, sample_ids = load_selected_layer_activations(
        target_dir=target_dir,
        sample_ids=sample_ids,
        label_map=label_map,
        layer=int(args.layer),
    )
    logger.info("Loaded activations shape: %s (B,S,D)", tuple(x_bsd.shape))

    set_status(meta_dir, run_id, "running", "load_probe")
    probe_w, probe_b, probe_path = resolve_probe_weight(Path(args.probe_dir), int(args.probe_layer))
    d_model = int(x_bsd.shape[-1])
    if probe_w.shape[0] != d_model:
        raise ValueError(
            f"Probe dim mismatch: probe weight dim {probe_w.shape[0]} vs activation dim {d_model}"
        )
    probe_w_hat = probe_w / (np.linalg.norm(probe_w) + 1e-8)

    pooled_x = x_bsd.mean(dim=1).numpy()  # (B,D)
    orig_logits = pooled_x @ probe_w + probe_b
    orig_auc = safe_auc(labels, orig_logits)

    set_status(meta_dir, run_id, "running", "load_glp")
    artifacts = load_glp_artifacts(
        model_id=args.glp_model,
        device=args.device,
        checkpoint=args.glp_checkpoint,
    )
    logger.info("Loaded GLP model: %s", args.glp_model)

    timesteps = parse_timesteps(args.num_timesteps)
    num_seeds = int(args.num_seeds)
    batch_size = int(args.batch_size)
    n = int(x_bsd.shape[0])

    all_per_sample_rows: List[dict] = []
    all_stability_rows: List[dict] = []
    by_timestep_rows: List[dict] = []
    by_class_rows: List[dict] = []
    results_by_timestep: Dict[str, dict] = {}

    set_status(meta_dir, run_id, "running", "denoise_and_metrics")
    completed = set(progress.get("completed_units", []))
    total_units = len(timesteps) * num_seeds
    overall_pbar = tqdm(total=total_units, desc="GLP units", unit="unit")
    for t in timesteps:
        pooled_prime_by_seed = np.zeros((num_seeds, n, d_model), dtype=np.float32)
        delta_norm_by_seed = np.zeros((num_seeds, n), dtype=np.float32)
        delta_parallel_by_seed = np.zeros((num_seeds, n), dtype=np.float32)
        logits_prime_by_seed = np.zeros((num_seeds, n), dtype=np.float32)
        start_timestep_values: Dict[int, Optional[float]] = {}

        for seed in range(num_seeds):
            unit_key = f"t{t}_s{seed}"
            ckpt_npz = checkpoints_dir / f"{unit_key}.npz"
            if unit_key in completed and ckpt_npz.exists() and args.resume:
                blob = np.load(ckpt_npz, allow_pickle=False)
                pooled_prime_by_seed[seed] = blob["pooled_x_prime"].astype(np.float32)
                delta_norm_by_seed[seed] = blob["delta_norm"].astype(np.float32)
                delta_parallel_by_seed[seed] = blob["delta_parallel"].astype(np.float32)
                logits_prime_by_seed[seed] = blob["logits_x_prime"].astype(np.float32)
                start_timestep_values[seed] = float(blob["start_timestep"].item()) if "start_timestep" in blob else None
                logger.info("Resumed unit %s from checkpoint.", unit_key)
                overall_pbar.update(1)
                continue

            logger.info("Running unit %s ...", unit_key)
            pooled_out_chunks: List[np.ndarray] = []
            delta_norm_chunks: List[np.ndarray] = []
            delta_parallel_chunks: List[np.ndarray] = []
            logits_chunks: List[np.ndarray] = []
            start_timestep_value: Optional[float] = None

            unit_pbar = tqdm(total=n, desc=f"{unit_key} samples", unit="sample", leave=False)
            for b0 in range(0, n, batch_size):
                b1 = min(n, b0 + batch_size)
                xb = x_bsd[b0:b1].to(args.device)
                x_prime, start_timestep_value = denoise_on_manifold(
                    artifacts=artifacts,
                    latents_bsd=xb,
                    layer_idx=int(args.layer),
                    num_timesteps=int(t),
                    start_timestep_mode=args.start_timestep_mode,
                    noise_scale=float(args.noise_scale),
                    seed=int(seed),
                )
                delta = x_prime - xb
                delta_norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=1).detach().cpu().numpy()
                pooled_prime = x_prime.mean(dim=1).detach().cpu().numpy()

                pooled_out_chunks.append(pooled_prime.astype(np.float32))
                delta_norm_chunks.append(delta_norm.astype(np.float32))

                d_parallel = (pooled_prime - pooled_x[b0:b1]) @ probe_w_hat
                logits_prime = pooled_prime @ probe_w + probe_b
                delta_parallel_chunks.append(d_parallel.astype(np.float32))
                logits_chunks.append(logits_prime.astype(np.float32))
                unit_pbar.update(b1 - b0)
            unit_pbar.close()

            pooled_prime_arr = np.concatenate(pooled_out_chunks, axis=0)
            delta_norm_arr = np.concatenate(delta_norm_chunks, axis=0)
            delta_parallel_arr = np.concatenate(delta_parallel_chunks, axis=0)
            logits_prime_arr = np.concatenate(logits_chunks, axis=0)

            pooled_prime_by_seed[seed] = pooled_prime_arr
            delta_norm_by_seed[seed] = delta_norm_arr
            delta_parallel_by_seed[seed] = delta_parallel_arr
            logits_prime_by_seed[seed] = logits_prime_arr
            start_timestep_values[seed] = start_timestep_value

            np.savez_compressed(
                ckpt_npz,
                pooled_x_prime=pooled_prime_arr,
                delta_norm=delta_norm_arr,
                delta_parallel=delta_parallel_arr,
                logits_x_prime=logits_prime_arr,
                start_timestep=np.array(start_timestep_value if start_timestep_value is not None else -1.0, dtype=np.float32),
            )
            completed.add(unit_key)
            progress["completed_units"] = sorted(completed)
            progress["updated_at_utc"] = utc_now_iso()
            write_json(progress_path, progress)
            overall_pbar.update(1)

        logits_prime_mean = logits_prime_by_seed.mean(axis=0)
        pooled_prime_mean = pooled_prime_by_seed.mean(axis=0)
        pooled_prime_var = pooled_prime_by_seed.var(axis=0)
        delta_norm_mean = delta_norm_by_seed.mean(axis=0)
        delta_parallel_mean = delta_parallel_by_seed.mean(axis=0)
        drift = pooled_prime_mean - pooled_x
        drift_norm = np.linalg.norm(drift, axis=1)
        var_trace = pooled_prime_var.sum(axis=1)

        auc_prime_mean = safe_auc(labels, logits_prime_mean)
        auc_prime_seed_mean = float(np.mean([safe_auc(labels, logits_prime_by_seed[s]) for s in range(num_seeds)]))

        pos = labels == 1
        neg = labels == 0

        mu_pos_x = pooled_x[pos].mean(axis=0) if np.any(pos) else np.zeros(d_model, dtype=np.float32)
        mu_neg_x = pooled_x[neg].mean(axis=0) if np.any(neg) else np.zeros(d_model, dtype=np.float32)
        mu_pos_p = pooled_prime_mean[pos].mean(axis=0) if np.any(pos) else np.zeros(d_model, dtype=np.float32)
        mu_neg_p = pooled_prime_mean[neg].mean(axis=0) if np.any(neg) else np.zeros(d_model, dtype=np.float32)

        sep_x = float(np.linalg.norm(mu_pos_x - mu_neg_x))
        sep_x_prime = float(np.linalg.norm(mu_pos_p - mu_neg_p))

        cov_trace_x_pos = class_cov_trace(pooled_x[pos])
        cov_trace_x_neg = class_cov_trace(pooled_x[neg])
        cov_trace_p_pos = class_cov_trace(pooled_prime_mean[pos])
        cov_trace_p_neg = class_cov_trace(pooled_prime_mean[neg])
        pooled_cov_x = 0.5 * (cov_trace_x_pos + cov_trace_x_neg)
        pooled_cov_p = 0.5 * (cov_trace_p_pos + cov_trace_p_neg)
        sep_norm_x = float(sep_x / (np.sqrt(pooled_cov_x) + 1e-8))
        sep_norm_p = float(sep_x_prime / (np.sqrt(pooled_cov_p) + 1e-8))

        timestep_key = f"t{t}"
        results_by_timestep[timestep_key] = {
            "num_timesteps": int(t),
            "start_timestep_mode": args.start_timestep_mode,
            "start_timestep_values": {str(k): (None if v is None else float(v)) for k, v in start_timestep_values.items()},
            "noise_scale": float(args.noise_scale),
            "num_seeds": num_seeds,
            "num_samples": n,
            "auc_x": float(orig_auc),
            "auc_xprime_mean": float(auc_prime_mean),
            "auc_xprime_seed_mean": float(auc_prime_seed_mean),
            "separation_l2_x": sep_x,
            "separation_l2_xprime": sep_x_prime,
            "separation_norm_x": sep_norm_x,
            "separation_norm_xprime": sep_norm_p,
            "cov_trace_x_pos": cov_trace_x_pos,
            "cov_trace_x_neg": cov_trace_x_neg,
            "cov_trace_xprime_pos": cov_trace_p_pos,
            "cov_trace_xprime_neg": cov_trace_p_neg,
            "delta_norm_all": describe(delta_norm_mean),
            "delta_parallel_all": describe(delta_parallel_mean),
            "drift_norm_all": describe(drift_norm),
            "var_trace_all": describe(var_trace),
        }
        by_timestep_rows.append(
            {
                "num_timesteps": int(t),
                "start_timestep_mode": args.start_timestep_mode,
                "noise_scale": float(args.noise_scale),
                "num_seeds": int(num_seeds),
                "num_samples": int(n),
                "auc_x": float(orig_auc),
                "auc_xprime_mean": float(auc_prime_mean),
                "auc_xprime_seed_mean": float(auc_prime_seed_mean),
                "separation_l2_x": sep_x,
                "separation_l2_xprime": sep_x_prime,
                "separation_norm_x": sep_norm_x,
                "separation_norm_xprime": sep_norm_p,
                "cov_trace_x_pos": cov_trace_x_pos,
                "cov_trace_x_neg": cov_trace_x_neg,
                "cov_trace_xprime_pos": cov_trace_p_pos,
                "cov_trace_xprime_neg": cov_trace_p_neg,
            }
        )

        for label_value, label_name in [(0, "honest"), (1, "deceptive")]:
            idx = labels == label_value
            by_class_rows.append(
                {
                    "num_timesteps": int(t),
                    "class": label_name,
                    "count": int(np.sum(idx)),
                    **{f"delta_norm_{k}": v for k, v in describe(delta_norm_mean[idx]).items()},
                    **{f"delta_parallel_{k}": v for k, v in describe(delta_parallel_mean[idx]).items()},
                    **{f"drift_norm_{k}": v for k, v in describe(drift_norm[idx]).items()},
                    **{f"var_trace_{k}": v for k, v in describe(var_trace[idx]).items()},
                }
            )

        # Seed-level per-sample rows.
        for i, sid in enumerate(sample_ids):
            for seed in range(num_seeds):
                all_per_sample_rows.append(
                    {
                        "sample_id": sid,
                        "label": int(labels[i]),
                        "num_timesteps": int(t),
                        "seed": int(seed),
                        "start_timestep": None if start_timestep_values.get(seed) is None else float(start_timestep_values[seed]),
                        "delta_norm": float(delta_norm_by_seed[seed, i]),
                        "delta_parallel": float(delta_parallel_by_seed[seed, i]),
                        "probe_logit_x": float(orig_logits[i]),
                        "probe_logit_xprime": float(logits_prime_by_seed[seed, i]),
                    }
                )

        # Stability rows.
        for i, sid in enumerate(sample_ids):
            all_stability_rows.append(
                {
                    "sample_id": sid,
                    "label": int(labels[i]),
                    "num_timesteps": int(t),
                    "delta_norm_mean": float(delta_norm_mean[i]),
                    "delta_parallel_mean": float(delta_parallel_mean[i]),
                    "drift_norm_mean": float(drift_norm[i]),
                    "var_trace": float(var_trace[i]),
                    "probe_logit_x": float(orig_logits[i]),
                    "probe_logit_xprime_mean": float(logits_prime_mean[i]),
                }
            )

    overall_pbar.close()

    set_status(meta_dir, run_id, "running", "write_results")
    write_jsonl(results_dir / "per_sample.jsonl", all_per_sample_rows)
    write_jsonl(results_dir / "per_sample_stability.jsonl", all_stability_rows)

    with (results_dir / "summary_by_timestep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_timestep_rows[0].keys()) if by_timestep_rows else [])
        writer.writeheader()
        writer.writerows(by_timestep_rows)

    with (results_dir / "summary_by_class.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_class_rows[0].keys()) if by_class_rows else [])
        writer.writeheader()
        writer.writerows(by_class_rows)

    results_payload = {
        "run_id": run_id,
        "target_dir": str(target_dir),
        "probe_path": probe_path,
        "layer": int(args.layer),
        "probe_layer": int(args.probe_layer),
        "timesteps": timesteps,
        "start_timestep_mode": args.start_timestep_mode,
        "noise_scale": float(args.noise_scale),
        "num_seeds": int(args.num_seeds),
        "num_samples": int(n),
        "results_by_timestep": results_by_timestep,
    }
    write_json(results_dir / "results.json", results_payload)

    progress["updated_at_utc"] = utc_now_iso()
    write_json(progress_path, progress)
    set_status(meta_dir, run_id, "completed", None)
    logger.info("Completed target experiment: %s", target_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLP target denoise metrics")
    parser.add_argument("--target_dir", type=str, action="append", required=True, help="Target activations split dir. Repeatable.")
    parser.add_argument("--probe_dir", type=str, required=True, help="Directory containing probe_layer_*.pt")
    parser.add_argument("--probe_layer", type=int, default=7, help="Probe layer index for direction extraction")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name for metadata")
    parser.add_argument("--glp_model", type=str, default="generative-latent-prior/glp-llama1b-d12")
    parser.add_argument("--glp_checkpoint", type=str, default="final")
    parser.add_argument("--layer", type=int, default=7, help="Activation layer index to denoise")
    parser.add_argument("--num_timesteps", type=str, default="50,100")
    parser.add_argument("--start_timestep_mode", type=str, default="half", help="none|half|idx:<n>|frac:<0..1>")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=100, help="<=0 for full split")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_root", type=str, default="results/GLP_Experiments")
    parser.add_argument("--run_id", type=str, default="pilot")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    # Global log stream; run-specific logs are written in each target run directory.
    logger = logging.getLogger("glp_target_denoise_main")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.addHandler(sh)

    for target_str in args.target_dir:
        target = Path(target_str).resolve()
        run_dir = resolve_run_dir(output_root, target, args.layer, args.run_id if args.run_id else make_run_id())
        run_logger = init_logger(run_dir / "logs" / "run.log", verbose=bool(args.verbose))
        run_logger.info("Starting target: %s", target)
        run_target_experiment(args, target, run_logger)

    logger.info("All target runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
