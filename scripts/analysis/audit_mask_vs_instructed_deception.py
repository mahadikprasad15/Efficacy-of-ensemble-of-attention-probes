#!/usr/bin/env python3
"""
Audit whether Deception-Mask and Deception-InstructedDeception artifacts are distinct.

This script compares four layers of the pipeline:
  1. raw dataset files
  2. cached activation manifests
  3. trained probe validation curves and probe checkpoints
  4. pairwise evaluation summaries and logits manifests

Outputs are written to a resumable artifact tree:
  artifacts/runs/mask_vs_instructed_deception_audit/<model_dir>/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


SEGMENTS = ["completion", "full"]
SPLITS = ["train", "validation", "test"]
POOLINGS = ["mean", "max", "last", "attn"]
SOURCE_BASES = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
    "Deception-Roleplaying",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_status(path: Path, state: str, message: str) -> None:
    payload = read_json(path, default={})
    payload["state"] = state
    payload["message"] = message
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def normalize_pooling(pooling: Any) -> str:
    p = str(pooling).strip().lower()
    if p in {"none", "final", "final_token"}:
        return "last"
    return p


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def dataset_name(base: str, segment: str) -> str:
    return f"{base}-{segment}"


def split_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    if root.name == model_dir:
        return root.parent, root
    candidate = root / model_dir
    if candidate.exists():
        return root, candidate
    return root.parent, root


def append_unique(paths: List[Path], candidate: Path) -> None:
    candidate_str = str(candidate)
    if all(str(existing) != candidate_str for existing in paths):
        paths.append(candidate)


def discover_pair_roots(ood_results_root: Path, model_dir: str) -> List[Path]:
    roots: List[Path] = []
    _, model_root = split_root_and_model(ood_results_root, model_dir)
    append_unique(roots, model_root)
    append_unique(roots, ood_results_root)
    if ood_results_root.name != model_dir:
        append_unique(roots, ood_results_root / model_dir)
    return roots


def normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if raw_messages is None:
        return []
    if hasattr(raw_messages, "tolist") and not isinstance(raw_messages, list):
        raw_messages = raw_messages.tolist()
    if isinstance(raw_messages, str):
        text = raw_messages.strip()
        if not text:
            return []
        try:
            raw_messages = json.loads(text)
        except Exception:
            return []
    if isinstance(raw_messages, dict):
        raw_messages = [raw_messages]
    if not isinstance(raw_messages, list):
        return []

    messages: List[Dict[str, str]] = []
    for msg in raw_messages:
        if hasattr(msg, "tolist") and not isinstance(msg, dict):
            msg = msg.tolist()
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        content = msg.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        messages.append({"role": role, "content": content})
    return messages


def coerce_binary_label(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        val = int(raw)
        return val if val in {0, 1} else None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"1", "true", "yes", "deceptive", "negative"}:
            return 1
        if s in {"0", "false", "no", "honest", "positive", "non_deceptive"}:
            return 0
    return None


def parse_typed_row(row: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    label = None
    for key in ("label", "deceptive", "gold_label"):
        if key in row and row.get(key) is not None:
            label = coerce_binary_label(row.get(key))
            if label is not None:
                break

    row_id = row.get("id", row.get("index", row.get("original_index", fallback_index)))
    row_id = str(row_id)
    messages = normalize_messages(row.get("messages_clean") if row.get("messages_clean") is not None else row.get("messages"))
    source_dataset = row.get("dataset", row.get("subdataset"))

    fingerprint_payload = {
        "messages": messages,
        "label": label,
        "source_dataset": source_dataset,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    return {
        "row_id": row_id,
        "label": label,
        "source_dataset": source_dataset,
        "fingerprint": fingerprint,
        "message_count": len(messages),
        "assistant_preview": messages[-1]["content"][:80] if messages else "",
    }


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_prefixes(ids: Iterable[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in ids:
        prefix = str(item).split("_", 1)[0] if "_" in str(item) else str(item)
        counts[prefix] += 1
    return dict(sorted(counts.items()))


def compare_raw_datasets(instructed_path: Optional[Path], mask_path: Optional[Path]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": "missing",
        "instructed_path": str(instructed_path) if instructed_path else None,
        "mask_path": str(mask_path) if mask_path else None,
    }
    if instructed_path is None or mask_path is None or not instructed_path.exists() or not mask_path.exists():
        summary["reason"] = "missing_raw_path"
        return summary

    instructed_raw = load_jsonl_rows(instructed_path)
    mask_raw = load_jsonl_rows(mask_path)
    instructed_rows = [parse_typed_row(row, i) for i, row in enumerate(instructed_raw)]
    mask_rows = [parse_typed_row(row, i) for i, row in enumerate(mask_raw)]

    instructed_ids = [row["row_id"] for row in instructed_rows]
    mask_ids = [row["row_id"] for row in mask_rows]
    instructed_fp = [row["fingerprint"] for row in instructed_rows]
    mask_fp = [row["fingerprint"] for row in mask_rows]

    id_overlap = len(set(instructed_ids) & set(mask_ids))
    fp_overlap = len(set(instructed_fp) & set(mask_fp))

    summary.update(
        {
            "status": "ok",
            "instructed_count": len(instructed_rows),
            "mask_count": len(mask_rows),
            "same_row_count": len(instructed_rows) == len(mask_rows),
            "same_id_sequence": instructed_ids == mask_ids,
            "same_fingerprint_sequence": instructed_fp == mask_fp,
            "id_overlap_count": id_overlap,
            "fingerprint_overlap_count": fp_overlap,
            "id_overlap_ratio_vs_instructed": (id_overlap / len(set(instructed_ids))) if instructed_ids else None,
            "fingerprint_overlap_ratio_vs_instructed": (fp_overlap / len(set(instructed_fp))) if instructed_fp else None,
            "instructed_id_prefixes": summarize_prefixes(instructed_ids),
            "mask_id_prefixes": summarize_prefixes(mask_ids),
            "instructed_source_dataset_counts": dict(Counter(str(row["source_dataset"]) for row in instructed_rows)),
            "mask_source_dataset_counts": dict(Counter(str(row["source_dataset"]) for row in mask_rows)),
            "first_instructed_id": instructed_ids[0] if instructed_ids else None,
            "first_mask_id": mask_ids[0] if mask_ids else None,
        }
    )
    return summary


def load_activation_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = load_jsonl_rows(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": str(row.get("id")),
                "label": parse_int(row.get("label")),
                "shard": row.get("shard"),
            }
        )
    return out


def compare_activation_manifests(activations_root: Path, model_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary = {"status": "ok", "pairs_checked": 0, "all_exact_match": True}
    model_root = activations_root / model_dir

    for segment in SEGMENTS:
        id_dataset = dataset_name("Deception-InstructedDeception", segment)
        mask_dataset = dataset_name("Deception-Mask", segment)
        for split in SPLITS:
            id_manifest = model_root / id_dataset / split / "manifest.jsonl"
            mask_manifest = model_root / mask_dataset / split / "manifest.jsonl"
            record: Dict[str, Any] = {
                "segment": segment,
                "split": split,
                "id_manifest": str(id_manifest),
                "mask_manifest": str(mask_manifest),
                "id_exists": id_manifest.exists(),
                "mask_exists": mask_manifest.exists(),
            }
            if not id_manifest.exists() or not mask_manifest.exists():
                record["status"] = "missing"
                summary["all_exact_match"] = False
                rows.append(record)
                continue

            id_rows = load_activation_manifest(id_manifest)
            mask_rows = load_activation_manifest(mask_manifest)
            id_ids = [row["id"] for row in id_rows]
            mask_ids = [row["id"] for row in mask_rows]
            id_labels = [row["label"] for row in id_rows]
            mask_labels = [row["label"] for row in mask_rows]
            id_shards = [row["shard"] for row in id_rows]
            mask_shards = [row["shard"] for row in mask_rows]
            equal = id_ids == mask_ids and id_labels == mask_labels and id_shards == mask_shards

            record.update(
                {
                    "status": "ok",
                    "id_count": len(id_rows),
                    "mask_count": len(mask_rows),
                    "ids_equal": id_ids == mask_ids,
                    "labels_equal": id_labels == mask_labels,
                    "shards_equal": id_shards == mask_shards,
                    "exact_match": equal,
                    "first_id": id_ids[0] if id_ids else None,
                    "id_prefixes_instructed": json.dumps(summarize_prefixes(id_ids), sort_keys=True),
                    "id_prefixes_mask": json.dumps(summarize_prefixes(mask_ids), sort_keys=True),
                }
            )
            summary["pairs_checked"] += 1
            summary["all_exact_match"] = summary["all_exact_match"] and equal
            rows.append(record)

    if summary["pairs_checked"] == 0:
        summary["status"] = "missing"
        summary["all_exact_match"] = False
    return rows, summary


def load_layer_results(path: Path) -> Dict[int, Dict[str, Optional[float]]]:
    if not path.exists():
        return {}
    raw = read_json(path, default=[])
    if not isinstance(raw, list):
        return {}
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        layer = parse_int(row.get("layer"))
        if layer is None:
            continue
        out[int(layer)] = {
            "val_auc": parse_float(row.get("val_auc", row.get("auc"))),
            "val_acc": parse_float(row.get("val_acc", row.get("accuracy"))),
        }
    return out


def hash_tensor_state(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        return None
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
            h.update(key.encode("utf-8"))
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(str(arr.shape).encode("utf-8"))
            h.update(arr.tobytes())
    return h.hexdigest()


def compare_probe_artifacts(probes_root: Path, model_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    metric_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    summary = {"status": "ok", "metric_pairs_checked": 0, "weight_pairs_checked": 0, "all_metrics_equal": True, "all_weights_equal": True}
    model_root = probes_root / model_dir

    for segment in SEGMENTS:
        id_dataset = dataset_name("Deception-InstructedDeception", segment)
        mask_dataset = dataset_name("Deception-Mask", segment)
        for pooling in POOLINGS:
            id_dir = model_root / "Deception-InstructedDeception_slices" / id_dataset / pooling
            mask_dir = model_root / "Deception-Mask_slices" / mask_dataset / pooling
            id_results = id_dir / "layer_results.json"
            mask_results = mask_dir / "layer_results.json"
            metric_row: Dict[str, Any] = {
                "segment": segment,
                "pooling": pooling,
                "id_layer_results": str(id_results),
                "mask_layer_results": str(mask_results),
                "id_exists": id_results.exists(),
                "mask_exists": mask_results.exists(),
            }

            if id_results.exists() and mask_results.exists():
                id_layers = load_layer_results(id_results)
                mask_layers = load_layer_results(mask_results)
                common_layers = sorted(set(id_layers.keys()) & set(mask_layers.keys()))
                auc_diffs = []
                acc_diffs = []
                exact_metrics = True
                for layer in common_layers:
                    id_auc = id_layers[layer].get("val_auc")
                    mask_auc = mask_layers[layer].get("val_auc")
                    id_acc = id_layers[layer].get("val_acc")
                    mask_acc = mask_layers[layer].get("val_acc")
                    if id_auc is not None and mask_auc is not None:
                        auc_diffs.append(abs(id_auc - mask_auc))
                    if id_acc is not None and mask_acc is not None:
                        acc_diffs.append(abs(id_acc - mask_acc))
                    exact_metrics = exact_metrics and (id_layers[layer] == mask_layers[layer])

                metric_row.update(
                    {
                        "status": "ok",
                        "layers_instructed": len(id_layers),
                        "layers_mask": len(mask_layers),
                        "common_layers": len(common_layers),
                        "metrics_equal": exact_metrics and (set(id_layers.keys()) == set(mask_layers.keys())),
                        "max_val_auc_abs_diff": max(auc_diffs) if auc_diffs else None,
                        "max_val_acc_abs_diff": max(acc_diffs) if acc_diffs else None,
                    }
                )
                summary["metric_pairs_checked"] += 1
                summary["all_metrics_equal"] = summary["all_metrics_equal"] and bool(metric_row["metrics_equal"])
            else:
                metric_row["status"] = "missing"
                summary["all_metrics_equal"] = False

            metric_rows.append(metric_row)

            id_layers = load_layer_results(id_results) if id_results.exists() else {}
            mask_layers = load_layer_results(mask_results) if mask_results.exists() else {}
            for layer in sorted(set(id_layers.keys()) & set(mask_layers.keys())):
                id_probe = id_dir / f"probe_layer_{layer}.pt"
                mask_probe = mask_dir / f"probe_layer_{layer}.pt"
                id_hash = hash_tensor_state(id_probe) if id_probe.exists() else None
                mask_hash = hash_tensor_state(mask_probe) if mask_probe.exists() else None
                record = {
                    "segment": segment,
                    "pooling": pooling,
                    "layer": layer,
                    "id_probe": str(id_probe),
                    "mask_probe": str(mask_probe),
                    "id_exists": id_probe.exists(),
                    "mask_exists": mask_probe.exists(),
                    "id_hash": id_hash,
                    "mask_hash": mask_hash,
                    "weights_equal": (id_hash is not None and mask_hash is not None and id_hash == mask_hash),
                }
                if id_probe.exists() and mask_probe.exists():
                    summary["weight_pairs_checked"] += 1
                    summary["all_weights_equal"] = summary["all_weights_equal"] and bool(record["weights_equal"])
                else:
                    summary["all_weights_equal"] = False
                weight_rows.append(record)

    if summary["metric_pairs_checked"] == 0:
        summary["status"] = "missing"
        summary["all_metrics_equal"] = False
    if summary["weight_pairs_checked"] == 0:
        summary["all_weights_equal"] = False
    return metric_rows, weight_rows, summary


def parse_pair_summary_metrics(data: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Optional[float]]]:
    out: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}
    poolings = data.get("poolings", {})
    if not isinstance(poolings, dict):
        return out
    for pooling, payload in poolings.items():
        p = normalize_pooling(pooling)
        if not isinstance(payload, dict):
            continue
        layers = payload.get("layers", [])
        if not isinstance(layers, list):
            continue
        for row in layers:
            if not isinstance(row, dict):
                continue
            layer = parse_int(row.get("layer"))
            if layer is None:
                continue
            out[(p, int(layer))] = {
                "auc": parse_float(row.get("auc")),
                "accuracy": parse_float(row.get("accuracy")),
                "f1": parse_float(row.get("f1")),
            }
    return out


def find_pair_file(pair_roots: Sequence[Path], source_dataset: str, target_dataset: str, name: str) -> Optional[Path]:
    for root in pair_roots:
        candidate = root / f"from-{source_dataset}" / f"to-{target_dataset}" / name
        if candidate.exists():
            return candidate
    return None


def compare_pairwise_outputs(pair_roots: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary = {"status": "ok", "pairs_checked": 0, "all_exact_match": True}

    for segment in SEGMENTS:
        id_target = dataset_name("Deception-InstructedDeception", segment)
        mask_target = dataset_name("Deception-Mask", segment)
        for base in SOURCE_BASES:
            source_dataset = dataset_name(base, segment)
            id_summary_path = find_pair_file(pair_roots, source_dataset, id_target, "pair_summary.json")
            mask_summary_path = find_pair_file(pair_roots, source_dataset, mask_target, "pair_summary.json")
            id_logits_path = find_pair_file(pair_roots, source_dataset, id_target, "pair_logits_manifest.json")
            mask_logits_path = find_pair_file(pair_roots, source_dataset, mask_target, "pair_logits_manifest.json")

            record: Dict[str, Any] = {
                "segment": segment,
                "source_dataset": source_dataset,
                "id_summary_path": str(id_summary_path) if id_summary_path else None,
                "mask_summary_path": str(mask_summary_path) if mask_summary_path else None,
                "id_logits_path": str(id_logits_path) if id_logits_path else None,
                "mask_logits_path": str(mask_logits_path) if mask_logits_path else None,
            }

            if not id_summary_path or not mask_summary_path or not id_logits_path or not mask_logits_path:
                record["status"] = "missing"
                summary["all_exact_match"] = False
                rows.append(record)
                continue

            id_summary = read_json(id_summary_path, default={})
            mask_summary = read_json(mask_summary_path, default={})
            id_logits = read_json(id_logits_path, default={})
            mask_logits = read_json(mask_logits_path, default={})

            id_metrics = parse_pair_summary_metrics(id_summary)
            mask_metrics = parse_pair_summary_metrics(mask_summary)
            common_metric_keys = sorted(set(id_metrics.keys()) & set(mask_metrics.keys()))
            auc_diffs = []
            for key in common_metric_keys:
                id_auc = id_metrics[key].get("auc")
                mask_auc = mask_metrics[key].get("auc")
                if id_auc is not None and mask_auc is not None:
                    auc_diffs.append(abs(id_auc - mask_auc))

            record.update(
                {
                    "status": "ok",
                    "overall_best_equal": id_summary.get("overall_best", {}) == mask_summary.get("overall_best", {}),
                    "layer_metrics_equal": id_metrics == mask_metrics,
                    "max_layer_auc_abs_diff": max(auc_diffs) if auc_diffs else None,
                    "sample_ids_equal": id_logits.get("sample_ids", []) == mask_logits.get("sample_ids", []),
                    "labels_equal": id_logits.get("labels", []) == mask_logits.get("labels", []),
                    "scores_meta_equal": id_logits.get("scores", []) == mask_logits.get("scores", []),
                    "sample_count_instructed": len(id_logits.get("sample_ids", [])),
                    "sample_count_mask": len(mask_logits.get("sample_ids", [])),
                    "first_sample_id": (id_logits.get("sample_ids") or [None])[0],
                }
            )
            record["exact_match"] = (
                record["overall_best_equal"]
                and record["layer_metrics_equal"]
                and record["sample_ids_equal"]
                and record["labels_equal"]
                and record["scores_meta_equal"]
            )
            summary["pairs_checked"] += 1
            summary["all_exact_match"] = summary["all_exact_match"] and bool(record["exact_match"])
            rows.append(record)

    return rows, summary


def resolve_optional_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    return Path(path_value)


def find_mask_raw_path(explicit_path: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path("data/apollo_raw/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl"),
            Path("data/apolloraw/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl"),
            Path.home() / "Downloads" / "mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
            Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl"),
            Path("/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/apollo_raw/mask/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Mask vs InstructedDeception artifacts.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--artifact_root", type=str, default="artifacts")
    p.add_argument("--output_root", type=str, default=None, help="Optional direct run-root override.")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--activations_root", type=str, default="data/activations_fullprompt")
    p.add_argument("--probes_root", type=str, default="data/probes")
    p.add_argument("--ood_results_root", type=str, default="results/ood_evaluation")
    p.add_argument("--instructed_raw_path", type=str, default=None)
    p.add_argument("--mask_raw_path", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()

    if args.output_root:
        run_root = Path(args.output_root) / run_id
    else:
        run_root = Path(args.artifact_root) / "runs" / "mask_vs_instructed_deception_audit" / model_dir / run_id

    meta_dir = run_root / "meta"
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    raw_dir = results_dir / "raw"
    cache_dir = results_dir / "cache"
    probes_dir = results_dir / "probes"
    eval_dir = results_dir / "eval"
    for directory in [meta_dir, checkpoints_dir, results_dir, raw_dir, cache_dir, probes_dir, eval_dir]:
        ensure_dir(directory)

    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    progress_path = checkpoints_dir / "progress.json"
    summary_path = results_dir / "summary.json"
    progress = read_json(progress_path, default={"completed_steps": []})

    instructed_raw_path = resolve_optional_path(args.instructed_raw_path)
    mask_raw_path = find_mask_raw_path(args.mask_raw_path)
    pair_roots = discover_pair_roots(Path(args.ood_results_root), model_dir)

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "paths": {
                "activations_root": args.activations_root,
                "probes_root": args.probes_root,
                "ood_results_root": args.ood_results_root,
                "pair_results_roots": [str(path) for path in pair_roots],
                "instructed_raw_path": str(instructed_raw_path) if instructed_raw_path else None,
                "mask_raw_path": str(mask_raw_path) if mask_raw_path else None,
                "run_root": str(run_root),
            },
        },
    )
    update_status(status_path, "running", "starting")

    def mark(step: str) -> None:
        done = set(progress.get("completed_steps", []))
        done.add(step)
        progress["completed_steps"] = sorted(done)
        progress["updated_at"] = utc_now()
        write_json(progress_path, progress)

    if args.resume and "finished" in set(progress.get("completed_steps", [])) and summary_path.exists():
        update_status(status_path, "completed", "resume: already finished")
        print(f"[resume] already complete: {run_root}")
        return 0

    raw_summary: Dict[str, Any]
    if args.resume and "raw" in set(progress.get("completed_steps", [])) and (raw_dir / "summary.json").exists():
        raw_summary = read_json(raw_dir / "summary.json", default={})
    else:
        raw_summary = compare_raw_datasets(instructed_raw_path, mask_raw_path)
        write_json(raw_dir / "summary.json", raw_summary)
        mark("raw")

    cache_rows: List[Dict[str, Any]]
    cache_summary: Dict[str, Any]
    if args.resume and "cache" in set(progress.get("completed_steps", [])) and (cache_dir / "summary.json").exists():
        cache_summary = read_json(cache_dir / "summary.json", default={})
        cache_rows = pd.read_csv(cache_dir / "manifest_comparison.csv").to_dict(orient="records") if (cache_dir / "manifest_comparison.csv").exists() else []
    else:
        cache_rows, cache_summary = compare_activation_manifests(Path(args.activations_root), model_dir)
        if cache_rows:
            write_csv_rows(cache_dir / "manifest_comparison.csv", cache_rows, list(cache_rows[0].keys()))
        write_json(cache_dir / "summary.json", cache_summary)
        mark("cache")

    probe_metric_rows: List[Dict[str, Any]]
    probe_weight_rows: List[Dict[str, Any]]
    probe_summary: Dict[str, Any]
    if args.resume and "probes" in set(progress.get("completed_steps", [])) and (probes_dir / "summary.json").exists():
        probe_summary = read_json(probes_dir / "summary.json", default={})
        probe_metric_rows = pd.read_csv(probes_dir / "metric_comparison.csv").to_dict(orient="records") if (probes_dir / "metric_comparison.csv").exists() else []
        probe_weight_rows = pd.read_csv(probes_dir / "weight_comparison.csv").to_dict(orient="records") if (probes_dir / "weight_comparison.csv").exists() else []
    else:
        probe_metric_rows, probe_weight_rows, probe_summary = compare_probe_artifacts(Path(args.probes_root), model_dir)
        if probe_metric_rows:
            write_csv_rows(probes_dir / "metric_comparison.csv", probe_metric_rows, list(probe_metric_rows[0].keys()))
        if probe_weight_rows:
            write_csv_rows(probes_dir / "weight_comparison.csv", probe_weight_rows, list(probe_weight_rows[0].keys()))
        write_json(probes_dir / "summary.json", probe_summary)
        mark("probes")

    eval_rows: List[Dict[str, Any]]
    eval_summary: Dict[str, Any]
    if args.resume and "eval" in set(progress.get("completed_steps", [])) and (eval_dir / "summary.json").exists():
        eval_summary = read_json(eval_dir / "summary.json", default={})
        eval_rows = pd.read_csv(eval_dir / "pair_comparison.csv").to_dict(orient="records") if (eval_dir / "pair_comparison.csv").exists() else []
    else:
        eval_rows, eval_summary = compare_pairwise_outputs(pair_roots)
        if eval_rows:
            write_csv_rows(eval_dir / "pair_comparison.csv", eval_rows, list(eval_rows[0].keys()))
        write_json(eval_dir / "summary.json", eval_summary)
        mark("eval")

    verdict = {
        "raw_same": bool(raw_summary.get("same_fingerprint_sequence", False)),
        "cached_activations_same": bool(cache_summary.get("all_exact_match", False)),
        "trained_probes_same": bool(probe_summary.get("all_metrics_equal", False) and probe_summary.get("all_weights_equal", False)),
        "pairwise_eval_same": bool(eval_summary.get("all_exact_match", False)),
        "instructed_raw_path_exists": bool(instructed_raw_path and instructed_raw_path.exists()),
        "mask_raw_path_exists": bool(mask_raw_path and mask_raw_path.exists()),
        "likely_root_cause": None,
    }

    if verdict["pairwise_eval_same"] and verdict["cached_activations_same"]:
        verdict["likely_root_cause"] = "cached_instructed_deception_activations_match_mask"
    elif verdict["pairwise_eval_same"] and not verdict["cached_activations_same"]:
        verdict["likely_root_cause"] = "evaluation_outputs_match_but_cache_audit_incomplete_or_split_specific"
    elif verdict["raw_same"]:
        verdict["likely_root_cause"] = "raw_datasets_identical"
    else:
        verdict["likely_root_cause"] = "requires_manual_followup"

    summary = {
        "run_id": run_id,
        "model": args.model,
        "model_dir": model_dir,
        "raw_summary": raw_summary,
        "cache_summary": cache_summary,
        "probe_summary": probe_summary,
        "eval_summary": eval_summary,
        "verdict": verdict,
        "written_at": utc_now(),
    }
    write_json(results_dir / "verdict.json", verdict)
    write_json(summary_path, summary)
    mark("finished")
    update_status(status_path, "completed", "audit complete")
    print(f"[done] wrote audit run to {run_root}")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
