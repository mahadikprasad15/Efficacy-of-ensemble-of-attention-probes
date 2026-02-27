#!/usr/bin/env python3
"""
Evaluate MultiAttention probes on OOD datasets.

This script is fully decoupled from training and is resumable shard-by-shard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))

from actprobe.probes.models import MultiAttentionProbe, MultiLayerMultiAttentionProbe


PROBE_FAMILIES = {
    "gmha": {"multilayer": False, "variant": "gmha"},
    "multimax": {"multilayer": False, "variant": "multimax"},
    "rolling": {"multilayer": False, "variant": "rolling"},
    "multilayer_gmha": {"multilayer": True, "variant": "gmha"},
    "multilayer_multimax": {"multilayer": True, "variant": "multimax"},
    "multilayer_rolling": {"multilayer": True, "variant": "rolling"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def model_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_status(run_dir: Path, status: str, message: str) -> None:
    write_json(
        run_dir / "meta" / "status.json",
        {
            "status": status,
            "message": message,
            "updated_at": utc_now(),
        },
    )


def build_model(
    probe_family: str,
    input_dim: int,
    num_heads: int,
    num_classes: int,
    input_layers: Sequence[int],
    topk_tokens: int,
    rolling_window: int,
    rolling_stride: int,
):
    cfg = PROBE_FAMILIES[probe_family]
    variant = cfg["variant"]
    if cfg["multilayer"]:
        return MultiLayerMultiAttentionProbe(
            input_dim=input_dim,
            input_layers=input_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            variant=variant,
            topk_tokens=topk_tokens,
            rolling_window=rolling_window,
            rolling_stride=rolling_stride,
        )
    return MultiAttentionProbe(
        input_dim=input_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        variant=variant,
        topk_tokens=topk_tokens,
        rolling_window=rolling_window,
        rolling_stride=rolling_stride,
    )


def evaluate_binary(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    logits = np.clip(logits, -60.0, 60.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    if len(np.unique(labels)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(labels, probs))
    acc = float(accuracy_score(labels, preds))
    return {"auc": auc, "accuracy": acc}


def evaluate_multiclass(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "f1_macro": f1_macro}


def load_existing_predictions(predictions_path: Path, num_classes: int) -> Tuple[List[float], List[int]]:
    if not predictions_path.exists():
        return [], []
    logits_all = []
    labels_all = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            labels_all.append(int(obj["label"]))
            if num_classes <= 1:
                logits_all.append(float(obj["logits"]))
            else:
                logits_all.append([float(x) for x in obj["logits"]])
    return logits_all, labels_all


def write_variant_csv(path: Path, payload: Dict) -> None:
    rows = [payload]
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> int:
    probe_run_dir = Path(args.probe_run_dir)
    manifest_path = probe_run_dir / "meta" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training run manifest missing: {manifest_path}")
    train_manifest = read_json(manifest_path, {})

    model_name = args.model or train_manifest.get("model")
    if not model_name:
        raise ValueError("Model name missing; pass --model or ensure it exists in training manifest")
    source_dataset = args.source_dataset or train_manifest.get("dataset")
    if not source_dataset:
        raise ValueError("Source dataset missing; pass --source_dataset or ensure it exists in training manifest")

    probe_family = train_manifest["probe_family"]
    num_heads = int(train_manifest["num_heads"])
    num_classes = int(train_manifest["num_classes"])
    input_layers = [int(x) for x in train_manifest.get("input_layers", [])]
    layer = train_manifest.get("layer")
    topk_tokens = int(train_manifest.get("topk_tokens", 4))
    rolling_window = int(train_manifest.get("rolling_window", 64))
    rolling_stride = int(train_manifest.get("rolling_stride", 32))

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else (probe_run_dir / "results" / "best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_dir = model_to_dir(model_name)
    target_split_dir = Path(args.activations_root) / model_dir / args.target_dataset / args.target_split
    if not target_split_dir.exists():
        raise FileNotFoundError(f"Target split directory not found: {target_split_dir}")

    pair_name = f"{source_dataset}_{args.target_dataset}"
    run_name = args.run_name or probe_run_dir.name
    eval_run_dir = (
        Path(args.output_root)
        / model_dir
        / "MultiAttention_probes"
        / pair_name
        / probe_family
        / run_name
    )
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    eval_manifest_path = eval_run_dir / "meta" / "run_manifest.json"
    eval_status_path = eval_run_dir / "meta" / "status.json"
    eval_progress_path = eval_run_dir / "checkpoints" / "progress.json"
    predictions_path = eval_run_dir / "results" / "predictions.jsonl"
    metrics_path = eval_run_dir / "results" / "metrics.json"
    variant_csv_path = eval_run_dir / "results" / "per_layer_or_variant_metrics.csv"

    if eval_status_path.exists():
        status = read_json(eval_status_path, {})
        if status.get("status") == "completed" and not args.force_recompute:
            print(f"[skip] evaluation already completed: {eval_run_dir}")
            return 0

    (eval_run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (eval_run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (eval_run_dir / "results").mkdir(parents=True, exist_ok=True)

    if args.force_recompute:
        if predictions_path.exists():
            predictions_path.unlink()
        if eval_progress_path.exists():
            eval_progress_path.unlink()

    if not eval_manifest_path.exists():
        write_json(
            eval_manifest_path,
            {
                "created_at": utc_now(),
                "model": model_name,
                "source_dataset": source_dataset,
                "target_dataset": args.target_dataset,
                "target_split": args.target_split,
                "probe_family": probe_family,
                "checkpoint_path": str(checkpoint_path),
                "probe_run_dir": str(probe_run_dir),
                "eval_run_dir": str(eval_run_dir),
            },
        )

    write_status(eval_run_dir, "running", "starting or resuming ood evaluation")

    # Labels lookup.
    label_map: Dict[str, int] = {}
    manifest_path_target = target_split_dir / "manifest.jsonl"
    with manifest_path_target.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label_map[obj["id"]] = int(obj.get("label", -1))

    shards = sorted(target_split_dir.glob("shard_*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {target_split_dir}")

    progress = read_json(eval_progress_path, {"processed_shards": [], "updated_at": utc_now()})
    processed_shards = set(progress.get("processed_shards", []))

    # Initialize model from first valid tensor.
    sample_tensor = None
    for shard in shards:
        tensors = load_file(str(shard))
        for eid, tensor in tensors.items():
            if label_map.get(eid, -1) < 0:
                continue
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(1)
            if tensor.ndim == 3:
                sample_tensor = tensor
                break
        if sample_tensor is not None:
            break
    if sample_tensor is None:
        raise RuntimeError("No valid labeled target samples found")

    input_dim = int(sample_tensor.shape[-1])
    model = build_model(
        probe_family=probe_family,
        input_dim=input_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        input_layers=input_layers,
        topk_tokens=topk_tokens,
        rolling_window=rolling_window,
        rolling_stride=rolling_stride,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logits_all, labels_all = load_existing_predictions(predictions_path, num_classes=num_classes)

    for shard in shards:
        shard_name = shard.name
        if shard_name in processed_shards:
            continue

        tensors = load_file(str(shard))
        batch_x: List[torch.Tensor] = []
        batch_y: List[int] = []
        batch_ids: List[str] = []

        def flush_batch():
            if not batch_x:
                return
            x = torch.stack(batch_x, dim=0).to(device)
            with torch.no_grad():
                logits = model(x).detach().cpu().numpy()

            with predictions_path.open("a", encoding="utf-8") as out_f:
                for i, eid in enumerate(batch_ids):
                    label = int(batch_y[i])
                    if num_classes <= 1:
                        logit_val = float(logits[i].reshape(-1)[0])
                        pred = int(logit_val >= 0.0)
                        record = {"id": eid, "label": label, "logits": logit_val, "pred": pred}
                        logits_all.append(logit_val)
                    else:
                        logit_vec = [float(v) for v in logits[i].tolist()]
                        pred = int(np.argmax(logit_vec))
                        record = {"id": eid, "label": label, "logits": logit_vec, "pred": pred}
                        logits_all.append(logit_vec)

                    labels_all.append(label)
                    out_f.write(json.dumps(record, ensure_ascii=True) + "\n")

            batch_x.clear()
            batch_y.clear()
            batch_ids.clear()

        for eid, tensor in tensors.items():
            label = label_map.get(eid, -1)
            if label < 0:
                continue
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(1)  # (L,D) -> (L,1,D)
            if tensor.ndim != 3:
                continue

            if PROBE_FAMILIES[probe_family]["multilayer"]:
                x = tensor.float()  # (L,T,D)
            else:
                if layer is None:
                    raise ValueError(f"Training manifest missing layer for single-layer family: {probe_family}")
                x = tensor[int(layer)].float()  # (T,D)

            batch_x.append(x)
            batch_y.append(int(label))
            batch_ids.append(eid)
            if len(batch_x) >= args.batch_size:
                flush_batch()
        flush_batch()

        processed_shards.add(shard_name)
        progress = {
            "processed_shards": sorted(processed_shards),
            "n_examples": len(labels_all),
            "updated_at": utc_now(),
        }
        write_json(eval_progress_path, progress)

    labels_np = np.array(labels_all, dtype=np.int64)
    if num_classes <= 1:
        logits_np = np.array(logits_all, dtype=np.float32)
        metrics = evaluate_binary(logits_np, labels_np)
    else:
        logits_np = np.array(logits_all, dtype=np.float32)
        metrics = evaluate_multiclass(logits_np, labels_np)

    payload = {
        "timestamp": utc_now(),
        "model": model_name,
        "source_dataset": source_dataset,
        "target_dataset": args.target_dataset,
        "target_split": args.target_split,
        "probe_family": probe_family,
        "checkpoint_path": str(checkpoint_path),
        "num_examples": int(len(labels_all)),
        **metrics,
    }
    write_json(metrics_path, payload)
    write_variant_csv(variant_csv_path, payload)
    write_status(eval_run_dir, "completed", "completed ood evaluation")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MultiAttention probes on OOD datasets")
    parser.add_argument("--probe_run_dir", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="validation")
    parser.add_argument("--activations_root", type=str, default="data/activations")
    parser.add_argument("--output_root", type=str, default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/ood_evaluation")

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
