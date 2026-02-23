#!/usr/bin/env python3
"""
Run normalized probe training + OOD evaluation for a Top-5 CSV.

For each row in the Top-5 CSV:
  1) Train a single-layer probe with --normalize on the source dataset.
  2) Evaluate the trained probe on the target dataset split using source stats.
  3) Save per-pair metrics as JSON + CSV under results_root/model_dir/normalized_probes_evaluation.

Example:
  python scripts/pipelines/run_normalized_top5_eval.py \
    --top5_csv /path/to/top5.csv \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --activations_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt \
    --results_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/ood_evaluation \
    --target_split validation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import logging
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

# Add src to path
import sys
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))

from actprobe.probes.models import LayerProbe
from actprobe.utils.norm_stats import (
    init_running_stats,
    update_running_stats,
    finalize_running_stats,
    save_layer_stats,
    load_layer_stats,
    stats_complete,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def model_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def normalize_pooling(value: str) -> str:
    x = str(value).strip().lower()
    mapping = {
        "mean": "mean",
        "max": "max",
        "last": "last",
        "attn": "attn",
        "attention": "attn",
    }
    if x not in mapping:
        raise ValueError(f"Unsupported pooling value: {value}")
    return mapping[x]


def parse_source_probe(source_probe: str) -> Tuple[str, str]:
    s = source_probe.strip()
    if s.startswith("Roleplaying "):
        return "Deception-Roleplaying", s[len("Roleplaying ") :].strip().lower()
    if s.startswith("AI Liar "):
        return "Deception-AILiar", s[len("AI Liar ") :].strip().lower()
    raise ValueError(f"Unrecognized Source Probe format: {source_probe}")


def map_source_probe_to_train_dataset_name(source_probe: str) -> str:
    base_dataset, segment = parse_source_probe(source_probe)
    return base_dataset if segment == "full" else f"{base_dataset}-{segment}"


def target_to_dataset_name(value: str) -> str:
    v = str(value).strip().lower()
    if "insidertrading" in v:
        base = "Deception-InsiderTrading"
    elif "roleplaying" in v:
        base = "Deception-Roleplaying"
    elif "ai liar" in v or "ailiar" in v:
        base = "Deception-AILiar"
    else:
        raise ValueError(f"Unsupported target dataset value: {value}")

    segment = "full"
    if "completion" in v:
        segment = "completion"
    elif "system" in v:
        segment = "system"
    elif "user" in v:
        segment = "user"
    elif "prompt" in v:
        segment = "prompt"
    elif "full" in v:
        segment = "full"

    return base if segment == "full" else f"{base}-{segment}"


def split_dir(activations_dir: Path, model_dir: str, dataset: str, split: str) -> Path:
    return activations_dir / model_dir / dataset / split


def read_top5_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def peek_first_tensor(activations_split_dir: Path) -> torch.Tensor:
    shards = sorted(activations_split_dir.glob("shard_*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {activations_split_dir}")
    tensors = load_file(str(shards[0]))
    if not tensors:
        raise RuntimeError(f"No tensors in shard {shards[0]}")
    return next(iter(tensors.values()))


def input_format_from_tensor(tensor: torch.Tensor) -> str:
    if tensor.dim() == 2:
        return "final_token"
    if tensor.dim() == 3:
        return "pooled"
    raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")


def iter_labeled_tensors(activations_split_dir: Path) -> Iterable[torch.Tensor]:
    manifest_path = activations_split_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    label_map: Dict[str, int] = {}
    with manifest_path.open("r") as f:
        for line in f:
            meta = json.loads(line)
            label_map[meta["id"]] = meta.get("label", -1)

    shards = sorted(activations_split_dir.glob("shard_*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {activations_split_dir}")

    for shard in shards:
        tensors = load_file(str(shard))
        for eid, tensor in tensors.items():
            label = label_map.get(eid, -1)
            if label == -1:
                continue
            yield tensor


def compute_stats_if_missing(
    activations_split_dir: Path,
    stats_dir: Path,
    pooling: str,
    eps: float,
) -> None:
    sample = peek_first_tensor(activations_split_dir)
    if sample.dim() == 3:
        num_layers, _, dim = sample.shape
    elif sample.dim() == 2:
        num_layers, dim = sample.shape
    else:
        raise ValueError(f"Unexpected tensor shape: {tuple(sample.shape)}")

    if stats_complete(str(stats_dir), num_layers):
        return

    running = init_running_stats(num_layers, dim)
    for tensor in iter_labeled_tensors(activations_split_dir):
        running = update_running_stats(running, tensor)
    finalized = finalize_running_stats(running, eps=eps)
    save_layer_stats(finalized, str(stats_dir), pooling=pooling, source_dir=str(activations_split_dir))


class CachedActivationsDataset(Dataset):
    def __init__(self, activations_split_dir: Path):
        self.items = []

        manifest_path = activations_split_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        label_map: Dict[str, int] = {}
        with manifest_path.open("r") as f:
            for line in f:
                meta = json.loads(line)
                label_map[meta["id"]] = meta.get("label", -1)

        shards = sorted(activations_split_dir.glob("shard_*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No shards found in {activations_split_dir}")

        for shard in shards:
            tensors = load_file(str(shard))
            for eid, tensor in tensors.items():
                label = label_map.get(eid, -1)
                if label == -1:
                    continue
                self.items.append({"tensor": tensor, "label": label})

        if not self.items:
            raise RuntimeError(f"No labeled samples found in {activations_split_dir}")

        sample_shape = self.items[0]["tensor"].shape
        if len(sample_shape) == 2:
            self.input_format = "final_token"
        elif len(sample_shape) == 3:
            self.input_format = "pooled"
        else:
            raise ValueError(f"Unexpected tensor shape: {sample_shape}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item["tensor"].float(), torch.tensor(item["label"], dtype=torch.float32)


class LayerDataset(Dataset):
    def __init__(self, base_dataset: Dataset, layer_idx: int):
        self.base = base_dataset
        self.layer_idx = layer_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x[self.layer_idx], y


def apply_batch_norm(x: torch.Tensor, norm: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if norm is None:
        return x
    mean, std = norm
    return (x - mean) / std


def evaluate_probe(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Dict[str, float]:
    model.eval()
    probs: List[float] = []
    preds: List[int] = []
    targets: List[int] = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x = apply_batch_norm(x, norm)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(prob)
            preds.extend((prob > 0.5).astype(int))
            targets.extend(y.numpy())

    probs_arr = np.array(probs)
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)

    metrics = {
        "auc": float(roc_auc_score(targets_arr, probs_arr)),
        "accuracy": float(accuracy_score(targets_arr, preds_arr)),
        "precision": float(precision_score(targets_arr, preds_arr, zero_division=0)),
        "recall": float(recall_score(targets_arr, preds_arr, zero_division=0)),
        "f1": float(f1_score(targets_arr, preds_arr, zero_division=0)),
        "count": int(len(targets_arr)),
    }
    return metrics


def run_training_if_needed(
    script: Path,
    model: str,
    dataset: str,
    activations_dir: Path,
    pooling: str,
    layer: int,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    force_retrain: bool,
) -> None:
    model_dir = model_to_dir(model)
    probe_path = output_dir / model_dir / dataset / pooling / f"probe_layer_{layer}.pt"
    if probe_path.exists() and not force_retrain:
        return

    cmd = [
        sys.executable,
        str(script),
        "--model", model,
        "--dataset", dataset,
        "--activations_dir", str(activations_dir),
        "--pooling", pooling,
        "--layer", str(layer),
        "--output_dir", str(output_dir),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--patience", str(patience),
        "--normalize",
        "--ood_dataset", "None",
    ]

    subprocess.check_call(cmd)


def write_metrics(out_dir: Path, payload: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    csv_path = out_dir / "metrics.csv"

    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    flat = payload.copy()
    for key, value in list(flat.items()):
        if isinstance(value, (dict, list)):
            flat.pop(key)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run normalized Top-5 probe evaluation")
    parser.add_argument("--top5_csv", type=str, required=True, help="Path to top-5 CSV")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--activations_dir", type=str, required=True, help="Base activations dir")
    parser.add_argument("--results_root", type=str, required=True, help="Base results root")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--target_split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--norm_eps", type=float, default=1e-8)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    top5_csv = Path(args.top5_csv)
    activations_dir = Path(args.activations_dir)
    results_root = Path(args.results_root)

    model_dir = model_to_dir(args.model)
    run_root = results_root / model_dir / "normalized_probes_evaluation"
    probes_root = run_root / "probes"
    run_root.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)

    rows = read_top5_rows(top5_csv)
    summary_rows = []

    train_script = Path("scripts/training/train_deception_probes.py")

    for row in rows:
        source_probe = row.get("Source Probe", "").strip()
        train_dataset = row.get("train_dataset_name", "").strip()
        if not train_dataset:
            train_dataset = map_source_probe_to_train_dataset_name(source_probe)

        target_label = row.get("Target Dataset (Test)", "").strip()
        target_dataset = target_to_dataset_name(target_label)

        pooling_raw = row.get("train_pooling", "") or row.get("Best Pooling", "")
        pooling = normalize_pooling(pooling_raw)
        layer = int(float(row.get("Best Layer", "0")))

        train_split_dir = split_dir(activations_dir, model_dir, train_dataset, args.train_split)
        target_split_dir = split_dir(activations_dir, model_dir, target_dataset, args.target_split)

        stats_dir = train_split_dir / "norm_stats" / pooling

        if not train_split_dir.exists():
            raise FileNotFoundError(f"Train split dir not found: {train_split_dir}")
        if not target_split_dir.exists():
            raise FileNotFoundError(f"Target split dir not found: {target_split_dir}")

        run_training_if_needed(
            script=train_script,
            model=args.model,
            dataset=train_dataset,
            activations_dir=activations_dir,
            pooling=pooling,
            layer=layer,
            output_dir=probes_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            force_retrain=args.force_retrain,
        )

        compute_stats_if_missing(train_split_dir, stats_dir, pooling, args.norm_eps)

        probe_path = probes_root / model_dir / train_dataset / pooling / f"probe_layer_{layer}.pt"
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe not found: {probe_path}")

        # Determine model pooling type using train input format.
        sample_tensor = peek_first_tensor(train_split_dir)
        input_format = input_format_from_tensor(sample_tensor)
        num_layers = sample_tensor.shape[0]
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range for {train_dataset} (num_layers={num_layers})")
        model_pooling_type = pooling
        if input_format == "final_token":
            model_pooling_type = "none"

        # Build evaluation dataset for target.
        target_dataset_obj = CachedActivationsDataset(target_split_dir)
        layer_dataset = LayerDataset(target_dataset_obj, layer)
        loader = DataLoader(layer_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        mean, std, _ = load_layer_stats(str(stats_dir), layer, device=device)
        norm = (mean, std)

        # Initialize and load probe.
        input_dim = sample_tensor.shape[-1]
        model = LayerProbe(input_dim=input_dim, pooling_type=model_pooling_type).to(device)
        model.load_state_dict(torch.load(probe_path, map_location=device))

        metrics = evaluate_probe(model, loader, device, norm=norm)

        pair_slug = f"from-{train_dataset}_to-{target_dataset}"
        out_dir = run_root / pair_slug

        payload = {
            "timestamp": utc_now(),
            "model": args.model,
            "source_probe": source_probe,
            "train_dataset": train_dataset,
            "target_dataset": target_dataset,
            "target_label": target_label,
            "pooling": pooling,
            "layer": layer,
            "train_split": args.train_split,
            "target_split": args.target_split,
            "probe_path": str(probe_path),
            "stats_dir": str(stats_dir),
            "input_format": input_format,
            "model_pooling_type": model_pooling_type,
            **metrics,
        }
        write_metrics(out_dir, payload)

        summary_rows.append(payload)

    # Write summary
    summary_json = run_root / "summary.json"
    summary_csv = run_root / "summary.csv"
    with summary_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    with summary_csv.open("w", newline="") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
