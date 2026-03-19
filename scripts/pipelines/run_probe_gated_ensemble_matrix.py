#!/usr/bin/env python3
"""
Train and evaluate source-specific gated ensembles over frozen probe banks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
from actprobe.probes.ensemble import ProbeLogitGatedEnsemble, mean_entropy  # noqa: E402
from actprobe.probes.models import LayerProbe  # noqa: E402


POOLINGS = ["mean", "max", "last", "attn"]
DEFAULT_LAYERS = list(range(28))
DEFAULT_SOURCES = [
    "Deception-ConvincingGame-completion",
    "Deception-HarmPressureChoice-completion",
    "Deception-InstructedDeception-completion",
    "Deception-Mask-completion",
    "Deception-AILiar-completion",
    "Deception-Roleplaying-completion",
]
DEFAULT_TARGETS = [
    "Deception-ConvincingGame-completion",
    "Deception-HarmPressureChoice-completion",
    "Deception-InstructedDeception-completion",
    "Deception-Mask-completion",
    "Deception-AILiar-completion",
    "Deception-InsiderTrading-completion",
    "Deception-Roleplaying-completion",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_status(status_path: Path, state: str, message: str) -> None:
    payload = read_json(status_path, default={})
    payload["state"] = state
    payload["message"] = message
    payload["updated_at"] = utc_now()
    write_json(status_path, payload)


def copy_tree_subset(src_root: Path, dst_root: Path, subdirs: Sequence[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        src = src_root / subdir
        if not src.exists():
            continue
        dst = dst_root / subdir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def parse_csv_list(value: str | None, default: Sequence[str]) -> List[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str | None, default: Sequence[int]) -> List[int]:
    if not value:
        return [int(x) for x in default]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def model_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def dataset_base(dataset_name: str) -> str:
    if dataset_name.endswith("-completion"):
        return dataset_name[: -len("-completion")]
    if dataset_name.endswith("-full"):
        return dataset_name[: -len("-full")]
    return dataset_name


def short_name(dataset_name: str) -> str:
    base = dataset_base(dataset_name).replace("Deception-", "")
    mapping = {
        "ConvincingGame": "CG",
        "HarmPressureChoice": "HPC",
        "InstructedDeception": "ID",
        "Mask": "M",
        "AILiar": "AL",
        "InsiderTrading": "IT",
        "InsiderTrading-SallyConcat": "ITS",
        "Roleplaying": "RP",
    }
    suffix = "c" if dataset_name.endswith("-completion") else "f"
    return f"{mapping.get(base, base)}-{suffix}"


class AttentionLinearProbeCompat(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return self.classifier(pooled)


class CachedDeceptionDataset(Dataset):
    def __init__(self, split_dir: Path):
        manifest_path = split_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        shard_paths = sorted(split_dir.glob("shard_*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {split_dir}")

        label_map: Dict[str, int] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                label_map[obj["id"]] = int(obj.get("label", -1))

        self.items: List[Dict[str, Any]] = []
        for shard_path in shard_paths:
            tensors = load_file(str(shard_path))
            for sample_id, tensor in tensors.items():
                label = label_map.get(sample_id, -1)
                if label < 0:
                    continue
                self.items.append(
                    {
                        "id": sample_id,
                        "tensor": tensor.float(),
                        "label": int(label),
                    }
                )
        if not self.items:
            raise RuntimeError(f"No labeled samples loaded from {split_dir}")

        sample = self.items[0]["tensor"]
        if sample.dim() == 2:
            self.input_format = "final_token"
            self.num_layers = int(sample.shape[0])
            self.hidden_dim = int(sample.shape[1])
        elif sample.dim() == 3:
            self.input_format = "pooled"
            self.num_layers = int(sample.shape[0])
            self.hidden_dim = int(sample.shape[2])
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(sample.shape)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return item["tensor"], item["label"], item["id"]


class LayerDataset(Dataset):
    def __init__(self, base_dataset: CachedDeceptionDataset, layer_idx: int):
        self.base = base_dataset
        self.layer_idx = int(layer_idx)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y, sample_id = self.base[idx]
        return x[self.layer_idx], y, sample_id


@dataclass(frozen=True)
class ExpertSpec:
    expert_idx: int
    source_dataset: str
    pooling: str
    layer: int
    probe_path: Path


def load_probe_model(probe_path: Path, pooling: str, input_dim: int, device: torch.device) -> nn.Module:
    state = torch.load(str(probe_path), map_location=device)
    if pooling == "attn" and any(key.startswith("attn.") for key in state.keys()):
        model = AttentionLinearProbeCompat(input_dim=input_dim).to(device)
    else:
        model = LayerProbe(input_dim=input_dim, pooling_type=pooling).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def ensure_split_dir(root: Path, model_dir: str, dataset_name: str, split: str) -> Path:
    split_dir = root / model_dir / dataset_name / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing activation split dir: {split_dir}")
    if not (split_dir / "manifest.jsonl").exists():
        raise FileNotFoundError(f"Missing manifest in {split_dir}")
    return split_dir


def discover_experts(
    probes_model_root: Path,
    source_dataset: str,
    *,
    poolings: Sequence[str],
    layers: Sequence[int],
) -> List[ExpertSpec]:
    base = dataset_base(source_dataset)
    layer_set = {int(layer) for layer in layers}
    expert_specs: List[ExpertSpec] = []
    expert_idx = 0
    for layer in sorted(layer_set):
        for pooling in poolings:
            probe_path = probes_model_root / f"{base}_slices" / source_dataset / pooling / f"probe_layer_{layer}.pt"
            if not probe_path.exists():
                raise FileNotFoundError(f"Missing expert probe: {probe_path}")
            expert_specs.append(
                ExpertSpec(
                    expert_idx=expert_idx,
                    source_dataset=source_dataset,
                    pooling=pooling,
                    layer=layer,
                    probe_path=probe_path,
                )
            )
            expert_idx += 1
    return expert_specs


def load_source_validation_metrics(probes_model_root: Path, source_dataset: str) -> Dict[tuple[str, int], Dict[str, float]]:
    base = dataset_base(source_dataset)
    lookup: Dict[tuple[str, int], Dict[str, float]] = {}
    for pooling in POOLINGS:
        layer_results_path = probes_model_root / f"{base}_slices" / source_dataset / pooling / "layer_results.json"
        if not layer_results_path.exists():
            continue
        payload = read_json(layer_results_path, default={})
        if not isinstance(payload, list):
            continue
        for row in payload:
            if "layer" not in row:
                continue
            layer = int(row["layer"])
            lookup[(pooling, layer)] = {
                "val_auc": float(row.get("val_auc", 0.5)),
                "val_acc": float(row.get("val_acc", 0.0)),
            }
    return lookup


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    preds = (probs >= 0.5).astype(np.int64)
    if len(np.unique(labels)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(labels, probs))
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, preds)),
    }


def stratified_split_indices(labels: np.ndarray, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            raise ValueError(f"Need at least 2 examples for label {label} to form gate train/val splits")
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
        train_idx.extend(shuffled[:n_train].tolist())
        val_idx.extend(shuffled[n_train:].tolist())
    return np.asarray(sorted(train_idx), dtype=np.int64), np.asarray(sorted(val_idx), dtype=np.int64)


def fit_logit_norm(logits: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = logits.mean(axis=0).astype(np.float32)
    std = logits.std(axis=0).astype(np.float32)
    std = np.where(std < eps, eps, std)
    return mean, std


def normalize_logits(logits: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((logits - mean) / std).astype(np.float32)


def choose_validation_selected_expert(
    experts: Sequence[ExpertSpec],
    metrics_lookup: Dict[tuple[str, int], Dict[str, float]],
) -> ExpertSpec:
    ranked: List[tuple[float, float, int, int, ExpertSpec]] = []
    pooling_rank = {name: idx for idx, name in enumerate(POOLINGS)}
    for spec in experts:
        metrics = metrics_lookup.get((spec.pooling, spec.layer), {})
        ranked.append(
            (
                float(metrics.get("val_auc", 0.5)),
                float(metrics.get("val_acc", 0.0)),
                -int(spec.layer),
                -int(pooling_rank.get(spec.pooling, 999)),
                spec,
            )
        )
    ranked.sort(reverse=True)
    return ranked[0][-1]


def choose_fixed_mean_l15_expert(experts: Sequence[ExpertSpec]) -> ExpertSpec:
    for spec in experts:
        if spec.pooling == "mean" and spec.layer == 15:
            return spec
    raise ValueError("Could not find fixed mean@L15 expert in discovered bank")


def topk_static_plan(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    per_expert_auc = []
    for idx in range(train_logits.shape[1]):
        metrics = compute_binary_metrics(train_logits[:, idx], train_labels)
        per_expert_auc.append(float(metrics["auc"]))
    ranked = np.argsort(np.asarray(per_expert_auc))[::-1]

    best: Dict[str, Any] | None = None
    for k in k_values:
        k = int(k)
        if k <= 0:
            continue
        selected = ranked[: min(k, len(ranked))]
        val_combined = val_logits[:, selected].mean(axis=1)
        metrics = compute_binary_metrics(val_combined, val_labels)
        candidate = {
            "k": int(len(selected)),
            "selected_indices": [int(x) for x in selected.tolist()],
            "val_auc": float(metrics["auc"]),
            "val_accuracy": float(metrics["accuracy"]),
            "per_expert_auc": per_expert_auc,
        }
        if best is None or candidate["val_auc"] > best["val_auc"]:
            best = candidate
    if best is None:
        raise ValueError("No valid top-k candidates found")
    return best


def weighted_static_weights(train_logits: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    aucs = []
    for idx in range(train_logits.shape[1]):
        aucs.append(float(compute_binary_metrics(train_logits[:, idx], train_labels)["auc"]))
    raw = np.maximum(np.asarray(aucs, dtype=np.float32) - 0.5, 0.0)
    if float(raw.sum()) <= 0.0:
        raw = np.ones_like(raw, dtype=np.float32)
    return (raw / raw.sum()).astype(np.float32)


def evaluate_gate(model: ProbeLogitGatedEnsemble, logits: np.ndarray, labels: np.ndarray, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(logits.astype(np.float32)).to(device)
        logits_out, weights = model(x, return_weights=True)
        logits_np = logits_out.squeeze(-1).detach().cpu().numpy()
        entropy = float(mean_entropy(weights).detach().cpu().item())
    metrics = compute_binary_metrics(logits_np, labels)
    metrics["mean_gate_entropy"] = entropy
    return metrics


def train_gate(
    *,
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    temperature: float,
    entropy_reg: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> tuple[ProbeLogitGatedEnsemble, Dict[str, Any]]:
    model = ProbeLogitGatedEnsemble(
        num_experts=int(train_logits.shape[1]),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        temperature=float(temperature),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_logits.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_metrics: Dict[str, Any] = {"val_auc": 0.0, "best_epoch": 0}
    history: List[Dict[str, float]] = []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_logits, batch_labels in train_loader:
            batch_logits = batch_logits.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits_out, weights = model(batch_logits, return_weights=True)
            bce = criterion(logits_out.squeeze(-1), batch_labels)
            entropy = mean_entropy(weights)
            loss = bce - float(entropy_reg) * entropy
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        train_metrics = evaluate_gate(model, train_logits, train_labels, device)
        val_metrics = evaluate_gate(model, val_logits, val_labels, device)
        row = {
            "epoch": float(epoch),
            "train_loss": epoch_loss / max(1, len(train_loader)),
            "train_auc": float(train_metrics["auc"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_entropy": float(train_metrics["mean_gate_entropy"]),
            "val_auc": float(val_metrics["auc"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_entropy": float(val_metrics["mean_gate_entropy"]),
        }
        history.append(row)

        if row["val_auc"] > float(best_metrics["val_auc"]):
            best_metrics = {
                "val_auc": float(row["val_auc"]),
                "val_accuracy": float(row["val_accuracy"]),
                "val_entropy": float(row["val_entropy"]),
                "best_epoch": int(epoch),
            }
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        raise RuntimeError("Gate training never produced a checkpoint")
    model.load_state_dict(best_state)
    return model, {"history": history, **best_metrics}


def cache_logits_for_source_target(
    *,
    cache_dir: Path,
    experts: Sequence[ExpertSpec],
    split_dir: Path,
    batch_size: int,
    device: torch.device,
    resume: bool,
) -> Dict[str, Any]:
    npz_path = cache_dir / "expert_logits.npz"
    index_path = cache_dir / "expert_index.json"
    if resume and npz_path.exists() and index_path.exists():
        payload = np.load(npz_path, allow_pickle=False)
        return {
            "logits": payload["logits"].astype(np.float32),
            "labels": payload["labels"].astype(np.int64),
            "ids": payload["ids"].astype(str).tolist(),
            "input_format": str(payload["input_format"][0]),
            "hidden_dim": int(payload["hidden_dim"][0]),
            "num_layers": int(payload["num_layers"][0]),
        }

    base_dataset = CachedDeceptionDataset(split_dir)
    logits = np.zeros((len(base_dataset), len(experts)), dtype=np.float32)
    labels = np.asarray([int(item["label"]) for item in base_dataset.items], dtype=np.int64)
    ids = [str(item["id"]) for item in base_dataset.items]

    grouped: Dict[int, List[ExpertSpec]] = {}
    for spec in experts:
        grouped.setdefault(int(spec.layer), []).append(spec)

    for layer, layer_specs in tqdm(sorted(grouped.items()), desc=f"expert bank {split_dir.parent.name}->{split_dir.name}"):
        layer_dataset = LayerDataset(base_dataset, layer_idx=layer)
        loader = DataLoader(layer_dataset, batch_size=batch_size, shuffle=False)
        models = [(spec.expert_idx, load_probe_model(spec.probe_path, spec.pooling, base_dataset.hidden_dim, device)) for spec in layer_specs]
        offset = 0
        for batch_x, _, _ in loader:
            batch_x = batch_x.to(device)
            if batch_x.dim() == 2:
                batch_x = batch_x.unsqueeze(1)
            batch_size_now = int(batch_x.shape[0])
            for expert_idx, model in models:
                with torch.no_grad():
                    batch_logits = model(batch_x.float()).reshape(-1).detach().cpu().numpy().astype(np.float32)
                logits[offset : offset + batch_size_now, int(expert_idx)] = batch_logits
            offset += batch_size_now

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        logits=logits,
        labels=labels,
        ids=np.asarray(ids),
        input_format=np.asarray([base_dataset.input_format]),
        hidden_dim=np.asarray([base_dataset.hidden_dim], dtype=np.int32),
        num_layers=np.asarray([base_dataset.num_layers], dtype=np.int32),
    )
    write_json(
        index_path,
        {
            "created_at": utc_now(),
            "split_dir": str(split_dir),
            "num_samples": int(len(ids)),
            "num_experts": int(len(experts)),
            "experts": [
                {
                    "expert_idx": int(spec.expert_idx),
                    "pooling": spec.pooling,
                    "layer": int(spec.layer),
                    "probe_path": str(spec.probe_path),
                }
                for spec in experts
            ],
        },
    )
    return {
        "logits": logits,
        "labels": labels,
        "ids": ids,
        "input_format": base_dataset.input_format,
        "hidden_dim": base_dataset.hidden_dim,
        "num_layers": base_dataset.num_layers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a source-specific gated ensemble matrix over frozen probe banks.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, default="data/activations_fullprompt")
    parser.add_argument("--probes_root", type=str, default="data/probes")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--targets", type=str, default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--poolings", type=str, default=",".join(POOLINGS))
    parser.add_argument("--layers", type=str, default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--gate_split", type=str, default="validation")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--gate_train_fraction", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--topk_values", type=str, default="4,8,16,32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mirror_results_root", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = parse_csv_list(args.sources, DEFAULT_SOURCES)
    targets = parse_csv_list(args.targets, DEFAULT_TARGETS)
    poolings = parse_csv_list(args.poolings, POOLINGS)
    layers = parse_int_csv(args.layers, DEFAULT_LAYERS)
    topk_values = parse_int_csv(args.topk_values, [4, 8, 16, 32])

    model_dir = model_to_dir(args.model)
    run_id = args.run_id or utc_run_id()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    run_root = (
        Path(args.artifact_root)
        / "runs"
        / "probe_gated_ensemble_matrix"
        / model_dir
        / "completion"
        / "frozen-probe-bank"
        / "gate-over-112"
        / run_id
    )
    inputs_dir = run_root / "inputs"
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    logs_dir = run_root / "logs"
    meta_dir = run_root / "meta"
    for directory in [inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    progress = read_json(
        progress_path,
        default={
            "completed_cache_units": [],
            "completed_sources": [],
        },
    )
    write_json(progress_path, progress)

    mirror_root = (
        Path(args.mirror_results_root)
        if args.mirror_results_root
        else Path("results") / "ood_evaluation" / model_dir / "probe_gated_ensemble"
    )
    mirrored_run_root = mirror_root / run_id

    activations_root = Path(args.activations_root)
    probes_root = Path(args.probes_root)
    probes_model_root = probes_root / model_dir
    if not probes_model_root.exists():
        raise FileNotFoundError(f"Probe root not found: {probes_model_root}")

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "paths": {
                "run_root": str(run_root),
                "activations_root": str(activations_root),
                "probes_root": str(probes_root),
                "mirror_results_root": str(mirror_root),
                "mirrored_run_root": str(mirrored_run_root),
            },
            "sources": sources,
            "targets": targets,
            "poolings": poolings,
            "layers": layers,
            "gate_split": args.gate_split,
            "eval_split": args.eval_split,
            "topk_values": topk_values,
        },
    )
    update_status(status_path, "running", "starting")

    try:
        source_results: List[Dict[str, Any]] = []
        matrix_rows: List[Dict[str, Any]] = []
        for source_dataset in sources:
            source_gate_dir = results_dir / "gates" / source_dataset
            source_metrics_path = results_dir / "source_rows" / source_dataset / "metrics.json"
            fit_summary_path = source_gate_dir / "fit_summary.json"
            if (
                args.resume
                and source_dataset in set(progress.get("completed_sources", []))
                and fit_summary_path.exists()
                and source_metrics_path.exists()
            ):
                matrix_rows.extend(read_json(source_metrics_path, default={"rows": []}).get("rows", []))
                source_results.append(
                    {
                        "source_dataset": source_dataset,
                        "gate_model_path": str(source_gate_dir / "gate_model.pt"),
                        "fit_summary_path": str(fit_summary_path),
                    }
                )
                continue

            experts = discover_experts(
                probes_model_root,
                source_dataset,
                poolings=poolings,
                layers=layers,
            )
            source_metric_lookup = load_source_validation_metrics(probes_model_root, source_dataset)
            fixed_expert = choose_fixed_mean_l15_expert(experts)
            validation_expert = choose_validation_selected_expert(experts, source_metric_lookup)

            gate_split_dir = ensure_split_dir(activations_root, model_dir, source_dataset, args.gate_split)
            gate_cache_dir = results_dir / "logit_cache" / source_dataset / f"to-{source_dataset}" / args.gate_split
            gate_cache_key = f"{source_dataset}::{source_dataset}::{args.gate_split}"
            gate_payload = cache_logits_for_source_target(
                cache_dir=gate_cache_dir,
                experts=experts,
                split_dir=gate_split_dir,
                batch_size=args.batch_size,
                device=device,
                resume=args.resume and gate_cache_key in set(progress.get("completed_cache_units", [])),
            )
            completed_cache = set(progress.get("completed_cache_units", []))
            completed_cache.add(gate_cache_key)
            progress["completed_cache_units"] = sorted(completed_cache)
            write_json(progress_path, progress)

            gate_train_idx, gate_val_idx = stratified_split_indices(
                gate_payload["labels"],
                train_fraction=float(args.gate_train_fraction),
                seed=int(args.split_seed),
            )
            gate_train_logits = gate_payload["logits"][gate_train_idx]
            gate_val_logits = gate_payload["logits"][gate_val_idx]
            gate_train_labels = gate_payload["labels"][gate_train_idx]
            gate_val_labels = gate_payload["labels"][gate_val_idx]

            norm_mean, norm_std = fit_logit_norm(gate_train_logits)
            train_norm = normalize_logits(gate_train_logits, norm_mean, norm_std)
            val_norm = normalize_logits(gate_val_logits, norm_mean, norm_std)

            gate_model, gate_fit = train_gate(
                train_logits=train_norm,
                train_labels=gate_train_labels,
                val_logits=val_norm,
                val_labels=gate_val_labels,
                device=device,
                hidden_dim=args.gate_hidden_dim,
                dropout=args.gate_dropout,
                temperature=args.gate_temperature,
                entropy_reg=args.entropy_reg,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )

            topk_plan = topk_static_plan(
                train_logits=train_norm,
                train_labels=gate_train_labels,
                val_logits=val_norm,
                val_labels=gate_val_labels,
                k_values=topk_values,
            )
            weighted_weights = weighted_static_weights(train_norm, gate_train_labels)

            source_gate_dir.mkdir(parents=True, exist_ok=True)
            gate_ckpt_path = source_gate_dir / "gate_model.pt"
            torch.save(gate_model.state_dict(), gate_ckpt_path)
            np.savez_compressed(source_gate_dir / "norm_stats.npz", mean=norm_mean, std=norm_std)
            write_json(
                fit_summary_path,
                {
                    "source_dataset": source_dataset,
                    "num_experts": len(experts),
                    "gate_fit": gate_fit,
                    "fixed_expert": {
                        "expert_idx": int(fixed_expert.expert_idx),
                        "pooling": fixed_expert.pooling,
                        "layer": int(fixed_expert.layer),
                    },
                    "validation_selected_expert": {
                        "expert_idx": int(validation_expert.expert_idx),
                        "pooling": validation_expert.pooling,
                        "layer": int(validation_expert.layer),
                    },
                    "topk_plan": topk_plan,
                    "weighted_weights": weighted_weights.tolist(),
                    "experts": [
                        {
                            "expert_idx": int(spec.expert_idx),
                            "pooling": spec.pooling,
                            "layer": int(spec.layer),
                            "probe_path": str(spec.probe_path),
                        }
                        for spec in experts
                    ],
                    "gate_train_indices": gate_train_idx.tolist(),
                    "gate_val_indices": gate_val_idx.tolist(),
                },
            )

            for target_dataset in targets:
                eval_split_dir = ensure_split_dir(activations_root, model_dir, target_dataset, args.eval_split)
                target_cache_dir = results_dir / "logit_cache" / source_dataset / f"to-{target_dataset}" / args.eval_split
                target_cache_key = f"{source_dataset}::{target_dataset}::{args.eval_split}"
                target_payload = cache_logits_for_source_target(
                    cache_dir=target_cache_dir,
                    experts=experts,
                    split_dir=eval_split_dir,
                    batch_size=args.batch_size,
                    device=device,
                    resume=args.resume and target_cache_key in set(progress.get("completed_cache_units", [])),
                )
                completed_cache = set(progress.get("completed_cache_units", []))
                completed_cache.add(target_cache_key)
                progress["completed_cache_units"] = sorted(completed_cache)
                write_json(progress_path, progress)

                target_logits_raw = target_payload["logits"]
                target_logits_norm = normalize_logits(target_logits_raw, norm_mean, norm_std)
                target_labels = target_payload["labels"]

                fixed_metrics = compute_binary_metrics(target_logits_raw[:, fixed_expert.expert_idx], target_labels)
                validation_metrics = compute_binary_metrics(target_logits_raw[:, validation_expert.expert_idx], target_labels)
                uniform_metrics = compute_binary_metrics(target_logits_norm.mean(axis=1), target_labels)
                topk_metrics = compute_binary_metrics(
                    target_logits_norm[:, topk_plan["selected_indices"]].mean(axis=1),
                    target_labels,
                )
                weighted_metrics = compute_binary_metrics(target_logits_norm @ weighted_weights, target_labels)
                gated_metrics = evaluate_gate(gate_model, target_logits_norm, target_labels, device)

                per_method = {
                    "fixed_mean_l15": fixed_metrics,
                    "validation_selected": validation_metrics,
                    "uniform_mean_112": uniform_metrics,
                    "topk_static": topk_metrics,
                    "weighted_static": weighted_metrics,
                    "gated": gated_metrics,
                }
                for method_name, metrics in per_method.items():
                    matrix_rows.append(
                        {
                            "source_dataset": source_dataset,
                            "target_dataset": target_dataset,
                            "source_short": short_name(source_dataset),
                            "target_short": short_name(target_dataset),
                            "method": method_name,
                            "auc": float(metrics["auc"]),
                            "accuracy": float(metrics["accuracy"]),
                            "mean_gate_entropy": float(metrics.get("mean_gate_entropy", np.nan)),
                        }
                    )

            source_row_payload = [row for row in matrix_rows if row["source_dataset"] == source_dataset]
            write_json(source_metrics_path, {"source_dataset": source_dataset, "rows": source_row_payload})
            source_results.append(
                {
                    "source_dataset": source_dataset,
                    "gate_model_path": str(gate_ckpt_path),
                    "fit_summary_path": str(fit_summary_path),
                }
            )
            completed_sources = set(progress.get("completed_sources", []))
            completed_sources.add(source_dataset)
            progress["completed_sources"] = sorted(completed_sources)
            write_json(progress_path, progress)

        fieldnames = ["source_dataset", "target_dataset", "source_short", "target_short", "method", "auc", "accuracy", "mean_gate_entropy"]
        write_csv_rows(results_dir / "metrics_long.csv", matrix_rows, fieldnames)

        matrix_methods = ["fixed_mean_l15", "validation_selected", "uniform_mean_112", "topk_static", "weighted_static", "gated"]
        outputs: Dict[str, str] = {}
        for method_name in matrix_methods:
            method_rows = [row for row in matrix_rows if row["method"] == method_name]
            wide_rows: List[Dict[str, Any]] = []
            for source_dataset in sources:
                row_payload: Dict[str, Any] = {"row": short_name(source_dataset)}
                for target_dataset in targets:
                    match = next(
                        row for row in method_rows
                        if row["source_dataset"] == source_dataset and row["target_dataset"] == target_dataset
                    )
                    row_payload[short_name(target_dataset)] = float(match["auc"])
                wide_rows.append(row_payload)
            matrix_path = results_dir / f"matrix_{method_name}_auc.csv"
            write_csv_rows(matrix_path, wide_rows, list(wide_rows[0].keys()))
            outputs[f"matrix_{method_name}_auc_csv"] = str(matrix_path)

        for baseline_name in ["fixed_mean_l15", "validation_selected"]:
            delta_rows: List[Dict[str, Any]] = []
            for source_dataset in sources:
                row_payload: Dict[str, Any] = {"row": short_name(source_dataset)}
                for target_dataset in targets:
                    gated_row = next(
                        row for row in matrix_rows
                        if row["method"] == "gated" and row["source_dataset"] == source_dataset and row["target_dataset"] == target_dataset
                    )
                    baseline_row = next(
                        row for row in matrix_rows
                        if row["method"] == baseline_name and row["source_dataset"] == source_dataset and row["target_dataset"] == target_dataset
                    )
                    row_payload[short_name(target_dataset)] = float(gated_row["auc"] - baseline_row["auc"])
                delta_rows.append(row_payload)
            delta_path = results_dir / f"matrix_delta_gated_vs_{baseline_name}_auc.csv"
            write_csv_rows(delta_path, delta_rows, list(delta_rows[0].keys()))
            outputs[f"matrix_delta_gated_vs_{baseline_name}_auc_csv"] = str(delta_path)

        plot_index = {
            "run_id": run_id,
            "model": args.model,
            "sources": sources,
            "targets": targets,
            "results_root": str(results_dir),
            **outputs,
            "metrics_long_csv": str(results_dir / "metrics_long.csv"),
        }
        write_json(results_dir / "plot_index.json", plot_index)
        write_json(results_dir / "plot_idx.json", plot_index)

        summary = {
            "run_id": run_id,
            "completed_at": utc_now(),
            "model": args.model,
            "sources": sources,
            "targets": targets,
            "num_experts_per_source": int(len(poolings) * len(layers)),
            "methods": matrix_methods,
            "source_results": source_results,
            "outputs": outputs,
            "metrics_long_csv": str(results_dir / "metrics_long.csv"),
            "plot_index_path": str(results_dir / "plot_index.json"),
            "plot_idx_path": str(results_dir / "plot_idx.json"),
        }
        write_json(results_dir / "results.json", summary)
        update_status(status_path, "completed", "finished")
        copy_tree_subset(run_root, mirrored_run_root, ["meta", "checkpoints", "results", "logs"])
        return 0
    except Exception as exc:
        update_status(status_path, "failed", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
