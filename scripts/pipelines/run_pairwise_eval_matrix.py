#!/usr/bin/env python3
"""
Train/evaluate pairwise probe matrices for completion/full deception segments.

Implements:
  - Stage 1 train source-row probes for the active segments
  - Stage 2 evaluate all off-diagonal row -> column pairs
  - Stage 3 optional compatibility collection from legacy artifact formats
  - Stage 4 evaluate same-dataset test diagonals
  - Stage 5 aggregate completion/full matrices and export CSV/JSON

Resumable + verbose progress:
  - stage dashboard
  - tqdm progress bars
  - checkpoints in artifacts/runs/pairwise_eval_matrix/<model_dir>/<run_id>/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
from actprobe.probes.models import LayerProbe  # noqa: E402


POOLING_ORDER = ["attn", "last", "max", "mean"]
ALL_POOLINGS = ["mean", "max", "last", "attn"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def pair_logits_tensor_path(pair_dir: Path) -> Path:
    return pair_dir / "pair_logits.safetensors"


def pair_logits_manifest_path(pair_dir: Path) -> Path:
    return pair_dir / "pair_logits_manifest.json"


def pair_logits_exist(pair_dir: Path) -> bool:
    return pair_logits_tensor_path(pair_dir).exists() and pair_logits_manifest_path(pair_dir).exists()


def write_pair_logits(
    pair_dir: Path,
    *,
    source_dataset: str,
    target_dataset: str,
    split: str,
    input_format: str,
    sample_ids: Sequence[str],
    labels: Sequence[int],
    score_tensors: Dict[str, torch.Tensor],
    score_index: Sequence[Dict[str, Any]],
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = pair_logits_tensor_path(pair_dir)
    manifest_path = pair_logits_manifest_path(pair_dir)
    save_file(score_tensors, str(tensor_path))
    write_json(
        manifest_path,
        {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "split": split,
            "input_format": input_format,
            "num_samples": len(sample_ids),
            "sample_ids": list(sample_ids),
            "labels": [int(x) for x in labels],
            "scores": list(score_index),
            "tensor_file": tensor_path.name,
            "created_at": utc_now(),
        },
    )


def update_status(status_path: Path, state: str, message: str) -> None:
    current = read_json(status_path, default={})
    current["state"] = state
    current["message"] = message
    current["updated_at"] = utc_now()
    write_json(status_path, current)


def split_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    """
    Returns:
      (base_root_without_model_dir, model_root_with_model_dir)
    """
    if root.name == model_dir:
        return root.parent, root
    candidate = root / model_dir
    if candidate.exists():
        return root, candidate
    # Fall back to assuming provided root is model root
    return root.parent, root


def resolve_output_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    """
    Resolve an output root without relying on pre-existing directories.

    For new runs we want canonical writes under <root>/<model_dir> whenever the
    caller passes the parent directory, even if that model-specific directory
    does not exist yet.
    """
    if root.name == model_dir:
        return root.parent, root
    return root, root / model_dir


def dataset_base(dataset_name: str) -> str:
    if dataset_name.endswith("-completion"):
        return dataset_name[: -len("-completion")]
    if dataset_name.endswith("-full"):
        return dataset_name[: -len("-full")]
    return dataset_name


def segment_of(dataset_name: str) -> str:
    if dataset_name.endswith("-completion"):
        return "completion"
    if dataset_name.endswith("-full"):
        return "full"
    raise ValueError(f"Dataset has no segment suffix: {dataset_name}")


def short_name(dataset_name: str) -> str:
    # Deception-ConvincingGame-completion -> CG-c
    seg = segment_of(dataset_name)
    base = dataset_base(dataset_name).replace("Deception-", "")
    m = {
        "ConvincingGame": "CG",
        "HarmPressureChoice": "HPC",
        "InstructedDeception": "ID",
        "Mask": "M",
        "AILiar": "AL",
        "InsiderTrading": "IT",
        "InsiderTrading-SallyConcat": "ITS",
        "Roleplaying": "RP",
    }
    return f"{m.get(base, base)}-{'c' if seg == 'completion' else 'f'}"


@dataclass(frozen=True)
class PairJob:
    source_dataset: str
    target_dataset: str
    segment: str
    cell_type: str  # matrix_off_diagonal
    action: str  # eval_run | collect_only

    @property
    def pair_id(self) -> str:
        return f"{self.source_dataset}__{self.target_dataset}"


class CachedDeceptionDataset(Dataset):
    def __init__(self, activations_dir: Path):
        self.items: List[Dict[str, Any]] = []
        manifest_path = activations_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        label_map: Dict[str, int] = {}
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                label_map[entry["id"]] = int(entry.get("label", -1))

        shards = sorted(activations_dir.glob("shard_*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No shards in {activations_dir}")

        for shard in shards:
            tensors = load_file(str(shard))
            for sid, tensor in tensors.items():
                y = label_map.get(sid, -1)
                if y == -1:
                    continue
                self.items.append({"id": sid, "tensor": tensor, "label": y})
        if not self.items:
            raise RuntimeError(f"No labeled samples in {activations_dir}")

        sample = self.items[0]["tensor"]
        if sample.dim() == 3:
            self.input_format = "pooled"  # (L,T,D)
        elif sample.dim() == 2:
            self.input_format = "final_token"  # (L,D)
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(sample.shape)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        return item["tensor"].float(), torch.tensor(item["label"], dtype=torch.float32)


class LayerDataset(Dataset):
    def __init__(self, base_dataset: Dataset, layer_idx: int):
        self.base = base_dataset
        self.layer_idx = layer_idx

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        return x[self.layer_idx], y


class AttentionLinearProbeCompat(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return self.classifier(pooled)


def normalize_pooling_for_input(pooling: str, input_format: str) -> str:
    if input_format == "final_token" and pooling not in ["none", "last"]:
        return "none"
    return pooling


def parse_layer_from_probe_name(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("probe_layer_") or not name.endswith(".pt"):
        return None
    try:
        return int(name.replace("probe_layer_", "").replace(".pt", ""))
    except ValueError:
        return None


def find_probe_layers(pooling_dir: Path) -> List[Tuple[int, Path]]:
    pairs: List[Tuple[int, Path]] = []
    for p in sorted(pooling_dir.glob("probe_layer_*.pt")):
        layer = parse_layer_from_probe_name(p)
        if layer is not None:
            pairs.append((layer, p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_probe_model(probe_path: Path, pooling: str, input_dim: int, device: torch.device) -> nn.Module:
    state = torch.load(str(probe_path), map_location=device)
    if pooling == "attn" and any(k.startswith("attn.") for k in state.keys()):
        model = AttentionLinearProbeCompat(input_dim=input_dim).to(device)
    else:
        model = LayerProbe(input_dim=input_dim, pooling_type=pooling).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_probe_layer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    all_probs: List[float] = []
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_logits: List[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # final-token case x: (B,D) -> (B,1,D)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            logits = model(x.float()).reshape(-1)
            logits_np = logits.detach().cpu().numpy()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(np.int64)
            all_logits.extend(logits_np.tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(y.numpy().astype(np.int64).tolist())
    targets = np.array(all_targets, dtype=np.int64)
    probs = np.array(all_probs, dtype=np.float64)
    preds = np.array(all_preds, dtype=np.int64)
    logits_arr = np.array(all_logits, dtype=np.float32)
    auc = 0.5
    if len(np.unique(targets)) >= 2:
        try:
            auc = float(roc_auc_score(targets, probs))
        except Exception:
            auc = 0.5
    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, zero_division=0))
    return {"auc": auc, "accuracy": acc, "f1": f1}, logits_arr


def pooling_priority(pooling: str) -> int:
    try:
        return POOLING_ORDER.index(pooling)
    except ValueError:
        return len(POOLING_ORDER)


def choose_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def key(r: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
        return (
            float(r.get("auc", -1.0)),
            float(r.get("accuracy", -1.0)),
            float(r.get("f1", -1.0)),
            -int(r.get("layer", 10_000)),
            -pooling_priority(str(r.get("pooling", ""))),
        )

    if not records:
        raise ValueError("No records to choose best from.")
    return max(records, key=key)


def evaluate_pair_all_poolings(
    source_dataset: str,
    target_dataset: str,
    split: str,
    model_root_activations: Path,
    model_root_probes: Path,
    poolings: Sequence[str],
    eval_batch_size: int,
    device: torch.device,
    use_tqdm: bool,
    save_pair_logits_enabled: bool = False,
    pair_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    target_dir = model_root_activations / target_dataset / split
    if not target_dir.exists():
        raise FileNotFoundError(f"Target split dir missing: {target_dir}")
    ds = CachedDeceptionDataset(target_dir)
    sample_x, _ = ds[0]
    input_dim = int(sample_x.shape[-1])
    input_format = ds.input_format

    source_base = dataset_base(source_dataset)
    source_probe_root = model_root_probes / f"{source_base}_slices" / source_dataset
    if not source_probe_root.exists():
        raise FileNotFoundError(f"Source probe root missing: {source_probe_root}")

    pooling_results: Dict[str, Any] = {}
    all_best_candidates: List[Dict[str, Any]] = []
    score_tensors: Dict[str, torch.Tensor] = {}
    score_index: List[Dict[str, Any]] = []
    sample_ids = [str(item["id"]) for item in ds.items]
    sample_labels = [int(item["label"]) for item in ds.items]

    for pooling in poolings:
        pooling_dir = source_probe_root / pooling
        if not pooling_dir.exists():
            pooling_results[pooling] = {"status": "missing_probe_pooling_dir", "layers": []}
            continue
        layer_files = find_probe_layers(pooling_dir)
        if not layer_files:
            pooling_results[pooling] = {"status": "no_probe_layers", "layers": []}
            continue

        model_pooling = normalize_pooling_for_input(pooling, input_format)
        layer_records: List[Dict[str, Any]] = []
        iterator: Iterable[Tuple[int, Path]] = layer_files
        if use_tqdm:
            iterator = tqdm(layer_files, desc=f"{short_name(source_dataset)}->{short_name(target_dataset)} {pooling}", leave=False)

        for layer_idx, probe_path in iterator:
            layer_ds = LayerDataset(ds, layer_idx)
            layer_loader = DataLoader(layer_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            model = load_probe_model(probe_path, pooling=model_pooling, input_dim=input_dim, device=device)
            metrics, logits_arr = evaluate_probe_layer(model, layer_loader, device)
            record = {
                "pooling": pooling,
                "layer": int(layer_idx),
                "auc": float(metrics["auc"]),
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
            }
            if save_pair_logits_enabled:
                score_key = f"{pooling}__L{int(layer_idx)}"
                score_tensors[score_key] = torch.from_numpy(logits_arr)
                score_index.append(
                    {
                        "score_key": score_key,
                        "pooling": pooling,
                        "layer": int(layer_idx),
                    }
                )
                record["score_key"] = score_key
            layer_records.append(record)
        best_pool = choose_best(layer_records)
        pooling_results[pooling] = {
            "status": "ok",
            "best": best_pool,
            "layers": layer_records,
        }
        all_best_candidates.append(best_pool)

    if not all_best_candidates:
        raise RuntimeError(f"No valid pooling results for {source_dataset} -> {target_dataset}")
    overall_best = choose_best(all_best_candidates)
    if save_pair_logits_enabled:
        if pair_dir is None:
            raise ValueError("pair_dir is required when save_pair_logits_enabled is True")
        write_pair_logits(
            pair_dir,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            split=split,
            input_format=input_format,
            sample_ids=sample_ids,
            labels=sample_labels,
            score_tensors=score_tensors,
            score_index=score_index,
        )
    return {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "split": split,
        "input_format": input_format,
        "poolings": pooling_results,
        "overall_best": overall_best,
        "evaluated_at": utc_now(),
        "logits_saved": bool(save_pair_logits_enabled),
        "logits_tensor_file": pair_logits_tensor_path(pair_dir).name if save_pair_logits_enabled and pair_dir is not None else None,
        "logits_manifest_file": pair_logits_manifest_path(pair_dir).name if save_pair_logits_enabled and pair_dir is not None else None,
    }


def parse_existing_pair_summary(pair_dir: Path) -> Optional[Dict[str, Any]]:
    # Preferred unified summary produced by this pipeline
    unified = pair_dir / "pair_summary.json"
    if unified.exists():
        return read_json(unified)

    # Fallback: ood_results_all_pooling.json
    ood_all = pair_dir / "ood_results_all_pooling.json"
    if ood_all.exists():
        data = read_json(ood_all)
        candidates: List[Dict[str, Any]] = []
        poolings: Dict[str, Any] = {}
        for p in ALL_POOLINGS:
            if p not in data or not isinstance(data[p], dict):
                continue
            ent = data[p]
            rec = {
                "pooling": p,
                "layer": int(ent.get("best_layer", -1)),
                "auc": float(ent.get("best_auc", 0.5)),
                "accuracy": float(ent.get("best_accuracy", 0.0)),
                "f1": float(ent.get("best_f1", np.nan)) if ent.get("best_f1") is not None else None,
            }
            poolings[p] = {"status": "ok", "best": rec, "layers": []}
            candidates.append(rec)
        if candidates:
            # existing format often has no F1
            for c in candidates:
                if c["f1"] is None or np.isnan(c["f1"]):
                    c["f1"] = None
            best = choose_best([{**c, "f1": c["f1"] if c["f1"] is not None else -1.0} for c in candidates])
            if best.get("f1", -1.0) == -1.0:
                best["f1"] = None
            return {"poolings": poolings, "overall_best": best, "source": "ood_results_all_pooling.json"}

    # Fallback: per-pooling eval_ood_*.json
    candidates = []
    poolings = {}
    for p in ALL_POOLINGS:
        pdir = pair_dir / p
        if not pdir.exists():
            continue
        files = sorted(pdir.glob("eval_ood_*_test.json")) + sorted(pdir.glob("eval_ood_*_validation.json"))
        if not files:
            continue
        data = read_json(files[0])
        rec = {
            "pooling": p,
            "layer": int(data.get("best_ood_layer", -1)),
            "auc": float(data.get("best_ood_auc", 0.5)),
            "accuracy": float(data.get("best_ood_acc", 0.0)),
            "f1": float(data.get("best_ood_f1")) if data.get("best_ood_f1") is not None else None,
        }
        poolings[p] = {"status": "ok", "best": rec, "layers": data.get("layer_results", [])}
        candidates.append(rec)
    if candidates:
        best = choose_best([{**c, "f1": c["f1"] if c["f1"] is not None else -1.0} for c in candidates])
        if best.get("f1", -1.0) == -1.0:
            best["f1"] = None
        return {"poolings": poolings, "overall_best": best, "source": "eval_ood_pooling_json"}

    return None


def resolve_validation_split(dataset_dir: Path) -> Optional[str]:
    for split in ["validation", "val"]:
        if (dataset_dir / split).exists():
            return split
    return None


def diagonal_from_validation(
    dataset_name: str,
    model_root_activations: Path,
    model_root_probes: Path,
    poolings: Sequence[str],
    eval_batch_size: int,
    device: torch.device,
    use_tqdm: bool,
    save_pair_logits_enabled: bool = False,
    pair_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset_dir = model_root_activations / dataset_name
    split = resolve_validation_split(dataset_dir)
    if split is None:
        raise FileNotFoundError(f"No validation split found for {dataset_name} (tried validation/val)")
    summary = evaluate_pair_all_poolings(
        source_dataset=dataset_name,
        target_dataset=dataset_name,
        split=split,
        model_root_activations=model_root_activations,
        model_root_probes=model_root_probes,
        poolings=poolings,
        eval_batch_size=eval_batch_size,
        device=device,
        use_tqdm=use_tqdm,
        save_pair_logits_enabled=save_pair_logits_enabled,
        pair_dir=pair_dir,
    )
    summary["source"] = "diagonal_validation"
    return summary


def diagonal_from_test(
    dataset_name: str,
    model_root_activations: Path,
    model_root_probes: Path,
    poolings: Sequence[str],
    eval_batch_size: int,
    device: torch.device,
    use_tqdm: bool,
    save_pair_logits_enabled: bool = False,
    pair_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset_dir = model_root_activations / dataset_name
    split = "test"
    if not (dataset_dir / split).exists():
        raise FileNotFoundError(f"No test split found for {dataset_name} at {dataset_dir / split}")
    summary = evaluate_pair_all_poolings(
        source_dataset=dataset_name,
        target_dataset=dataset_name,
        split=split,
        model_root_activations=model_root_activations,
        model_root_probes=model_root_probes,
        poolings=poolings,
        eval_batch_size=eval_batch_size,
        device=device,
        use_tqdm=use_tqdm,
        save_pair_logits_enabled=save_pair_logits_enabled,
        pair_dir=pair_dir,
    )
    summary["source"] = "diagonal_test"
    summary["cell_type"] = "matrix_diagonal"
    summary["action"] = "eval_run"
    return summary


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_dataset_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise probe train/eval matrix pipeline.")
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--probes_root", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poolings", type=str, default="mean,max,last,attn")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--force_reeval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_training", action="store_true", help="Skip training stage entirely.")
    parser.add_argument("--skip_diagonals", action="store_true", help="Skip diagonal validation stage.")
    parser.add_argument(
        "--save_pair_logits",
        action="store_true",
        help=(
            "Persist per-sample probe logits for every evaluated source->target pair. "
            "Writes pair_logits.safetensors + pair_logits_manifest.json beside pair_summary.json."
        ),
    )
    parser.add_argument(
        "--collect_existing_pairs",
        action="store_true",
        help=(
            "Compatibility mode: if a pair lacks pair_summary.json, "
            "backfill it from older artifact formats instead of reevaluating."
        ),
    )
    parser.add_argument(
        "--only_sources",
        type=str,
        default=None,
        help="Comma-separated list of source datasets to evaluate (filters eval/collect).",
    )
    parser.add_argument(
        "--only_targets",
        type=str,
        default=None,
        help="Comma-separated list of target datasets to evaluate (filters eval/collect).",
    )
    parser.add_argument(
        "--only_segments",
        type=str,
        default=None,
        help="Comma-separated list of segments to evaluate (completion,full).",
    )
    parser.add_argument(
        "--completion_rows",
        type=str,
        default=None,
        help="Optional comma-separated override for completion row datasets.",
    )
    parser.add_argument(
        "--completion_cols",
        type=str,
        default=None,
        help="Optional comma-separated override for completion column datasets.",
    )
    parser.add_argument(
        "--full_rows",
        type=str,
        default=None,
        help="Optional comma-separated override for full row datasets.",
    )
    parser.add_argument(
        "--full_cols",
        type=str,
        default=None,
        help="Optional comma-separated override for full column datasets.",
    )
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument(
        "--pipeline_results_root",
        type=str,
        default=None,
        help=(
            "Optional external directory to mirror pipeline run artifacts "
            "(results/meta/checkpoints). "
            "Default: <results_model_root>/All_dataset_pairwise_results"
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def stage_spec(args: argparse.Namespace) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    rows = {
        "completion": [
            "Deception-ConvincingGame-completion",
            "Deception-HarmPressureChoice-completion",
            "Deception-InstructedDeception-completion",
            "Deception-Mask-completion",
            "Deception-AILiar-completion",
            "Deception-Roleplaying-completion",
        ],
        "full": [
            "Deception-ConvincingGame-full",
            "Deception-HarmPressureChoice-full",
            "Deception-InstructedDeception-full",
            "Deception-Mask-full",
            "Deception-AILiar-full",
            "Deception-Roleplaying-full",
        ],
    }
    cols = {
        "completion": [
            "Deception-ConvincingGame-completion",
            "Deception-HarmPressureChoice-completion",
            "Deception-InstructedDeception-completion",
            "Deception-Mask-completion",
            "Deception-AILiar-completion",
            "Deception-InsiderTrading-SallyConcat-completion",
            "Deception-Roleplaying-completion",
        ],
        "full": [
            "Deception-ConvincingGame-full",
            "Deception-HarmPressureChoice-full",
            "Deception-InstructedDeception-full",
            "Deception-Mask-full",
            "Deception-AILiar-full",
            "Deception-InsiderTrading-SallyConcat-full",
            "Deception-Roleplaying-full",
        ],
    }
    completion_rows_override = parse_dataset_csv(args.completion_rows)
    completion_cols_override = parse_dataset_csv(args.completion_cols)
    full_rows_override = parse_dataset_csv(args.full_rows)
    full_cols_override = parse_dataset_csv(args.full_cols)
    if completion_rows_override is not None:
        rows["completion"] = completion_rows_override
    if completion_cols_override is not None:
        cols["completion"] = completion_cols_override
    if full_rows_override is not None:
        rows["full"] = full_rows_override
    if full_cols_override is not None:
        cols["full"] = full_cols_override
    return rows, cols


def build_expected_pairs(rows: Dict[str, List[str]], cols: Dict[str, List[str]]) -> List[PairJob]:
    jobs: List[PairJob] = []
    for seg in [seg for seg in ["completion", "full"] if seg in rows or seg in cols]:
        for src in rows.get(seg, []):
            for tgt in cols.get(seg, []):
                if src == tgt:
                    continue
                jobs.append(PairJob(src, tgt, seg, "matrix_off_diagonal", "eval_run"))
    return jobs


def print_stage_dashboard(progress: Dict[str, Any]) -> None:
    stages = [
        ("stage1_train", "Train Source Probes"),
        ("stage2_eval", "Evaluate Off-Diagonal Pairs"),
        ("stage3_collect", "Collect Legacy Pairs"),
        ("stage4_diag", "Evaluate Diagonal Test Cells"),
        ("stage5_aggregate", "Aggregate Matrices"),
    ]
    done = sum(1 for key, _ in stages if key in set(progress.get("completed_steps", [])))
    print(f"[progress] stages done={done}/{len(stages)} remaining={len(stages)-done}")
    for idx, (key, label) in enumerate(stages, start=1):
        status = "done" if key in set(progress.get("completed_steps", [])) else "pending"
        print(f"[progress] [{idx}/{len(stages)}] {label}: {status}")


def main() -> int:
    args = parse_args()
    poolings = [p.strip() for p in args.poolings.split(",") if p.strip()]
    for p in poolings:
        if p not in ALL_POOLINGS:
            raise ValueError(f"Unsupported pooling '{p}'. Supported: {ALL_POOLINGS}")

    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()
    use_tqdm = not args.no_tqdm
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    only_sources = set(x.strip() for x in args.only_sources.split(",") if x.strip()) if args.only_sources else None
    only_targets = set(x.strip() for x in args.only_targets.split(",") if x.strip()) if args.only_targets else None
    only_segments = set(x.strip() for x in args.only_segments.split(",") if x.strip()) if args.only_segments else None

    # Resolve roots (supports passing either .../<model_dir> or its parent)
    acts_base, acts_model_root = split_root_and_model(Path(args.activations_root), model_dir)
    probes_base, probes_model_root = split_root_and_model(Path(args.probes_root), model_dir)
    results_base, results_model_root = resolve_output_root_and_model(Path(args.results_root), model_dir)
    results_model_root.mkdir(parents=True, exist_ok=True)

    # Canonical run artifacts
    run_root = Path(args.artifact_root) / "runs" / "pairwise_eval_matrix" / model_dir / run_id
    meta_dir = run_root / "meta"
    chk_dir = run_root / "checkpoints"
    out_dir = run_root / "results"
    logs_dir = run_root / "logs"
    for d in [meta_dir, chk_dir, out_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = meta_dir / "run_manifest.json"
    status_path = meta_dir / "status.json"
    progress_path = chk_dir / "progress.json"
    progress = read_json(
        progress_path,
        default={
            "completed_steps": [],
            "completed_train_jobs": [],
            "completed_eval_pairs": [],
            "completed_collect_pairs": [],
            "completed_diag_jobs": [],
        },
    )
    for key in ["completed_steps", "completed_train_jobs", "completed_eval_pairs", "completed_collect_pairs", "completed_diag_jobs"]:
        if key not in progress:
            progress[key] = []

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "paths": {
                "activations_base": str(acts_base),
                "activations_model_root": str(acts_model_root),
                "probes_base": str(probes_base),
                "probes_model_root": str(probes_model_root),
                "results_base": str(results_base),
                "results_model_root": str(results_model_root),
                "run_root": str(run_root),
            },
            "poolings": poolings,
            "resume": bool(args.resume),
            "collect_existing_pairs": bool(args.collect_existing_pairs),
            "save_pair_logits": bool(args.save_pair_logits),
            "pipeline_results_root": args.pipeline_results_root,
        },
    )
    update_status(status_path, "running", "starting")
    print(f"[start] run_id={run_id}")
    print(f"[start] model={args.model} device={device}")
    print(f"[start] activations_model_root={acts_model_root}")
    print(f"[start] probes_model_root={probes_model_root}")
    print(f"[start] results_model_root={results_model_root}")
    if only_sources or only_targets or only_segments:
        print(f"[filter] only_sources={sorted(only_sources) if only_sources else None}")
        print(f"[filter] only_targets={sorted(only_targets) if only_targets else None}")
        print(f"[filter] only_segments={sorted(only_segments) if only_segments else None}")
    print_stage_dashboard(progress)

    external_pipeline_root = (
        Path(args.pipeline_results_root)
        if args.pipeline_results_root
        else (results_model_root / "All_dataset_pairwise_results")
    )
    external_run_root = external_pipeline_root / run_id

    def mirror_pipeline_outputs() -> None:
        external_run_root.mkdir(parents=True, exist_ok=True)
        for sub in ["results", "meta", "checkpoints"]:
            src = run_root / sub
            if not src.exists():
                continue
            dst = external_run_root / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    def mark_step(step: str) -> None:
        done_set = set(progress["completed_steps"])
        done_set.add(step)
        progress["completed_steps"] = sorted(done_set)
        progress["updated_at"] = utc_now()
        write_json(progress_path, progress)

    rows_map, cols_map = stage_spec(args)
    pair_jobs = build_expected_pairs(rows_map, cols_map)
    eval_jobs = [j for j in pair_jobs if j.action == "eval_run"]
    collect_jobs: List[PairJob] = []
    configured_segments = [seg for seg in ["completion", "full"] if rows_map.get(seg) and cols_map.get(seg)]
    if only_segments:
        eval_jobs = [j for j in eval_jobs if j.segment in only_segments]
    if only_sources:
        eval_jobs = [j for j in eval_jobs if j.source_dataset in only_sources]
    if only_targets:
        eval_jobs = [j for j in eval_jobs if j.target_dataset in only_targets]

    active_segments = list(configured_segments)
    if only_segments:
        active_segments = [seg for seg in active_segments if seg in only_segments]
    if not active_segments:
        raise ValueError("No active segments remain after applying dataset and segment filters.")

    train_sources: List[str] = []
    for seg in active_segments:
        for ds_name in rows_map.get(seg, []):
            if ds_name not in train_sources:
                train_sources.append(ds_name)

    if only_sources or only_targets or only_segments:
        filtered_sources = {j.source_dataset for j in eval_jobs}
        train_sources = [s for s in train_sources if s in filtered_sources]

    # Export expected pair tables
    expected_rows: List[Dict[str, Any]] = []
    for j in pair_jobs:
        expected_rows.append(
            {
                "segment": j.segment,
                "source_dataset": j.source_dataset,
                "target_dataset": j.target_dataset,
                "cell_type": j.cell_type,
                "action": j.action,
                "pair_id": j.pair_id,
            }
        )
    write_csv_rows(
        out_dir / "pairs_expected.csv",
        expected_rows,
        ["segment", "source_dataset", "target_dataset", "cell_type", "action", "pair_id"],
    )
    write_csv_rows(
        out_dir / "pairs_to_run.csv",
        [r for r in expected_rows if r["action"] == "eval_run"],
        ["segment", "source_dataset", "target_dataset", "cell_type", "action", "pair_id"],
    )
    write_csv_rows(
        out_dir / "pairs_collected.csv",
        [],
        ["segment", "source_dataset", "target_dataset", "cell_type", "action", "pair_id"],
    )
    print(f"[info] eval jobs={len(eval_jobs)} collect-only jobs={len(collect_jobs)}")

    if args.dry_run:
        print("[dry_run] exiting before execution.")
        update_status(status_path, "completed", "dry_run complete")
        return 0

    # Stage 1: Train source-row probes
    update_status(status_path, "running", "stage 1 training")
    print("[stage 1/5] train source-row probes")
    train_jobs: List[Tuple[str, str]] = []
    for ds in train_sources:
        for pooling in poolings:
            train_jobs.append((ds, pooling))

    if args.skip_training:
        print("[train] skip_training enabled; skipping stage 1")
        mark_step("stage1_train")
    else:
        pbar = tqdm(train_jobs, desc="train_jobs", disable=not use_tqdm)
        for idx, (dataset_name, pooling) in enumerate(pbar, start=1):
            job_id = f"{dataset_name}__{pooling}"
            if args.resume and job_id in set(progress["completed_train_jobs"]):
                print(f"[train] skip completed {job_id}")
                continue
            base = dataset_base(dataset_name)
            out_pool_dir = probes_model_root / f"{base}_slices" / dataset_name / pooling
            has_probes = bool(find_probe_layers(out_pool_dir))
            if has_probes and not args.force_retrain:
                print(f"[train] skip existing probes {job_id}")
                progress["completed_train_jobs"].append(job_id)
                write_json(progress_path, progress)
                continue

            cmd = [
                sys.executable,
                "scripts/training/train_deception_probes.py",
                "--model",
                args.model,
                "--dataset",
                dataset_name,
                "--activations_dir",
                str(acts_base),
                "--pooling",
                pooling,
                "--output_dir",
                str(probes_base),
                "--output_subdir",
                f"{base}_slices",
                "--output_dataset_name",
                dataset_name,
                "--batch_size",
                str(args.train_batch_size),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.weight_decay),
                "--resume",
            ]
            t0 = time.time()
            print(f"[train] ({idx}/{len(train_jobs)}) start {job_id}")
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                update_status(status_path, "failed", f"training failed for {job_id}")
                raise RuntimeError(f"Training failed for {job_id}")
            elapsed = time.time() - t0
            print(f"[train] done {job_id} in {elapsed:.1f}s")
            progress["completed_train_jobs"].append(job_id)
            progress["updated_at"] = utc_now()
            write_json(progress_path, progress)
            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(f"[train] checkpoint at {idx}/{len(train_jobs)}")

    mark_step("stage1_train")
    print("[done] stage 1")

    # Stage 2: evaluate off-diagonal pairs
    update_status(status_path, "running", "stage 2 evaluation")
    print("[stage 2/5] evaluate off-diagonal pairs")
    pair_summary_index: Dict[str, Dict[str, Any]] = {}
    eval_pbar = tqdm(eval_jobs, desc="eval_pairs", disable=not use_tqdm)
    for i, job in enumerate(eval_pbar, start=1):
        pair_id = job.pair_id
        pair_dir = results_model_root / f"from-{job.source_dataset}" / f"to-{job.target_dataset}"
        pair_summary_path = pair_dir / "pair_summary.json"
        logits_ready = pair_logits_exist(pair_dir)
        logits_missing = bool(args.save_pair_logits and not logits_ready)

        if args.resume and pair_id in set(progress["completed_eval_pairs"]) and not logits_missing and not args.force_reeval:
            if pair_summary_path.exists():
                pair_summary_index[pair_id] = read_json(pair_summary_path)
            print(f"[eval] skip completed {pair_id}")
            continue
        if pair_summary_path.exists() and not args.force_reeval and not logits_missing:
            print(f"[eval] skip existing summary {pair_id}")
            pair_summary_index[pair_id] = read_json(pair_summary_path)
            progress["completed_eval_pairs"].append(pair_id)
            write_json(progress_path, progress)
            continue
        if args.collect_existing_pairs and not args.force_reeval and not logits_missing:
            parsed = parse_existing_pair_summary(pair_dir)
            if parsed is not None:
                payload = {
                    "source_dataset": job.source_dataset,
                    "target_dataset": job.target_dataset,
                    "split": "test",
                    "poolings": parsed.get("poolings", {}),
                    "overall_best": parsed.get("overall_best", {}),
                    "cell_type": job.cell_type,
                    "action": "collect_only",
                    "collected_at": utc_now(),
                    "source_format": parsed.get("source", "unknown"),
                }
                pair_dir.mkdir(parents=True, exist_ok=True)
                write_json(pair_summary_path, payload)
                pair_summary_index[pair_id] = payload
                progress["completed_eval_pairs"].append(pair_id)
                progress["completed_collect_pairs"].append(pair_id)
                progress["updated_at"] = utc_now()
                write_json(progress_path, progress)
                print(f"[eval] collected legacy summary {pair_id}")
                continue

        pair_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[eval] ({i}/{len(eval_jobs)}) start {job.source_dataset} -> {job.target_dataset}")
        try:
            summary = evaluate_pair_all_poolings(
                source_dataset=job.source_dataset,
                target_dataset=job.target_dataset,
                split="test",
                model_root_activations=acts_model_root,
                model_root_probes=probes_model_root,
                poolings=poolings,
                eval_batch_size=args.eval_batch_size,
                device=device,
                use_tqdm=use_tqdm,
                save_pair_logits_enabled=bool(args.save_pair_logits),
                pair_dir=pair_dir,
            )
            summary["cell_type"] = job.cell_type
            summary["action"] = "eval_run"
            write_json(pair_summary_path, summary)
            pair_summary_index[pair_id] = summary
            progress["completed_eval_pairs"].append(pair_id)
            progress["updated_at"] = utc_now()
            write_json(progress_path, progress)
            elapsed = time.time() - t0
            best = summary["overall_best"]
            print(
                f"[eval] done {pair_id} in {elapsed:.1f}s "
                f"(best={best['pooling']} L{best['layer']} auc={best['auc']:.4f})"
            )
        except Exception as exc:
            append_jsonl(out_dir / "pairs_missing.jsonl", {"pair_id": pair_id, "reason": str(exc), "stage": "eval"})
            print(f"[eval] missing/failed {pair_id}: {exc}")
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"[eval] checkpoint at {i}/{len(eval_jobs)}")

    mark_step("stage2_eval")
    print("[done] stage 2")

    # Stage 3: optional legacy collection compatibility stage
    update_status(status_path, "running", "stage 3 legacy collection")
    print("[stage 3/5] legacy collection compatibility")
    if args.collect_existing_pairs:
        print("[collect] compatibility mode is enabled; legacy summaries were collected during stage 2 when available.")
    else:
        print("[collect] compatibility mode disabled; no legacy pair collection performed.")
    mark_step("stage3_collect")
    print("[done] stage 3")

    # Stage 4: same-dataset test diagonals
    update_status(status_path, "running", "stage 4 diagonals")
    print("[stage 4/5] evaluate same-dataset test diagonal cells")
    filtered_eval_sources = {j.source_dataset for j in eval_jobs}
    filtered_eval_targets = {j.target_dataset for j in eval_jobs}
    diag_datasets: List[str] = []
    for seg in active_segments:
        for ds_name in rows_map.get(seg, []):
            if ds_name not in filtered_eval_sources:
                continue
            if ds_name not in filtered_eval_targets:
                continue
            if ds_name not in diag_datasets:
                diag_datasets.append(ds_name)
    diag_results: Dict[str, Dict[str, Any]] = {}
    if args.skip_diagonals:
        print("[diag] skip_diagonals enabled; skipping stage 4")
        mark_step("stage4_diag")
    else:
        if not diag_datasets:
            print("[diag] no diagonal datasets remain after filtering; skipping stage 4")
            mark_step("stage4_diag")
            print("[done] stage 4")
        else:
            diag_pbar = tqdm(diag_datasets, desc="diag_cells", disable=not use_tqdm)
            for i, ds_name in enumerate(diag_pbar, start=1):
                pair_id = f"{ds_name}__{ds_name}"
                pair_dir = results_model_root / f"from-{ds_name}" / f"to-{ds_name}"
                pair_summary_path = pair_dir / "pair_summary.json"
                diag_path = out_dir / "diagonals" / f"{ds_name}.json"
                logits_ready = pair_logits_exist(pair_dir)
                logits_missing = bool(args.save_pair_logits and not logits_ready)
                if args.resume and ds_name in set(progress["completed_diag_jobs"]) and not logits_missing and not args.force_reeval:
                    if pair_summary_path.exists():
                        summary = read_json(pair_summary_path)
                        diag_results[ds_name] = summary
                        pair_summary_index[pair_id] = summary
                    elif diag_path.exists():
                        diag_results[ds_name] = read_json(diag_path)
                    print(f"[diag] skip completed {ds_name}")
                    continue
                if pair_summary_path.exists() and not args.force_reeval and not logits_missing:
                    summary = read_json(pair_summary_path)
                    diag_results[ds_name] = summary
                    pair_summary_index[pair_id] = summary
                    progress["completed_diag_jobs"].append(ds_name)
                    progress["updated_at"] = utc_now()
                    write_json(progress_path, progress)
                    print(f"[diag] skip existing test summary {ds_name}")
                    continue
                t0 = time.time()
                try:
                    diag = diagonal_from_test(
                        dataset_name=ds_name,
                        model_root_activations=acts_model_root,
                        model_root_probes=probes_model_root,
                        poolings=poolings,
                        eval_batch_size=args.eval_batch_size,
                        device=device,
                        use_tqdm=use_tqdm,
                        save_pair_logits_enabled=bool(args.save_pair_logits),
                        pair_dir=pair_dir,
                    )
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    write_json(pair_summary_path, diag)
                    write_json(diag_path, diag)
                    diag_results[ds_name] = diag
                    pair_summary_index[pair_id] = diag
                    progress["completed_diag_jobs"].append(ds_name)
                    progress["updated_at"] = utc_now()
                    write_json(progress_path, progress)
                    best = diag["overall_best"]
                    elapsed = time.time() - t0
                    print(
                        f"[diag] done {ds_name} in {elapsed:.1f}s "
                        f"(best={best['pooling']} L{best['layer']} auc={best['auc']:.4f})"
                    )
                except Exception as exc:
                    append_jsonl(out_dir / "pairs_missing.jsonl", {"pair_id": f"{ds_name}__{ds_name}", "reason": str(exc), "stage": "diag"})
                    print(f"[diag] missing {ds_name}: {exc}")
                if args.progress_every > 0 and i % args.progress_every == 0:
                    print(f"[diag] checkpoint at {i}/{len(diag_datasets)}")

            mark_step("stage4_diag")
            print("[done] stage 4")

    # Stage 5: aggregate matrices
    update_status(status_path, "running", "stage 5 aggregate")
    print("[stage 5/5] aggregate matrices")
    missing_rows = []
    missing_jsonl = out_dir / "pairs_missing.jsonl"
    if missing_jsonl.exists():
        with missing_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                missing_rows.append(json.loads(line))

    def cell_metrics(source: str, target: str) -> Dict[str, Any]:
        if source == target:
            pair_id = f"{source}__{target}"
            summary = pair_summary_index.get(pair_id)
            if summary is None:
                path = results_model_root / f"from-{source}" / f"to-{target}" / "pair_summary.json"
                if path.exists():
                    summary = read_json(path)
            if summary is None:
                diag = diag_results.get(source)
                if not diag:
                    return {"status": "missing", "auc": None, "accuracy": None, "f1": None, "pooling": None, "layer": None, "source": "diag_missing"}
                summary = diag
            best = summary.get("overall_best", {})
            f1v = best.get("f1")
            return {
                "status": "ok" if best else "missing",
                "auc": float(best["auc"]) if "auc" in best else None,
                "accuracy": float(best["accuracy"]) if "accuracy" in best else None,
                "f1": float(f1v) if f1v is not None else None,
                "pooling": best.get("pooling"),
                "layer": int(best["layer"]) if "layer" in best and best["layer"] is not None else None,
                "source": summary.get("source", summary.get("action", "diag_test")),
            }
        pair_id = f"{source}__{target}"
        summary = pair_summary_index.get(pair_id)
        if summary is None:
            path = results_model_root / f"from-{source}" / f"to-{target}" / "pair_summary.json"
            if path.exists():
                summary = read_json(path)
            else:
                return {"status": "missing", "auc": None, "accuracy": None, "f1": None, "pooling": None, "layer": None, "source": "pair_eval"}
        best = summary.get("overall_best", {})
        f1v = best.get("f1")
        return {
            "status": "ok" if best else "missing",
            "auc": float(best["auc"]) if "auc" in best else None,
            "accuracy": float(best["accuracy"]) if "accuracy" in best else None,
            "f1": float(f1v) if f1v is not None else None,
            "pooling": best.get("pooling"),
            "layer": int(best["layer"]) if "layer" in best and best["layer"] is not None else None,
            "source": summary.get("action", "pair_eval"),
        }

    def build_matrix(segment: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        rows = rows_map[segment]
        cols = cols_map[segment]
        long_rows: List[Dict[str, Any]] = []
        auc_wide: Dict[str, Dict[str, Any]] = {}
        acc_wide: Dict[str, Dict[str, Any]] = {}
        f1_wide: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            auc_wide[short_name(r)] = {"row": short_name(r)}
            acc_wide[short_name(r)] = {"row": short_name(r)}
            f1_wide[short_name(r)] = {"row": short_name(r)}
            for c in cols:
                m = cell_metrics(r, c)
                col_name = short_name(c)
                long_rows.append(
                    {
                        "segment": segment,
                        "row_dataset": r,
                        "col_dataset": c,
                        "row_short": short_name(r),
                        "col_short": col_name,
                        "auc": m["auc"],
                        "accuracy": m["accuracy"],
                        "f1": m["f1"],
                        "status": m["status"],
                        "pooling": m["pooling"],
                        "layer": m["layer"],
                        "source": m["source"],
                    }
                )
                auc_wide[short_name(r)][col_name] = m["auc"]
                acc_wide[short_name(r)][col_name] = m["accuracy"]
                f1_wide[short_name(r)][col_name] = m["f1"]
        return long_rows, auc_wide, acc_wide, f1_wide

    segment_cells: Dict[str, List[Dict[str, Any]]] = {}
    for segment in active_segments:
        seg_long, seg_auc, seg_acc, seg_f1 = build_matrix(segment)
        segment_cells[segment] = seg_long
        write_csv_rows(
            out_dir / f"matrix_{segment}.csv",
            seg_long,
            ["segment", "row_dataset", "col_dataset", "row_short", "col_short", "auc", "accuracy", "f1", "status", "pooling", "layer", "source"],
        )
        write_csv_rows(
            out_dir / f"matrix_{segment}_auc.csv",
            list(seg_auc.values()),
            list(list(seg_auc.values())[0].keys()),
        )
        write_csv_rows(
            out_dir / f"matrix_{segment}_accuracy.csv",
            list(seg_acc.values()),
            list(list(seg_acc.values())[0].keys()),
        )
        write_csv_rows(
            out_dir / f"matrix_{segment}_f1.csv",
            list(seg_f1.values()),
            list(list(seg_f1.values())[0].keys()),
        )
        write_json(out_dir / f"matrix_{segment}.json", {"segment": segment, "cells": seg_long})

    missing_table_rows: List[Dict[str, Any]] = []
    for mr in missing_rows:
        missing_table_rows.append(
            {
                "pair_id": mr.get("pair_id"),
                "stage": mr.get("stage"),
                "reason": mr.get("reason"),
            }
        )
    write_csv_rows(out_dir / "pairs_missing.csv", missing_table_rows, ["pair_id", "stage", "reason"])

    summary = {
        "run_id": run_id,
        "completed_at": utc_now(),
        "model": args.model,
        "active_segments": active_segments,
        "counts": {
            "train_jobs_total": len(train_jobs),
            "train_jobs_completed": len(set(progress["completed_train_jobs"])),
            "eval_pairs_total": len(eval_jobs),
            "eval_pairs_completed": len(set(progress["completed_eval_pairs"])),
            "collect_pairs_total": len(collect_jobs),
            "collect_pairs_completed": len(set(progress["completed_collect_pairs"])),
            "diag_total": len(diag_datasets),
            "diag_completed": len(set(progress["completed_diag_jobs"])),
            "missing_rows": len(missing_table_rows),
        },
        "outputs": {
            "run_root": str(run_root),
            "external_run_root": str(external_run_root),
            "results_model_root": str(results_model_root),
            "pairs_expected_csv": str(out_dir / "pairs_expected.csv"),
            "pairs_missing_csv": str(out_dir / "pairs_missing.csv"),
        },
    }
    for segment in active_segments:
        summary["outputs"][f"matrix_{segment}_csv"] = str(out_dir / f"matrix_{segment}.csv")
        summary["counts"][f"{segment}_cells"] = len(segment_cells.get(segment, []))
    write_json(out_dir / "summary.json", summary)
    mirror_pipeline_outputs()
    mark_step("stage5_aggregate")

    update_status(status_path, "completed", "finished")
    print("[done] stage 5")
    print(f"[done] summary -> {out_dir / 'summary.json'}")
    print(f"[done] mirrored pipeline outputs -> {external_run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
