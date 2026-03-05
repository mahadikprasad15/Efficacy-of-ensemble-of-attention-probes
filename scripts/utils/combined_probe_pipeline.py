#!/usr/bin/env python3
"""
Shared utilities for combined deception probe pipelines.

This module provides:
- Manifest/shard indexing across multiple split directories
- Lazy shard-cached dataset for per-layer training/evaluation
- Probe training/evaluation helpers with resume-friendly outputs
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Repo-local import
import sys
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
from actprobe.probes.models import LayerProbe  # noqa: E402


POOLING_ORDER = ["attn", "last", "max", "mean"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def read_json(path: Path, default: Optional[Dict] = None) -> Dict:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def model_dir_name(model: str) -> str:
    return model.replace("/", "_")


def dataset_segment_name(dataset_base: str, segment: str) -> str:
    if dataset_base.endswith("-completion") or dataset_base.endswith("-full"):
        return dataset_base
    return f"{dataset_base}-{segment}"


def safe_auc(labels: Sequence[int], probs: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if y.size == 0:
        return 0.5
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return 0.5


@dataclass(frozen=True)
class SampleRef:
    shard_path: str
    sample_id: str
    label: int


class SplitIndex:
    """
    Indexes samples across one or more split directories without loading all tensors.
    """

    def __init__(self, split_dirs: Sequence[Path], positive_labels: Sequence[int] = (0, 1)):
        self.split_dirs = [Path(p) for p in split_dirs]
        self.positive_labels = set(int(x) for x in positive_labels)
        self.entries: List[SampleRef] = []
        self.input_type: Optional[str] = None  # token|final
        self.num_layers: Optional[int] = None
        self.d_model: Optional[int] = None
        self.label_hist: Dict[int, int] = {}

        for split_dir in self.split_dirs:
            self._index_split_dir(split_dir)

        if not self.entries:
            raise RuntimeError(f"No labeled samples found in split dirs: {self.split_dirs}")

        self._infer_shape_from_first_sample()

    def _index_split_dir(self, split_dir: Path) -> None:
        manifest_path = split_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        shard_paths = sorted(split_dir.glob("shard_*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards in: {split_dir}")

        shard_by_idx: Dict[int, Path] = {}
        for p in shard_paths:
            name = p.stem  # shard_000
            try:
                idx = int(name.split("_")[-1])
                shard_by_idx[idx] = p
            except Exception:
                continue

        fallback_sorted = [str(p) for p in shard_paths]

        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = str(row.get("id", "")).strip()
                if not sid:
                    continue
                label = int(row.get("label", -1))
                if label not in self.positive_labels:
                    continue

                shard_idx_raw = row.get("shard", None)
                shard_path: Optional[str] = None
                if shard_idx_raw is not None:
                    try:
                        shard_idx = int(shard_idx_raw)
                        if shard_idx in shard_by_idx:
                            shard_path = str(shard_by_idx[shard_idx])
                    except Exception:
                        shard_path = None

                if shard_path is None:
                    # Safe fallback if manifest has no shard field.
                    # We do not know exact shard for sid; search lazily via all shards later.
                    # Store sentinel with first shard path and fallback-search in dataset loader.
                    shard_path = fallback_sorted[0]

                self.entries.append(SampleRef(shard_path=shard_path, sample_id=sid, label=label))
                self.label_hist[label] = self.label_hist.get(label, 0) + 1

    def _infer_shape_from_first_sample(self) -> None:
        first = self.entries[0]
        shard = load_file(first.shard_path)
        if first.sample_id not in shard:
            # Try scanning siblings for fallback-manifest case.
            parent = Path(first.shard_path).parent
            found = None
            for p in sorted(parent.glob("shard_*.safetensors")):
                sh = load_file(str(p))
                if first.sample_id in sh:
                    found = sh[first.sample_id]
                    break
            if found is None:
                raise KeyError(f"Sample id {first.sample_id} not found in any shard under {parent}")
            tensor = found
        else:
            tensor = shard[first.sample_id]

        if tensor.dim() == 3:
            self.input_type = "token"
            self.num_layers = int(tensor.shape[0])
            self.d_model = int(tensor.shape[2])
        elif tensor.dim() == 2:
            self.input_type = "final"
            self.num_layers = int(tensor.shape[0])
            self.d_model = int(tensor.shape[1])
        else:
            raise ValueError(f"Unexpected activation shape: {tuple(tensor.shape)}")


class LayerSampleDataset(Dataset):
    """
    Lazy per-layer dataset with shard cache.
    """

    def __init__(
        self,
        split_index: SplitIndex,
        layer: int,
        max_shards_cached: int = 4,
        shuffle_seed: Optional[int] = None,
    ):
        if split_index.num_layers is None:
            raise ValueError("split_index has no shape info")
        if layer < 0 or layer >= int(split_index.num_layers):
            raise ValueError(f"Layer {layer} out of range [0, {split_index.num_layers})")

        self.index = split_index
        self.layer = int(layer)
        self.max_shards_cached = int(max_shards_cached)
        self.items = list(split_index.entries)
        if shuffle_seed is not None:
            rng = random.Random(int(shuffle_seed))
            rng.shuffle(self.items)

        self._cache: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def _get_shard(self, path: str) -> Dict[str, torch.Tensor]:
        if path in self._cache:
            v = self._cache.pop(path)
            self._cache[path] = v
            return v

        loaded = load_file(path)
        self._cache[path] = loaded
        while len(self._cache) > self.max_shards_cached:
            self._cache.popitem(last=False)
        return loaded

    def _find_sample_tensor(self, ref: SampleRef) -> torch.Tensor:
        shard = self._get_shard(ref.shard_path)
        if ref.sample_id in shard:
            return shard[ref.sample_id]

        # Manifest fallback: scan same directory shards.
        parent = Path(ref.shard_path).parent
        for p in sorted(parent.glob("shard_*.safetensors")):
            sh = self._get_shard(str(p))
            if ref.sample_id in sh:
                return sh[ref.sample_id]

        raise KeyError(f"Sample id {ref.sample_id} not found under {parent}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ref = self.items[idx]
        tensor = self._find_sample_tensor(ref)

        if tensor.dim() == 3:
            x = tensor[self.layer, :, :].float()  # (T,D)
        elif tensor.dim() == 2:
            x = tensor[self.layer, :].float()  # (D,)
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")

        y = torch.tensor(ref.label, dtype=torch.float32)
        return x, y


def _model_forward(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    # LayerProbe expects (B,T,D). For final-token features, use T=1.
    if xb.dim() == 2:
        xb = xb.unsqueeze(1)
    logits = model(xb)
    return logits.reshape(-1)


def evaluate_probe_on_dataset(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    probs: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = _model_forward(model, xb)
            p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
            probs.extend(p.tolist())
            labels.extend(yb.detach().cpu().numpy().astype(np.int64).tolist())

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)

    out = {
        "auc": safe_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)) if y.size else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)) if y.size else 0.0,
        "count": int(y.size),
    }
    return out


def train_layer_probe(
    *,
    pooling: str,
    layer: int,
    d_model: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    model = LayerProbe(input_dim=d_model, pooling_type=pooling).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = -1.0
    best_epoch = 0
    patience_ctr = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = _model_forward(model, xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_probe_on_dataset(
            model=model,
            dataset=val_dataset,
            device=device,
            batch_size=batch_size,
        )
        val_auc = float(val_metrics["auc"])
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate_probe_on_dataset(
        model=model,
        dataset=val_dataset,
        device=device,
        batch_size=batch_size,
    )
    final_val["best_epoch"] = int(best_epoch)
    final_val["pooling"] = pooling
    final_val["layer"] = int(layer)
    return model, final_val


def load_probe_state_with_compat(probe_path: Path, pooling: str, input_dim: int, device: torch.device) -> nn.Module:
    state = torch.load(str(probe_path), map_location=device)
    model = LayerProbe(input_dim=input_dim, pooling_type=pooling).to(device)
    # Keep compatibility with checkpoints that store logits head with exact same key names.
    model.load_state_dict(state)
    model.eval()
    return model


def choose_best_row(rows: Sequence[Dict]) -> Dict:
    if not rows:
        raise ValueError("No rows to rank")

    def _key(r: Dict) -> Tuple[float, float, float, int, int]:
        pooling = str(r.get("pooling", "mean"))
        try:
            pidx = POOLING_ORDER.index(pooling)
        except ValueError:
            pidx = len(POOLING_ORDER)
        return (
            float(r.get("auc", -1.0)),
            float(r.get("accuracy", -1.0)),
            float(r.get("f1", -1.0)),
            -int(r.get("layer", 10_000)),
            -pidx,
        )

    return max(rows, key=_key)


def format_sps(count: int, t0: float) -> float:
    dt = max(time.time() - t0, 1e-9)
    return float(count / dt)
