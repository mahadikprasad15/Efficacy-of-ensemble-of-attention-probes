#!/usr/bin/env python3
"""
Replay probe attribution from saved checkpoints (post-hoc, no retraining).

This script targets per-layer probes trained with train_deception_probes.py and
reconstructs attribution CSV outputs for selected layers and poolings.

Typical usage (Colab):
    python scripts/analysis/replay_attribution_from_checkpoints.py \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --dataset Deception-Roleplaying \
      --pooling attn \
      --layers 4,6 \
      --activations_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations \
      --source_attr_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/probe_attribution \
      --final_weights_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes \
      --output_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/probe_attribution_replay \
      --save_full_matrix
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Keep compatibility with project layout when running from repo root.
import sys
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
from actprobe.probes.models import LayerProbe  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("replay_attribution")


@dataclass
class ReplayPaths:
    model_dir: str
    source_pool_dir: str
    checkpoint_root: str
    final_weight_pool_dir: str
    output_pool_dir: str
    train_dir: str
    val_dir: str
    ood_dir: Optional[str]


class CachedDeceptionDataset(Dataset):
    """Loads shard activations + manifest labels and returns IDs for attribution."""

    def __init__(
        self,
        activations_dir: str,
        pool_before_batch: bool = False,
        pooling_type: str = "mean",
        return_ids: bool = True,
    ):
        self.items: List[Dict[str, object]] = []
        self.pool_before_batch = pool_before_batch
        self.pooling_type = pooling_type
        self.return_ids = return_ids
        self.input_format = "pooled"

        shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
        shards = sorted(glob.glob(shard_pattern))
        if not shards:
            raise FileNotFoundError(f"No shard files found: {shard_pattern}")

        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        manifest: Dict[str, dict] = {}
        with open(manifest_path, "r") as f:
            for line in f:
                row = json.loads(line)
                manifest[row["id"]] = row

        for shard_path in tqdm(shards, desc=f"Loading {os.path.basename(activations_dir)} shards"):
            tensors = load_file(shard_path)
            for eid, tensor in tensors.items():
                meta = manifest.get(eid)
                if not meta:
                    continue
                label = int(meta.get("label", -1))
                if label not in (0, 1):
                    continue

                x = tensor
                if pool_before_batch and len(x.shape) == 3:
                    if pooling_type == "mean":
                        x = x.mean(dim=1)
                    elif pooling_type == "max":
                        x = x.max(dim=1).values
                    elif pooling_type == "last":
                        x = x[:, -1, :]
                    else:
                        raise ValueError(f"pool_before_batch does not support pooling '{pooling_type}'")

                self.items.append({"id": eid, "tensor": x, "label": label})

        if not self.items:
            raise ValueError(f"No valid labeled items found in {activations_dir}")

        shape = self.items[0]["tensor"].shape
        if len(shape) == 2:
            self.input_format = "final_token"
        elif len(shape) == 3:
            self.input_format = "pooled"
        else:
            raise ValueError(f"Unexpected tensor shape: {shape}")

        logger.info(
            "Loaded %d samples from %s (format=%s)",
            len(self.items),
            activations_dir,
            self.input_format,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        if self.return_ids:
            return (
                item["tensor"].float(),
                torch.tensor(item["label"], dtype=torch.float32),
                item["id"],
            )
        return item["tensor"].float(), torch.tensor(item["label"], dtype=torch.float32)


class LayerSliceDataset(Dataset):
    """Select a single layer from cached per-sample tensors."""

    def __init__(self, base_dataset: CachedDeceptionDataset, layer_idx: int):
        self.base = base_dataset
        self.layer_idx = int(layer_idx)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if len(item) == 3:
            x, y, sid = item
            return x[self.layer_idx], y, sid
        x, y = item
        return x[self.layer_idx], y


def parse_layers(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    out: List[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return sorted(set(out))


def parse_epoch_from_checkpoint(path: str) -> Optional[int]:
    m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def discover_layers_from_checkpoints(checkpoint_root: str) -> List[int]:
    layers = []
    for p in glob.glob(os.path.join(checkpoint_root, "layer_*")):
        if not os.path.isdir(p):
            continue
        m = re.search(r"layer_(\d+)$", os.path.basename(p))
        if m:
            layers.append(int(m.group(1)))
    return sorted(set(layers))


def discover_poolings(model_dir: str, dataset: str, source_attr_root: str) -> List[str]:
    base = os.path.join(source_attr_root, model_dir, dataset)
    if not os.path.isdir(base):
        return []
    out = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    return sorted(out)


def evaluate_binary(model: nn.Module, loader: Optional[DataLoader], device: str) -> Tuple[float, float]:
    if loader is None:
        return np.nan, np.nan

    model.eval()
    preds: List[float] = []
    targets: List[float] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            logits = model(x).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend(probs.tolist())
            targets.extend(y.numpy().tolist())

    if not preds:
        return np.nan, np.nan

    p = np.array(preds)
    t = np.array(targets)
    try:
        auc = float(roc_auc_score(t, p))
    except Exception:
        auc = 0.5
    acc = float(accuracy_score(t, (p > 0.5).astype(int)))
    return auc, acc


def _flatten_params(model: nn.Module) -> torch.Tensor:
    params = [p.detach().flatten().cpu() for p in model.parameters()]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten().cpu())
        else:
            grads.append(p.grad.detach().flatten().cpu())
    if not grads:
        return torch.empty(0)
    return torch.cat(grads)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = (torch.norm(a) * torch.norm(b)).item()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def _compute_per_sample_grads(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> List[torch.Tensor]:
    logits = model(x).squeeze(-1)
    losses = loss_fn(logits, y)
    grads: List[torch.Tensor] = []
    for i in range(len(losses)):
        model.zero_grad(set_to_none=True)
        losses[i].backward(retain_graph=True)
        grads.append(_flatten_grads(model))
    model.zero_grad(set_to_none=True)
    return grads


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)


def infer_model_pooling_type(sample_shape: torch.Size, pooling: str, pool_before_batch: bool) -> str:
    model_pooling_type = pooling
    if len(sample_shape) == 2 or pool_before_batch:
        if pooling not in ("none", "last"):
            model_pooling_type = "none"
    if pool_before_batch:
        model_pooling_type = "none"
    return model_pooling_type


def load_checkpoints_for_layer(checkpoint_root: str, layer_idx: int) -> List[Tuple[int, str]]:
    layer_dir = os.path.join(checkpoint_root, f"layer_{layer_idx}")
    if not os.path.isdir(layer_dir):
        return []

    out: List[Tuple[int, str]] = []
    for p in glob.glob(os.path.join(layer_dir, "epoch_*.pt")):
        ep = parse_epoch_from_checkpoint(p)
        if ep is None:
            continue
        out.append((ep, p))
    out.sort(key=lambda x: x[0])
    return out


def choose_w_star(
    model: nn.Module,
    final_weight_path: str,
    checkpoints: List[Tuple[int, str]],
    device: str,
) -> torch.Tensor:
    if os.path.exists(final_weight_path):
        state = torch.load(final_weight_path, map_location=device)
        model.load_state_dict(state)
        logger.info("Using final probe weights: %s", final_weight_path)
        return _flatten_params(model)

    if checkpoints:
        last_ckpt = checkpoints[-1][1]
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state)
        logger.warning("Final probe weights missing, using last checkpoint as w*: %s", last_ckpt)
        return _flatten_params(model)

    raise FileNotFoundError(
        f"No final weights at {final_weight_path} and no checkpoints available for fallback"
    )


def run_replay_for_layer(
    layer_idx: int,
    model: nn.Module,
    checkpoints: List[Tuple[int, str]],
    w_star: torch.Tensor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    ood_loader: Optional[DataLoader],
    attr_every_n: int,
    top_n: int,
    out_dir: str,
    device: str,
    save_full_matrix: bool,
    skip_checkpoint_eval: bool,
    max_train_batches: int,
) -> Dict[str, object]:
    training_dynamics: List[List[object]] = []
    checkpoint_metrics: List[List[object]] = []

    sample_progress: Dict[str, float] = {}
    sample_influence: Dict[str, float] = {}
    sample_counts: Dict[str, int] = {}

    layer_progress_total = 0.0
    layer_influence_total = 0.0
    layer_count = 0

    best_id_auc = -1.0
    best_ood_auc = -1.0

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for epoch, ckpt_path in tqdm(checkpoints, desc=f"Replay layer {layer_idx}"):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        w_t = _flatten_params(model)
        cos_to_final = _cosine(w_t, w_star)
        w_norm = float(torch.norm(w_t).item()) if w_t.numel() else 0.0
        training_dynamics.append([epoch, cos_to_final, w_norm])

        if not skip_checkpoint_eval:
            id_auc, id_acc = evaluate_binary(model, val_loader, device)
            if np.isfinite(id_auc):
                best_id_auc = max(best_id_auc, id_auc)
                checkpoint_metrics.append([epoch, "id", id_auc, id_acc, 1 if id_auc >= best_id_auc else 0])

            if ood_loader is not None:
                ood_auc, ood_acc = evaluate_binary(model, ood_loader, device)
                if np.isfinite(ood_auc):
                    best_ood_auc = max(best_ood_auc, ood_auc)
                    checkpoint_metrics.append([epoch, "ood", ood_auc, ood_acc, 1 if ood_auc >= best_ood_auc else 0])

        delta = w_star - w_t

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            if attr_every_n > 1 and batch_idx % attr_every_n != 0:
                continue
            if len(batch) == 3:
                x, y, ids = batch
            else:
                x, y = batch
                ids = [f"idx_{batch_idx}_{i}" for i in range(len(y))]

            x = x.to(device)
            y = y.to(device)

            grads = _compute_per_sample_grads(model, x, y, loss_fn)
            for i, g in enumerate(grads):
                sid = str(ids[i])
                alignment = -_cosine(g, w_star)
                influence = float(torch.dot(g, delta)) if g.numel() else 0.0

                sample_progress[sid] = sample_progress.get(sid, 0.0) + alignment
                sample_influence[sid] = sample_influence.get(sid, 0.0) + influence
                sample_counts[sid] = sample_counts.get(sid, 0) + 1

                layer_progress_total += alignment
                layer_influence_total += influence
                layer_count += 1

    # Canonical outputs mirroring train_deception_probes attribution files.
    _write_csv(
        os.path.join(out_dir, f"training_dynamics_layer_{layer_idx}.csv"),
        ["epoch", "cos_to_final", "w_norm"],
        training_dynamics,
    )

    _write_csv(
        os.path.join(out_dir, f"checkpoint_metrics_layer_{layer_idx}.csv"),
        ["epoch", "split", "auc", "acc", "best_so_far"],
        checkpoint_metrics,
    )

    mean_progress = layer_progress_total / max(layer_count, 1)
    mean_influence = layer_influence_total / max(layer_count, 1)

    _write_csv(
        os.path.join(out_dir, f"layer_progress_layer_{layer_idx}.csv"),
        ["layer", "mean_grad_alignment", "count"],
        [[layer_idx, mean_progress, layer_count]],
    )

    _write_csv(
        os.path.join(out_dir, f"layer_influence_layer_{layer_idx}.csv"),
        ["layer", "mean_influence", "count"],
        [[layer_idx, mean_influence, layer_count]],
    )

    top_prog = sorted(sample_progress.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    _write_csv(
        os.path.join(out_dir, f"sample_progress_top{top_n}_layer_{layer_idx}.csv"),
        ["sample_id", "grad_alignment"],
        [[sid, score] for sid, score in top_prog],
    )

    top_inf = sorted(sample_influence.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    _write_csv(
        os.path.join(out_dir, f"sample_influence_top{top_n}_layer_{layer_idx}.csv"),
        ["sample_id", "influence"],
        [[sid, score] for sid, score in top_inf],
    )

    if save_full_matrix:
        full_progress_rows = []
        full_influence_rows = []
        for sid in sample_counts:
            cnt = sample_counts[sid]
            full_progress_rows.append([sid, layer_idx, sample_progress[sid], cnt])
            full_influence_rows.append([sid, layer_idx, sample_influence[sid], cnt])

        _write_csv(
            os.path.join(out_dir, f"sample_progress_full_layer_{layer_idx}.csv"),
            ["sample_id", "layer", "grad_alignment", "count_updates"],
            full_progress_rows,
        )
        _write_csv(
            os.path.join(out_dir, f"sample_influence_full_layer_{layer_idx}.csv"),
            ["sample_id", "layer", "influence", "count_updates"],
            full_influence_rows,
        )

    return {
        "layer": layer_idx,
        "num_checkpoints": len(checkpoints),
        "num_samples_seen": len(sample_counts),
        "mean_grad_alignment": mean_progress,
        "mean_influence": mean_influence,
    }


def build_paths(args: argparse.Namespace, pooling: str) -> ReplayPaths:
    model_dir = args.model.replace("/", "_")

    source_pool_dir = os.path.join(args.source_attr_root, model_dir, args.dataset, pooling)
    checkpoint_root = args.checkpoint_root or os.path.join(source_pool_dir, "checkpoints")
    final_weight_pool_dir = os.path.join(args.final_weights_root, model_dir, args.dataset, pooling)
    output_pool_dir = os.path.join(args.output_root, model_dir, args.dataset, pooling)

    train_dir = os.path.join(args.activations_root, model_dir, args.dataset, args.train_split)
    val_dir = os.path.join(args.activations_root, model_dir, args.dataset, args.val_split)

    ood_dir = None
    if args.ood_dataset:
        ood_root = args.ood_activations_root or args.activations_root
        ood_dir = os.path.join(ood_root, model_dir, args.ood_dataset, args.ood_split)

    return ReplayPaths(
        model_dir=model_dir,
        source_pool_dir=source_pool_dir,
        checkpoint_root=checkpoint_root,
        final_weight_pool_dir=final_weight_pool_dir,
        output_pool_dir=output_pool_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        ood_dir=ood_dir,
    )


def run_pooling_replay(args: argparse.Namespace, pooling: str, layers_override: List[int]) -> Dict[str, object]:
    paths = build_paths(args, pooling)

    if not os.path.exists(paths.train_dir):
        raise FileNotFoundError(f"Train activations not found: {paths.train_dir}")
    if not os.path.isdir(paths.checkpoint_root):
        raise FileNotFoundError(f"Checkpoint root not found: {paths.checkpoint_root}")

    os.makedirs(paths.output_pool_dir, exist_ok=True)

    train_ds = CachedDeceptionDataset(
        paths.train_dir,
        pool_before_batch=args.pool_before_batch,
        pooling_type=pooling,
        return_ids=True,
    )

    val_ds = None
    if os.path.exists(paths.val_dir):
        val_ds = CachedDeceptionDataset(
            paths.val_dir,
            pool_before_batch=args.pool_before_batch,
            pooling_type=pooling,
            return_ids=True,
        )
    else:
        logger.warning("Validation split not found for pooling '%s': %s", pooling, paths.val_dir)

    ood_ds = None
    if paths.ood_dir and os.path.exists(paths.ood_dir):
        ood_ds = CachedDeceptionDataset(
            paths.ood_dir,
            pool_before_batch=args.pool_before_batch,
            pooling_type=pooling,
            return_ids=True,
        )
    elif paths.ood_dir:
        logger.warning("OOD split not found for pooling '%s': %s", pooling, paths.ood_dir)

    sample_tensor = train_ds.items[0]["tensor"]
    if len(sample_tensor.shape) == 2:
        num_layers, hidden_dim = sample_tensor.shape
    else:
        num_layers, _, hidden_dim = sample_tensor.shape

    model_pooling_type = infer_model_pooling_type(sample_tensor.shape, pooling, args.pool_before_batch)
    logger.info(
        "Pooling '%s': inferred model_pooling_type=%s (L=%d, D=%d)",
        pooling,
        model_pooling_type,
        num_layers,
        hidden_dim,
    )

    layers = layers_override or discover_layers_from_checkpoints(paths.checkpoint_root)
    if not layers:
        raise ValueError(f"No layers provided/discovered under {paths.checkpoint_root}")

    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError("No valid layers remain after filtering by activation tensor layer count")

    summary_rows = []

    for layer_idx in layers:
        checkpoints = load_checkpoints_for_layer(paths.checkpoint_root, layer_idx)
        if not checkpoints:
            logger.warning("Skipping layer %d: no checkpoints found", layer_idx)
            continue
        if args.max_checkpoints > 0 and len(checkpoints) > args.max_checkpoints:
            checkpoints = checkpoints[-args.max_checkpoints:]

        layer_train_loader = DataLoader(
            LayerSliceDataset(train_ds, layer_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        layer_val_loader = None
        if val_ds is not None:
            layer_val_loader = DataLoader(
                LayerSliceDataset(val_ds, layer_idx),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
        layer_ood_loader = None
        if ood_ds is not None:
            layer_ood_loader = DataLoader(
                LayerSliceDataset(ood_ds, layer_idx),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

        model = LayerProbe(input_dim=hidden_dim, pooling_type=model_pooling_type).to(args.device)

        final_weight_path = os.path.join(paths.final_weight_pool_dir, f"probe_layer_{layer_idx}.pt")
        w_star = choose_w_star(model, final_weight_path, checkpoints, args.device)

        row = run_replay_for_layer(
            layer_idx=layer_idx,
            model=model,
            checkpoints=checkpoints,
            w_star=w_star,
            train_loader=layer_train_loader,
            val_loader=layer_val_loader,
            ood_loader=layer_ood_loader,
            attr_every_n=args.attr_every_n,
            top_n=args.top_n,
            out_dir=paths.output_pool_dir,
            device=args.device,
            save_full_matrix=args.save_full_matrix,
            skip_checkpoint_eval=args.skip_checkpoint_eval,
            max_train_batches=args.max_train_batches,
        )
        summary_rows.append(row)

    if summary_rows:
        _write_csv(
            os.path.join(paths.output_pool_dir, "replay_summary.csv"),
            ["layer", "num_checkpoints", "num_samples_seen", "mean_grad_alignment", "mean_influence"],
            [
                [
                    r["layer"],
                    r["num_checkpoints"],
                    r["num_samples_seen"],
                    r["mean_grad_alignment"],
                    r["mean_influence"],
                ]
                for r in summary_rows
            ],
        )

    return {
        "pooling": pooling,
        "layers_requested": layers_override,
        "layers_processed": [r["layer"] for r in summary_rows],
        "output_dir": paths.output_pool_dir,
        "num_layers_processed": len(summary_rows),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay attribution from saved probe checkpoints (post-hoc).")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--probe_type", type=str, default="single", choices=["single", "combined"])

    parser.add_argument("--pooling", type=str, default=None, help="Single pooling to process")
    parser.add_argument("--poolings", nargs="+", default=None, help="Optional list of poolings")
    parser.add_argument("--layers", type=str, default="", help="Comma-separated layer indices, e.g. 4,6")

    parser.add_argument("--activations_root", type=str, default="data/activations")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")

    parser.add_argument("--ood_dataset", type=str, default="Deception-InsiderTrading")
    parser.add_argument("--ood_split", type=str, default="test")
    parser.add_argument("--ood_activations_root", type=str, default=None)

    parser.add_argument("--source_attr_root", type=str, default="results/probe_attribution")
    parser.add_argument("--checkpoint_root", type=str, default=None, help="Override checkpoints root for one pooling run")
    parser.add_argument("--final_weights_root", type=str, default="data/probes")
    parser.add_argument("--output_root", type=str, default="results/probe_attribution_replay")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--attr_every_n", type=int, default=50)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--save_full_matrix", action="store_true")
    parser.add_argument("--skip_checkpoint_eval", action="store_true")
    parser.add_argument("--max_checkpoints", type=int, default=0, help="If >0, replay only the latest N checkpoints per layer")
    parser.add_argument("--max_train_batches", type=int, default=0, help="If >0, cap attribution batches per checkpoint (debug/smoke)")
    parser.add_argument("--pool_before_batch", action="store_true")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> int:
    args = make_parser().parse_args()

    layers = parse_layers(args.layers)

    if args.poolings:
        poolings = args.poolings
    elif args.pooling:
        poolings = [args.pooling]
    else:
        model_dir = args.model.replace("/", "_")
        poolings = discover_poolings(model_dir, args.dataset, args.source_attr_root)
        if not poolings:
            raise ValueError("No pooling specified and none discovered under source_attr_root")

    logger.info("Running replay for probe_type=%s, poolings=%s, layers=%s", args.probe_type, poolings, layers or "auto")

    overall = []
    for pooling in poolings:
        result = run_pooling_replay(args, pooling, layers)
        overall.append(result)

    out_manifest = {
        "model": args.model,
        "dataset": args.dataset,
        "probe_type": args.probe_type,
        "results": overall,
    }

    model_dir = args.model.replace("/", "_")
    manifest_dir = os.path.join(args.output_root, model_dir, args.dataset)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "replay_run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(out_manifest, f, indent=2)

    logger.info("Replay complete. Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
