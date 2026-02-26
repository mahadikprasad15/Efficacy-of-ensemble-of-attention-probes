#!/usr/bin/env python3
"""
Train MultiAttention probe families with resumable checkpoints.

This script is intentionally decoupled from OOD evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


def parse_input_layers(value: str) -> List[int]:
    if not value.strip():
        return []
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


class CachedActivationsDataset(Dataset):
    """
    Loads all labeled activation samples into memory.

    Each item returns:
      tensor: (L, T, D)
      label: int
      id: sample id
    """

    def __init__(self, split_dir: Path):
        manifest_path = split_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        shards = sorted(split_dir.glob("shard_*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No shards found in {split_dir}")

        label_map: Dict[str, int] = {}
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                label_map[obj["id"]] = int(obj.get("label", -1))

        self.items = []
        for shard in tqdm(shards, desc=f"Loading {split_dir.name} shards"):
            tensors = load_file(str(shard))
            for eid, tensor in tensors.items():
                label = label_map.get(eid, -1)
                if label < 0:
                    continue
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(1)  # (L,D) -> (L,1,D)
                if tensor.ndim != 3:
                    continue
                self.items.append(
                    {
                        "id": eid,
                        "tensor": tensor.float(),  # (L,T,D)
                        "label": label,
                    }
                )

        if not self.items:
            raise RuntimeError(f"No labeled items loaded from {split_dir}")

        sample = self.items[0]["tensor"]
        self.num_layers = int(sample.shape[0])
        self.hidden_dim = int(sample.shape[-1])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return item["tensor"], item["label"], item["id"]


class SingleLayerDataset(Dataset):
    def __init__(self, base: CachedActivationsDataset, layer: int):
        self.base = base
        self.layer = layer

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, eid = self.base[idx]
        return x[self.layer], y, eid  # (T,D), label, id


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate_binary(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            logits_all.append(logits.detach().cpu())
            labels_all.append(y.detach().cpu().long())

    logits_np = torch.cat(logits_all, dim=0).numpy()
    labels_np = torch.cat(labels_all, dim=0).numpy()
    if num_classes <= 1:
        return evaluate_binary(logits_np.reshape(-1), labels_np.reshape(-1))
    return evaluate_multiclass(logits_np, labels_np)


def default_run_name(
    probe_family: str,
    num_heads: int,
    layer: Optional[int],
    input_layers: Sequence[int],
) -> str:
    if layer is not None:
        layer_part = f"layer{layer}"
    else:
        layer_part = "layers" + "-".join(str(x) for x in input_layers)
    return f"{probe_family}_h{num_heads}_{layer_part}"


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
    family_cfg = PROBE_FAMILIES[probe_family]
    variant = family_cfg["variant"]
    if family_cfg["multilayer"]:
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


def find_latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    candidates = sorted(checkpoints_dir.glob("epoch_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def train(args: argparse.Namespace) -> int:
    model_dir = model_to_dir(args.model)
    train_dir = Path(args.activations_root) / model_dir / args.dataset / args.train_split
    val_dir = Path(args.activations_root) / model_dir / args.dataset / args.val_split

    family_cfg = PROBE_FAMILIES[args.probe_family]
    is_multilayer = family_cfg["multilayer"]

    input_layers = parse_input_layers(args.input_layers)
    if is_multilayer:
        if not input_layers:
            raise ValueError("--input_layers is required for multilayer families")
    else:
        if args.layer is None:
            raise ValueError("--layer is required for single-layer families")

    dataset_root = Path(args.output_root) / model_dir / "MultiAttentionProbes" / args.dataset
    run_name = args.run_name or default_run_name(args.probe_family, args.num_heads, args.layer, input_layers)
    run_dir = dataset_root / run_name

    manifest_path = run_dir / "meta" / "run_manifest.json"
    status_path = run_dir / "meta" / "status.json"
    progress_path = run_dir / "checkpoints" / "progress.json"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    best_model_path = run_dir / "results" / "best_model.pt"
    train_metrics_path = run_dir / "results" / "train_metrics.json"
    val_metrics_path = run_dir / "results" / "val_metrics.json"

    if status_path.exists():
        status = read_json(status_path, {})
        if status.get("status") == "completed" and not args.force_retrain:
            print(f"[skip] run already completed: {run_dir}")
            return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "created_at": utc_now(),
                "model": args.model,
                "dataset": args.dataset,
                "probe_family": args.probe_family,
                "variant": family_cfg["variant"],
                "multilayer": is_multilayer,
                "num_heads": args.num_heads,
                "num_classes": args.num_classes,
                "layer": args.layer,
                "input_layers": input_layers,
                "topk_tokens": args.topk_tokens,
                "rolling_window": args.rolling_window,
                "rolling_stride": args.rolling_stride,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "run_dir": str(run_dir),
                "data_train_dir": str(train_dir),
                "data_val_dir": str(val_dir),
            },
        )

    write_status(run_dir, "running", "starting or resuming training")

    train_base = CachedActivationsDataset(train_dir)
    val_base = CachedActivationsDataset(val_dir)

    if is_multilayer:
        train_ds = train_base
        val_ds = val_base
    else:
        if args.layer < 0 or args.layer >= train_base.num_layers:
            raise ValueError(f"Invalid layer {args.layer}; num_layers={train_base.num_layers}")
        train_ds = SingleLayerDataset(train_base, args.layer)
        val_ds = SingleLayerDataset(val_base, args.layer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        probe_family=args.probe_family,
        input_dim=train_base.hidden_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        input_layers=input_layers,
        topk_tokens=args.topk_tokens,
        rolling_window=args.rolling_window,
        rolling_stride=args.rolling_stride,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)

    if args.num_classes <= 1:
        pos_weight = None
        if args.positive_class_weight and args.positive_class_weight > 0:
            pos_weight = torch.tensor([args.positive_class_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        monitor_key = "auc"
    else:
        criterion = nn.CrossEntropyLoss()
        monitor_key = "f1_macro"

    progress = read_json(
        progress_path,
        {
            "current_epoch": 0,
            "best_metric": -1e18,
            "best_epoch": 0,
            "global_step": 0,
            "patience_counter": 0,
            "checkpoint_path": None,
            "history_train": [],
            "history_val": [],
        },
    )

    start_epoch = int(progress.get("current_epoch", 0))
    best_metric = float(progress.get("best_metric", -1e18))
    best_epoch = int(progress.get("best_epoch", 0))
    global_step = int(progress.get("global_step", 0))
    patience_counter = int(progress.get("patience_counter", 0))
    history_train = list(progress.get("history_train", []))
    history_val = list(progress.get("history_val", []))

    ckpt_path = progress.get("checkpoint_path")
    if not ckpt_path:
        latest = find_latest_checkpoint(checkpoints_dir)
        if latest is not None:
            ckpt_path = str(latest)

    if ckpt_path and Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        global_step = int(checkpoint.get("global_step", global_step))
        patience_counter = int(checkpoint.get("patience_counter", patience_counter))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = []
        epoch_logits = []
        epoch_labels = []

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{args.epochs}")
        for x, y, _ in pbar:
            x = x.to(device)
            if args.num_classes <= 1:
                y_t = y.to(device=device, dtype=torch.float32)
            else:
                y_t = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            if args.num_classes <= 1:
                loss = criterion(logits.squeeze(-1), y_t)
                epoch_logits.append(logits.detach().cpu().squeeze(-1))
                epoch_labels.append(y_t.detach().cpu().long())
            else:
                loss = criterion(logits, y_t)
                epoch_logits.append(logits.detach().cpu())
                epoch_labels.append(y_t.detach().cpu())

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_losses.append(float(loss.item()))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if args.num_classes <= 1:
            train_metrics = evaluate_binary(
                logits=torch.cat(epoch_logits).numpy(),
                labels=torch.cat(epoch_labels).numpy(),
            )
        else:
            train_metrics = evaluate_multiclass(
                logits=torch.cat(epoch_logits).numpy(),
                labels=torch.cat(epoch_labels).numpy(),
            )
        train_metrics["loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        train_metrics["epoch"] = epoch + 1

        val_metrics = run_eval(model, val_loader, device=device, num_classes=args.num_classes)
        val_metrics["epoch"] = epoch + 1

        history_train.append(train_metrics)
        history_val.append(val_metrics)

        metric_value = float(val_metrics[monitor_key])
        improved = metric_value > best_metric
        if improved:
            best_metric = metric_value
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        epoch_ckpt = checkpoints_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "global_step": global_step,
                "patience_counter": patience_counter,
            },
            epoch_ckpt,
        )

        progress = {
            "current_epoch": epoch + 1,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "global_step": global_step,
            "patience_counter": patience_counter,
            "checkpoint_path": str(epoch_ckpt),
            "history_train": history_train,
            "history_val": history_val,
            "updated_at": utc_now(),
        }
        write_json(progress_path, progress)
        write_json(train_metrics_path, {"history": history_train})
        write_json(val_metrics_path, {"history": history_val})

        if patience_counter >= args.patience:
            break

    write_status(run_dir, "completed", f"completed training at epoch {progress['current_epoch']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MultiAttention probe families")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--probe_family", type=str, choices=sorted(PROBE_FAMILIES), required=True)
    parser.add_argument("--activations_root", type=str, default="data/activations")
    parser.add_argument("--output_root", type=str, default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes")

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--force_retrain", action="store_true")

    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--input_layers", type=str, default="")

    parser.add_argument("--topk_tokens", type=int, default=4)
    parser.add_argument("--rolling_window", type=int, default=64)
    parser.add_argument("--rolling_stride", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--positive_class_weight", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    try:
        return train(args)
    except Exception as exc:
        # Best-effort status update; if run_dir was never created this simply no-ops.
        try:
            model_dir = model_to_dir(args.model)
            dataset_root = Path(args.output_root) / model_dir / "MultiAttentionProbes" / args.dataset
            run_name = args.run_name or default_run_name(
                args.probe_family,
                args.num_heads,
                args.layer,
                parse_input_layers(args.input_layers),
            )
            write_status(dataset_root / run_name, "failed", str(exc))
        finally:
            raise


if __name__ == "__main__":
    raise SystemExit(main())
