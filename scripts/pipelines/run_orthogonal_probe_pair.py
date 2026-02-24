#!/usr/bin/env python3
"""
Run orthogonal probe decomposition for an explicit source/target pair.

This script is intentionally CSV-independent:
  - Provide train/val/target split directories directly.
  - Projection-only orthogonal decomposition (leakage-safe).
  - Fixed K probes per selected layer(s).
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
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))
sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))

from actprobe.probes.models import LayerProbe
from orthogonal_probe import (
    evaluate_probe_with_projection,
    max_cos_to_previous,
    train_probe_with_projection,
    update_q_basis,
)


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


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_jsonl_rows(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_logger(log_path: Path, verbose: bool) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("orthogonal_probe_pair")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

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
    write_json(
        meta_dir / "status.json",
        {
            "run_id": run_id,
            "state": state,
            "current_step": current_step,
            "last_updated_utc": utc_now_iso(),
        },
    )


def git_commit_or_unknown() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def model_to_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def normalize_pooling(value: str) -> str:
    v = str(value).strip().lower()
    mapping = {
        "mean": "mean",
        "max": "max",
        "last": "last",
        "attn": "attn",
        "attention": "attn",
        "none": "none",
    }
    if v not in mapping:
        raise ValueError(f"Unsupported pooling value: {value}")
    return mapping[v]


def parse_layers(value: str) -> List[int]:
    out: List[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No valid layers provided.")
    return sorted(set(out))


def load_label_map(split_path: Path) -> Dict[str, int]:
    manifest = split_path / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {split_path}")
    label_map: Dict[str, int] = {}
    with manifest.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("id", ""))
            if not sid:
                continue
            label_map[sid] = int(row.get("label", -1))
    return label_map


def load_split_layer_tensors(
    split_path: Path,
    layer: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
    label_map = load_label_map(split_path)
    shards = sorted(glob.glob(str(split_path / "shard_*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.safetensors found in {split_path}")

    xs: List[torch.Tensor] = []
    ys: List[int] = []
    ids: List[str] = []
    input_type: Optional[str] = None

    for shard_path in shards:
        shard = load_file(shard_path)
        for sid, tensor in shard.items():
            label = label_map.get(str(sid), -1)
            if label not in (0, 1):
                continue

            if tensor.dim() == 3:
                if layer >= int(tensor.shape[0]):
                    raise ValueError(
                        f"Layer {layer} out of range for sample {sid}; tensor has {int(tensor.shape[0])} layers."
                    )
                x = tensor[layer, :, :].detach().cpu().float()
                cur_type = "token"
            elif tensor.dim() == 2:
                if layer >= int(tensor.shape[0]):
                    raise ValueError(
                        f"Layer {layer} out of range for sample {sid}; tensor has {int(tensor.shape[0])} layers."
                    )
                x = tensor[layer, :].detach().cpu().float()
                cur_type = "final"
            else:
                raise ValueError(f"Unexpected tensor shape {tuple(tensor.shape)} for sample {sid}")

            if input_type is None:
                input_type = cur_type
            elif input_type != cur_type:
                raise ValueError(f"Mixed input types found in split {split_path}: {input_type} vs {cur_type}")

            xs.append(x)
            ys.append(int(label))
            ids.append(str(sid))

    if not xs:
        raise RuntimeError(f"No labeled samples found in {split_path}")

    x_tensor = torch.stack(xs, dim=0).float()
    y_tensor = torch.tensor(ys, dtype=torch.float32)
    return x_tensor, y_tensor, ids, (input_type or "unknown")


def extract_probe_vector_and_bias(model: LayerProbe) -> Tuple[torch.Tensor, float]:
    w = model.classifier.weight.detach().reshape(-1).clone()
    b = float(model.classifier.bias.detach().reshape(-1)[0].item())
    return w, b


def probe_path(probes_dir: Path, k: int) -> Path:
    return probes_dir / f"probe_k{k:03d}.pt"


def run_layer(
    *,
    args: argparse.Namespace,
    layer: int,
    run_dir: Path,
    probes_dir: Path,
    logger: logging.Logger,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    pooling: str,
) -> None:
    meta_dir = run_dir / "meta"
    ckpt_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    ensure_dir(meta_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(results_dir)
    ensure_dir(probes_dir)

    progress_path = ckpt_dir / "progress.json"
    per_probe_jsonl = results_dir / "per_probe.jsonl"
    per_probe_csv = results_dir / "per_probe.csv"

    if args.resume and progress_path.exists():
        progress = read_json(progress_path)
    else:
        progress = {
            "run_id": args.run_id,
            "layer": layer,
            "completed_k": [],
            "updated_at_utc": utc_now_iso(),
        }
        write_json(progress_path, progress)

    completed_k = {int(v) for v in progress.get("completed_k", [])}
    existing_rows = read_jsonl(per_probe_jsonl)
    existing_by_k = {int(r["k"]): r for r in existing_rows if "k" in r}
    if bool(args.eval_only):
        # Recompute eval rows from saved probes; do not carry stale metrics.
        existing_by_k = {}

    input_dim = int(x_train.shape[-1])
    model_pooling = pooling
    if x_train.dim() == 2:
        model_pooling = "none"
        if pooling != "none":
            logger.info("Layer %d input is final-token; overriding pooling %s -> none", layer, pooling)

    logger.info(
        "Layer %d | input_shape(train)=%s | pooling=%s | completed=%d/%d",
        layer,
        tuple(x_train.shape),
        model_pooling,
        len(completed_k),
        args.k_probes,
    )

    q_basis: Optional[torch.Tensor] = None
    prev_w: List[torch.Tensor] = []
    if not bool(args.eval_only):
        for k in sorted(completed_k):
            p = probe_path(probes_dir, k)
            if not p.exists():
                raise FileNotFoundError(f"Progress says k={k} completed but probe missing: {p}")
            m = LayerProbe(input_dim=input_dim, pooling_type=model_pooling).to(args.device)
            m.load_state_dict(torch.load(p, map_location=args.device, weights_only=True))
            w, _ = extract_probe_vector_and_bias(m)
            w = w.to(args.device)
            q_basis, _ = update_q_basis(q_basis, w)
            prev_w.append(w.detach().cpu())

    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=0)
    target_loader = DataLoader(
        TensorDataset(x_target, y_target), batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    if bool(args.eval_only):
        logger.info("Eval-only mode: loading saved probes and recomputing metrics; no training.")
        for k in range(1, args.k_probes + 1):
            p = probe_path(probes_dir, k)
            if not p.exists():
                logger.warning("Skipping k=%d: probe not found at %s", k, p)
                continue

            set_status(meta_dir, args.run_id, state="running", current_step=f"layer_{layer}_eval_k_{k}")
            model = LayerProbe(input_dim=input_dim, pooling_type=model_pooling).to(args.device)
            model.load_state_dict(torch.load(p, map_location=args.device, weights_only=True))

            val_metrics_resid = evaluate_probe_with_projection(
                model=model,
                loader=val_loader,
                device=args.device,
                q_basis=q_basis,
            )
            target_metrics_resid = evaluate_probe_with_projection(
                model=model,
                loader=target_loader,
                device=args.device,
                q_basis=q_basis,
            )
            val_metrics_raw = evaluate_probe_with_projection(
                model=model,
                loader=val_loader,
                device=args.device,
                q_basis=None,
            )
            target_metrics_raw = evaluate_probe_with_projection(
                model=model,
                loader=target_loader,
                device=args.device,
                q_basis=None,
            )

            w, b = extract_probe_vector_and_bias(model)
            w = w.to(args.device)
            w_norm = float(torch.linalg.vector_norm(w).item())
            max_cos = max_cos_to_previous(w.detach().cpu(), prev_w)
            bias_over_norm = float(b / (w_norm + 1e-12))

            row = {
                "timestamp_utc": utc_now_iso(),
                "run_id": args.run_id,
                "layer": int(layer),
                "k": int(k),
                "method": "projection",
                "pooling": model_pooling,
                "eval_only": True,
                "auc_A_val": float(val_metrics_resid["auc"]),
                "acc_A_val": float(val_metrics_resid["accuracy"]),
                "auc_A_val_resid": float(val_metrics_resid["auc"]),
                "acc_A_val_resid": float(val_metrics_resid["accuracy"]),
                "auc_A_val_raw": float(val_metrics_raw["auc"]),
                "acc_A_val_raw": float(val_metrics_raw["accuracy"]),
                "best_epoch": None,
                "auc_B_target": float(target_metrics_resid["auc"]),
                "acc_B_target": float(target_metrics_resid["accuracy"]),
                "auc_B_target_resid": float(target_metrics_resid["auc"]),
                "acc_B_target_resid": float(target_metrics_resid["accuracy"]),
                "auc_B_target_raw": float(target_metrics_raw["auc"]),
                "acc_B_target_raw": float(target_metrics_raw["accuracy"]),
                "delta_auc_target_raw_minus_resid": float(target_metrics_raw["auc"] - target_metrics_resid["auc"]),
                "w_norm": w_norm,
                "bias": float(b),
                "bias_over_w_norm": bias_over_norm,
                "max_cos_prev": float(max_cos),
                "probe_path": str(p),
            }
            existing_by_k[k] = row

            q_basis, residual_norm = update_q_basis(q_basis, w)
            if residual_norm < 1e-8:
                logger.warning("Layer %d k=%d residual norm near zero after orthogonalization.", layer, k)
            prev_w.append(w.detach().cpu())

    for k in range(1, args.k_probes + 1):
        if bool(args.eval_only):
            break
        if k in completed_k:
            continue

        set_status(meta_dir, args.run_id, state="running", current_step=f"layer_{layer}_k_{k}")
        logger.info("Training layer=%d k=%d ...", layer, k)

        model = LayerProbe(input_dim=input_dim, pooling_type=model_pooling).to(args.device)
        trained_model, val_metrics = train_probe_with_projection(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            device=args.device,
            q_basis=q_basis,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )

        target_metrics_resid = evaluate_probe_with_projection(
            model=trained_model,
            loader=target_loader,
            device=args.device,
            q_basis=q_basis,
        )
        target_metrics_raw = evaluate_probe_with_projection(
            model=trained_model,
            loader=target_loader,
            device=args.device,
            q_basis=None,
        )

        w, b = extract_probe_vector_and_bias(trained_model)
        w = w.to(args.device)
        w_norm = float(torch.linalg.vector_norm(w).item())
        max_cos = max_cos_to_previous(w.detach().cpu(), prev_w)
        bias_over_norm = float(b / (w_norm + 1e-12))

        p_path = probe_path(probes_dir, k)
        torch.save(trained_model.state_dict(), p_path)

        row = {
            "timestamp_utc": utc_now_iso(),
            "run_id": args.run_id,
            "layer": int(layer),
            "k": int(k),
            "method": "projection",
            "pooling": model_pooling,
            "auc_A_val": float(val_metrics["auc"]),
            "acc_A_val": float(val_metrics["accuracy"]),
            "best_epoch": int(val_metrics["best_epoch"]),
            # Backward-compatible keys (residual/projection-space metrics).
            "auc_B_target": float(target_metrics_resid["auc"]),
            "acc_B_target": float(target_metrics_resid["accuracy"]),
            "auc_B_target_resid": float(target_metrics_resid["auc"]),
            "acc_B_target_resid": float(target_metrics_resid["accuracy"]),
            # Raw OOD metrics on original activations (no projection removal).
            "auc_B_target_raw": float(target_metrics_raw["auc"]),
            "acc_B_target_raw": float(target_metrics_raw["accuracy"]),
            "delta_auc_target_raw_minus_resid": float(target_metrics_raw["auc"] - target_metrics_resid["auc"]),
            "w_norm": w_norm,
            "bias": float(b),
            "bias_over_w_norm": bias_over_norm,
            "max_cos_prev": float(max_cos),
            "probe_path": str(p_path),
        }
        append_jsonl(per_probe_jsonl, row)
        existing_by_k[k] = row

        q_basis, residual_norm = update_q_basis(q_basis, w)
        if residual_norm < 1e-8:
            logger.warning("Layer %d k=%d residual norm near zero after orthogonalization.", layer, k)
        prev_w.append(w.detach().cpu())

        completed_k.add(k)
        progress["completed_k"] = sorted(completed_k)
        progress["updated_at_utc"] = utc_now_iso()
        write_json(progress_path, progress)

    rows = [existing_by_k[k] for k in sorted(existing_by_k.keys())]
    write_jsonl_rows(per_probe_jsonl, rows)
    write_csv(per_probe_csv, rows)

    if rows:
        auc_a = [float(r["auc_A_val"]) for r in rows]
        auc_b_resid = [float(r.get("auc_B_target_resid", r.get("auc_B_target", 0.5))) for r in rows]
        auc_b_raw = [float(r.get("auc_B_target_raw", r.get("auc_B_target", 0.5))) for r in rows]
        summary = {
            "run_id": args.run_id,
            "layer": int(layer),
            "method": "projection",
            "k_completed": len(rows),
            "auc_A_val_curve": auc_a,
            # Backward-compatible key (residual/projection-space target AUC).
            "auc_B_target_curve": auc_b_resid,
            "auc_B_target_curve_resid": auc_b_resid,
            "auc_B_target_curve_raw": auc_b_raw,
            "transfer_gap_curve": [a - b for a, b in zip(auc_a, auc_b_resid)],
            "transfer_gap_curve_raw": [a - b for a, b in zip(auc_a, auc_b_raw)],
            "best_A_idx": int(np.argmax(np.asarray(auc_a)) + 1),
            "best_B_idx": int(np.argmax(np.asarray(auc_b_resid)) + 1),
            "best_B_idx_resid": int(np.argmax(np.asarray(auc_b_resid)) + 1),
            "best_B_idx_raw": int(np.argmax(np.asarray(auc_b_raw)) + 1),
            "best_A_auc": float(np.max(np.asarray(auc_a))),
            "best_B_auc": float(np.max(np.asarray(auc_b_resid))),
            "best_B_auc_resid": float(np.max(np.asarray(auc_b_resid))),
            "best_B_auc_raw": float(np.max(np.asarray(auc_b_raw))),
            "updated_at_utc": utc_now_iso(),
        }
        write_json(results_dir / "summary.json", summary)

    set_status(meta_dir, args.run_id, state="completed", current_step=None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run orthogonal probes for explicit source/target pair")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--source_name", type=str, default=None, help="Name used in output slug")
    parser.add_argument("--target_name", type=str, default=None, help="Name used in output slug")
    parser.add_argument("--layers", type=str, required=True, help="Comma-separated list")
    parser.add_argument("--k_probes", type=int, default=50)
    parser.add_argument("--pooling", type=str, required=True)
    parser.add_argument("--method", type=str, default="projection", choices=["projection"])
    parser.add_argument("--output_root", type=str, default="results/orthogonal_probe_decomp")
    parser.add_argument(
        "--probes_output_root",
        type=str,
        default=None,
        help="Optional separate root for probe checkpoints. Defaults to --output_root.",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only evaluate saved probes (no training). Requires probe checkpoints in probes_output_root path.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device(args.device)
    args.run_id = args.run_id or make_run_id()

    pooling = normalize_pooling(args.pooling)
    layers = parse_layers(args.layers)

    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    target_dir = Path(args.target_dir).resolve()

    if not train_dir.exists():
        raise FileNotFoundError(f"Train split dir not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation split dir not found: {val_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target split dir not found: {target_dir}")

    source_name = args.source_name or train_dir.parent.name
    target_name = args.target_name or target_dir.parent.name
    pair_slug = f"{source_name}__to__{target_name}"
    model_dir = model_to_dir(args.model)
    output_root = Path(args.output_root).resolve()
    probes_output_root = Path(args.probes_output_root).resolve() if args.probes_output_root else output_root

    for layer in layers:
        run_dir = output_root / model_dir / pair_slug / pooling / f"layer_{layer}" / args.run_id
        probes_dir = probes_output_root / model_dir / pair_slug / pooling / f"layer_{layer}" / args.run_id
        logger = init_logger(run_dir / "logs" / "run.log", verbose=bool(args.verbose))
        set_status(run_dir / "meta", args.run_id, state="running", current_step="load_data")

        logger.info("Run id: %s", args.run_id)
        logger.info("Method: projection")
        logger.info("Source: %s", source_name)
        logger.info("Target: %s", target_name)
        logger.info("Layer: %d", layer)
        logger.info("Pooling: %s", pooling)
        logger.info("Train dir: %s", train_dir)
        logger.info("Val dir: %s", val_dir)
        logger.info("Target dir: %s", target_dir)
        logger.info("Results root: %s", output_root)
        logger.info("Probes root: %s", probes_output_root)
        logger.info("Probes dir: %s", probes_dir)

        x_train, y_train, ids_train, train_type = load_split_layer_tensors(train_dir, layer=layer)
        x_val, y_val, ids_val, val_type = load_split_layer_tensors(val_dir, layer=layer)
        x_target, y_target, ids_target, target_type = load_split_layer_tensors(target_dir, layer=layer)

        if not (train_type == val_type == target_type):
            raise ValueError(
                f"Input type mismatch across splits for layer {layer}: "
                f"train={train_type}, val={val_type}, target={target_type}"
            )

        write_json(
            run_dir / "meta" / "run_manifest.json",
            {
                "run_id": args.run_id,
                "timestamp_utc": utc_now_iso(),
                "git_commit": git_commit_or_unknown(),
                "model": args.model,
                "model_dir": model_dir,
                "method": "projection",
                "k_probes": int(args.k_probes),
                "layer": int(layer),
                "pooling": pooling,
                "source_name": source_name,
                "target_name": target_name,
                "train_dir": str(train_dir),
                "val_dir": str(val_dir),
                "target_dir": str(target_dir),
                "output_root": str(output_root),
                "probes_output_root": str(probes_output_root),
                "probes_dir": str(probes_dir),
                "counts": {
                    "train": int(y_train.numel()),
                    "val": int(y_val.numel()),
                    "target": int(y_target.numel()),
                },
                "sample_ids": {
                    "train": ids_train,
                    "val": ids_val,
                    "target": ids_target,
                },
                "input_type": train_type,
                "input_shape_train": list(x_train.shape),
                "input_shape_val": list(x_val.shape),
                "input_shape_target": list(x_target.shape),
                "hyperparams": {
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "epochs": int(args.epochs),
                    "patience": int(args.patience),
                    "batch_size": int(args.batch_size),
                    "seed": int(args.seed),
                },
                "eval_only": bool(args.eval_only),
            },
        )

        try:
            run_layer(
                args=args,
                layer=layer,
                run_dir=run_dir,
                probes_dir=probes_dir,
                logger=logger,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_target=x_target,
                y_target=y_target,
                pooling=pooling,
            )
            logger.info("Completed layer=%d run.", layer)
        except Exception:
            set_status(run_dir / "meta", args.run_id, state="failed", current_step=f"layer_{layer}")
            logger.exception("Run failed for layer=%d", layer)
            raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
