#!/usr/bin/env python3
"""
Train combined probes for one deception segment and evaluate on test datasets.

Stage A pipeline:
- Train probes over all layers and poolings on combined train split (multiple datasets)
- Validate on combined validation split
- Evaluate each trained (pooling, layer) on each test dataset
- Export matrices/heatmaps and resumable run metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
from combined_probe_pipeline import (  # noqa: E402
    LayerSampleDataset,
    SplitIndex,
    append_jsonl,
    choose_best_row,
    dataset_segment_name,
    default_run_id,
    ensure_dir,
    load_probe_state_with_compat,
    model_dir_name,
    read_json,
    train_layer_probe,
    utc_now_iso,
    write_json,
    evaluate_probe_on_dataset,
)

ALL_POOLINGS = ["mean", "max", "last", "attn"]
DEFAULT_TRAIN_DATASETS = [
    "Deception-AILiar",
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-Roleplaying",
]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"{self.formatTime(record)} | {record.levelname} | {record.getMessage()}"


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_layers(value: str, num_layers: int) -> List[int]:
    if not value:
        return list(range(num_layers))

    out: List[int] = []
    for part in parse_csv_list(value):
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))

    uniq = sorted(set(out))
    for layer in uniq:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} outside [0, {num_layers})")
    return uniq


def make_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    output_path: Path,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.3), 4.8))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt = "NA" if np.isnan(val) else format(val, fmt)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_matrix_csv(path: Path, row_labels: Sequence[str], col_labels: Sequence[str], matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pooling"] + list(col_labels))
        for i, r in enumerate(row_labels):
            row: List[str] = [r]
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                row.append("" if np.isnan(val) else str(val))
            writer.writerow(row)


def configure_logger(log_path: Path, verbose: bool) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("combined_segment_probes")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = JsonFormatter()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)

    return logger


def resolve_split_dir(activations_root: Path, model_dir: str, dataset: str, split: str) -> Path:
    return activations_root / model_dir / dataset / split


def save_progress(path: Path, payload: Dict) -> None:
    payload["updated_at"] = utc_now_iso()
    write_json(path, payload)


def dedupe_rows(rows: Sequence[Dict], key_fields: Sequence[str]) -> List[Dict]:
    seen: Dict[Tuple, Dict] = {}
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        seen[key] = r
    return list(seen.values())


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run combined deception segment probe pipeline")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--segment", type=str, choices=["completion", "full"], required=True)

    parser.add_argument("--train_datasets", type=str, default=",".join(DEFAULT_TRAIN_DATASETS))
    parser.add_argument("--test_only_datasets", type=str, default="Deception-InsiderTrading")

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--poolings", type=str, default="mean,max,last,attn")
    parser.add_argument("--layers", type=str, default="")

    parser.add_argument("--probe_output_root", type=str, required=True)
    parser.add_argument("--results_output_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name.strip() or f"combined_datasets_all_{args.segment}s"

    model_dir = model_dir_name(args.model)
    activations_root = Path(args.activations_root)
    probe_run_dir = Path(args.probe_output_root) / run_name
    result_run_dir = Path(args.results_output_root) / run_name

    meta_dir = result_run_dir / "meta"
    ckpt_dir = result_run_dir / "checkpoints"
    out_dir = result_run_dir / "results"
    log_dir = result_run_dir / "logs"
    ensure_dir(meta_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    logger = configure_logger(log_dir / "run.log", verbose=args.verbose)

    status_path = meta_dir / "status.json"
    progress_path = ckpt_dir / "progress.json"
    train_metrics_path = out_dir / "train_metrics.jsonl"
    eval_rows_path = out_dir / "layerwise_eval.jsonl"

    train_bases = parse_csv_list(args.train_datasets)
    test_only_bases = parse_csv_list(args.test_only_datasets)
    test_bases: List[str] = []
    for d in train_bases + test_only_bases:
        if d not in test_bases:
            test_bases.append(d)

    poolings = parse_csv_list(args.poolings)
    for p in poolings:
        if p not in ALL_POOLINGS:
            raise ValueError(f"Unsupported pooling: {p}")

    run_manifest = {
        "run_name": run_name,
        "created_at": utc_now_iso(),
        "model": args.model,
        "model_dir": model_dir,
        "segment": args.segment,
        "train_datasets": train_bases,
        "test_only_datasets": test_only_bases,
        "test_datasets": test_bases,
        "splits": {
            "train": args.train_split,
            "validation": args.val_split,
            "test": args.test_split,
        },
        "poolings": poolings,
        "probe_run_dir": str(probe_run_dir),
        "result_run_dir": str(result_run_dir),
    }
    write_json(meta_dir / "run_manifest.json", run_manifest)

    progress = read_json(progress_path, default={
        "train_done": [],
        "eval_done": [],
    })
    train_done = set(progress.get("train_done", []))
    eval_done = set(progress.get("eval_done", []))

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    logger.info(f"[start] run={run_name} segment={args.segment} model={args.model} device={device}")

    write_json(status_path, {
        "state": "running",
        "stage": "index",
        "updated_at": utc_now_iso(),
        "message": "building split indices",
    })

    train_dirs = [
        resolve_split_dir(
            activations_root,
            model_dir,
            dataset_segment_name(d, args.segment),
            args.train_split,
        )
        for d in train_bases
    ]
    val_dirs = [
        resolve_split_dir(
            activations_root,
            model_dir,
            dataset_segment_name(d, args.segment),
            args.val_split,
        )
        for d in train_bases
    ]
    test_dirs = {
        d: resolve_split_dir(
            activations_root,
            model_dir,
            dataset_segment_name(d, args.segment),
            args.test_split,
        )
        for d in test_bases
    }

    for p in train_dirs + val_dirs + list(test_dirs.values()):
        if not p.exists():
            raise FileNotFoundError(f"Missing split dir: {p}")

    logger.info("[stage 1/4] indexing train/val")
    train_index = SplitIndex(train_dirs)
    val_index = SplitIndex(val_dirs)
    if train_index.d_model != val_index.d_model or train_index.num_layers != val_index.num_layers:
        raise ValueError("Train/val tensor shape mismatch")

    num_layers = int(train_index.num_layers or 0)
    d_model = int(train_index.d_model or 0)
    layers = parse_layers(args.layers, num_layers=num_layers)

    logger.info(
        f"[info] train_n={len(train_index.entries)} val_n={len(val_index.entries)} "
        f"layers={len(layers)} d_model={d_model}"
    )

    write_json(status_path, {
        "state": "running",
        "stage": "train",
        "updated_at": utc_now_iso(),
        "message": "training probes",
    })

    # Stage train: pooling x layer
    for pooling in poolings:
        pool_dir = probe_run_dir / pooling
        ensure_dir(pool_dir)

        for layer in layers:
            unit = f"{pooling}:L{layer}"
            probe_path = pool_dir / f"probe_layer_{layer}.pt"
            if args.resume and unit in train_done and probe_path.exists():
                continue

            logger.info(f"[train] {unit}")
            train_ds = LayerSampleDataset(train_index, layer=layer)
            val_ds = LayerSampleDataset(val_index, layer=layer)

            model, val_metrics = train_layer_probe(
                pooling=pooling,
                layer=layer,
                d_model=d_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )

            ensure_dir(probe_path.parent)
            torch.save(model.state_dict(), str(probe_path))

            row = {
                "timestamp": utc_now_iso(),
                "segment": args.segment,
                "pooling": pooling,
                "layer": int(layer),
                "val_auc": float(val_metrics.get("auc", 0.5)),
                "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
                "val_f1": float(val_metrics.get("f1", 0.0)),
                "best_epoch": int(val_metrics.get("best_epoch", 0)),
                "probe_path": str(probe_path),
            }
            append_jsonl(train_metrics_path, row)

            train_done.add(unit)
            progress["train_done"] = sorted(train_done)
            progress["eval_done"] = sorted(eval_done)
            save_progress(progress_path, progress)

    logger.info("[stage 2/4] indexing test datasets")
    test_indices: Dict[str, SplitIndex] = {}
    for base_name, split_dir in test_dirs.items():
        idx = SplitIndex([split_dir])
        if idx.d_model != d_model or idx.num_layers != num_layers:
            raise ValueError(f"Test shape mismatch for {base_name}: {split_dir}")
        test_indices[base_name] = idx

    write_json(status_path, {
        "state": "running",
        "stage": "evaluate",
        "updated_at": utc_now_iso(),
        "message": "evaluating all pooling/layer on test datasets",
    })

    for test_base, test_index in test_indices.items():
        logger.info(f"[eval] dataset={test_base} n={len(test_index.entries)}")

        for pooling in poolings:
            for layer in layers:
                unit = f"{test_base}:{pooling}:L{layer}"
                if args.resume and unit in eval_done:
                    continue

                probe_path = probe_run_dir / pooling / f"probe_layer_{layer}.pt"
                if not probe_path.exists():
                    raise FileNotFoundError(f"Missing trained probe: {probe_path}")

                model = load_probe_state_with_compat(
                    probe_path=probe_path,
                    pooling=pooling,
                    input_dim=d_model,
                    device=device,
                )
                test_ds = LayerSampleDataset(test_index, layer=layer)
                m = evaluate_probe_on_dataset(
                    model=model,
                    dataset=test_ds,
                    device=device,
                    batch_size=args.batch_size,
                )

                row = {
                    "timestamp": utc_now_iso(),
                    "segment": args.segment,
                    "test_dataset": dataset_segment_name(test_base, args.segment),
                    "test_dataset_base": test_base,
                    "pooling": pooling,
                    "layer": int(layer),
                    "auc": float(m["auc"]),
                    "accuracy": float(m["accuracy"]),
                    "f1": float(m["f1"]),
                    "count": int(m["count"]),
                    "probe_path": str(probe_path),
                }
                append_jsonl(eval_rows_path, row)

                eval_done.add(unit)
                progress["train_done"] = sorted(train_done)
                progress["eval_done"] = sorted(eval_done)
                save_progress(progress_path, progress)

    # Final aggregation
    logger.info("[stage 3/4] aggregating outputs")
    eval_rows = dedupe_rows(read_jsonl(eval_rows_path), key_fields=["segment", "test_dataset_base", "pooling", "layer"])
    eval_rows.sort(key=lambda r: (r["test_dataset_base"], r["pooling"], int(r["layer"])))

    layerwise_csv = out_dir / "layerwise_eval.csv"
    with layerwise_csv.open("w", newline="", encoding="utf-8") as f:
        if eval_rows:
            fields = [
                "segment", "test_dataset", "test_dataset_base", "pooling", "layer",
                "auc", "accuracy", "f1", "count", "probe_path", "timestamp",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in eval_rows:
                writer.writerow({k: r.get(k) for k in fields})

    best_rows: List[Dict] = []
    for test_base in test_bases:
        for pooling in poolings:
            candidates = [r for r in eval_rows if r["test_dataset_base"] == test_base and r["pooling"] == pooling]
            if not candidates:
                continue
            best = choose_best_row(candidates)
            best_rows.append({
                "segment": args.segment,
                "test_dataset": dataset_segment_name(test_base, args.segment),
                "test_dataset_base": test_base,
                "pooling": pooling,
                "best_layer": int(best["layer"]),
                "best_auc": float(best["auc"]),
                "best_accuracy": float(best["accuracy"]),
                "best_f1": float(best["f1"]),
            })

    best_rows.sort(key=lambda r: (r["test_dataset_base"], r["pooling"]))
    best_csv = out_dir / "best_by_test_dataset.csv"
    with best_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "segment", "test_dataset", "test_dataset_base", "pooling",
            "best_layer", "best_auc", "best_accuracy", "best_f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(best_rows)

    row_labels = poolings
    col_labels = [dataset_segment_name(x, args.segment) for x in test_bases]
    auc_mat = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float64)
    layer_mat = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float64)

    for i, pooling in enumerate(row_labels):
        for j, test_base in enumerate(test_bases):
            rec = next((r for r in best_rows if r["pooling"] == pooling and r["test_dataset_base"] == test_base), None)
            if rec is None:
                continue
            auc_mat[i, j] = float(rec["best_auc"])
            layer_mat[i, j] = float(rec["best_layer"])

    write_matrix_csv(out_dir / "matrix_pooling_vs_test_auc.csv", row_labels, col_labels, auc_mat)
    write_matrix_csv(out_dir / "matrix_pooling_vs_test_best_layer.csv", row_labels, col_labels, layer_mat)

    make_heatmap(
        matrix=auc_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{args.segment}: Best AUC by Pooling x Test Dataset",
        output_path=out_dir / "heatmap_pooling_vs_test_auc.png",
        fmt=".3f",
        vmin=0.0,
        vmax=1.0,
    )
    make_heatmap(
        matrix=layer_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{args.segment}: Best Layer by Pooling x Test Dataset",
        output_path=out_dir / "heatmap_pooling_vs_test_best_layer.png",
        fmt=".0f",
        vmin=0.0,
        vmax=float(max(layers)) if layers else None,
    )

    summary = {
        "run_name": run_name,
        "segment": args.segment,
        "model": args.model,
        "device": str(device),
        "train_datasets": train_bases,
        "test_datasets": [dataset_segment_name(x, args.segment) for x in test_bases],
        "test_only_datasets": [dataset_segment_name(x, args.segment) for x in test_only_bases],
        "poolings": poolings,
        "layers": layers,
        "n_train_samples": len(train_index.entries),
        "n_val_samples": len(val_index.entries),
        "n_test_samples": {
            dataset_segment_name(k, args.segment): len(v.entries)
            for k, v in test_indices.items()
        },
        "probe_run_dir": str(probe_run_dir),
        "results_dir": str(out_dir),
        "artifacts": {
            "layerwise_eval_csv": str(layerwise_csv),
            "best_by_test_dataset_csv": str(best_csv),
            "matrix_auc_csv": str(out_dir / "matrix_pooling_vs_test_auc.csv"),
            "matrix_best_layer_csv": str(out_dir / "matrix_pooling_vs_test_best_layer.csv"),
            "heatmap_auc_png": str(out_dir / "heatmap_pooling_vs_test_auc.png"),
            "heatmap_best_layer_png": str(out_dir / "heatmap_pooling_vs_test_best_layer.png"),
        },
        "updated_at": utc_now_iso(),
    }
    write_json(out_dir / "summary.json", summary)

    write_json(status_path, {
        "state": "completed",
        "stage": "done",
        "updated_at": utc_now_iso(),
        "message": "completed successfully",
    })
    logger.info(f"[done] results -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
