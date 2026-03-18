#!/usr/bin/env python3
"""
Evaluate saved RP-learned rank-1 directions on InsiderTrading against fixed baselines.

This script:
1. Loads saved learned directions from an RP source-sweep run.
2. Loads the fixed `mean@L15` baseline AUROCs for InsiderTrading from an existing
   principled pairwise score matrix.
3. Applies each saved direction to InsiderTrading activations.
4. Scores fixed single-source probes on the transformed activations.
5. Compares transformed AUROC to the reused fixed baseline AUROC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
from combined_probe_pipeline import (  # noqa: E402
    default_run_id,
    ensure_dir,
    model_dir_name,
    read_json,
    utc_now_iso,
    write_json,
)


SUPPORTED_POOLINGS = ["mean", "max", "last"]
DEFAULT_DIRECTION_SOURCES = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
]
SHORT_LABELS = {
    "Deception-ConvincingGame": "CG",
    "Deception-HarmPressureChoice": "HPC",
    "Deception-InstructedDeception": "ID",
    "Deception-Mask": "Mask",
    "Deception-AILiar": "AL",
}


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def dataset_segment_name(dataset_base: str, segment: str) -> str:
    if dataset_base.endswith("-completion") or dataset_base.endswith("-full"):
        return dataset_base
    return f"{dataset_base}-{segment}"


def slugify_dataset_base(dataset_base: str) -> str:
    return dataset_base.replace("Deception-", "").replace("_", "-").lower()


def get_git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        out = proc.stdout.strip()
        return out or None
    except Exception:
        return None


def load_label_map(activations_dir: Path) -> Dict[str, int]:
    manifest_path = activations_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl in {activations_dir}")
    label_map: Dict[str, int] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label_map[str(row["id"])] = int(row.get("label", -1))
    return label_map


def pool_tokens(x_layer: torch.Tensor, pooling: str) -> np.ndarray:
    if pooling == "mean":
        pooled = x_layer.mean(dim=0)
    elif pooling == "max":
        pooled = x_layer.max(dim=0).values
    elif pooling == "last":
        pooled = x_layer[-1, :]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def load_pooled_split(
    activations_dir: Path,
    layer: int,
    pooling: str,
    desc: str,
    use_tqdm: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    label_map = load_label_map(activations_dir)
    shard_paths = sorted(activations_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

    features: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[str] = []

    iterator: Iterable[Path] = shard_paths
    if use_tqdm:
        iterator = tqdm(shard_paths, desc=f"Loading {desc} shards")

    for shard_path in iterator:
        shard = load_file(str(shard_path))
        for sid, tensor in shard.items():
            label = label_map.get(sid, -1)
            if label == -1:
                continue
            if tensor.dim() != 3:
                raise ValueError(
                    f"Expected tensor shape (L,T,D), got {tuple(tensor.shape)} for sample {sid}"
                )
            if layer >= int(tensor.shape[0]):
                raise ValueError(
                    f"Layer index {layer} out of range for sample {sid} with shape {tuple(tensor.shape)}"
                )
            x_layer = tensor[layer, :, :]
            features.append(pool_tokens(x_layer, pooling))
            labels.append(label)
            sample_ids.append(str(sid))

    if not labels:
        raise ValueError(f"No labeled examples loaded from {activations_dir}")

    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, sample_ids


def extract_classifier_params(state_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, float]:
    weight_key = None
    bias_key = None
    for k in state_dict:
        if k.endswith("classifier.weight"):
            weight_key = k
        if k.endswith("classifier.bias"):
            bias_key = k
    if weight_key is None or bias_key is None:
        raise KeyError("Could not find classifier.weight/classifier.bias in probe state dict")
    w = state_dict[weight_key].detach().cpu().numpy().reshape(-1).astype(np.float32)
    b = float(state_dict[bias_key].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def load_linear_probe(probe_path: Path) -> Tuple[np.ndarray, float]:
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing probe checkpoint: {probe_path}")
    state = torch.load(str(probe_path), map_location="cpu")
    return extract_classifier_params(state)


def single_probe_path(
    single_probes_root: Path,
    model_dir: str,
    dataset_base: str,
    segment: str,
    pooling: str,
    layer: int,
) -> Path:
    ds = dataset_segment_name(dataset_base, segment)
    return single_probes_root / model_dir / f"{dataset_base}_slices" / ds / pooling / f"probe_layer_{layer}.pt"


def normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        raise ValueError("Direction norm is zero")
    return (v / norm).astype(np.float32)


def apply_rank1_transform_np(x: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    proj = (x @ direction).astype(np.float32, copy=False)
    transformed = x - float(alpha) * proj[:, None] * direction[None, :]
    return transformed.astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = sigmoid_np(logits)
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "count": int(y_true.shape[0]),
        "logit_mean": float(np.mean(logits)),
        "logit_std": float(np.std(logits)),
    }


def evaluate_probe_logits(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, float]:
    logits = x @ w + b
    return compute_metrics(y, logits)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_matrix_csv(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Malformed matrix CSV: {path}")
    col_labels = [c.strip() for c in rows[0][1:]]
    row_labels: List[str] = []
    matrix_rows: List[List[float]] = []
    for row in rows[1:]:
        if not row:
            continue
        row_labels.append(row[0].strip())
        vals = [np.nan if str(cell).strip() == "" else float(cell) for cell in row[1:]]
        matrix_rows.append(vals)
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    return row_labels, col_labels, matrix


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    output_path: Path,
    *,
    fmt: str = ".3f",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "YlGnBu",
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(8.5, len(col_labels) * 1.0), max(4.8, len(row_labels) * 0.72)))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_matched_delta_bar(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    labels = [SHORT_LABELS.get(str(r["direction_source_dataset"]), str(r["direction_source_dataset"])) for r in rows]
    values = [float(r["matched_delta_auc"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x, values, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta AUROC vs fixed IT baseline")
    ax.set_title("Matched-probe delta on InsiderTrading")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_tree_subset(src_root: Path, dst_root: Path, subdirs: Sequence[str]) -> None:
    ensure_dir(dst_root)
    for subdir in subdirs:
        src = src_root / subdir
        if not src.exists():
            continue
        dst = dst_root / subdir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def mirror_png(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved RP-learned directions on InsiderTrading against fixed baselines.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--single_probes_root", type=str, default="data/probes")
    parser.add_argument("--source_sweep_root", type=str, required=True)
    parser.add_argument("--fixed_results_dir", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, default="Deception-InsiderTrading")
    parser.add_argument("--segment", type=str, choices=["completion", "full"], default="completion")
    parser.add_argument("--pooling", type=str, choices=SUPPORTED_POOLINGS, default="mean")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--direction_source_datasets", type=str, default=",".join(DEFAULT_DIRECTION_SOURCES))
    parser.add_argument("--evaluation_single_datasets", type=str, default="")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--mirror_results_root", type=str, default="")
    parser.add_argument("--gallery_root", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    direction_source_datasets = parse_csv_list(args.direction_source_datasets)
    if not direction_source_datasets:
        raise ValueError("At least one direction source dataset is required")
    evaluation_single_datasets = parse_csv_list(args.evaluation_single_datasets) or list(direction_source_datasets)
    if not evaluation_single_datasets:
        raise ValueError("At least one evaluation probe dataset is required")

    model_dir = model_dir_name(args.model)
    target_segment = dataset_segment_name(args.target_dataset, args.segment)
    run_id = args.run_id.strip() or f"{default_run_id()}-it-rank1-transfer-v1"

    artifact_root = Path(args.artifact_root)
    run_root = (
        artifact_root
        / "runs"
        / "transferred_rank1_direction_eval"
        / model_dir
        / target_segment
        / "from-rp-source-sweep-vs-fixed-baseline"
        / f"{args.pooling}_l{args.layer}"
        / run_id
    )
    inputs_dir = run_root / "inputs"
    results_dir = run_root / "results"
    plots_dir = results_dir / "plots"
    meta_dir = run_root / "meta"
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    ensure_dir(inputs_dir)
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    ensure_dir(meta_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)
    log_path = logs_dir / "run.log"

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "transferred_rank1_direction_eval"
    mirror_root = Path(args.mirror_results_root) if args.mirror_results_root else default_mirror_root
    external_run_root = mirror_root / target_segment / f"{args.pooling}_l{args.layer}" / run_id

    default_gallery_root = (
        Path("results")
        / "ood_evaluation"
        / model_dir
        / "all_pairwise_results_final"
        / "generated_plots_after_id_fix"
    )
    gallery_root = Path(args.gallery_root) if args.gallery_root else default_gallery_root

    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    summary_path = results_dir / "summary.json"
    metrics_long_path = results_dir / "metrics_long.csv"
    metrics_jsonl_path = results_dir / "metrics_rows.jsonl"

    if args.resume and read_json(status_path, default={}).get("state") == "completed" and summary_path.exists():
        print(f"[skip] run already completed -> {run_root}")
        return 0

    progress = read_json(progress_path, default={"completed_direction_sources": [], "updated_at": utc_now_iso()})
    completed_sources = set(progress.get("completed_direction_sources", []))

    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "git_commit": get_git_commit(),
            "model": args.model,
            "model_dir": model_dir,
            "target_dataset": args.target_dataset,
            "target_segment": target_segment,
            "segment": args.segment,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "direction_source_datasets": direction_source_datasets,
            "evaluation_single_datasets": evaluation_single_datasets,
            "source_sweep_root": str(Path(args.source_sweep_root)),
            "fixed_results_dir": str(Path(args.fixed_results_dir)),
            "run_root": str(run_root),
            "mirror_results_root": str(external_run_root),
            "gallery_root": str(gallery_root),
        },
    )
    write_json(status_path, {"state": "running", "message": "starting run", "updated_at": utc_now_iso()})

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[start] run_id={run_id}\n")

    # Load target test activations once.
    activations_root = Path(args.activations_root)
    target_root = activations_root / model_dir / target_segment
    test_dir = target_root / "test"
    x_test, y_test, test_ids = load_pooled_split(
        test_dir,
        args.layer,
        args.pooling,
        f"{target_segment} test",
        use_tqdm=not args.no_tqdm,
    )
    write_json(
        inputs_dir / "target_stats.json",
        {
            "target_dataset": target_segment,
            "n_test": int(x_test.shape[0]),
            "dim": int(x_test.shape[1]),
            "test_label_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
            "sample_id_count": len(test_ids),
        },
    )

    # Load fixed baselines from existing matrix.
    fixed_csv = Path(args.fixed_results_dir) / args.segment / f"matrix_fixed_{args.pooling}_L{args.layer}_auc.csv"
    if not fixed_csv.exists():
        raise FileNotFoundError(f"Missing fixed baseline matrix CSV: {fixed_csv}")
    row_labels, col_labels, baseline_matrix = read_matrix_csv(fixed_csv)
    target_col = target_segment
    if target_col not in col_labels:
        raise KeyError(f"Target column {target_col} not found in {fixed_csv}")
    col_idx = col_labels.index(target_col)

    baseline_by_probe: Dict[str, float] = {}
    baseline_rows: List[Dict[str, Any]] = []
    for dataset_base in evaluation_single_datasets:
        row_name = dataset_segment_name(dataset_base, args.segment)
        if row_name not in row_labels:
            raise KeyError(f"Baseline row {row_name} not found in {fixed_csv}")
        row_idx = row_labels.index(row_name)
        auc = float(baseline_matrix[row_idx, col_idx])
        baseline_by_probe[dataset_base] = auc
        baseline_rows.append(
            {
                "evaluation_probe_dataset": dataset_base,
                "row_dataset": row_name,
                "target_dataset": target_col,
                "baseline_auc": auc,
                "fixed_matrix_csv": str(fixed_csv),
            }
        )
    write_csv(
        results_dir / "baseline_it_auc_vector.csv",
        baseline_rows,
        ["evaluation_probe_dataset", "row_dataset", "target_dataset", "baseline_auc", "fixed_matrix_csv"],
    )
    write_json(
        inputs_dir / "fixed_baseline_manifest.json",
        {
            "fixed_matrix_csv": str(fixed_csv),
            "target_dataset": target_col,
            "evaluation_single_datasets": evaluation_single_datasets,
            "baseline_by_probe": baseline_by_probe,
        },
    )

    # Load evaluation probes once.
    probe_weights: Dict[str, Tuple[np.ndarray, float]] = {}
    probe_manifest_rows: List[Dict[str, Any]] = []
    for dataset_base in evaluation_single_datasets:
        probe_path = single_probe_path(
            single_probes_root=Path(args.single_probes_root),
            model_dir=model_dir,
            dataset_base=dataset_base,
            segment=args.segment,
            pooling=args.pooling,
            layer=args.layer,
        )
        w, b = load_linear_probe(probe_path)
        if int(w.shape[0]) != int(x_test.shape[1]):
            raise ValueError(
                f"Probe dim mismatch for {dataset_base}: probe={w.shape[0]} features={x_test.shape[1]}"
            )
        probe_weights[dataset_base] = (w, b)
        probe_manifest_rows.append(
            {
                "probe_dataset": dataset_base,
                "probe_path": str(probe_path),
                "dim": int(w.shape[0]),
                "baseline_auc": baseline_by_probe[dataset_base],
            }
        )
    write_csv(
        inputs_dir / "probe_manifest.csv",
        probe_manifest_rows,
        ["probe_dataset", "probe_path", "dim", "baseline_auc"],
    )

    direction_manifest_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    if metrics_jsonl_path.exists():
        with metrics_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics_rows.append(json.loads(line))

    source_sweep_root = Path(args.source_sweep_root)
    for direction_source_dataset in direction_source_datasets:
        source_slug = slugify_dataset_base(direction_source_dataset)
        if args.resume and source_slug in completed_sources:
            continue

        learned_npz = source_sweep_root / "source_runs" / source_slug / "results" / "learned_direction.npz"
        selection_summary = source_sweep_root / "source_runs" / source_slug / "results" / "selection_summary.json"
        if not learned_npz.exists():
            raise FileNotFoundError(f"Missing learned direction for {direction_source_dataset}: {learned_npz}")
        arr = np.load(str(learned_npz))
        direction = normalize_np(np.asarray(arr["direction"], dtype=np.float32))
        alpha = float(np.asarray(arr["alpha"]).reshape(-1)[0])
        epoch = int(np.asarray(arr["epoch"]).reshape(-1)[0])

        x_test_trans = apply_rank1_transform_np(x_test, direction, alpha)
        for probe_dataset in evaluation_single_datasets:
            w, b = probe_weights[probe_dataset]
            metrics = evaluate_probe_logits(x_test_trans, y_test, w, b)
            baseline_auc = float(baseline_by_probe[probe_dataset])
            baseline_row = {
                "timestamp": utc_now_iso(),
                "direction_source_dataset": direction_source_dataset,
                "direction_slug": source_slug,
                "target_dataset": target_segment,
                "evaluation_probe_dataset": probe_dataset,
                "split": "test",
                "method": "baseline_from_fixed",
                "probe_path": next(r["probe_path"] for r in probe_manifest_rows if r["probe_dataset"] == probe_dataset),
                "direction_npz": str(learned_npz),
                "alpha": alpha,
                "epoch": epoch,
                "auc": baseline_auc,
                "accuracy": "",
                "f1": "",
                "count": int(y_test.shape[0]),
                "logit_mean": "",
                "logit_std": "",
                "delta_auc_vs_baseline": 0.0,
            }
            transformed_row = {
                "timestamp": utc_now_iso(),
                "direction_source_dataset": direction_source_dataset,
                "direction_slug": source_slug,
                "target_dataset": target_segment,
                "evaluation_probe_dataset": probe_dataset,
                "split": "test",
                "method": "transferred_rank1",
                "probe_path": next(r["probe_path"] for r in probe_manifest_rows if r["probe_dataset"] == probe_dataset),
                "direction_npz": str(learned_npz),
                "alpha": alpha,
                "epoch": epoch,
                "auc": float(metrics["auc"]),
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "count": int(metrics["count"]),
                "logit_mean": float(metrics["logit_mean"]),
                "logit_std": float(metrics["logit_std"]),
                "delta_auc_vs_baseline": float(metrics["auc"] - baseline_auc),
            }
            metrics_rows.append(baseline_row)
            metrics_rows.append(transformed_row)
            with metrics_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(baseline_row, ensure_ascii=True) + "\n")
                f.write(json.dumps(transformed_row, ensure_ascii=True) + "\n")

        completed_sources.add(source_slug)
        progress["completed_direction_sources"] = sorted(completed_sources)
        progress["updated_at"] = utc_now_iso()
        write_json(progress_path, progress)

    for direction_source_dataset in direction_source_datasets:
        source_slug = slugify_dataset_base(direction_source_dataset)
        learned_npz = source_sweep_root / "source_runs" / source_slug / "results" / "learned_direction.npz"
        selection_summary = source_sweep_root / "source_runs" / source_slug / "results" / "selection_summary.json"
        if not learned_npz.exists():
            raise FileNotFoundError(f"Missing learned direction for {direction_source_dataset}: {learned_npz}")
        arr = np.load(str(learned_npz))
        direction_manifest_rows.append(
            {
                "direction_source_dataset": direction_source_dataset,
                "direction_slug": source_slug,
                "learned_direction_npz": str(learned_npz),
                "selection_summary_json": str(selection_summary),
                "alpha": float(np.asarray(arr["alpha"]).reshape(-1)[0]),
                "epoch": int(np.asarray(arr["epoch"]).reshape(-1)[0]),
            }
        )

    write_csv(
        inputs_dir / "direction_manifest.csv",
        direction_manifest_rows,
        ["direction_source_dataset", "direction_slug", "learned_direction_npz", "selection_summary_json", "alpha", "epoch"],
    )

    # Deduplicate rows for clean resume behavior.
    deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in metrics_rows:
        key = (
            str(row["direction_source_dataset"]),
            str(row["evaluation_probe_dataset"]),
            str(row["method"]),
        )
        deduped[key] = row
    metrics_rows = list(deduped.values())
    metrics_rows.sort(key=lambda r: (str(r["direction_source_dataset"]), str(r["evaluation_probe_dataset"]), str(r["method"])))

    write_csv(
        metrics_long_path,
        metrics_rows,
        [
            "timestamp",
            "direction_source_dataset",
            "direction_slug",
            "target_dataset",
            "evaluation_probe_dataset",
            "split",
            "method",
            "probe_path",
            "direction_npz",
            "alpha",
            "epoch",
            "auc",
            "accuracy",
            "f1",
            "count",
            "logit_mean",
            "logit_std",
            "delta_auc_vs_baseline",
        ],
    )

    row_labels = list(direction_source_datasets)
    col_labels = list(evaluation_single_datasets)
    auc_matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float64)
    delta_matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float64)

    transformed_rows = [r for r in metrics_rows if str(r["method"]) == "transferred_rank1"]
    for i, direction_source_dataset in enumerate(row_labels):
        for j, probe_dataset in enumerate(col_labels):
            rec = next(
                (
                    r for r in transformed_rows
                    if str(r["direction_source_dataset"]) == direction_source_dataset
                    and str(r["evaluation_probe_dataset"]) == probe_dataset
                ),
                None,
            )
            if rec is None:
                continue
            auc_matrix[i, j] = float(rec["auc"])
            delta_matrix[i, j] = float(rec["delta_auc_vs_baseline"])

    # CSV matrices
    ensure_dir(results_dir)
    with (results_dir / "test_auc_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["direction_source_dataset"] + col_labels)
        for i, row_name in enumerate(row_labels):
            writer.writerow([row_name] + [("" if np.isnan(v) else float(v)) for v in auc_matrix[i]])
    with (results_dir / "test_delta_auc_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["direction_source_dataset"] + col_labels)
        for i, row_name in enumerate(row_labels):
            writer.writerow([row_name] + [("" if np.isnan(v) else float(v)) for v in delta_matrix[i]])

    # Summaries
    summary_by_direction_rows: List[Dict[str, Any]] = []
    for i, direction_source_dataset in enumerate(row_labels):
        direction_values = delta_matrix[i]
        matched_j = col_labels.index(direction_source_dataset) if direction_source_dataset in col_labels else None
        matched_probe = direction_source_dataset if matched_j is not None else ""
        matched_delta = float(direction_values[matched_j]) if matched_j is not None and not np.isnan(direction_values[matched_j]) else np.nan
        matched_auc = float(auc_matrix[i, matched_j]) if matched_j is not None and not np.isnan(auc_matrix[i, matched_j]) else np.nan
        direction_rec = next((r for r in direction_manifest_rows if r["direction_source_dataset"] == direction_source_dataset), {})

        valid_pairs = [(col_labels[j], float(direction_values[j])) for j in range(len(col_labels)) if not np.isnan(direction_values[j])]
        mean_delta = float(np.mean([v for _, v in valid_pairs])) if valid_pairs else np.nan
        best_probe_name, best_probe_delta = ("", np.nan)
        worst_probe_name, worst_probe_delta = ("", np.nan)
        if valid_pairs:
            best_probe_name, best_probe_delta = max(valid_pairs, key=lambda x: x[1])
            worst_probe_name, worst_probe_delta = min(valid_pairs, key=lambda x: x[1])

        summary_by_direction_rows.append(
            {
                "direction_source_dataset": direction_source_dataset,
                "matched_probe_dataset": matched_probe,
                "matched_transformed_auc": matched_auc,
                "matched_baseline_auc": baseline_by_probe.get(matched_probe, np.nan) if matched_probe else np.nan,
                "matched_delta_auc": matched_delta,
                "mean_delta_auc": mean_delta,
                "best_probe_dataset": best_probe_name,
                "best_probe_delta_auc": best_probe_delta,
                "worst_probe_dataset": worst_probe_name,
                "worst_probe_delta_auc": worst_probe_delta,
                "alpha": direction_rec.get("alpha", np.nan),
                "epoch": direction_rec.get("epoch", ""),
                "learned_direction_npz": direction_rec.get("learned_direction_npz", ""),
            }
        )

    summary_by_probe_rows: List[Dict[str, Any]] = []
    for j, probe_dataset in enumerate(col_labels):
        probe_values = delta_matrix[:, j]
        valid = [(row_labels[i], float(probe_values[i])) for i in range(len(row_labels)) if not np.isnan(probe_values[i])]
        best_direction, best_delta = ("", np.nan)
        best_auc = np.nan
        if valid:
            best_direction, best_delta = max(valid, key=lambda x: x[1])
            i = row_labels.index(best_direction)
            best_auc = float(auc_matrix[i, j])
        summary_by_probe_rows.append(
            {
                "evaluation_probe_dataset": probe_dataset,
                "baseline_auc": baseline_by_probe[probe_dataset],
                "best_direction_source_dataset": best_direction,
                "best_transformed_auc": best_auc,
                "best_delta_auc": best_delta,
            }
        )

    write_csv(
        results_dir / "summary_by_direction.csv",
        summary_by_direction_rows,
        [
            "direction_source_dataset",
            "matched_probe_dataset",
            "matched_transformed_auc",
            "matched_baseline_auc",
            "matched_delta_auc",
            "mean_delta_auc",
            "best_probe_dataset",
            "best_probe_delta_auc",
            "worst_probe_dataset",
            "worst_probe_delta_auc",
            "alpha",
            "epoch",
            "learned_direction_npz",
        ],
    )
    write_csv(
        results_dir / "summary_by_probe.csv",
        summary_by_probe_rows,
        [
            "evaluation_probe_dataset",
            "baseline_auc",
            "best_direction_source_dataset",
            "best_transformed_auc",
            "best_delta_auc",
        ],
    )

    # Plots
    row_short = [SHORT_LABELS.get(r, r) for r in row_labels]
    col_short = [SHORT_LABELS.get(c, c) for c in col_labels]
    auc_heatmap_path = plots_dir / "test_auc_heatmap.png"
    delta_heatmap_path = plots_dir / "test_delta_auc_heatmap.png"
    matched_bar_path = plots_dir / "matched_probe_delta_bar.png"
    plot_heatmap(
        auc_matrix,
        row_short,
        col_short,
        "Transformed InsiderTrading test AUROC",
        auc_heatmap_path,
        vmin=0.0,
        vmax=1.0,
        cmap="YlGnBu",
    )
    delta_abs = float(np.nanmax(np.abs(delta_matrix))) if np.any(~np.isnan(delta_matrix)) else 0.05
    delta_abs = max(delta_abs, 0.05)
    plot_heatmap(
        delta_matrix,
        row_short,
        col_short,
        "Delta AUROC vs fixed InsiderTrading baseline",
        delta_heatmap_path,
        vmin=-delta_abs,
        vmax=delta_abs,
        cmap="RdBu_r",
    )
    plot_matched_delta_bar(summary_by_direction_rows, matched_bar_path)

    gallery_paths: List[str] = []
    if gallery_root:
        gallery_paths.extend(
            p for p in [
                mirror_png(auc_heatmap_path, gallery_root / "it_completion_mean_l15_transferred_rank1_test_auc_heatmap.png"),
                mirror_png(delta_heatmap_path, gallery_root / "it_completion_mean_l15_transferred_rank1_test_delta_auc_heatmap.png"),
                mirror_png(matched_bar_path, gallery_root / "it_completion_mean_l15_transferred_rank1_matched_probe_delta_bar.png"),
            ] if p
        )

    summary = {
        "run_id": run_id,
        "completed_at": utc_now_iso(),
        "model": args.model,
        "target_dataset": target_segment,
        "segment": args.segment,
        "pooling": args.pooling,
        "layer": int(args.layer),
        "direction_source_datasets": direction_source_datasets,
        "evaluation_single_datasets": evaluation_single_datasets,
        "fixed_results_dir": str(Path(args.fixed_results_dir)),
        "source_sweep_root": str(source_sweep_root),
        "baseline_matrix_csv": str(fixed_csv),
        "baseline_target_column": target_segment,
        "n_test_samples": int(x_test.shape[0]),
        "artifacts": {
            "metrics_long_csv": str(metrics_long_path),
            "baseline_it_auc_vector_csv": str(results_dir / "baseline_it_auc_vector.csv"),
            "test_auc_matrix_csv": str(results_dir / "test_auc_matrix.csv"),
            "test_delta_auc_matrix_csv": str(results_dir / "test_delta_auc_matrix.csv"),
            "summary_by_direction_csv": str(results_dir / "summary_by_direction.csv"),
            "summary_by_probe_csv": str(results_dir / "summary_by_probe.csv"),
            "test_auc_heatmap_png": str(auc_heatmap_path),
            "test_delta_auc_heatmap_png": str(delta_heatmap_path),
            "matched_probe_delta_bar_png": str(matched_bar_path),
        },
        "gallery_paths": gallery_paths,
    }
    write_json(summary_path, summary)

    if external_run_root:
        copy_tree_subset(run_root, external_run_root, ["meta", "checkpoints", "inputs", "results", "logs"])

    write_json(status_path, {"state": "completed", "message": "completed successfully", "updated_at": utc_now_iso()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
