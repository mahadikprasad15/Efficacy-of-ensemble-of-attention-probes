#!/usr/bin/env python3
"""
Evaluate frozen single-source and combined probes on a target dataset after
plain target-side PCA direction removal.

Workflow:
1. Load pooled target activations for target train/validation/test.
2. Fit PCA on target train only.
3. Evaluate frozen probes on baseline (k=0), target validation, and target test.
4. Remove top-k target PCs and reevaluate the same frozen probes.
5. Save resumable metrics, summaries, plots, and optional mirrored outputs.

This script currently supports pooled probes only: mean/max/last.
Removal modes:
  - cumulative: remove top-k PCs (1..k)
  - single: remove exactly one PC index at a time
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
from combined_probe_pipeline import (  # noqa: E402
    default_run_id,
    ensure_dir,
    model_dir_name,
    utc_now_iso,
    write_json,
    read_json,
)


SUPPORTED_POOLINGS = ["mean", "max", "last"]


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_k_values(raw: str) -> List[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    vals = sorted(set(vals))
    if any(k <= 0 for k in vals):
        raise ValueError(f"All k values must be positive. Got: {vals}")
    return vals


def dataset_segment_name(dataset_base: str, segment: str) -> str:
    if dataset_base.endswith("-completion") or dataset_base.endswith("-full"):
        return dataset_base
    return f"{dataset_base}-{segment}"


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
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    label_map = load_label_map(activations_dir)
    shard_paths = sorted(activations_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

    features: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[str] = []

    for shard_path in tqdm(shard_paths, desc=f"Loading {desc} shards"):
        shard = load_file(str(shard_path))
        for sid, tensor in shard.items():
            if sid not in label_map:
                continue
            label = label_map[sid]
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = sigmoid(logits)
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
    }


def remove_top_k_pcs(
    x: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    k: int,
) -> np.ndarray:
    x_centered = x - mean[None, :]
    u = components[:k, :]
    proj = (x_centered @ u.T) @ u
    return (x_centered - proj) + mean[None, :]


def remove_specific_pcs(
    x: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    component_indices_1based: Sequence[int],
) -> np.ndarray:
    if not component_indices_1based:
        return x.copy()
    zero_based = [int(i) - 1 for i in component_indices_1based]
    if any(i < 0 or i >= components.shape[0] for i in zero_based):
        raise ValueError(
            f"Component index out of range. Requested={component_indices_1based}, available=1..{components.shape[0]}"
        )
    x_centered = x - mean[None, :]
    u = components[zero_based, :]
    proj = (x_centered @ u.T) @ u
    return (x_centered - proj) + mean[None, :]


def fit_pca(
    x_train: np.ndarray,
    n_components_to_save: int,
    pca_solver: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    n_samples, dim = x_train.shape
    n_components = min(n_components_to_save, n_samples, dim)
    if n_components < 1:
        raise ValueError("PCA needs at least one component")
    pca = PCA(n_components=n_components, svd_solver=pca_solver, random_state=seed)
    pca.fit(x_train)
    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "explained_variance": pca.explained_variance_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }


def save_pca_artifact(path: Path, artifact: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        str(path),
        mean=artifact["mean"],
        components=artifact["components"],
        explained_variance=artifact["explained_variance"],
        explained_variance_ratio=artifact["explained_variance_ratio"],
    )


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


def combined_probe_path(
    combined_run_root: Path,
    combo_name: str,
    pooling: str,
    layer: int,
) -> Path:
    return combined_run_root / "probes" / combo_name / pooling / f"probe_layer_{layer}.pt"


def load_linear_probe(probe_path: Path) -> Tuple[np.ndarray, float]:
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing probe checkpoint: {probe_path}")
    state = torch.load(str(probe_path), map_location="cpu")
    return extract_classifier_params(state)


def evaluate_linear_probe(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, float]:
    logits = x @ w + b
    metrics = compute_metrics(y, logits)
    metrics["logit_mean"] = float(np.mean(logits))
    metrics["logit_std"] = float(np.std(logits))
    return metrics


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def append_log_line(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_probe_by_k(
    rows: Sequence[Dict[str, Any]],
    split: str,
    value_key: str,
    title: str,
    out_path: Path,
    x_label: str,
) -> None:
    ensure_dir(out_path.parent)
    by_probe: Dict[str, List[Tuple[int, float]]] = {}
    for row in rows:
        if row["split"] != split:
            continue
        val = row.get(value_key)
        if val in [None, ""]:
            continue
        by_probe.setdefault(str(row["probe_name"]), []).append((int(row["k"]), float(val)))

    plt.figure(figsize=(10, 6))
    for probe_name, pts in sorted(by_probe.items()):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=probe_name)
    plt.xlabel(x_label)
    plt.ylabel(value_key.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: Path,
    cbar_label: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = ".3f",
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.0), max(4.5, len(row_labels) * 0.65)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = "NA" if np.isnan(v) else format(v, fmt)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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
    parser = argparse.ArgumentParser(description="Evaluate frozen probes after target-side PCA removal.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--single_probes_root", type=str, default="data/probes")
    parser.add_argument("--target_dataset", type=str, default="Deception-Roleplaying")
    parser.add_argument("--segment", type=str, choices=["completion", "full"], default="completion")
    parser.add_argument("--pooling", type=str, choices=SUPPORTED_POOLINGS, default="mean")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--source_datasets",
        type=str,
        default="Deception-ConvincingGame,Deception-HarmPressureChoice,Deception-InstructedDeception,Deception-Mask,Deception-AILiar",
    )
    parser.add_argument("--include_target_self", action="store_true")
    parser.add_argument("--combined_run_root", type=str, default="")
    parser.add_argument("--combined_probe_names", type=str, default="")
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--removal_mode", type=str, choices=["cumulative", "single"], default="cumulative")
    parser.add_argument("--target_train_split", type=str, default="train")
    parser.add_argument("--target_val_split", type=str, default="validation")
    parser.add_argument("--target_test_split", type=str, default="test")
    parser.add_argument("--pca_components_to_save", type=int, default=128)
    parser.add_argument("--pca_solver", type=str, default="randomized")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--mirror_results_root", type=str, default="")
    parser.add_argument("--gallery_root", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_datasets = parse_csv_list(args.source_datasets)
    combined_probe_names = parse_csv_list(args.combined_probe_names)
    k_values = parse_k_values(args.k_values)
    model_dir = model_dir_name(args.model)
    target_segment = dataset_segment_name(args.target_dataset, args.segment)
    config_slug = f"plain-pca-{args.removal_mode}_{args.pooling}_l{args.layer}"
    run_id = args.run_id.strip() or f"{default_run_id()}-{args.target_dataset.split('-')[-1].lower()}-{args.segment}-{args.pooling}-l{args.layer}-pca-{args.removal_mode}-v1"

    artifact_root = Path(args.artifact_root)
    run_root = (
        artifact_root
        / "runs"
        / "target_pca_probe_transfer"
        / model_dir
        / target_segment
        / "frozen-single-and-combined"
        / config_slug
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

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "target_pca_probe_transfer"
    mirror_root = Path(args.mirror_results_root) if args.mirror_results_root else default_mirror_root
    external_run_root = mirror_root / target_segment / config_slug / run_id

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
    rows_jsonl_path = results_dir / "metrics_long.jsonl"
    summary_path = results_dir / "summary.json"

    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "model": args.model,
            "model_dir": model_dir,
            "target_dataset": args.target_dataset,
            "target_segment": target_segment,
            "segment": args.segment,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "source_datasets": source_datasets,
            "include_target_self": bool(args.include_target_self),
            "combined_run_root": args.combined_run_root,
            "combined_probe_names": combined_probe_names,
            "k_values": k_values,
            "removal_mode": args.removal_mode,
            "splits": {
                "train": args.target_train_split,
                "validation": args.target_val_split,
                "test": args.target_test_split,
            },
            "paths": {
                "run_root": str(run_root),
                "inputs_dir": str(inputs_dir),
                "results_dir": str(results_dir),
                "mirror_results_root": str(external_run_root),
                "gallery_root": str(gallery_root),
            },
        },
    )
    append_log_line(log_path, f"[start] run_id={run_id}")
    append_log_line(log_path, f"[target] {target_segment} pooling={args.pooling} layer={args.layer} removal_mode={args.removal_mode}")

    if args.resume and summary_path.exists():
        write_json(status_path, {"state": "completed", "message": "resume requested and summary already exists", "updated_at": utc_now_iso()})
        append_log_line(log_path, f"[resume] summary exists: {summary_path}")
        print(f"[resume] summary exists: {summary_path}")
        return 0

    write_json(status_path, {"state": "running", "stage": "load_inputs", "updated_at": utc_now_iso()})

    activations_model_root = Path(args.activations_root) / model_dir
    target_train_dir = activations_model_root / target_segment / args.target_train_split
    target_val_dir = activations_model_root / target_segment / args.target_val_split
    target_test_dir = activations_model_root / target_segment / args.target_test_split

    x_train, y_train, train_ids = load_pooled_split(target_train_dir, args.layer, args.pooling, desc=f"{target_segment} train")
    x_val, y_val, _ = load_pooled_split(target_val_dir, args.layer, args.pooling, desc=f"{target_segment} val")
    x_test, y_test, test_ids = load_pooled_split(target_test_dir, args.layer, args.pooling, desc=f"{target_segment} test")
    append_log_line(log_path, f"[loaded] n_train={x_train.shape[0]} n_val={x_val.shape[0]} n_test={x_test.shape[0]} dim={x_train.shape[1]}")

    pca_artifact = fit_pca(
        x_train=x_train,
        n_components_to_save=max(max(k_values), args.pca_components_to_save),
        pca_solver=args.pca_solver,
        seed=args.seed,
    )
    if pca_artifact["components"].shape[0] < max(k_values):
        raise ValueError(
            f"PCA saved only {pca_artifact['components'].shape[0]} components but max k={max(k_values)}."
        )
    save_pca_artifact(inputs_dir / "target_pca.npz", pca_artifact)
    append_log_line(log_path, f"[pca] saved target_pca.npz components={pca_artifact['components'].shape[0]}")
    write_json(
        inputs_dir / "target_pca_manifest.json",
        {
            "target_dataset": target_segment,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "train_dir": str(target_train_dir),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_test": int(x_test.shape[0]),
            "dim": int(x_train.shape[1]),
            "k_values": k_values,
            "removal_mode": args.removal_mode,
            "pca_solver": args.pca_solver,
            "created_at": utc_now_iso(),
        },
    )

    probe_specs: List[Dict[str, Any]] = []
    single_probes_root = Path(args.single_probes_root)
    for dataset_base in source_datasets:
        probe_specs.append(
            {
                "probe_name": dataset_base,
                "probe_kind": "single_source",
                "source_dataset": dataset_base,
                "probe_path": str(single_probe_path(single_probes_root, model_dir, dataset_base, args.segment, args.pooling, args.layer)),
            }
        )
    if args.include_target_self:
        probe_specs.append(
            {
                "probe_name": f"{args.target_dataset}_self",
                "probe_kind": "target_self",
                "source_dataset": args.target_dataset,
                "probe_path": str(single_probe_path(single_probes_root, model_dir, args.target_dataset, args.segment, args.pooling, args.layer)),
            }
        )
    if args.combined_run_root:
        combined_root = Path(args.combined_run_root)
        for combo_name in combined_probe_names:
            probe_specs.append(
                {
                    "probe_name": combo_name,
                    "probe_kind": "combined",
                    "source_dataset": combo_name,
                    "probe_path": str(combined_probe_path(combined_root, combo_name, args.pooling, args.layer)),
                }
            )

    if not probe_specs:
        raise ValueError("No probes selected for evaluation.")
    append_log_line(log_path, f"[probes] count={len(probe_specs)} names={[p['probe_name'] for p in probe_specs]}")

    progress = read_json(progress_path, default={"completed_units": []})
    completed_units = set(str(x) for x in progress.get("completed_units", []))

    write_json(status_path, {"state": "running", "stage": "evaluate", "updated_at": utc_now_iso()})

    baseline_by_probe_split: Dict[Tuple[str, str], Dict[str, float]] = {}

    for spec in probe_specs:
        w, b = load_linear_probe(Path(spec["probe_path"]))
        if w.shape[0] != x_train.shape[1]:
            raise ValueError(
                f"Probe dim mismatch for {spec['probe_name']}: probe={w.shape[0]} features={x_train.shape[1]}"
            )

        for k in [0] + k_values:
            unit = f"{spec['probe_name']}::k={k}"
            if args.resume and unit in completed_units:
                append_log_line(log_path, f"[skip] {unit}")
                continue

            if k == 0:
                x_val_use = x_val
                x_test_use = x_test
            else:
                if args.removal_mode == "cumulative":
                    x_val_use = remove_top_k_pcs(x_val, pca_artifact["mean"], pca_artifact["components"], k)
                    x_test_use = remove_top_k_pcs(x_test, pca_artifact["mean"], pca_artifact["components"], k)
                else:
                    x_val_use = remove_specific_pcs(x_val, pca_artifact["mean"], pca_artifact["components"], [k])
                    x_test_use = remove_specific_pcs(x_test, pca_artifact["mean"], pca_artifact["components"], [k])

            for split_name, x_split, y_split in [
                ("validation", x_val_use, y_val),
                ("test", x_test_use, y_test),
            ]:
                metrics = evaluate_linear_probe(x_split, y_split, w, b)
                if k == 0:
                    baseline_by_probe_split[(spec["probe_name"], split_name)] = metrics
                baseline = baseline_by_probe_split[(spec["probe_name"], split_name)]
                row = {
                    "timestamp": utc_now_iso(),
                    "target_dataset": target_segment,
                    "probe_name": spec["probe_name"],
                    "probe_kind": spec["probe_kind"],
                    "source_dataset": spec["source_dataset"],
                    "probe_path": spec["probe_path"],
                    "pooling": args.pooling,
                    "layer": int(args.layer),
                    "split": split_name,
                    "k": int(k),
                    "removal_mode": args.removal_mode,
                    "auc": float(metrics["auc"]),
                    "accuracy": float(metrics["accuracy"]),
                    "f1": float(metrics["f1"]),
                    "count": int(metrics["count"]),
                    "delta_auc_vs_baseline": float(metrics["auc"] - baseline["auc"]),
                    "delta_accuracy_vs_baseline": float(metrics["accuracy"] - baseline["accuracy"]),
                    "delta_f1_vs_baseline": float(metrics["f1"] - baseline["f1"]),
                }
                append_jsonl(rows_jsonl_path, row)

            completed_units.add(unit)
            progress["completed_units"] = sorted(completed_units)
            progress["updated_at"] = utc_now_iso()
            write_json(progress_path, progress)
            append_log_line(log_path, f"[done] {unit}")

    rows = read_jsonl(rows_jsonl_path)
    rows.sort(key=lambda r: (str(r["probe_name"]), str(r["split"]), int(r["k"])))

    fieldnames = [
        "timestamp", "target_dataset", "probe_name", "probe_kind", "source_dataset", "probe_path",
        "pooling", "layer", "split", "k", "removal_mode", "auc", "accuracy", "f1", "count",
        "delta_auc_vs_baseline", "delta_accuracy_vs_baseline", "delta_f1_vs_baseline",
    ]
    write_csv(results_dir / "metrics_long.csv", rows, fieldnames)

    baseline_rows = [r for r in rows if int(r["k"]) == 0]
    write_csv(
        results_dir / "baseline_metrics.csv",
        baseline_rows,
        [f for f in fieldnames if f != "timestamp"],
    )

    selection_rows: List[Dict[str, Any]] = []
    for probe_name in sorted({str(r["probe_name"]) for r in rows}):
        val_rows = [r for r in rows if r["probe_name"] == probe_name and r["split"] == "validation"]
        test_rows = [r for r in rows if r["probe_name"] == probe_name and r["split"] == "test"]
        if not val_rows or not test_rows:
            continue
        best_val = max(val_rows, key=lambda r: (float(r["auc"]), float(r["accuracy"]), float(r["f1"]), -int(r["k"])))
        best_k = int(best_val["k"])
        matching_test = next(r for r in test_rows if int(r["k"]) == best_k)
        baseline_test = next(r for r in test_rows if int(r["k"]) == 0)
        selection_rows.append(
            {
                "probe_name": probe_name,
                "probe_kind": best_val["probe_kind"],
                "source_dataset": best_val["source_dataset"],
                "best_val_k": best_k,
                "best_val_auc": float(best_val["auc"]),
                "test_auc_at_best_val_k": float(matching_test["auc"]),
                "test_delta_auc_at_best_val_k": float(matching_test["auc"] - baseline_test["auc"]),
                "baseline_test_auc": float(baseline_test["auc"]),
            }
        )
    write_csv(
        results_dir / "selection_summary.csv",
        selection_rows,
        [
            "probe_name",
            "probe_kind",
            "source_dataset",
            "best_val_k",
            "best_val_auc",
            "test_auc_at_best_val_k",
            "test_delta_auc_at_best_val_k",
            "baseline_test_auc",
        ],
    )

    probe_names = sorted({str(r["probe_name"]) for r in rows})
    k_axis = [0] + k_values
    test_auc_mat = np.full((len(probe_names), len(k_axis)), np.nan, dtype=np.float64)
    test_delta_mat = np.full((len(probe_names), len(k_axis)), np.nan, dtype=np.float64)
    for i, probe_name in enumerate(probe_names):
        for j, k in enumerate(k_axis):
            rec = next((r for r in rows if r["probe_name"] == probe_name and r["split"] == "test" and int(r["k"]) == int(k)), None)
            if rec is None:
                continue
            test_auc_mat[i, j] = float(rec["auc"])
            test_delta_mat[i, j] = float(rec["delta_auc_vs_baseline"])

    axis_label = "k removed PCs" if args.removal_mode == "cumulative" else "single removed PC index"
    mode_title = "target PCA removal" if args.removal_mode == "cumulative" else "single target PC removal"
    test_auc_heatmap = plots_dir / "target_test_auc_by_probe_k.png"
    test_delta_heatmap = plots_dir / "target_test_delta_auc_by_probe_k.png"
    test_auc_lines = plots_dir / "target_test_auc_lines.png"
    val_auc_lines = plots_dir / "target_validation_auc_lines.png"

    plot_heatmap(
        matrix=test_auc_mat,
        row_labels=probe_names,
        col_labels=[str(k) for k in k_axis],
        title=f"{target_segment}: frozen probe AUC after {mode_title}",
        out_path=test_auc_heatmap,
        cbar_label="AUROC",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    delta_abs = float(np.nanmax(np.abs(test_delta_mat))) if np.any(~np.isnan(test_delta_mat)) else 0.05
    plot_heatmap(
        matrix=test_delta_mat,
        row_labels=probe_names,
        col_labels=[str(k) for k in k_axis],
        title=f"{target_segment}: test AUC delta vs baseline ({args.removal_mode})",
        out_path=test_delta_heatmap,
        cbar_label="Delta AUROC",
        cmap="RdBu_r",
        vmin=-delta_abs,
        vmax=delta_abs,
    )
    plot_probe_by_k(
        rows=rows,
        split="test",
        value_key="auc",
        title=f"{target_segment}: test AUROC by {axis_label}",
        out_path=test_auc_lines,
        x_label=axis_label,
    )
    plot_probe_by_k(
        rows=rows,
        split="validation",
        value_key="auc",
        title=f"{target_segment}: validation AUROC by {axis_label}",
        out_path=val_auc_lines,
        x_label=axis_label,
    )

    gallery_paths: List[str] = []
    gallery_prefix = f"{target_segment}_{args.pooling}_l{args.layer}_plain_pca_{args.removal_mode}"
    for src, name in [
        (test_auc_heatmap, f"{gallery_prefix}_test_auc_by_probe_k.png"),
        (test_delta_heatmap, f"{gallery_prefix}_test_delta_auc_by_probe_k.png"),
        (test_auc_lines, f"{gallery_prefix}_test_auc_lines.png"),
        (val_auc_lines, f"{gallery_prefix}_validation_auc_lines.png"),
    ]:
        mirrored = mirror_png(src, gallery_root / name)
        if mirrored:
            gallery_paths.append(mirrored)

    summary = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "model": args.model,
        "target_dataset": target_segment,
        "pooling": args.pooling,
        "layer": int(args.layer),
        "removal_mode": args.removal_mode,
        "k_values": [0] + k_values,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "dim": int(x_train.shape[1]),
        "probe_names": probe_names,
        "outputs": {
            "metrics_long_csv": str(results_dir / "metrics_long.csv"),
            "baseline_metrics_csv": str(results_dir / "baseline_metrics.csv"),
            "selection_summary_csv": str(results_dir / "selection_summary.csv"),
            "test_auc_heatmap_png": str(test_auc_heatmap),
            "test_delta_heatmap_png": str(test_delta_heatmap),
            "test_auc_lines_png": str(test_auc_lines),
            "validation_auc_lines_png": str(val_auc_lines),
        },
        "gallery_pngs": gallery_paths,
        "mirror_results_root": str(external_run_root),
    }
    write_json(summary_path, summary)

    write_json(status_path, {"state": "completed", "message": "completed successfully", "updated_at": utc_now_iso()})
    append_log_line(log_path, f"[done] canonical run root -> {run_root}")
    append_log_line(log_path, f"[done] mirrored results -> {external_run_root}")
    append_log_line(log_path, f"[done] mirrored gallery pngs -> {gallery_root}")

    copy_tree_subset(run_root, external_run_root, ["meta", "checkpoints", "inputs", "results", "logs"])
    print(f"[done] canonical run root -> {run_root}")
    print(f"[done] mirrored results -> {external_run_root}")
    print(f"[done] mirrored gallery pngs -> {gallery_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
