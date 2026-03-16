#!/usr/bin/env python3
"""
Evaluate frozen probes on a target dataset after learning a target-side rank-1
direction removal transform.

Workflow:
1. Load pooled target activations for train/validation/test.
2. Load frozen source probes and optional target self probe.
3. Fit PCA-PC1 on target train for comparison.
4. Learn one direction v and scalar alpha on target train:
      x' = x - alpha * (x^T v) * v
5. Select the learned checkpoint on target validation.
6. Compare no-transform, PCA-PC1, learned, and random rank-1 baselines.
7. Save resumable artifacts, summaries, and optional mirrored outputs.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.decomposition import PCA
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
    }


def fit_pca_pc1(x_train: np.ndarray, pca_solver: str, seed: int) -> Dict[str, np.ndarray]:
    n_samples, dim = x_train.shape
    n_components = min(1, n_samples, dim)
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


def save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(str(path), **payload)


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
    return ((x_centered - proj) + mean[None, :]).astype(np.float32)


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


def load_trusted_torch_checkpoint(path: Path) -> Dict[str, Any]:
    """
    Load an internal checkpoint created by this script.

    PyTorch 2.6 changed torch.load to default to weights_only=True, which breaks
    our resumable training checkpoint because it stores NumPy arrays and the
    optimizer state, not just tensors. This helper opts back into the previous
    trusted behavior for this script-owned checkpoint format.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def append_log_line(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


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


def normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        raise ValueError("Direction norm is zero")
    return (v / norm).astype(np.float32)


def normalize_torch(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v), min=eps)


def apply_rank1_transform_np(x: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    proj = (x @ direction).astype(np.float32, copy=False)
    transformed = x - float(alpha) * proj[:, None] * direction[None, :]
    return transformed.astype(np.float32, copy=False)


def apply_rank1_transform_torch(x: torch.Tensor, direction: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    proj = torch.matmul(x, direction)
    return x - alpha * proj.unsqueeze(1) * direction.unsqueeze(0)


def evaluate_probe_logits(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, float]:
    logits = x @ w + b
    metrics = compute_metrics(y, logits)
    metrics["logit_mean"] = float(np.mean(logits))
    metrics["logit_std"] = float(np.std(logits))
    return metrics


def plot_grouped_bar(
    rows: Sequence[Dict[str, Any]],
    split: str,
    value_key: str,
    title: str,
    out_path: Path,
    method_order: Sequence[str],
    probe_order: Sequence[str],
    y_label: str,
) -> None:
    ensure_dir(out_path.parent)
    width = 0.14
    x = np.arange(len(probe_order))
    fig, ax = plt.subplots(figsize=(max(9, len(probe_order) * 2.3), 6))
    for idx, method in enumerate(method_order):
        vals = []
        for probe_name in probe_order:
            rec = next(
                (
                    r
                    for r in rows
                    if str(r["split"]) == split
                    and str(r["method"]) == method
                    and str(r["probe_name"]) == probe_name
                ),
                None,
            )
            vals.append(np.nan if rec is None else float(rec[value_key]))
        ax.bar(x + (idx - (len(method_order) - 1) / 2.0) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(probe_order, rotation=25, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history_rows: Sequence[Dict[str, Any]],
    out_path: Path,
    self_probe_present: bool,
) -> None:
    ensure_dir(out_path.parent)
    if not history_rows:
        return
    epochs = [int(r["epoch"]) for r in history_rows]
    train_total = [float(r["train_total_loss"]) for r in history_rows]
    val_source_auc = [float(r["val_avg_source_auc"]) for r in history_rows]
    alpha_vals = [float(r["alpha"]) for r in history_rows]

    nrows = 3 if self_probe_present else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(epochs, train_total, marker="o", linewidth=1.8)
    axes[0].set_ylabel("Train Total Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, val_source_auc, marker="o", linewidth=1.8, label="Val Avg Source AUC")
    if self_probe_present:
        val_self_auc = [float(r["val_self_auc"]) for r in history_rows]
        axes[1].plot(epochs, val_self_auc, marker="o", linewidth=1.8, label="Val Self AUC")
    axes[1].set_ylabel("Validation AUROC")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    axes[-1].plot(epochs, alpha_vals, marker="o", linewidth=1.8)
    axes[-1].set_ylabel("Alpha")
    axes[-1].set_xlabel("Epoch")
    axes[-1].grid(alpha=0.25)

    fig.suptitle("Learned Rank-1 Direction Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_random_distribution(
    random_rows: Sequence[Dict[str, Any]],
    learned_metrics: Dict[str, Dict[str, float]],
    pca_metrics: Dict[str, Dict[str, float]],
    baseline_metrics: Dict[str, Dict[str, float]],
    source_probe_name: str,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    val_vals = [
        float(r["auc"])
        for r in random_rows
        if r["split"] == "validation" and r["probe_name"] == source_probe_name
    ]
    test_vals = [
        float(r["auc"])
        for r in random_rows
        if r["split"] == "test" and r["probe_name"] == source_probe_name
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, split in zip(axes, [val_vals, test_vals], ["validation", "test"]):
        ax.hist(vals, bins=min(20, max(5, len(vals))), alpha=0.75, color="#8fbcd4")
        ax.axvline(float(baseline_metrics[split]["auc"]), color="black", linewidth=2, label="baseline")
        ax.axvline(float(pca_metrics[split]["auc"]), color="#f28e2b", linewidth=2, label="pca_pc1")
        ax.axvline(float(learned_metrics[split]["auc"]), color="#d62728", linewidth=2, label="learned")
        ax.set_title(f"{split.title()} random directions")
        ax.set_xlabel("Mask AUROC")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=9)
    fig.suptitle("Random Direction Baseline vs Learned Direction", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_probe_specs(
    single_probes_root: Path,
    model_dir: str,
    source_datasets: Sequence[str],
    target_dataset: str,
    segment: str,
    pooling: str,
    layer: int,
    include_target_self: bool,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for dataset_base in source_datasets:
        probe_path = single_probe_path(single_probes_root, model_dir, dataset_base, segment, pooling, layer)
        specs.append(
            {
                "probe_name": dataset_base,
                "probe_kind": "single_source",
                "source_dataset": dataset_base,
                "probe_path": str(probe_path),
            }
        )
    if include_target_self:
        probe_path = single_probe_path(single_probes_root, model_dir, target_dataset, segment, pooling, layer)
        specs.append(
            {
                "probe_name": f"{target_dataset}_self",
                "probe_kind": "target_self",
                "source_dataset": target_dataset,
                "probe_path": str(probe_path),
            }
        )
    return specs


def build_evaluation_probe_specs(
    *,
    single_probes_root: Path,
    model_dir: str,
    evaluation_single_datasets: Sequence[str],
    target_dataset: str,
    segment: str,
    pooling: str,
    layer: int,
    include_target_self: bool,
    combined_run_root: str,
    combined_probe_names: Sequence[str],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = build_probe_specs(
        single_probes_root=single_probes_root,
        model_dir=model_dir,
        source_datasets=evaluation_single_datasets,
        target_dataset=target_dataset,
        segment=segment,
        pooling=pooling,
        layer=layer,
        include_target_self=include_target_self,
    )

    if combined_run_root:
        combined_root = Path(combined_run_root)
        for combo_name in combined_probe_names:
            specs.append(
                {
                    "probe_name": combo_name,
                    "probe_kind": "combined",
                    "source_dataset": combo_name,
                    "probe_path": str(combined_probe_path(combined_root, combo_name, pooling, layer)),
                }
            )

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        name = str(spec["probe_name"])
        if name in seen:
            continue
        seen.add(name)
        deduped.append(spec)
    return deduped


def evaluate_probe_collection(
    x: np.ndarray,
    y: np.ndarray,
    probe_specs: Sequence[Dict[str, Any]],
    weights: Dict[str, Tuple[np.ndarray, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for spec in probe_specs:
        w, b = weights[spec["probe_name"]]
        out[spec["probe_name"]] = evaluate_probe_logits(x, y, w, b)
    return out


def metrics_to_rows(
    metrics_by_probe: Dict[str, Dict[str, float]],
    probe_specs: Sequence[Dict[str, Any]],
    split: str,
    method: str,
    baseline_by_probe_split: Dict[Tuple[str, str], Dict[str, float]],
    optimization_probe_names: Optional[Sequence[str]] = None,
    self_probe_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    optimization_probe_name_set = set(str(x) for x in (optimization_probe_names or []))
    for spec in probe_specs:
        probe_name = spec["probe_name"]
        metrics = metrics_by_probe[probe_name]
        baseline = baseline_by_probe_split[(probe_name, split)]
        row = {
            "timestamp": utc_now_iso(),
            "probe_name": probe_name,
            "probe_kind": spec["probe_kind"],
            "source_dataset": spec["source_dataset"],
            "probe_path": spec["probe_path"],
            "split": split,
            "method": method,
            "is_optimization_probe": bool(probe_name in optimization_probe_name_set),
            "is_target_self_probe": bool(self_probe_name is not None and probe_name == self_probe_name),
            "is_combined_probe": bool(str(spec["probe_kind"]) == "combined"),
            "is_heldout_probe": bool(
                probe_name not in optimization_probe_name_set
                and (self_probe_name is None or probe_name != self_probe_name)
            ),
            "auc": float(metrics["auc"]),
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "count": int(metrics["count"]),
            "logit_mean": float(metrics["logit_mean"]),
            "logit_std": float(metrics["logit_std"]),
            "delta_auc_vs_baseline": float(metrics["auc"] - baseline["auc"]),
            "delta_accuracy_vs_baseline": float(metrics["accuracy"] - baseline["accuracy"]),
            "delta_f1_vs_baseline": float(metrics["f1"] - baseline["f1"]),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def source_probe_names_from_specs(probe_specs: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(spec["probe_name"]) for spec in probe_specs if str(spec["probe_kind"]) == "single_source"]


def target_self_probe_name(probe_specs: Sequence[Dict[str, Any]]) -> Optional[str]:
    for spec in probe_specs:
        if str(spec["probe_kind"]) == "target_self":
            return str(spec["probe_name"])
    return None


def aggregate_source_auc(metrics_by_probe: Dict[str, Dict[str, float]], source_probe_names: Sequence[str]) -> float:
    if not source_probe_names:
        return 0.5
    vals = [float(metrics_by_probe[name]["auc"]) for name in source_probe_names]
    return float(np.mean(vals))


def selection_score(
    avg_source_auc: float,
    self_auc: Optional[float],
    self_baseline_auc: Optional[float],
    self_auc_drop_tolerance: float,
    infeasible_penalty: float,
) -> Tuple[bool, float, float]:
    if self_auc is None or self_baseline_auc is None:
        return True, avg_source_auc, 0.0
    drop = float(self_baseline_auc - self_auc)
    feasible = drop <= float(self_auc_drop_tolerance)
    if feasible:
        return True, avg_source_auc, drop
    score = avg_source_auc - float(infeasible_penalty) * max(0.0, drop - float(self_auc_drop_tolerance))
    return False, score, drop


def summarize_checkpoint_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    return {
        "epoch": int(payload["epoch"]),
        "alpha": float(payload["alpha"]),
        "alpha_logit": float(payload["alpha_logit"]),
        "val_avg_source_auc": float(payload["val_avg_source_auc"]),
        "val_self_auc": None if payload["val_self_auc"] is None else float(payload["val_self_auc"]),
        "val_self_drop": float(payload["val_self_drop"]),
        "selection_score": float(payload["selection_score"]),
        "feasible": bool(payload["feasible"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen probes after learned target-side rank-1 removal.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--single_probes_root", type=str, default="data/probes")
    parser.add_argument("--target_dataset", type=str, default="Deception-Roleplaying")
    parser.add_argument("--segment", type=str, choices=["completion", "full"], default="completion")
    parser.add_argument("--pooling", type=str, choices=SUPPORTED_POOLINGS, default="mean")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--source_datasets", type=str, default="Deception-Mask")
    parser.add_argument("--optimization_source_datasets", type=str, default="")
    parser.add_argument("--evaluation_single_datasets", type=str, default="")
    parser.add_argument("--include_target_self", action="store_true")
    parser.add_argument("--combined_run_root", type=str, default="")
    parser.add_argument("--combined_probe_names", type=str, default="")
    parser.add_argument("--target_train_split", type=str, default="train")
    parser.add_argument("--target_val_split", type=str, default="validation")
    parser.add_argument("--target_test_split", type=str, default="test")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--mirror_results_root", type=str, default="")
    parser.add_argument("--gallery_root", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--alpha_init", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--lambda_self", type=float, default=1.0)
    parser.add_argument("--lambda_dist", type=float, default=1e-3)
    parser.add_argument("--self_loss_tolerance", type=float, default=0.0)
    parser.add_argument("--self_auc_drop_tolerance", type=float, default=0.01)
    parser.add_argument("--infeasible_penalty", type=float, default=5.0)

    parser.add_argument("--pca_solver", type=str, default="randomized")
    parser.add_argument("--random_directions", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_datasets = parse_csv_list(args.source_datasets)
    optimization_source_datasets = parse_csv_list(args.optimization_source_datasets) or source_datasets
    evaluation_single_datasets = parse_csv_list(args.evaluation_single_datasets) or source_datasets
    combined_probe_names = parse_csv_list(args.combined_probe_names)
    model_dir = model_dir_name(args.model)
    target_segment = dataset_segment_name(args.target_dataset, args.segment)
    source_slug = (
        "-".join(slugify_dataset_base(ds) for ds in optimization_source_datasets)
        if optimization_source_datasets
        else "no-source"
    )
    probe_set_slug = f"frozen-{source_slug}" + ("-self" if args.include_target_self else "")
    run_id = args.run_id.strip() or (
        f"{default_run_id()}-{args.target_dataset.split('-')[-1].lower()}-{args.segment}-{args.pooling}-l{args.layer}-rank1-v1"
    )

    artifact_root = Path(args.artifact_root)
    run_root = (
        artifact_root
        / "runs"
        / "target_rank1_probe_transfer"
        / model_dir
        / target_segment
        / probe_set_slug
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

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "target_rank1_probe_transfer"
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
    training_ckpt_path = checkpoints_dir / "training_state.pt"
    training_history_path = results_dir / "training_history.jsonl"
    metrics_long_path = results_dir / "metrics_long.csv"
    random_metrics_path = results_dir / "random_direction_metrics.csv"
    summary_path = results_dir / "summary.json"

    optimization_probe_specs = build_probe_specs(
        single_probes_root=Path(args.single_probes_root),
        model_dir=model_dir,
        source_datasets=optimization_source_datasets,
        target_dataset=args.target_dataset,
        segment=args.segment,
        pooling=args.pooling,
        layer=args.layer,
        include_target_self=False,
    )
    evaluation_probe_specs = build_evaluation_probe_specs(
        single_probes_root=Path(args.single_probes_root),
        model_dir=model_dir,
        evaluation_single_datasets=evaluation_single_datasets,
        target_dataset=args.target_dataset,
        segment=args.segment,
        pooling=args.pooling,
        layer=args.layer,
        include_target_self=args.include_target_self,
        combined_run_root=args.combined_run_root,
        combined_probe_names=combined_probe_names,
    )
    source_probe_names = source_probe_names_from_specs(optimization_probe_specs)
    self_probe_name = target_self_probe_name(evaluation_probe_specs)
    heldout_probe_names = [
        str(spec["probe_name"])
        for spec in evaluation_probe_specs
        if str(spec["probe_name"]) not in set(source_probe_names)
        and str(spec["probe_name"]) != str(self_probe_name)
    ]

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
            "source_datasets": source_datasets,
            "optimization_source_datasets": optimization_source_datasets,
            "evaluation_single_datasets": evaluation_single_datasets,
            "include_target_self": bool(args.include_target_self),
            "combined_run_root": args.combined_run_root,
            "combined_probe_names": combined_probe_names,
            "probe_set": probe_set_slug,
            "variant": "learned-rank1",
            "optimization": {
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "patience": int(args.patience),
                "alpha_init": float(args.alpha_init),
                "alpha_max": float(args.alpha_max),
                "lambda_self": float(args.lambda_self),
                "lambda_dist": float(args.lambda_dist),
                "self_loss_tolerance": float(args.self_loss_tolerance),
                "self_auc_drop_tolerance": float(args.self_auc_drop_tolerance),
                "infeasible_penalty": float(args.infeasible_penalty),
            },
            "random_direction_count": int(args.random_directions),
            "paths": {
                "run_root": str(run_root),
                "inputs_dir": str(inputs_dir),
                "results_dir": str(results_dir),
                "mirror_results_root": str(external_run_root),
                "gallery_root": str(gallery_root),
            },
        },
    )

    prior_status = read_json(status_path, default={})
    if args.resume and prior_status.get("state") == "completed" and summary_path.exists():
        print(f"[skip] run already completed -> {run_root}")
        return 0

    write_json(status_path, {"state": "running", "message": "starting run", "updated_at": utc_now_iso()})
    append_log_line(log_path, f"[start] run_id={run_id}")
    append_log_line(
        log_path,
        (
            f"[target] {target_segment} pooling={args.pooling} layer={args.layer} "
            f"opt_sources={optimization_source_datasets} eval_singles={evaluation_single_datasets} "
            f"combined={combined_probe_names} include_target_self={args.include_target_self}"
        ),
    )

    progress = read_json(progress_path, default={})
    completed_units = set(progress.get("completed_units", []))

    activations_root = Path(args.activations_root)
    target_root = activations_root / model_dir / target_segment
    train_dir = target_root / args.target_train_split
    val_dir = target_root / args.target_val_split
    test_dir = target_root / args.target_test_split

    append_log_line(log_path, "[stage] loading pooled activations")
    x_train, y_train, train_ids = load_pooled_split(train_dir, args.layer, args.pooling, f"{target_segment} train")
    x_val, y_val, val_ids = load_pooled_split(val_dir, args.layer, args.pooling, f"{target_segment} validation")
    x_test, y_test, test_ids = load_pooled_split(test_dir, args.layer, args.pooling, f"{target_segment} test")
    save_npz(
        inputs_dir / "target_split_shapes.npz",
        {
            "train_shape": np.asarray(x_train.shape, dtype=np.int64),
            "val_shape": np.asarray(x_val.shape, dtype=np.int64),
            "test_shape": np.asarray(x_test.shape, dtype=np.int64),
        },
    )
    write_json(
        inputs_dir / "target_stats.json",
        {
            "target_dataset": target_segment,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_test": int(x_test.shape[0]),
            "dim": int(x_train.shape[1]),
            "train_label_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "val_label_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "test_label_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
            "sample_id_counts": {
                "train": len(train_ids),
                "validation": len(val_ids),
                "test": len(test_ids),
            },
        },
    )

    append_log_line(log_path, "[stage] loading frozen probes")
    probe_weights: Dict[str, Tuple[np.ndarray, float]] = {}
    probe_manifest_rows: List[Dict[str, Any]] = []
    optimization_probe_name_set = set(source_probe_names)
    heldout_probe_name_set = set(heldout_probe_names)
    for spec in evaluation_probe_specs:
        w, b = load_linear_probe(Path(spec["probe_path"]))
        if w.shape[0] != x_train.shape[1]:
            raise ValueError(
                f"Probe dim mismatch for {spec['probe_name']}: probe={w.shape[0]} features={x_train.shape[1]}"
            )
        probe_weights[spec["probe_name"]] = (w, b)
        probe_manifest_rows.append(
            {
                "probe_name": spec["probe_name"],
                "probe_kind": spec["probe_kind"],
                "source_dataset": spec["source_dataset"],
                "probe_path": spec["probe_path"],
                "dim": int(w.shape[0]),
                "is_optimization_probe": bool(spec["probe_name"] in optimization_probe_name_set),
                "is_target_self_probe": bool(spec["probe_name"] == self_probe_name),
                "is_heldout_probe": bool(spec["probe_name"] in heldout_probe_name_set),
            }
        )
    write_csv(
        inputs_dir / "probe_manifest.csv",
        probe_manifest_rows,
        ["probe_name", "probe_kind", "source_dataset", "probe_path", "dim", "is_optimization_probe", "is_target_self_probe", "is_heldout_probe"],
    )

    append_log_line(log_path, "[stage] fitting PCA-PC1 baseline")
    pca_artifact = fit_pca_pc1(x_train, args.pca_solver, args.seed)
    save_npz(inputs_dir / "pca_pc1.npz", pca_artifact)

    baseline_by_probe_split: Dict[Tuple[str, str], Dict[str, float]] = {}
    baseline_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []

    append_log_line(log_path, "[stage] evaluating no-transform baseline")
    baseline_val = evaluate_probe_collection(x_val, y_val, evaluation_probe_specs, probe_weights)
    baseline_test = evaluate_probe_collection(x_test, y_test, evaluation_probe_specs, probe_weights)
    for split_name, metrics_by_probe in [("validation", baseline_val), ("test", baseline_test)]:
        for spec in evaluation_probe_specs:
            probe_name = spec["probe_name"]
            baseline_by_probe_split[(probe_name, split_name)] = metrics_by_probe[probe_name]
        rows = metrics_to_rows(
            metrics_by_probe=metrics_by_probe,
            probe_specs=evaluation_probe_specs,
            split=split_name,
            method="baseline",
            baseline_by_probe_split=baseline_by_probe_split,
            optimization_probe_names=source_probe_names,
            self_probe_name=self_probe_name,
            extra={"alpha": 0.0, "direction_label": "none"},
        )
        baseline_rows.extend(rows)
        metrics_rows.extend(rows)

    write_csv(
        results_dir / "baseline_metrics.csv",
        baseline_rows,
        [
            "timestamp",
            "probe_name",
            "probe_kind",
            "source_dataset",
            "probe_path",
            "split",
            "method",
            "is_optimization_probe",
            "is_target_self_probe",
            "is_combined_probe",
            "is_heldout_probe",
            "auc",
            "accuracy",
            "f1",
            "count",
            "logit_mean",
            "logit_std",
            "delta_auc_vs_baseline",
            "delta_accuracy_vs_baseline",
            "delta_f1_vs_baseline",
            "alpha",
            "direction_label",
        ],
    )

    append_log_line(log_path, "[stage] evaluating PCA-PC1 baseline")
    x_val_pca = remove_specific_pcs(x_val, pca_artifact["mean"], pca_artifact["components"], [1])
    x_test_pca = remove_specific_pcs(x_test, pca_artifact["mean"], pca_artifact["components"], [1])
    pca_val = evaluate_probe_collection(x_val_pca, y_val, evaluation_probe_specs, probe_weights)
    pca_test = evaluate_probe_collection(x_test_pca, y_test, evaluation_probe_specs, probe_weights)
    for split_name, metrics_by_probe in [("validation", pca_val), ("test", pca_test)]:
        metrics_rows.extend(
            metrics_to_rows(
                metrics_by_probe=metrics_by_probe,
                probe_specs=evaluation_probe_specs,
                split=split_name,
                method="pca_pc1",
                baseline_by_probe_split=baseline_by_probe_split,
                optimization_probe_names=source_probe_names,
                self_probe_name=self_probe_name,
                extra={"alpha": 1.0, "direction_label": "pc1"},
            )
        )

    append_log_line(log_path, "[stage] learning rank-1 direction")
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    train_probe_tensors = {
        name: (
            torch.from_numpy(probe_weights[name][0]),
            torch.tensor(probe_weights[name][1], dtype=torch.float32),
        )
        for name in probe_weights
    }

    source_probe_train_baselines: Dict[str, float] = {}
    for name in source_probe_names:
        w_t, b_t = train_probe_tensors[name]
        logits = x_train_t @ w_t + b_t
        source_probe_train_baselines[name] = float(
            F.binary_cross_entropy_with_logits(logits, y_train_t).detach().cpu().item()
        )

    baseline_self_train_loss = None
    baseline_self_val_auc = None
    if self_probe_name:
        w_t, b_t = train_probe_tensors[self_probe_name]
        logits = x_train_t @ w_t + b_t
        baseline_self_train_loss = float(
            F.binary_cross_entropy_with_logits(logits, y_train_t).detach().cpu().item()
        )
        baseline_self_val_auc = float(baseline_val[self_probe_name]["auc"])

    resume_state: Dict[str, Any] = {}
    if args.resume and training_ckpt_path.exists():
        resume_state = load_trusted_torch_checkpoint(training_ckpt_path)
        append_log_line(log_path, f"[resume] loaded training checkpoint epoch={resume_state.get('epoch', 0)}")

    start_epoch = int(resume_state.get("epoch", 0)) + 1 if resume_state else 1
    u = torch.nn.Parameter(
        torch.from_numpy(
            resume_state.get(
                "u",
                np.random.default_rng(args.seed).normal(size=x_train.shape[1]).astype(np.float32),
            )
        )
    )
    alpha_init = float(np.clip(args.alpha_init, 1e-4, 1.0 - 1e-4))
    alpha_logit_init = float(np.log(alpha_init / (1.0 - alpha_init)))
    alpha_logit = torch.nn.Parameter(
        torch.tensor(float(resume_state.get("alpha_logit", alpha_logit_init)), dtype=torch.float32)
    )

    optimizer = torch.optim.Adam([u, alpha_logit], lr=args.lr, weight_decay=args.weight_decay)
    if resume_state.get("optimizer_state"):
        optimizer.load_state_dict(resume_state["optimizer_state"])

    best_feasible = resume_state.get("best_feasible")
    best_any = resume_state.get("best_any")
    patience_counter = int(resume_state.get("patience_counter", 0))
    best_score_for_patience = float(resume_state.get("best_score_for_patience", -1e18))
    history_rows: List[Dict[str, Any]] = []
    if training_history_path.exists():
        with training_history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history_rows.append(json.loads(line))

    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad()
        direction = normalize_torch(u)
        alpha = torch.sigmoid(alpha_logit) * float(args.alpha_max)
        x_train_trans = apply_rank1_transform_torch(x_train_t, direction, alpha)

        source_losses: List[torch.Tensor] = []
        for name in source_probe_names:
            w_t, b_t = train_probe_tensors[name]
            logits = x_train_trans @ w_t + b_t
            source_losses.append(F.binary_cross_entropy_with_logits(logits, y_train_t))
        if not source_losses:
            raise ValueError("At least one source probe is required")
        train_source_loss = torch.stack(source_losses).mean()

        if self_probe_name:
            w_t, b_t = train_probe_tensors[self_probe_name]
            self_logits = x_train_trans @ w_t + b_t
            self_loss = F.binary_cross_entropy_with_logits(self_logits, y_train_t)
            self_excess = F.relu(self_loss - float(baseline_self_train_loss) - float(args.self_loss_tolerance))
        else:
            self_loss = torch.tensor(0.0, dtype=torch.float32)
            self_excess = torch.tensor(0.0, dtype=torch.float32)

        dist_loss = torch.mean((x_train_trans - x_train_t) ** 2)
        total_loss = train_source_loss + float(args.lambda_self) * self_excess + float(args.lambda_dist) * dist_loss
        total_loss.backward()
        optimizer.step()

        direction_np = normalize_np(u.detach().cpu().numpy())
        alpha_np = float(torch.sigmoid(alpha_logit).detach().cpu().item() * float(args.alpha_max))
        x_val_trans = apply_rank1_transform_np(x_val, direction_np, alpha_np)
        val_metrics = evaluate_probe_collection(x_val_trans, y_val, evaluation_probe_specs, probe_weights)
        val_avg_source_auc = aggregate_source_auc(val_metrics, source_probe_names)
        val_self_auc = float(val_metrics[self_probe_name]["auc"]) if self_probe_name else None
        feasible, score, self_drop = selection_score(
            avg_source_auc=val_avg_source_auc,
            self_auc=val_self_auc,
            self_baseline_auc=baseline_self_val_auc,
            self_auc_drop_tolerance=float(args.self_auc_drop_tolerance),
            infeasible_penalty=float(args.infeasible_penalty),
        )

        record = {
            "epoch": int(epoch),
            "timestamp": utc_now_iso(),
            "train_total_loss": float(total_loss.detach().cpu().item()),
            "train_avg_source_loss": float(train_source_loss.detach().cpu().item()),
            "train_self_loss": float(self_loss.detach().cpu().item()),
            "train_self_excess": float(self_excess.detach().cpu().item()),
            "train_dist_loss": float(dist_loss.detach().cpu().item()),
            "alpha": alpha_np,
            "val_avg_source_auc": float(val_avg_source_auc),
            "val_self_auc": None if val_self_auc is None else float(val_self_auc),
            "val_self_drop": float(self_drop),
            "selection_score": float(score),
            "feasible": bool(feasible),
        }
        history_rows.append(record)
        append_jsonl(training_history_path, record)

        if feasible and (best_feasible is None or float(val_avg_source_auc) > float(best_feasible["val_avg_source_auc"])):
            best_feasible = {
                "epoch": int(epoch),
                "u": u.detach().cpu().numpy(),
                "direction": direction_np,
                "alpha": alpha_np,
                "alpha_logit": float(alpha_logit.detach().cpu().item()),
                "val_avg_source_auc": float(val_avg_source_auc),
                "val_self_auc": None if val_self_auc is None else float(val_self_auc),
                "val_self_drop": float(self_drop),
                "selection_score": float(score),
                "feasible": True,
            }

        if best_any is None or float(score) > float(best_any["selection_score"]):
            best_any = {
                "epoch": int(epoch),
                "u": u.detach().cpu().numpy(),
                "direction": direction_np,
                "alpha": alpha_np,
                "alpha_logit": float(alpha_logit.detach().cpu().item()),
                "val_avg_source_auc": float(val_avg_source_auc),
                "val_self_auc": None if val_self_auc is None else float(val_self_auc),
                "val_self_drop": float(self_drop),
                "selection_score": float(score),
                "feasible": bool(feasible),
            }

        if float(score) > best_score_for_patience:
            best_score_for_patience = float(score)
            patience_counter = 0
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": int(epoch),
                "u": u.detach().cpu().numpy(),
                "alpha_logit": float(alpha_logit.detach().cpu().item()),
                "optimizer_state": optimizer.state_dict(),
                "best_feasible": best_feasible,
                "best_any": best_any,
                "patience_counter": int(patience_counter),
                "best_score_for_patience": float(best_score_for_patience),
            },
            str(training_ckpt_path),
        )
        write_json(
            progress_path,
            {
                "updated_at": utc_now_iso(),
                "completed_units": sorted(completed_units),
                "train_epoch_complete": int(epoch),
                "best_feasible_epoch": None if best_feasible is None else int(best_feasible["epoch"]),
                "best_any_epoch": None if best_any is None else int(best_any["epoch"]),
            },
        )
        append_log_line(
            log_path,
            f"[epoch {epoch}] train_total={record['train_total_loss']:.4f} val_source_auc={record['val_avg_source_auc']:.4f} val_self_drop={record['val_self_drop']:.4f} alpha={alpha_np:.4f} feasible={feasible}",
        )

        if patience_counter >= int(args.patience):
            append_log_line(log_path, f"[early-stop] epoch={epoch} patience={args.patience}")
            break

    chosen = best_feasible if best_feasible is not None else best_any
    if chosen is None:
        raise RuntimeError("Failed to learn any direction")

    chosen_direction = normalize_np(np.asarray(chosen["direction"], dtype=np.float32))
    chosen_alpha = float(chosen["alpha"])
    save_npz(
        results_dir / "learned_direction.npz",
        {
            "direction": chosen_direction,
            "alpha": np.asarray([chosen_alpha], dtype=np.float32),
            "epoch": np.asarray([int(chosen["epoch"])], dtype=np.int64),
        },
    )
    write_json(
        results_dir / "selection_summary.json",
        {
            "selected": summarize_checkpoint_payload(chosen),
            "used_fallback": bool(best_feasible is None),
            "best_feasible": summarize_checkpoint_payload(best_feasible),
            "best_any": summarize_checkpoint_payload(best_any),
            "artifacts": {
                "learned_direction_npz": str(results_dir / "learned_direction.npz"),
                "training_checkpoint_pt": str(training_ckpt_path),
            },
        },
    )

    append_log_line(log_path, "[stage] evaluating learned direction")
    x_val_learned = apply_rank1_transform_np(x_val, chosen_direction, chosen_alpha)
    x_test_learned = apply_rank1_transform_np(x_test, chosen_direction, chosen_alpha)
    learned_val = evaluate_probe_collection(x_val_learned, y_val, evaluation_probe_specs, probe_weights)
    learned_test = evaluate_probe_collection(x_test_learned, y_test, evaluation_probe_specs, probe_weights)
    for split_name, metrics_by_probe in [("validation", learned_val), ("test", learned_test)]:
        metrics_rows.extend(
            metrics_to_rows(
                metrics_by_probe=metrics_by_probe,
                probe_specs=evaluation_probe_specs,
                split=split_name,
                method="learned",
                baseline_by_probe_split=baseline_by_probe_split,
                optimization_probe_names=source_probe_names,
                self_probe_name=self_probe_name,
                extra={"alpha": chosen_alpha, "direction_label": "learned_rank1", "selected_epoch": int(chosen["epoch"])},
            )
        )

    append_log_line(log_path, "[stage] evaluating random directions")
    random_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 17)
    random_dirs = rng.normal(size=(int(args.random_directions), x_train.shape[1])).astype(np.float32)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-8
    save_npz(results_dir / "random_directions.npz", {"directions": random_dirs.astype(np.float32)})

    random_val_source_scores: List[Tuple[int, float]] = []
    for idx, direction in enumerate(random_dirs):
        label = f"random_{idx:03d}"
        x_val_rand = apply_rank1_transform_np(x_val, direction, chosen_alpha)
        x_test_rand = apply_rank1_transform_np(x_test, direction, chosen_alpha)
        val_metrics = evaluate_probe_collection(x_val_rand, y_val, evaluation_probe_specs, probe_weights)
        test_metrics = evaluate_probe_collection(x_test_rand, y_test, evaluation_probe_specs, probe_weights)
        val_avg_source_auc = aggregate_source_auc(val_metrics, source_probe_names)
        random_val_source_scores.append((idx, val_avg_source_auc))
        for split_name, metrics_by_probe in [("validation", val_metrics), ("test", test_metrics)]:
            rows = metrics_to_rows(
                metrics_by_probe=metrics_by_probe,
                probe_specs=evaluation_probe_specs,
                split=split_name,
                method=label,
                baseline_by_probe_split=baseline_by_probe_split,
                optimization_probe_names=source_probe_names,
                self_probe_name=self_probe_name,
                extra={"alpha": chosen_alpha, "direction_label": label, "direction_index": int(idx)},
            )
            random_rows.extend(rows)

    write_csv(
        random_metrics_path,
        random_rows,
        [
            "timestamp",
            "probe_name",
            "probe_kind",
            "source_dataset",
            "probe_path",
            "split",
            "method",
            "auc",
            "accuracy",
            "f1",
            "count",
            "logit_mean",
            "logit_std",
            "delta_auc_vs_baseline",
            "delta_accuracy_vs_baseline",
            "delta_f1_vs_baseline",
            "alpha",
            "direction_label",
            "direction_index",
        ],
    )

    random_best_idx = max(random_val_source_scores, key=lambda item: item[1])[0]
    random_best_rows = [r for r in random_rows if str(r["method"]) == f"random_{random_best_idx:03d}"]

    random_mean_rows: List[Dict[str, Any]] = []
    for split_name in ["validation", "test"]:
        for spec in evaluation_probe_specs:
            probe_name = spec["probe_name"]
            probe_rows = [
                r for r in random_rows if str(r["split"]) == split_name and str(r["probe_name"]) == probe_name
            ]
            aucs = [float(r["auc"]) for r in probe_rows]
            accs = [float(r["accuracy"]) for r in probe_rows]
            f1s = [float(r["f1"]) for r in probe_rows]
            logit_means = [float(r["logit_mean"]) for r in probe_rows]
            logit_stds = [float(r["logit_std"]) for r in probe_rows]
            baseline = baseline_by_probe_split[(probe_name, split_name)]
            random_mean_rows.append(
                {
                    "timestamp": utc_now_iso(),
                    "probe_name": probe_name,
                    "probe_kind": spec["probe_kind"],
                    "source_dataset": spec["source_dataset"],
                    "probe_path": spec["probe_path"],
                    "split": split_name,
                    "method": "random_mean",
                    "is_optimization_probe": bool(probe_name in optimization_probe_name_set),
                    "is_target_self_probe": bool(self_probe_name is not None and probe_name == self_probe_name),
                    "is_combined_probe": bool(str(spec["probe_kind"]) == "combined"),
                    "is_heldout_probe": bool(probe_name in heldout_probe_name_set),
                    "auc": float(np.mean(aucs)),
                    "accuracy": float(np.mean(accs)),
                    "f1": float(np.mean(f1s)),
                    "count": int(probe_rows[0]["count"]) if probe_rows else 0,
                    "logit_mean": float(np.mean(logit_means)),
                    "logit_std": float(np.mean(logit_stds)),
                    "delta_auc_vs_baseline": float(np.mean(aucs) - baseline["auc"]),
                    "delta_accuracy_vs_baseline": float(np.mean(accs) - baseline["accuracy"]),
                    "delta_f1_vs_baseline": float(np.mean(f1s) - baseline["f1"]),
                    "alpha": chosen_alpha,
                    "direction_label": "random_mean",
                }
            )

    random_best_summary_rows: List[Dict[str, Any]] = []
    for row in random_best_rows:
        row = dict(row)
        row["method"] = "random_best"
        row["direction_label"] = f"random_{random_best_idx:03d}"
        random_best_summary_rows.append(row)

    metrics_rows.extend(random_mean_rows)
    metrics_rows.extend(random_best_summary_rows)

    fieldnames = [
        "timestamp",
        "probe_name",
        "probe_kind",
        "source_dataset",
        "probe_path",
        "split",
        "method",
        "is_optimization_probe",
        "is_target_self_probe",
        "is_combined_probe",
        "is_heldout_probe",
        "auc",
        "accuracy",
        "f1",
        "count",
        "logit_mean",
        "logit_std",
        "delta_auc_vs_baseline",
        "delta_accuracy_vs_baseline",
        "delta_f1_vs_baseline",
        "alpha",
        "direction_label",
        "selected_epoch",
    ]
    metrics_rows.sort(key=lambda r: (str(r["probe_name"]), str(r["split"]), str(r["method"])))
    write_csv(metrics_long_path, metrics_rows, fieldnames)

    random_summary_rows: List[Dict[str, Any]] = []
    for split_name in ["validation", "test"]:
        for spec in evaluation_probe_specs:
            probe_name = spec["probe_name"]
            probe_rows = [
                r for r in random_rows if str(r["split"]) == split_name and str(r["probe_name"]) == probe_name
            ]
            aucs = [float(r["auc"]) for r in probe_rows]
            random_summary_rows.append(
                {
                    "split": split_name,
                    "probe_name": probe_name,
                    "random_auc_mean": float(np.mean(aucs)),
                    "random_auc_std": float(np.std(aucs)),
                    "random_auc_best": float(np.max(aucs)),
                    "random_best_direction_index": int(random_best_idx),
                    "learned_auc": float(
                        next(
                            r["auc"]
                            for r in metrics_rows
                            if str(r["method"]) == "learned"
                            and str(r["split"]) == split_name
                            and str(r["probe_name"]) == probe_name
                        )
                    ),
                }
            )
    write_csv(
        results_dir / "random_baseline_summary.csv",
        random_summary_rows,
        ["split", "probe_name", "random_auc_mean", "random_auc_std", "random_auc_best", "random_best_direction_index", "learned_auc"],
    )

    append_log_line(log_path, "[stage] generating plots")
    probe_order = [str(spec["probe_name"]) for spec in evaluation_probe_specs]
    method_order = ["baseline", "pca_pc1", "random_mean", "random_best", "learned"]

    test_auc_plot = plots_dir / "test_auc_method_comparison.png"
    test_delta_plot = plots_dir / "test_delta_method_comparison.png"
    train_curves_plot = plots_dir / "training_curves.png"
    random_plot = plots_dir / "random_baseline_comparison.png"

    plot_grouped_bar(
        rows=metrics_rows,
        split="test",
        value_key="auc",
        title=f"{target_segment}: baseline vs learned rank-1 test AUROC",
        out_path=test_auc_plot,
        method_order=method_order,
        probe_order=probe_order,
        y_label="AUROC",
    )
    plot_grouped_bar(
        rows=metrics_rows,
        split="test",
        value_key="delta_auc_vs_baseline",
        title=f"{target_segment}: test AUROC delta vs baseline",
        out_path=test_delta_plot,
        method_order=method_order,
        probe_order=probe_order,
        y_label="Delta AUROC",
    )
    plot_training_curves(history_rows, train_curves_plot, self_probe_present=bool(self_probe_name))

    learned_metrics_by_split = {
        split: {
            "auc": next(
                float(r["auc"])
                for r in metrics_rows
                if str(r["method"]) == "learned"
                and str(r["split"]) == split
                and str(r["probe_name"]) == source_probe_names[0]
            )
        }
        for split in ["validation", "test"]
    }
    pca_metrics_by_split = {
        split: {
            "auc": next(
                float(r["auc"])
                for r in metrics_rows
                if str(r["method"]) == "pca_pc1"
                and str(r["split"]) == split
                and str(r["probe_name"]) == source_probe_names[0]
            )
        }
        for split in ["validation", "test"]
    }
    baseline_metrics_by_split = {
        split: {
            "auc": float(baseline_by_probe_split[(source_probe_names[0], split)]["auc"])
        }
        for split in ["validation", "test"]
    }
    plot_random_distribution(
        random_rows=random_rows,
        learned_metrics=learned_metrics_by_split,
        pca_metrics=pca_metrics_by_split,
        baseline_metrics=baseline_metrics_by_split,
        source_probe_name=source_probe_names[0],
        out_path=random_plot,
    )

    gallery_paths: List[str] = []
    gallery_prefix = f"{slugify_dataset_base(args.target_dataset)}_{args.segment}_{args.pooling}_l{args.layer}_rank1_{source_slug}"
    for src, name in [
        (test_auc_plot, f"{gallery_prefix}_test_auc_method_comparison.png"),
        (test_delta_plot, f"{gallery_prefix}_test_delta_method_comparison.png"),
        (train_curves_plot, f"{gallery_prefix}_training_curves.png"),
        (random_plot, f"{gallery_prefix}_random_baseline_comparison.png"),
    ]:
        mirrored = mirror_png(src, gallery_root / name)
        if mirrored:
            gallery_paths.append(mirrored)

    summary = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "git_commit": get_git_commit(),
        "model": args.model,
        "target_dataset": target_segment,
        "pooling": args.pooling,
        "layer": int(args.layer),
        "source_datasets": source_datasets,
        "optimization_source_datasets": optimization_source_datasets,
        "evaluation_single_datasets": evaluation_single_datasets,
        "combined_probe_names": combined_probe_names,
        "optimization_probe_names": source_probe_names,
        "evaluation_probe_names": [str(spec["probe_name"]) for spec in evaluation_probe_specs],
        "heldout_probe_names": heldout_probe_names,
        "include_target_self": bool(args.include_target_self),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "dim": int(x_train.shape[1]),
        "selected_epoch": int(chosen["epoch"]),
        "selected_alpha": float(chosen_alpha),
        "selected_feasible": bool(chosen["feasible"]) if "feasible" in chosen else bool(best_feasible is not None),
        "random_best_direction_index": int(random_best_idx),
        "outputs": {
            "metrics_long_csv": str(metrics_long_path),
            "baseline_metrics_csv": str(results_dir / "baseline_metrics.csv"),
            "random_direction_metrics_csv": str(random_metrics_path),
            "random_baseline_summary_csv": str(results_dir / "random_baseline_summary.csv"),
            "selection_summary_json": str(results_dir / "selection_summary.json"),
            "learned_direction_npz": str(results_dir / "learned_direction.npz"),
            "training_history_jsonl": str(training_history_path),
            "test_auc_method_comparison_png": str(test_auc_plot),
            "test_delta_method_comparison_png": str(test_delta_plot),
            "training_curves_png": str(train_curves_plot),
            "random_baseline_comparison_png": str(random_plot),
        },
        "gallery_pngs": gallery_paths,
        "mirror_results_root": str(external_run_root),
    }
    write_json(summary_path, summary)

    completed_units.update({"baseline_eval", "pca_eval", "learn_direction", "random_baselines", "plots", "summary"})
    write_json(
        progress_path,
        {
            "updated_at": utc_now_iso(),
            "completed_units": sorted(completed_units),
            "train_epoch_complete": int(chosen["epoch"]),
        },
    )
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
