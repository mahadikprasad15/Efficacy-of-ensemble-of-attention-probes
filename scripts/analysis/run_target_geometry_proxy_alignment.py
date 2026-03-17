#!/usr/bin/env python3
"""
Build fixed-config target-geometry proxy alignment matrices.

For each selected segment, this script:
1. Reads the existing fixed `mean@L15` transfer AUROC matrix.
2. Computes train-split dataset statistics at the same fixed config:
   - class means
   - class-mean difference
   - within-class covariance
   - inverse covariance
   - source-side LDA proxy direction
3. Loads fixed trained probe directions.
4. Builds three source->target alignment matrices under target inverse covariance:
   - Score 1: trained probe vs trained target probe
   - Score 2: source LDA proxy vs trained target probe
   - Score 3: source mean-diff vs target mean-diff
5. Compares each score family to transfer AUROC on a shared off-diagonal mask.

The canonical output root is:
  artifacts/runs/target_geometry_proxy_alignment/<model_dir>/all-datasets/fixed-mean-l15/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import rankdata
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent / "utils"))
from combined_probe_pipeline import (  # noqa: E402
    default_run_id,
    ensure_dir,
    model_dir_name,
    read_json,
    utc_now_iso,
    write_json,
)


SUPPORTED_POOLINGS = ["mean", "max", "last"]
SUPPORTED_SEGMENTS = ["completion", "full"]
SCORE_VARIANTS = [
    "score1_trained_probe",
    "score2_lda_to_trained_target",
    "score3_mean_diff",
]
SEGMENT_ORDER_ROWS = {
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
SEGMENT_ORDER_COLS = {
    "completion": [
        "Deception-ConvincingGame-completion",
        "Deception-HarmPressureChoice-completion",
        "Deception-InstructedDeception-completion",
        "Deception-Mask-completion",
        "Deception-AILiar-completion",
        "Deception-InsiderTrading-completion",
        "Deception-Roleplaying-completion",
    ],
    "full": [
        "Deception-ConvincingGame-full",
        "Deception-HarmPressureChoice-full",
        "Deception-InstructedDeception-full",
        "Deception-Mask-full",
        "Deception-AILiar-full",
        "Deception-InsiderTrading-full",
        "Deception-Roleplaying-full",
    ],
}


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build target-geometry proxy alignment matrices.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--activations_root", type=str, required=True)
    p.add_argument("--probes_root", type=str, required=True)
    p.add_argument(
        "--fixed_results_dir",
        type=str,
        required=True,
        help=(
            "Results dir from build_pairwise_score_matrices_from_artifacts.py "
            "(contains completion/full subdirs with matrix_fixed_<pooling>_L<layer>_auc.csv)"
        ),
    )
    p.add_argument("--artifact_root", type=str, default="artifacts")
    p.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional mirror root; full run tree is copied to <output_root>/<run_id>/",
    )
    p.add_argument(
        "--gallery_root",
        type=str,
        default=None,
        help="Optional directory for headline PNG copies.",
    )
    p.add_argument("--segments", type=str, default="completion,full")
    p.add_argument("--pooling", type=str, default="mean", choices=SUPPORTED_POOLINGS)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--ridge_lambda", type=float, default=1e-4)
    p.add_argument("--use_pinv", action="store_true")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--progress_every", type=int, default=200)
    return p.parse_args()


def get_git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR.parent.parent),
        )
        out = proc.stdout.strip()
        return out or None
    except Exception:
        return None


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


def base_label_map() -> Dict[str, str]:
    return {
        "Deception-ConvincingGame": "ConvincingGame",
        "Deception-HarmPressureChoice": "HarmPressureChoice",
        "Deception-InstructedDeception": "InstructedDeception",
        "Deception-Mask": "Mask",
        "Deception-AILiar": "AILiar",
        "Deception-InsiderTrading": "IT",
        "Deception-Roleplaying": "Roleplaying",
    }


def dataset_tick_label(dataset_name: str) -> str:
    seg = "Completion" if dataset_name.endswith("-completion") else "Full-prompt"
    base = dataset_base(dataset_name)
    nice = base_label_map().get(base, base.replace("Deception-", ""))
    return f"{nice}\n({seg})"


def short_name(dataset_name: str) -> str:
    seg = segment_of(dataset_name)
    base = dataset_base(dataset_name).replace("Deception-", "")
    short = {
        "ConvincingGame": "CG",
        "HarmPressureChoice": "HPC",
        "InstructedDeception": "ID",
        "Mask": "M",
        "AILiar": "AL",
        "InsiderTrading": "IT",
        "Roleplaying": "RP",
    }.get(base, base)
    return f"{short}-{'c' if seg == 'completion' else 'f'}"


def log_message(log_path: Path, message: str) -> None:
    print(message)
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


def update_status(status_path: Path, state: str, message: str) -> None:
    payload = read_json(status_path, default={})
    payload["state"] = state
    payload["message"] = message
    payload["updated_at"] = utc_now_iso()
    write_json(status_path, payload)


def load_progress(progress_path: Path) -> Dict[str, Any]:
    payload = read_json(progress_path, default={})
    payload.setdefault("completed_units", [])
    payload.setdefault("updated_at", utc_now_iso())
    return payload


def save_progress(progress_path: Path, progress: Dict[str, Any]) -> None:
    progress["updated_at"] = utc_now_iso()
    write_json(progress_path, progress)


def split_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    if root.name == model_dir:
        return root.parent, root
    candidate = root / model_dir
    if candidate.exists():
        return root, candidate
    return root.parent, root


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
        vals: List[float] = []
        for cell in row[1:]:
            raw = str(cell).strip()
            vals.append(np.nan if raw == "" else float(raw))
        matrix_rows.append(vals)
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.shape != (len(row_labels), len(col_labels)):
        raise ValueError(f"Shape mismatch while reading {path}: {matrix.shape} vs ({len(row_labels)}, {len(col_labels)})")
    return row_labels, col_labels, matrix


def write_matrix_csv(path: Path, rows: Sequence[str], cols: Sequence[str], matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_dataset"] + list(cols))
        for i, row_label in enumerate(rows):
            out: List[Any] = [row_label]
            for j in range(len(cols)):
                value = matrix[i, j]
                out.append("" if np.isnan(value) else float(value))
            writer.writerow(out)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def iter_shards(shard_paths: Sequence[Path], desc: str, use_tqdm: bool) -> Iterable[Path]:
    if use_tqdm:
        return tqdm(shard_paths, desc=desc)
    return shard_paths


def load_pooled_split(
    activations_dir: Path,
    layer: int,
    pooling: str,
    desc: str,
    use_tqdm: bool,
    progress_every: int,
    log_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    label_map = load_label_map(activations_dir)
    shard_paths = sorted(activations_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.safetensors in {activations_dir}")

    features: List[np.ndarray] = []
    labels: List[int] = []
    count = 0
    for shard_path in iter_shards(shard_paths, desc=f"Loading {desc} shards", use_tqdm=use_tqdm):
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
                    f"Layer {layer} out of range for sample {sid} with shape {tuple(tensor.shape)}"
                )
            features.append(pool_tokens(tensor[layer, :, :], pooling))
            labels.append(label)
            count += 1
            if (not use_tqdm) and progress_every and count % progress_every == 0:
                log_message(log_path, f"[load] {desc}: processed {count} samples")
    if not labels:
        raise ValueError(f"No labeled examples loaded from {activations_dir}")
    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def compute_dataset_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    idx0 = y == 0
    idx1 = y == 1
    n0 = int(idx0.sum())
    n1 = int(idx1.sum())
    if n0 < 2 or n1 < 2:
        raise ValueError(f"Need at least 2 examples per class, got n0={n0}, n1={n1}")

    x0 = x[idx0].astype(np.float64, copy=False)
    x1 = x[idx1].astype(np.float64, copy=False)
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    delta_mu = mu1 - mu0

    r0 = x0 - mu0[None, :]
    r1 = x1 - mu1[None, :]
    residuals = np.concatenate([r0, r1], axis=0)
    denom = max(int(residuals.shape[0]) - 1, 1)
    cov = (residuals.T @ residuals) / float(denom)

    return {
        "mu0": mu0,
        "mu1": mu1,
        "delta_mu": delta_mu,
        "cov": cov,
        "n0": n0,
        "n1": n1,
        "n_total": int(x.shape[0]),
        "dim": int(x.shape[1]),
    }


def invert_covariance(
    cov: np.ndarray,
    ridge_lambda: float,
    use_pinv: bool,
) -> Tuple[np.ndarray, str]:
    dim = cov.shape[0]
    ridge = cov + np.eye(dim, dtype=np.float64) * float(ridge_lambda)
    if use_pinv:
        return np.linalg.pinv(ridge, hermitian=True), "pinv_ridge"
    try:
        return np.linalg.inv(ridge), "ridge_inv"
    except np.linalg.LinAlgError:
        return np.linalg.pinv(ridge, hermitian=True), "pinv_ridge_fallback"


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
    w = state_dict[weight_key].detach().cpu().numpy().reshape(-1).astype(np.float64)
    b = float(state_dict[bias_key].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def resolve_probe_path(probes_model_root: Path, dataset_name: str, pooling: str, layer: int) -> Path:
    base = dataset_base(dataset_name)
    return probes_model_root / f"{base}_slices" / dataset_name / pooling / f"probe_layer_{layer}.pt"


def load_probe_direction(probe_path: Path) -> Tuple[np.ndarray, float]:
    state = torch.load(str(probe_path), map_location="cpu")
    return extract_classifier_params(state)


def metric_cosine(a: np.ndarray, b: np.ndarray, cov_inv: np.ndarray, eps: float = 1e-12) -> float:
    a64 = np.asarray(a, dtype=np.float64).reshape(-1)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(a64.T @ cov_inv @ b64)
    da = float(a64.T @ cov_inv @ a64)
    db = float(b64.T @ cov_inv @ b64)
    if da <= eps or db <= eps:
        return float("nan")
    den = np.sqrt(da) * np.sqrt(db)
    if den <= eps:
        return float("nan")
    return float(num / den)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    xr = rankdata(x)
    yr = rankdata(y)
    return safe_pearson(np.asarray(xr, dtype=np.float64), np.asarray(yr, dtype=np.float64))


def format_float(value: float) -> str:
    return "NA" if np.isnan(value) else f"{value:.2f}"


def plot_matrix_heatmap(
    output_path: Path,
    matrix: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str],
    title: str,
    segment: str,
    cell_text: Optional[Sequence[Sequence[str]]] = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    cbar_label: str = "Alignment score",
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([dataset_tick_label(c) for c in cols], rotation=30, ha="right")
    ax.set_yticklabels([dataset_tick_label(r) for r in rows])
    ax.set_xlabel("Target dataset")
    ax.set_ylabel("Source dataset" if segment == "completion" else "Source dataset")
    ax.set_title(title, fontweight="bold", fontsize=14)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    midpoint = (vmin + vmax) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = cell_text[i][j] if cell_text is not None else format_float(value)
            if np.isnan(value):
                color = "#4a4a4a"
            else:
                color = "white" if value >= midpoint else "#4a4a4a"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    for i, row_name in enumerate(rows):
        for j, col_name in enumerate(cols):
            if row_name == col_name:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=2.4)
                ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_proxy_vs_auc_scatter(
    output_path: Path,
    segment: str,
    data_by_variant: Dict[str, Tuple[np.ndarray, np.ndarray]],
    summary_by_variant: Dict[str, Dict[str, Any]],
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), sharey=True)
    titles = {
        "score1_trained_probe": "Score 1: trained probe",
        "score2_lda_to_trained_target": "Score 2: LDA to target probe",
        "score3_mean_diff": "Score 3: raw mean-diff",
    }
    for ax, variant in zip(axes, SCORE_VARIANTS):
        x, y = data_by_variant.get(variant, (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)))
        if x.size:
            ax.scatter(x, y, s=34, alpha=0.85, color="#2c7fb8")
        ax.set_title(titles[variant])
        ax.set_xlabel("Alignment score")
        ax.grid(alpha=0.25)
        stats = summary_by_variant.get(variant, {})
        n = int(stats.get("num_valid_shared_offdiag", 0))
        pearson = stats.get("pearson_shared_offdiag")
        spearman = stats.get("spearman_shared_offdiag")
        lines = [f"n={n}"]
        if pearson is not None and not np.isnan(pearson):
            lines.append(f"Pearson={pearson:.3f}")
        if spearman is not None and not np.isnan(spearman):
            lines.append(f"Spearman={spearman:.3f}")
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
        )
    axes[0].set_ylabel("Transfer AUROC")
    fig.suptitle(f"Proxy alignment vs fixed transfer AUROC | {segment}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_png(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def rank_desc(values: np.ndarray) -> np.ndarray:
    return rankdata(-values, method="average")


def main() -> int:
    args = parse_args()
    selected_segments = parse_csv_list(args.segments)
    if not selected_segments:
        raise ValueError("At least one segment is required")
    for seg in selected_segments:
        if seg not in SUPPORTED_SEGMENTS:
            raise ValueError(f"Unsupported segment '{seg}'. Choose from {SUPPORTED_SEGMENTS}.")

    model_dir = model_dir_name(args.model)
    run_id = args.run_id or default_run_id()
    run_root = (
        Path(args.artifact_root)
        / "runs"
        / "target_geometry_proxy_alignment"
        / model_dir
        / "all-datasets"
        / f"fixed-{args.pooling}-l{args.layer}"
        / run_id
    )
    inputs_dir = run_root / "inputs"
    stats_root = inputs_dir / "dataset_stats"
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    matrices_dir = results_dir / "matrices"
    long_dir = results_dir / "long"
    plots_dir = results_dir / "plots"
    logs_dir = run_root / "logs"
    meta_dir = run_root / "meta"
    manifest_path = meta_dir / "run_manifest.json"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    results_json_path = results_dir / "results.json"
    summary_json_path = results_dir / "summary.json"
    log_path = logs_dir / "run.log"

    if args.resume and status_path.exists():
        existing_status = read_json(status_path, default={})
        if existing_status.get("state") == "completed":
            log_message(log_path, f"[resume] run already completed: {run_root}")
            return 0

    ensure_dir(run_root)
    progress = load_progress(progress_path)
    completed_units = set(str(x) for x in progress.get("completed_units", []))

    manifest = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "git_commit": get_git_commit(),
        "model": args.model,
        "model_dir": model_dir,
        "config": {
            "segments": selected_segments,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "ridge_lambda": float(args.ridge_lambda),
            "use_pinv": bool(args.use_pinv),
            "progress_every": int(args.progress_every),
        },
        "inputs": {
            "activations_root": str(args.activations_root),
            "probes_root": str(args.probes_root),
            "fixed_results_dir": str(args.fixed_results_dir),
        },
        "outputs": {
            "run_root": str(run_root),
            "output_root": None if args.output_root is None else str(Path(args.output_root) / run_id),
            "gallery_root": None if args.gallery_root is None else str(args.gallery_root),
        },
    }
    if not manifest_path.exists():
        write_json(manifest_path, manifest)

    update_status(status_path, "running", "starting target-geometry proxy alignment")
    log_message(log_path, f"[start] run_id={run_id} model={args.model} pooling={args.pooling} layer={args.layer}")

    fixed_results_dir = Path(args.fixed_results_dir)
    activations_root = Path(args.activations_root)
    probes_root = Path(args.probes_root)
    _, probes_model_root = split_root_and_model(probes_root, model_dir)

    try:
        fixed_matrices: Dict[str, Dict[str, Any]] = {}
        fixed_manifest_payload: Dict[str, Any] = {
            "model": args.model,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "segments": {},
        }

        for segment in selected_segments:
            fixed_csv_path = fixed_results_dir / segment / f"matrix_fixed_{args.pooling}_L{args.layer}_auc.csv"
            if not fixed_csv_path.exists():
                raise FileNotFoundError(f"Missing fixed matrix CSV: {fixed_csv_path}")
            rows, cols, auc_matrix = read_matrix_csv(fixed_csv_path)
            fixed_matrices[segment] = {
                "rows": rows,
                "cols": cols,
                "auc_matrix": auc_matrix,
                "source_csv": fixed_csv_path,
            }
            fixed_manifest_payload["segments"][segment] = {
                "source_csv": str(fixed_csv_path),
                "rows": rows,
                "cols": cols,
                "num_rows": len(rows),
                "num_cols": len(cols),
            }
            ensure_dir(matrices_dir / segment)
            shutil.copy2(fixed_csv_path, matrices_dir / segment / f"transfer_auc_fixed_{args.pooling}_l{args.layer}.csv")
        write_json(inputs_dir / "fixed_matrix_manifest.json", fixed_manifest_payload)

        # Build stats caches first.
        unique_datasets_by_segment: Dict[str, List[str]] = {}
        for segment, payload in fixed_matrices.items():
            merged = payload["rows"] + [x for x in payload["cols"] if x not in payload["rows"]]
            unique_datasets_by_segment[segment] = merged

        lightweight_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        probe_dirs: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        probe_biases: Dict[str, Dict[str, Optional[float]]] = {}

        for segment in selected_segments:
            lightweight_stats[segment] = {}
            probe_dirs[segment] = {}
            probe_biases[segment] = {}
            for dataset_name in unique_datasets_by_segment[segment]:
                unit_id = f"stats::{segment}::{dataset_name}"
                stats_path = stats_root / segment / f"{dataset_name}.npz"
                if unit_id not in completed_units or not stats_path.exists():
                    update_status(status_path, "running", f"computing stats for {dataset_name}")
                    log_message(log_path, f"[stats] computing {dataset_name}")
                    split_dir = activations_root / model_dir / dataset_name / "train"
                    x_train, y_train = load_pooled_split(
                        split_dir,
                        layer=args.layer,
                        pooling=args.pooling,
                        desc=f"{dataset_name} train",
                        use_tqdm=not args.no_tqdm,
                        progress_every=args.progress_every,
                        log_path=log_path,
                    )
                    stats = compute_dataset_stats(x_train, y_train)
                    cov_inv, inversion_method = invert_covariance(
                        stats["cov"],
                        ridge_lambda=args.ridge_lambda,
                        use_pinv=args.use_pinv,
                    )
                    lda_dir = cov_inv @ stats["delta_mu"]
                    save_payload = {
                        "mu0": stats["mu0"].astype(np.float32),
                        "mu1": stats["mu1"].astype(np.float32),
                        "delta_mu": stats["delta_mu"].astype(np.float32),
                        "cov": stats["cov"].astype(np.float32),
                        "cov_inv": cov_inv.astype(np.float32),
                        "lda_dir": lda_dir.astype(np.float32),
                        "n0": np.asarray([stats["n0"]], dtype=np.int64),
                        "n1": np.asarray([stats["n1"]], dtype=np.int64),
                        "n_total": np.asarray([stats["n_total"]], dtype=np.int64),
                        "dim": np.asarray([stats["dim"]], dtype=np.int64),
                        "ridge_lambda": np.asarray([float(args.ridge_lambda)], dtype=np.float64),
                        "inversion_method": np.asarray([inversion_method]),
                    }
                    ensure_dir(stats_path.parent)
                    np.savez_compressed(str(stats_path), **save_payload)
                    completed_units.add(unit_id)
                    progress["completed_units"] = sorted(completed_units)
                    save_progress(progress_path, progress)

                cached = np.load(str(stats_path), allow_pickle=False)
                lightweight_stats[segment][dataset_name] = {
                    "delta_mu": np.asarray(cached["delta_mu"], dtype=np.float64),
                    "lda_dir": np.asarray(cached["lda_dir"], dtype=np.float64),
                    "n0": int(np.asarray(cached["n0"]).reshape(-1)[0]),
                    "n1": int(np.asarray(cached["n1"]).reshape(-1)[0]),
                    "dim": int(np.asarray(cached["dim"]).reshape(-1)[0]),
                    "stats_path": str(stats_path),
                }

                probe_path = resolve_probe_path(probes_model_root, dataset_name, args.pooling, args.layer)
                if probe_path.exists():
                    w, b = load_probe_direction(probe_path)
                    probe_dirs[segment][dataset_name] = w
                    probe_biases[segment][dataset_name] = b
                else:
                    probe_dirs[segment][dataset_name] = None
                    probe_biases[segment][dataset_name] = None
                    log_message(log_path, f"[warn] missing fixed probe for {dataset_name}: {probe_path}")

        proxy_long_rows: List[Dict[str, Any]] = []
        proxy_vs_auc_long_rows: List[Dict[str, Any]] = []
        proxy_summary_rows: List[Dict[str, Any]] = []
        results_payload: Dict[str, Any] = {
            "run_id": run_id,
            "completed_at": None,
            "model": args.model,
            "config": manifest["config"],
            "segments": {},
        }
        gallery_copies: List[str] = []

        for segment in selected_segments:
            update_status(status_path, "running", f"building matrices for {segment}")
            row_labels = fixed_matrices[segment]["rows"]
            col_labels = fixed_matrices[segment]["cols"]
            auc_matrix = fixed_matrices[segment]["auc_matrix"]
            num_rows = len(row_labels)
            num_cols = len(col_labels)

            score_mats = {
                "score1_trained_probe": np.full((num_rows, num_cols), np.nan, dtype=np.float64),
                "score2_lda_to_trained_target": np.full((num_rows, num_cols), np.nan, dtype=np.float64),
                "score3_mean_diff": np.full((num_rows, num_cols), np.nan, dtype=np.float64),
            }

            # Load one target covariance inverse at a time.
            for j, target_name in enumerate(col_labels):
                target_stats_path = Path(lightweight_stats[segment][target_name]["stats_path"])
                target_cached = np.load(str(target_stats_path), allow_pickle=False)
                target_cov_inv = np.asarray(target_cached["cov_inv"], dtype=np.float64)
                target_delta = lightweight_stats[segment][target_name]["delta_mu"]
                target_probe = probe_dirs[segment].get(target_name)

                for i, source_name in enumerate(row_labels):
                    source_delta = lightweight_stats[segment][source_name]["delta_mu"]
                    source_lda = lightweight_stats[segment][source_name]["lda_dir"]
                    source_probe = probe_dirs[segment].get(source_name)

                    if source_probe is not None and target_probe is not None:
                        score_mats["score1_trained_probe"][i, j] = metric_cosine(source_probe, target_probe, target_cov_inv)
                    if target_probe is not None:
                        score_mats["score2_lda_to_trained_target"][i, j] = metric_cosine(source_lda, target_probe, target_cov_inv)
                    score_mats["score3_mean_diff"][i, j] = metric_cosine(source_delta, target_delta, target_cov_inv)

            # Write matrices and plots.
            segment_plot_dir = plots_dir / segment
            ensure_dir(segment_plot_dir)
            variant_titles = {
                "score1_trained_probe": f"{segment} | Score 1 | trained probe vs trained target probe",
                "score2_lda_to_trained_target": f"{segment} | Score 2 | source LDA vs trained target probe",
                "score3_mean_diff": f"{segment} | Score 3 | raw mean-diff vs raw mean-diff",
            }
            variant_plot_names = {
                "score1_trained_probe": "score1_trained_probe_alignment_heatmap.png",
                "score2_lda_to_trained_target": "score2_lda_to_trained_target_alignment_heatmap.png",
                "score3_mean_diff": "score3_mean_diff_alignment_heatmap.png",
            }
            variant_csv_names = {
                "score1_trained_probe": "score1_trained_probe_alignment.csv",
                "score2_lda_to_trained_target": "score2_lda_to_trained_target_alignment.csv",
                "score3_mean_diff": "score3_mean_diff_alignment.csv",
            }
            for variant in SCORE_VARIANTS:
                matrix_unit = f"matrix::{segment}::{variant}"
                csv_path = matrices_dir / segment / variant_csv_names[variant]
                if matrix_unit not in completed_units or not csv_path.exists():
                    write_matrix_csv(csv_path, row_labels, col_labels, score_mats[variant])
                    completed_units.add(matrix_unit)
                    progress["completed_units"] = sorted(completed_units)
                    save_progress(progress_path, progress)

                plot_unit = f"plot::{segment}::{variant_plot_names[variant]}"
                plot_path = segment_plot_dir / variant_plot_names[variant]
                if plot_unit not in completed_units or not plot_path.exists():
                    plot_matrix_heatmap(
                        output_path=plot_path,
                        matrix=score_mats[variant],
                        rows=row_labels,
                        cols=col_labels,
                        title=variant_titles[variant],
                        segment=segment,
                    )
                    completed_units.add(plot_unit)
                    progress["completed_units"] = sorted(completed_units)
                    save_progress(progress_path, progress)
                if args.gallery_root:
                    gallery_copies.append(
                        copy_png(
                            plot_path,
                            Path(args.gallery_root) / f"{segment}_{variant_plot_names[variant]}",
                        )
                    )

            # Long rows and summary correlations.
            shared_mask = np.isfinite(auc_matrix)
            for variant in SCORE_VARIANTS:
                shared_mask &= np.isfinite(score_mats[variant])
            off_diag_mask = np.ones_like(shared_mask, dtype=bool)
            for i, row_name in enumerate(row_labels):
                for j, col_name in enumerate(col_labels):
                    if row_name == col_name:
                        off_diag_mask[i, j] = False
            shared_offdiag_mask = shared_mask & off_diag_mask

            variant_metric_rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            variant_summary: Dict[str, Dict[str, Any]] = {}
            for variant in SCORE_VARIANTS:
                x_valid = score_mats[variant][shared_offdiag_mask]
                y_valid = auc_matrix[shared_offdiag_mask]
                variant_metric_rows[variant] = (x_valid, y_valid)
                variant_summary[variant] = {
                    "num_valid_shared_offdiag": int(x_valid.size),
                    "pearson_shared_offdiag": None if x_valid.size == 0 else safe_pearson(x_valid, y_valid),
                    "spearman_shared_offdiag": None if x_valid.size == 0 else safe_spearman(x_valid, y_valid),
                    "csv": str(matrices_dir / segment / variant_csv_names[variant]),
                    "png": str(segment_plot_dir / variant_plot_names[variant]),
                }

            scatter_path = segment_plot_dir / "proxy_vs_auc_scatter.png"
            scatter_unit = f"plot::{segment}::proxy_vs_auc_scatter.png"
            if scatter_unit not in completed_units or not scatter_path.exists():
                plot_proxy_vs_auc_scatter(
                    output_path=scatter_path,
                    segment=segment,
                    data_by_variant=variant_metric_rows,
                    summary_by_variant=variant_summary,
                )
                completed_units.add(scatter_unit)
                progress["completed_units"] = sorted(completed_units)
                save_progress(progress_path, progress)
            if args.gallery_root:
                gallery_copies.append(copy_png(scatter_path, Path(args.gallery_root) / f"{segment}_proxy_vs_auc_scatter.png"))

            # Build long rows after shared mask is known.
            for variant in SCORE_VARIANTS:
                matrix = score_mats[variant]
                valid_scores = matrix[shared_offdiag_mask]
                alignment_ranks = rank_desc(valid_scores) if valid_scores.size else np.asarray([], dtype=np.float64)
                auc_ranks = rank_desc(auc_matrix[shared_offdiag_mask]) if valid_scores.size else np.asarray([], dtype=np.float64)
                shared_positions = np.argwhere(shared_offdiag_mask)
                shared_rank_lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
                for idx, (i, j) in enumerate(shared_positions):
                    shared_rank_lookup[(int(i), int(j))] = (float(alignment_ranks[idx]), float(auc_ranks[idx]))

                for i, source_name in enumerate(row_labels):
                    for j, target_name in enumerate(col_labels):
                        is_diagonal = source_name == target_name
                        alignment_score = matrix[i, j]
                        transfer_auc = auc_matrix[i, j]
                        shared_valid = bool(shared_offdiag_mask[i, j])
                        proxy_long_rows.append(
                            {
                                "segment": segment,
                                "source_dataset": source_name,
                                "target_dataset": target_name,
                                "score_variant": variant,
                                "alignment_score": "" if np.isnan(alignment_score) else float(alignment_score),
                                "transfer_auc": "" if np.isnan(transfer_auc) else float(transfer_auc),
                                "is_diagonal": int(is_diagonal),
                                "shared_valid_offdiag": int(shared_valid),
                                "used_target_covariance_dataset": target_name,
                                "pooling": args.pooling,
                                "layer": int(args.layer),
                            }
                        )
                        if shared_valid:
                            alignment_rank, auc_rank = shared_rank_lookup[(i, j)]
                            proxy_vs_auc_long_rows.append(
                                {
                                    "segment": segment,
                                    "source_dataset": source_name,
                                    "target_dataset": target_name,
                                    "score_variant": variant,
                                    "alignment_score": float(alignment_score),
                                    "transfer_auc": float(transfer_auc),
                                    "rank_alignment_within_variant": alignment_rank,
                                    "rank_auc": auc_rank,
                                    "used_target_covariance_dataset": target_name,
                                    "pooling": args.pooling,
                                    "layer": int(args.layer),
                                }
                            )

            best_by_pearson = None
            best_by_spearman = None
            pearson_candidates = [
                (variant, variant_summary[variant]["pearson_shared_offdiag"])
                for variant in SCORE_VARIANTS
                if variant_summary[variant]["pearson_shared_offdiag"] is not None
                and not np.isnan(variant_summary[variant]["pearson_shared_offdiag"])
            ]
            spearman_candidates = [
                (variant, variant_summary[variant]["spearman_shared_offdiag"])
                for variant in SCORE_VARIANTS
                if variant_summary[variant]["spearman_shared_offdiag"] is not None
                and not np.isnan(variant_summary[variant]["spearman_shared_offdiag"])
            ]
            if pearson_candidates:
                best_by_pearson = max(pearson_candidates, key=lambda x: float(x[1]))[0]
            if spearman_candidates:
                best_by_spearman = max(spearman_candidates, key=lambda x: float(x[1]))[0]

            results_payload["segments"][segment] = {
                "rows": row_labels,
                "cols": col_labels,
                "shared_valid_offdiag_count": int(np.sum(shared_offdiag_mask)),
                "transfer_auc_csv": str(matrices_dir / segment / f"transfer_auc_fixed_{args.pooling}_l{args.layer}.csv"),
                "scatter_png": str(scatter_path),
                "variants": variant_summary,
                "best_by_pearson": best_by_pearson,
                "best_by_spearman": best_by_spearman,
            }

            for variant in SCORE_VARIANTS:
                proxy_summary_rows.append(
                    {
                        "segment": segment,
                        "score_variant": variant,
                        "num_valid_shared_offdiag": variant_summary[variant]["num_valid_shared_offdiag"],
                        "pearson_shared_offdiag": variant_summary[variant]["pearson_shared_offdiag"],
                        "spearman_shared_offdiag": variant_summary[variant]["spearman_shared_offdiag"],
                        "best_by_pearson": int(variant == best_by_pearson),
                        "best_by_spearman": int(variant == best_by_spearman),
                    }
                )

        write_csv_rows(
            long_dir / "proxy_alignment_long.csv",
            proxy_long_rows,
            [
                "segment",
                "source_dataset",
                "target_dataset",
                "score_variant",
                "alignment_score",
                "transfer_auc",
                "is_diagonal",
                "shared_valid_offdiag",
                "used_target_covariance_dataset",
                "pooling",
                "layer",
            ],
        )
        write_csv_rows(
            long_dir / "proxy_vs_auc_long.csv",
            proxy_vs_auc_long_rows,
            [
                "segment",
                "source_dataset",
                "target_dataset",
                "score_variant",
                "alignment_score",
                "transfer_auc",
                "rank_alignment_within_variant",
                "rank_auc",
                "used_target_covariance_dataset",
                "pooling",
                "layer",
            ],
        )
        write_csv_rows(
            results_dir / "proxy_vs_auc_summary.csv",
            proxy_summary_rows,
            [
                "segment",
                "score_variant",
                "num_valid_shared_offdiag",
                "pearson_shared_offdiag",
                "spearman_shared_offdiag",
                "best_by_pearson",
                "best_by_spearman",
            ],
        )
        results_payload["completed_at"] = utc_now_iso()
        results_payload["gallery_copies"] = gallery_copies
        write_json(results_json_path, results_payload)
        write_json(summary_json_path, results_payload)

        if args.output_root:
            mirror_root = Path(args.output_root) / run_id
            if mirror_root.exists():
                shutil.rmtree(mirror_root)
            shutil.copytree(run_root, mirror_root)
            log_message(log_path, f"[mirror] copied run tree to {mirror_root}")

        update_status(status_path, "completed", "finished")
        log_message(log_path, f"[done] outputs -> {run_root}")
        return 0
    except Exception as exc:
        update_status(status_path, "failed", str(exc))
        log_message(log_path, "[error] run failed")
        for line in traceback.format_exc().splitlines():
            log_message(log_path, line)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
