#!/usr/bin/env python3
"""
Run a suite of probe-similarity analyses from existing pairwise OOD artifacts.

This script computes, per segment:
  1. source-level Pearson vs Spearman similarity from per-sample probe logits
  2. source-level linear CKA (aggregate + layerwise)
  3. label-conditioned source similarity (positive / negative / delta)
  4. probe clustering from 7D AUROC profiles across target datasets
  5. spectral clustering from probe-logit similarity matrices

Canonical outputs are written under:
  artifacts/runs/probe_similarity_suite/<model_dir>/all-segments/pair-logits/<variant>/<run_id>/

Optionally, PNG plots are mirrored into a flat gallery folder, for example:
  results/ood_evaluation/<model_dir>/all_pairwise_results_final/generated_plots_after_id_fix/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.cluster import KMeans


POOLING_ORDER = ["attn", "last", "max", "mean"]
SEGMENTS = ["completion", "full"]
SOURCE_COLOR_MAP = {
    "Deception-ConvincingGame": "#4c78a8",
    "Deception-HarmPressureChoice": "#f58518",
    "Deception-InstructedDeception": "#54a24b",
    "Deception-Mask": "#e45756",
    "Deception-AILiar": "#72b7b2",
    "Deception-InsiderTrading": "#b279a2",
    "Deception-Roleplaying": "#ff9da6",
}
POOLING_MARKER_MAP = {
    "mean": "o",
    "max": "s",
    "last": "^",
    "attn": "D",
}


@dataclass
class SegmentBundle:
    segment: str
    rows: List[str]
    cols: List[str]
    matrix: np.ndarray
    probe_manifest: pd.DataFrame
    sample_manifest: pd.DataFrame


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_status(path: Path, state: str, message: str) -> None:
    payload = read_json(path, default={})
    payload["state"] = state
    payload["message"] = message
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def split_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    if root.name == model_dir:
        return root.parent, root
    candidate = root / model_dir
    if candidate.exists():
        return root, candidate
    return root.parent, root


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
    seg = segment_of(dataset_name)
    base = dataset_base(dataset_name).replace("Deception-", "")
    mapping = {
        "ConvincingGame": "CG",
        "HarmPressureChoice": "HPC",
        "InstructedDeception": "ID",
        "Mask": "M",
        "AILiar": "AL",
        "InsiderTrading": "IT",
        "Roleplaying": "RP",
    }
    return f"{mapping.get(base, base)}-{'c' if seg == 'completion' else 'f'}"


def normalize_pooling(pooling: Any) -> str:
    p = str(pooling).strip().lower()
    if p in {"none", "final", "final_token"}:
        return "last"
    return p


def parse_float(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def parse_int(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(v)
    except Exception:
        return None


def stage_spec() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
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
    return rows, cols


def pair_manifest_path(model_root: Path, source_dataset: str, target_dataset: str) -> Path:
    return model_root / f"from-{source_dataset}" / f"to-{target_dataset}" / "pair_logits_manifest.json"


def pair_tensor_path(model_root: Path, source_dataset: str, target_dataset: str) -> Path:
    return model_root / f"from-{source_dataset}" / f"to-{target_dataset}" / "pair_logits.safetensors"


def pair_summary_path(model_root: Path, source_dataset: str, target_dataset: str) -> Path:
    return model_root / f"from-{source_dataset}" / f"to-{target_dataset}" / "pair_summary.json"


def find_any_manifest_for_target(model_root: Path, sources: Sequence[str], target_dataset: str) -> Tuple[Path, str]:
    for source_dataset in sources:
        path = pair_manifest_path(model_root, source_dataset, target_dataset)
        if path.exists():
            return path, source_dataset
    raise FileNotFoundError(f"No pair logits manifest found for target {target_dataset}")


def find_any_manifest_for_source(model_root: Path, source_dataset: str, targets: Sequence[str]) -> Tuple[Path, str]:
    for target_dataset in targets:
        path = pair_manifest_path(model_root, source_dataset, target_dataset)
        if path.exists():
            return path, target_dataset
    raise FileNotFoundError(f"No pair logits manifest found for source {source_dataset}")


def load_wide_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing matrix CSV: {path}")
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Matrix CSV has too few columns: {path}")
    df = df.set_index(df.columns[0])
    df.index = [str(x).strip() for x in df.index]
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def save_matrix_csv(path: Path, labels: Sequence[str], matrix: np.ndarray, index_name: str = "row_dataset") -> None:
    df = pd.DataFrame(matrix, index=list(labels), columns=list(labels))
    df.index.name = index_name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def score_key_sort_tuple(score_key: str) -> Tuple[int, str]:
    pooling, layer = score_key.split(":")
    return (parse_int(layer) or 0, pooling)


def sort_probe_manifest(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {p: i for i, p in enumerate(POOLING_ORDER)}
    out = df.copy()
    out["pooling_order"] = out["pooling"].map(lambda p: order_map.get(str(p), 999))
    out = out.sort_values(
        by=["source_dataset", "layer", "pooling_order", "pooling"],
        kind="stable",
    )
    out = out.drop(columns=["pooling_order"])
    return out


def correlation_similarity(matrix: np.ndarray) -> np.ndarray:
    x = matrix.astype(np.float64)
    x = x - np.mean(x, axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = norms.squeeze(-1) > 0
    x_norm = np.zeros_like(x)
    x_norm[safe] = x[safe] / norms[safe]
    sim = x_norm @ x_norm.T
    sim = np.clip(sim, -1.0, 1.0)
    sim[~safe, :] = 0.0
    sim[:, ~safe] = 0.0
    np.fill_diagonal(sim, 1.0)
    return sim


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.size != yv.size or xv.size < 2:
        return float("nan")
    xv = xv - np.mean(xv)
    yv = yv - np.mean(yv)
    den = float(np.linalg.norm(xv) * np.linalg.norm(yv))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(xv, yv) / den)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(rankdata(x), rankdata(y))


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape[0] != ya.shape[0] or xa.shape[0] < 2:
        return float("nan")
    xa = xa - np.mean(xa, axis=0, keepdims=True)
    ya = ya - np.mean(ya, axis=0, keepdims=True)
    cross = xa.T @ ya
    xx = xa.T @ xa
    yy = ya.T @ ya
    num = float(np.sum(cross * cross))
    den = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)) + eps)
    return float(num / den)


def plot_small_heatmap(
    path: Path,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    if center is not None:
        vmax_eff = vmax if vmax is not None else np.nanmax(matrix)
        vmin_eff = vmin if vmin is not None else np.nanmin(matrix)
        lim = max(abs(vmin_eff - center), abs(vmax_eff - center))
        vmin_eff = center - lim
        vmax_eff = center + lim
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff, aspect="auto")
    else:
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = "nan" if np.isnan(val) else f"{val:.2f}"
            color = "white" if np.isfinite(val) and abs(val) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_large_heatmap(
    path: Path,
    matrix: np.ndarray,
    title: str,
    cmap: str,
    colorbar_label: str,
    xticklabels: Optional[Sequence[str]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    if center is not None:
        vmax_eff = vmax if vmax is not None else np.nanmax(matrix)
        vmin_eff = vmin if vmin is not None else np.nanmin(matrix)
        lim = max(abs(vmin_eff - center), abs(vmax_eff - center))
        vmin_eff = center - lim
        vmax_eff = center + lim
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff, aspect="auto", interpolation="nearest")
    else:
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    else:
        ax.set_xticks([])
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if xv.size < 2 or np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return None, None, None
    slope, intercept = np.polyfit(xv, yv, deg=1)
    r = float(np.corrcoef(xv, yv)[0, 1])
    return float(slope), float(intercept), r


def plot_scatter(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: Optional[Sequence[Any]] = None,
    color_label: Optional[str] = None,
    discrete_legend: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, Optional[float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    stats: Dict[str, Optional[float]] = {"n_points": 0, "slope": None, "intercept": None, "pearson_r": None}
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if colors is not None:
        colors_arr = np.asarray(colors)[mask]
    else:
        colors_arr = None
    if xv.size > 0:
        if discrete_legend:
            for key, (label, style) in discrete_legend.items():
                sub = colors_arr == key
                if np.any(sub):
                    ax.scatter(xv[sub], yv[sub], s=24, alpha=0.6, label=label, marker=style)
            ax.legend(frameon=False, fontsize=9)
        elif colors_arr is not None:
            sc = ax.scatter(xv, yv, c=colors_arr, s=24, alpha=0.6, cmap="viridis")
            if color_label:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(color_label)
        else:
            ax.scatter(xv, yv, s=24, alpha=0.6)
        slope, intercept, r = fit_line(xv, yv)
        stats = {
            "n_points": int(xv.size),
            "slope": slope,
            "intercept": intercept,
            "pearson_r": r,
        }
        if slope is not None and intercept is not None:
            xs = np.linspace(float(np.min(xv)), float(np.max(xv)), 200)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color="black", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stats


def plot_line(path: Path, values: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(values, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def mirror_pngs(source_dir: Path, mirror_root: Optional[Path]) -> List[str]:
    mirrored: List[str] = []
    if mirror_root is None:
        return mirrored
    mirror_root.mkdir(parents=True, exist_ok=True)
    for png in sorted(source_dir.rglob("*.png")):
        dst = mirror_root / png.name
        shutil.copy2(png, dst)
        mirrored.append(str(dst))
    return mirrored


def load_segment_probe_scores(
    model_root: Path,
    segment: str,
    seg_rows: Sequence[str],
    seg_cols: Sequence[str],
    cache_dir: Path,
    resume: bool,
) -> SegmentBundle:
    probe_manifest_path = cache_dir / f"{segment}_probe_manifest.csv"
    sample_manifest_path = cache_dir / f"{segment}_sample_manifest.csv"
    matrix_path = cache_dir / f"{segment}_score_matrix.npy"

    if resume and probe_manifest_path.exists() and sample_manifest_path.exists() and matrix_path.exists():
        probe_manifest = pd.read_csv(probe_manifest_path)
        sample_manifest = pd.read_csv(sample_manifest_path)
        matrix = np.load(matrix_path)
        return SegmentBundle(
            segment=segment,
            rows=list(seg_rows),
            cols=list(seg_cols),
            matrix=matrix,
            probe_manifest=sort_probe_manifest(probe_manifest),
            sample_manifest=sample_manifest,
        )

    sample_rows: List[Dict[str, Any]] = []
    sample_offsets: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for target_dataset in seg_cols:
        manifest_path, source_used = find_any_manifest_for_target(model_root, seg_rows, target_dataset)
        manifest = read_json(manifest_path)
        target_sample_ids = manifest["sample_ids"]
        target_labels = manifest["labels"]
        start = offset
        end = start + len(target_sample_ids)
        sample_offsets[target_dataset] = (start, end)
        for local_idx, (sample_id, label) in enumerate(zip(target_sample_ids, target_labels)):
            sample_rows.append(
                {
                    "segment": segment,
                    "target_dataset": target_dataset,
                    "target_short": short_name(target_dataset),
                    "source_reference": source_used,
                    "global_sample_index": start + local_idx,
                    "target_sample_index": local_idx,
                    "sample_id": sample_id,
                    "label": int(label),
                }
            )
        offset = end

    probe_rows: List[Dict[str, Any]] = []
    probe_index: Dict[Tuple[str, str, int], int] = {}
    for source_dataset in seg_rows:
        manifest_path, target_used = find_any_manifest_for_source(model_root, source_dataset, seg_cols)
        manifest = read_json(manifest_path)
        for rec in manifest["scores"]:
            key = (source_dataset, str(rec["pooling"]), int(rec["layer"]))
            if key in probe_index:
                continue
            probe_index[key] = len(probe_rows)
            probe_rows.append(
                {
                    "segment": segment,
                    "source_dataset": source_dataset,
                    "source_short": short_name(source_dataset),
                    "pooling": str(rec["pooling"]),
                    "layer": int(rec["layer"]),
                    "score_key": str(rec["score_key"]),
                    "target_reference": target_used,
                }
            )

    matrix = np.full((len(probe_rows), len(sample_rows)), np.nan, dtype=np.float32)
    for source_dataset in seg_rows:
        for target_dataset in seg_cols:
            manifest_path = pair_manifest_path(model_root, source_dataset, target_dataset)
            tensor_path = pair_tensor_path(model_root, source_dataset, target_dataset)
            if not manifest_path.exists() or not tensor_path.exists():
                raise FileNotFoundError(
                    f"Missing pair logits for {source_dataset} -> {target_dataset}: "
                    f"{manifest_path} / {tensor_path}"
                )
            manifest = read_json(manifest_path)
            tensors = load_file(str(tensor_path))
            start, end = sample_offsets[target_dataset]
            for rec in manifest["scores"]:
                row_idx = probe_index[(source_dataset, str(rec["pooling"]), int(rec["layer"]))]
                values = tensors[str(rec["score_key"])].detach().cpu().numpy().astype(np.float32)
                matrix[row_idx, start:end] = values

    if np.isnan(matrix).any():
        raise RuntimeError(f"Segment {segment}: score matrix still contains NaNs")

    probe_manifest = pd.DataFrame(probe_rows)
    probe_manifest["orig_idx"] = np.arange(len(probe_manifest), dtype=np.int64)
    probe_manifest = sort_probe_manifest(probe_manifest)
    ordered_indices = probe_manifest["orig_idx"].to_list()
    matrix = matrix[ordered_indices]
    probe_manifest = probe_manifest.drop(columns=["orig_idx"]).reset_index(drop=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    probe_manifest.to_csv(probe_manifest_path, index=False)
    pd.DataFrame(sample_rows).to_csv(sample_manifest_path, index=False)
    np.save(matrix_path, matrix)
    return SegmentBundle(
        segment=segment,
        rows=list(seg_rows),
        cols=list(seg_cols),
        matrix=matrix,
        probe_manifest=probe_manifest,
        sample_manifest=pd.DataFrame(sample_rows),
    )


def source_config_index(bundle: SegmentBundle) -> Dict[str, Dict[Tuple[str, int], int]]:
    out: Dict[str, Dict[Tuple[str, int], int]] = {}
    for idx, row in bundle.probe_manifest.iterrows():
        source = str(row["source_dataset"])
        out.setdefault(source, {})
        out[source][(normalize_pooling(row["pooling"]), int(row["layer"]))] = idx
    return out


def common_probe_configs(
    bundle: SegmentBundle,
    source_a: str,
    source_b: str,
) -> List[Tuple[str, int]]:
    idx_map = source_config_index(bundle)
    configs = sorted(
        set(idx_map[source_a].keys()) & set(idx_map[source_b].keys()),
        key=lambda x: (x[1], POOLING_ORDER.index(x[0]) if x[0] in POOLING_ORDER else 999, x[0]),
    )
    return configs


def source_similarity_matrix(
    bundle: SegmentBundle,
    metric: str,
    sample_mask: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    idx_map = source_config_index(bundle)
    sources = list(bundle.rows)
    sample_mask = np.ones(bundle.matrix.shape[1], dtype=bool) if sample_mask is None else sample_mask.astype(bool)
    matrix = np.full((len(sources), len(sources)), np.nan, dtype=np.float64)
    counts = np.zeros((len(sources), len(sources)), dtype=np.int32)
    long_rows: List[Dict[str, Any]] = []
    per_config_rows: List[Dict[str, Any]] = []

    corr_fn = safe_corr if metric == "pearson" else safe_spearman

    for i, source_a in enumerate(sources):
        for j, source_b in enumerate(sources):
            configs = sorted(
                set(idx_map[source_a].keys()) & set(idx_map[source_b].keys()),
                key=lambda x: (x[1], POOLING_ORDER.index(x[0]) if x[0] in POOLING_ORDER else 999, x[0]),
            )
            vals: List[float] = []
            for pooling, layer in configs:
                vec_a = bundle.matrix[idx_map[source_a][(pooling, layer)], sample_mask]
                vec_b = bundle.matrix[idx_map[source_b][(pooling, layer)], sample_mask]
                sim_val = corr_fn(vec_a, vec_b)
                per_config_rows.append(
                    {
                        "segment": bundle.segment,
                        "metric": metric,
                        "source_a": source_a,
                        "source_b": source_b,
                        "pooling": pooling,
                        "layer": int(layer),
                        "similarity": sim_val,
                    }
                )
                if np.isfinite(sim_val):
                    vals.append(float(sim_val))
            if vals:
                matrix[i, j] = float(np.mean(vals))
                counts[i, j] = len(vals)
            if i == j and not vals:
                matrix[i, j] = 1.0
            long_rows.append(
                {
                    "segment": bundle.segment,
                    "metric": metric,
                    "source_a": source_a,
                    "source_b": source_b,
                    "similarity": float(matrix[i, j]) if np.isfinite(matrix[i, j]) else None,
                    "n_common_configs": int(counts[i, j]),
                }
            )
    return sources, matrix, counts, long_rows, per_config_rows


def source_feature_matrix(
    bundle: SegmentBundle,
    source_dataset: str,
    layer: Optional[int] = None,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    rows = bundle.probe_manifest[bundle.probe_manifest["source_dataset"] == source_dataset]
    if layer is not None:
        rows = rows[rows["layer"] == int(layer)]
    rows = rows.copy()
    rows["pooling"] = rows["pooling"].map(normalize_pooling)
    rows = rows.sort_values(
        by=["layer", "pooling"],
        key=lambda col: col.map(lambda x: POOLING_ORDER.index(x) if x in POOLING_ORDER else 999) if col.name == "pooling" else col,
        kind="stable",
    )
    idxs = rows.index.to_list()
    configs = [(normalize_pooling(r["pooling"]), int(r["layer"])) for _, r in rows.iterrows()]
    x = bundle.matrix[idxs, :].T.astype(np.float64)
    return x, configs


def source_cka_matrix(bundle: SegmentBundle) -> Tuple[List[str], np.ndarray]:
    sources = list(bundle.rows)
    mat = np.full((len(sources), len(sources)), np.nan, dtype=np.float64)
    features: Dict[str, np.ndarray] = {}
    for source in sources:
        x, _ = source_feature_matrix(bundle, source)
        features[source] = x
    for i, source_a in enumerate(sources):
        for j, source_b in enumerate(sources):
            if source_a == source_b:
                mat[i, j] = 1.0
            else:
                mat[i, j] = linear_cka(features[source_a], features[source_b])
    return sources, mat


def layerwise_cka_rows(bundle: SegmentBundle) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sources = list(bundle.rows)
    for i, source_a in enumerate(sources):
        for j, source_b in enumerate(sources):
            if j <= i:
                continue
            for layer in range(28):
                xa, cfg_a = source_feature_matrix(bundle, source_a, layer=layer)
                xb, cfg_b = source_feature_matrix(bundle, source_b, layer=layer)
                common_poolings = sorted(
                    set(pool for pool, _ in cfg_a) & set(pool for pool, _ in cfg_b),
                    key=lambda p: POOLING_ORDER.index(p) if p in POOLING_ORDER else 999,
                )
                if not common_poolings:
                    cka_val = float("nan")
                    n_features = 0
                else:
                    order_a = [cfg_a.index((p, layer)) for p in common_poolings]
                    order_b = [cfg_b.index((p, layer)) for p in common_poolings]
                    cka_val = linear_cka(xa[:, order_a], xb[:, order_b])
                    n_features = len(common_poolings)
                rows.append(
                    {
                        "segment": bundle.segment,
                        "source_a": source_a,
                        "source_b": source_b,
                        "layer": int(layer),
                        "cka": cka_val,
                        "n_features": n_features,
                    }
                )
    return rows


def parse_pair_summary_metrics(data: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Optional[float]]]:
    out: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}
    poolings = data.get("poolings", {})
    if not isinstance(poolings, dict):
        return out
    for pooling, payload in poolings.items():
        p = normalize_pooling(pooling)
        if not isinstance(payload, dict):
            continue
        layers = payload.get("layers", [])
        if isinstance(layers, list):
            for r in layers:
                if not isinstance(r, dict):
                    continue
                layer = parse_int(r.get("layer"))
                auc = parse_float(r.get("auc"))
                if layer is None or auc is None:
                    continue
                out[(p, int(layer))] = {
                    "auc": float(auc),
                    "accuracy": parse_float(r.get("accuracy")),
                    "f1": parse_float(r.get("f1")),
                }
        best = payload.get("best")
        if isinstance(best, dict):
            layer = parse_int(best.get("layer"))
            auc = parse_float(best.get("auc"))
            if layer is not None and auc is not None and (p, int(layer)) not in out:
                out[(p, int(layer))] = {
                    "auc": float(auc),
                    "accuracy": parse_float(best.get("accuracy")),
                    "f1": parse_float(best.get("f1")),
                }
    return out


def build_auroc_profile_matrix(
    model_root: Path,
    segment: str,
    seg_rows: Sequence[str],
    seg_cols: Sequence[str],
    out_dir: Path,
    resume: bool,
) -> Tuple[pd.DataFrame, np.ndarray]:
    manifest_path = out_dir / f"{segment}_probe_manifest.csv"
    matrix_path = out_dir / f"{segment}_profile_matrix.npy"
    if resume and manifest_path.exists() and matrix_path.exists():
        return pd.read_csv(manifest_path), np.load(matrix_path)

    metrics_by_pair: Dict[Tuple[str, str], Dict[Tuple[str, int], Dict[str, Optional[float]]]] = {}
    all_probe_keys: set[Tuple[str, str, int]] = set()
    for source_dataset in seg_rows:
        for target_dataset in seg_cols:
            path = pair_summary_path(model_root, source_dataset, target_dataset)
            if not path.exists():
                raise FileNotFoundError(f"Missing pair summary: {path}")
            metrics = parse_pair_summary_metrics(read_json(path, default={}))
            metrics_by_pair[(source_dataset, target_dataset)] = metrics
            for pooling, layer in metrics.keys():
                all_probe_keys.add((source_dataset, pooling, int(layer)))

    probe_rows: List[Dict[str, Any]] = []
    ordered_keys = sorted(
        all_probe_keys,
        key=lambda x: (
            list(seg_rows).index(x[0]),
            x[2],
            POOLING_ORDER.index(x[1]) if x[1] in POOLING_ORDER else 999,
            x[1],
        ),
    )
    matrix = np.full((len(ordered_keys), len(seg_cols)), np.nan, dtype=np.float64)
    for i, (source_dataset, pooling, layer) in enumerate(ordered_keys):
        probe_rows.append(
            {
                "segment": segment,
                "source_dataset": source_dataset,
                "source_short": short_name(source_dataset),
                "pooling": pooling,
                "layer": int(layer),
            }
        )
        for j, target_dataset in enumerate(seg_cols):
            metrics = metrics_by_pair.get((source_dataset, target_dataset), {})
            auc = metrics.get((pooling, layer), {}).get("auc")
            if auc is not None:
                matrix[i, j] = float(auc)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probe_rows).to_csv(manifest_path, index=False)
    np.save(matrix_path, matrix)
    return pd.DataFrame(probe_rows), matrix


def ordered_heatmap_indices(linkage_mat: np.ndarray) -> List[int]:
    dendro = dendrogram(linkage_mat, no_plot=True)
    return [int(x) for x in dendro["leaves"]]


def plot_dendrogram(path: Path, linkage_mat: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    dendrogram(linkage_mat, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def spectral_decomposition(
    similarity: np.ndarray,
    mode: str,
    max_clusters: int,
    random_state: int = 0,
) -> Dict[str, Any]:
    sim = np.asarray(similarity, dtype=np.float64)
    np.fill_diagonal(sim, 1.0)
    if mode == "shifted":
        affinity = (sim + 1.0) / 2.0
        degree = np.sum(affinity, axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
        norm_mat = d_inv_sqrt @ affinity @ d_inv_sqrt
    elif mode == "squared":
        affinity = np.square(sim)
        degree = np.sum(affinity, axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
        norm_mat = d_inv_sqrt @ affinity @ d_inv_sqrt
    elif mode == "signed":
        affinity = sim
        degree = np.sum(np.abs(affinity), axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
        norm_mat = d_inv_sqrt @ affinity @ d_inv_sqrt
    else:
        raise ValueError(f"Unsupported spectral mode: {mode}")

    eigvals, eigvecs = np.linalg.eigh(norm_mat)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k_limit = max(3, min(max_clusters + 1, eigvals.shape[0] - 1))
    gap_scores: List[Tuple[int, float]] = []
    for k in range(2, k_limit):
        gap_scores.append((k, float(eigvals[k - 1] - eigvals[k])))
    chosen_k = 2
    if gap_scores:
        chosen_k = max(gap_scores, key=lambda x: x[1])[0]
    embed_k = eigvecs[:, : max(2, chosen_k)]
    embed_2d = eigvecs[:, :2]

    rows = np.linalg.norm(embed_k, axis=1, keepdims=True)
    rows[rows <= 1e-12] = 1.0
    norm_embed = embed_k / rows
    km = KMeans(n_clusters=chosen_k, n_init=20, random_state=random_state)
    labels = km.fit_predict(norm_embed)

    return {
        "mode": mode,
        "affinity": affinity,
        "normalized_matrix": norm_mat,
        "eigenvalues": eigvals,
        "embedding_2d": embed_2d,
        "embedding_k": embed_k,
        "labels": labels,
        "chosen_k": int(chosen_k),
        "gap_scores": gap_scores,
    }


def plot_spectral_eigenvalues(path: Path, eigenvalues: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = eigenvalues[: min(20, len(eigenvalues))]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(np.arange(1, len(top) + 1), top, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_embedding(path: Path, probe_manifest: pd.DataFrame, embedding_2d: np.ndarray, labels: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    base_names = probe_manifest["source_dataset"].map(dataset_base)
    poolings = probe_manifest["pooling"].map(normalize_pooling)
    for source_base in sorted(base_names.unique()):
        for pooling in sorted(poolings.unique(), key=lambda p: POOLING_ORDER.index(p) if p in POOLING_ORDER else 999):
            mask = (base_names == source_base) & (poolings == pooling)
            if not np.any(mask):
                continue
            ax.scatter(
                embedding_2d[mask.to_numpy(), 0],
                embedding_2d[mask.to_numpy(), 1],
                s=22,
                alpha=0.65,
                label=f"{source_base.replace('Deception-', '')}:{pooling}",
                c=SOURCE_COLOR_MAP.get(source_base, "#4c78a8"),
                marker=POOLING_MARKER_MAP.get(pooling, "o"),
            )
    ax.set_title(title)
    ax.set_xlabel("Eigenvector 1")
    ax.set_ylabel("Eigenvector 2")
    handles, labels_txt = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels_txt, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_segment(
    bundle: SegmentBundle,
    principled_results_dir: Path,
    fixed_pooling: str,
    fixed_layer: int,
    model_root: Path,
    seg_out_dir: Path,
) -> Dict[str, Any]:
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    labels = bundle.sample_manifest["label"].to_numpy(dtype=np.int64)
    mask_pos = labels == 1
    mask_neg = labels == 0
    source_short_labels = [short_name(ds) for ds in bundle.rows]

    # 1. Pearson / Spearman source-level similarity
    ranked_dir = seg_out_dir / "ranked_similarity"
    ranked_dir.mkdir(parents=True, exist_ok=True)
    sources, pearson_mat, pearson_counts, pearson_rows, pearson_per_cfg = source_similarity_matrix(bundle, metric="pearson")
    _, spearman_mat, spearman_counts, spearman_rows, spearman_per_cfg = source_similarity_matrix(bundle, metric="spearman")
    delta_mat = spearman_mat - pearson_mat

    save_matrix_csv(ranked_dir / f"{bundle.segment}_source_similarity_pearson.csv", sources, pearson_mat)
    save_matrix_csv(ranked_dir / f"{bundle.segment}_source_similarity_spearman.csv", sources, spearman_mat)
    save_matrix_csv(ranked_dir / f"{bundle.segment}_source_similarity_spearman_minus_pearson.csv", sources, delta_mat)
    write_csv_rows(
        ranked_dir / f"{bundle.segment}_source_similarity_pearson_long.csv",
        pearson_rows,
        ["segment", "metric", "source_a", "source_b", "similarity", "n_common_configs"],
    )
    write_csv_rows(
        ranked_dir / f"{bundle.segment}_source_similarity_spearman_long.csv",
        spearman_rows,
        ["segment", "metric", "source_a", "source_b", "similarity", "n_common_configs"],
    )
    write_csv_rows(
        ranked_dir / f"{bundle.segment}_source_similarity_per_config.csv",
        pearson_per_cfg + spearman_per_cfg,
        ["segment", "metric", "source_a", "source_b", "pooling", "layer", "similarity"],
    )

    plot_small_heatmap(
        ranked_dir / f"{bundle.segment}_source_similarity_pearson_heatmap.png",
        pearson_mat,
        source_short_labels,
        source_short_labels,
        f"{bundle.segment}: source similarity (Pearson)",
        "Pearson similarity",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_small_heatmap(
        ranked_dir / f"{bundle.segment}_source_similarity_spearman_heatmap.png",
        spearman_mat,
        source_short_labels,
        source_short_labels,
        f"{bundle.segment}: source similarity (Spearman)",
        "Spearman similarity",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_small_heatmap(
        ranked_dir / f"{bundle.segment}_source_similarity_delta_heatmap.png",
        delta_mat,
        source_short_labels,
        source_short_labels,
        f"{bundle.segment}: Spearman - Pearson",
        "Delta similarity",
        cmap="RdBu_r",
        center=0.0,
    )

    scatter_rows: List[Dict[str, Any]] = []
    for source_a in bundle.rows:
        for source_b in bundle.rows:
            if source_b <= source_a:
                continue
            common_cfgs = common_probe_configs(bundle, source_a, source_b)
            idx_map = source_config_index(bundle)
            for pooling, layer in common_cfgs:
                vec_a = bundle.matrix[idx_map[source_a][(pooling, layer)]]
                vec_b = bundle.matrix[idx_map[source_b][(pooling, layer)]]
                scatter_rows.append(
                    {
                        "segment": bundle.segment,
                        "source_a": source_a,
                        "source_b": source_b,
                        "pooling": pooling,
                        "layer": int(layer),
                        "pearson": safe_corr(vec_a, vec_b),
                        "spearman": safe_spearman(vec_a, vec_b),
                    }
                )
    scatter_df = pd.DataFrame(scatter_rows)
    scatter_df.to_csv(ranked_dir / f"{bundle.segment}_pearson_vs_spearman_points.csv", index=False)
    discrete_legend = {
        "mean": ("mean", "o"),
        "max": ("max", "s"),
        "last": ("last", "^"),
        "attn": ("attn", "D"),
    }
    plot_scatter(
        ranked_dir / f"{bundle.segment}_pearson_vs_spearman_scatter.png",
        scatter_df["pearson"].to_numpy(dtype=np.float64),
        scatter_df["spearman"].to_numpy(dtype=np.float64),
        f"{bundle.segment}: Pearson vs Spearman by probe config",
        "Pearson similarity",
        "Spearman similarity",
        colors=scatter_df["pooling"].to_numpy(),
        discrete_legend=discrete_legend,
    )

    fixed_auc_path = principled_results_dir / bundle.segment / f"matrix_fixed_{fixed_pooling}_L{fixed_layer}_auc.csv"
    fixed_auc = load_wide_matrix(fixed_auc_path)
    auc_points: List[Dict[str, Any]] = []
    for source_a in bundle.rows:
        for target_b in bundle.rows:
            if source_a == target_b:
                continue
            if target_b not in fixed_auc.columns or source_a not in fixed_auc.index:
                continue
            i = bundle.rows.index(source_a)
            j = bundle.rows.index(target_b)
            auc_val = fixed_auc.at[source_a, target_b]
            auc_points.append(
                {
                    "segment": bundle.segment,
                    "source_dataset": source_a,
                    "target_dataset": target_b,
                    "pearson_similarity": pearson_mat[i, j],
                    "spearman_similarity": spearman_mat[i, j],
                    "auc": auc_val,
                }
            )
    auc_points_df = pd.DataFrame(auc_points)
    auc_points_df.to_csv(ranked_dir / f"{bundle.segment}_source_similarity_vs_fixed_auc_points.csv", index=False)
    pearson_stats = plot_scatter(
        ranked_dir / f"{bundle.segment}_pearson_vs_fixed_auc_scatter.png",
        auc_points_df["pearson_similarity"].to_numpy(dtype=np.float64),
        auc_points_df["auc"].to_numpy(dtype=np.float64),
        f"{bundle.segment}: Pearson source similarity vs fixed {fixed_pooling} L{fixed_layer} AUROC",
        "Pearson source similarity",
        "Fixed-config AUROC",
    )
    spearman_stats = plot_scatter(
        ranked_dir / f"{bundle.segment}_spearman_vs_fixed_auc_scatter.png",
        auc_points_df["spearman_similarity"].to_numpy(dtype=np.float64),
        auc_points_df["auc"].to_numpy(dtype=np.float64),
        f"{bundle.segment}: Spearman source similarity vs fixed {fixed_pooling} L{fixed_layer} AUROC",
        "Spearman source similarity",
        "Fixed-config AUROC",
    )
    summary["ranked_similarity"] = {
        "pearson_csv": str(ranked_dir / f"{bundle.segment}_source_similarity_pearson.csv"),
        "spearman_csv": str(ranked_dir / f"{bundle.segment}_source_similarity_spearman.csv"),
        "delta_csv": str(ranked_dir / f"{bundle.segment}_source_similarity_spearman_minus_pearson.csv"),
        "pearson_vs_auc_stats": pearson_stats,
        "spearman_vs_auc_stats": spearman_stats,
    }

    # 2. CKA
    cka_dir = seg_out_dir / "cka"
    cka_dir.mkdir(parents=True, exist_ok=True)
    cka_sources, cka_mat = source_cka_matrix(bundle)
    save_matrix_csv(cka_dir / f"{bundle.segment}_source_cka.csv", cka_sources, cka_mat)
    plot_small_heatmap(
        cka_dir / f"{bundle.segment}_source_cka_heatmap.png",
        cka_mat,
        source_short_labels,
        source_short_labels,
        f"{bundle.segment}: source CKA",
        "Linear CKA",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    layerwise_rows = layerwise_cka_rows(bundle)
    write_csv_rows(
        cka_dir / f"{bundle.segment}_layerwise_source_pair_cka.csv",
        layerwise_rows,
        ["segment", "source_a", "source_b", "layer", "cka", "n_features"],
    )
    pair_labels = sorted({f"{short_name(r['source_a'])}|{short_name(r['source_b'])}" for r in layerwise_rows})
    pair_index = {label: i for i, label in enumerate(pair_labels)}
    heat = np.full((len(pair_labels), 28), np.nan, dtype=np.float64)
    for row in layerwise_rows:
        label = f"{short_name(row['source_a'])}|{short_name(row['source_b'])}"
        heat[pair_index[label], int(row["layer"])] = float(row["cka"]) if row["cka"] is not None else np.nan
    plot_large_heatmap(
        cka_dir / f"{bundle.segment}_layerwise_source_pair_cka_heatmap.png",
        heat,
        f"{bundle.segment}: layerwise CKA by source pair",
        cmap="viridis",
        colorbar_label="Linear CKA",
        xticklabels=[str(i) for i in range(28)],
        yticklabels=pair_labels,
        vmin=0.0,
        vmax=1.0,
    )
    summary["cka"] = {
        "source_cka_csv": str(cka_dir / f"{bundle.segment}_source_cka.csv"),
        "layerwise_pair_csv": str(cka_dir / f"{bundle.segment}_layerwise_source_pair_cka.csv"),
    }

    # 3. Label-conditioned similarity
    label_dir = seg_out_dir / "label_conditioned"
    label_dir.mkdir(parents=True, exist_ok=True)
    _, pos_mat, _, pos_rows, _ = source_similarity_matrix(bundle, metric="pearson", sample_mask=mask_pos)
    _, neg_mat, _, neg_rows, _ = source_similarity_matrix(bundle, metric="pearson", sample_mask=mask_neg)
    delta_label = pos_mat - neg_mat
    save_matrix_csv(label_dir / f"{bundle.segment}_source_similarity_positive.csv", bundle.rows, pos_mat)
    save_matrix_csv(label_dir / f"{bundle.segment}_source_similarity_negative.csv", bundle.rows, neg_mat)
    save_matrix_csv(label_dir / f"{bundle.segment}_source_similarity_positive_minus_negative.csv", bundle.rows, delta_label)
    write_csv_rows(
        label_dir / f"{bundle.segment}_source_similarity_positive_long.csv",
        pos_rows,
        ["segment", "metric", "source_a", "source_b", "similarity", "n_common_configs"],
    )
    write_csv_rows(
        label_dir / f"{bundle.segment}_source_similarity_negative_long.csv",
        neg_rows,
        ["segment", "metric", "source_a", "source_b", "similarity", "n_common_configs"],
    )
    plot_small_heatmap(
        label_dir / f"{bundle.segment}_source_similarity_positive_heatmap.png",
        pos_mat,
        source_short_labels,
        source_short_labels,
        f"{bundle.segment}: positive-only source similarity",
        "Pearson similarity",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_small_heatmap(
        label_dir / f"{bundle.segment}_source_similarity_negative_heatmap.png",
        neg_mat,
        source_short_labels,
        source_short_labels,
        f"{bundle.segment}: negative-only source similarity",
        "Pearson similarity",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_small_heatmap(
        label_dir / f"{bundle.segment}_source_similarity_positive_minus_negative_heatmap.png",
        delta_label,
        source_short_labels,
        source_short_labels,
        f"{bundle.segment}: positive - negative source similarity",
        "Delta similarity",
        cmap="RdBu_r",
        center=0.0,
    )
    summary["label_conditioned"] = {
        "positive_csv": str(label_dir / f"{bundle.segment}_source_similarity_positive.csv"),
        "negative_csv": str(label_dir / f"{bundle.segment}_source_similarity_negative.csv"),
        "delta_csv": str(label_dir / f"{bundle.segment}_source_similarity_positive_minus_negative.csv"),
    }

    # 4. AUROC profile clustering
    profile_dir = seg_out_dir / "auroc_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_manifest, profile_matrix = build_auroc_profile_matrix(
        model_root=model_root,
        segment=bundle.segment,
        seg_rows=bundle.rows,
        seg_cols=bundle.cols,
        out_dir=profile_dir,
        resume=False,
    )
    profile_filled = np.where(np.isfinite(profile_matrix), profile_matrix, 0.5)
    signed_profiles = profile_filled - 0.5
    linkage_mat = linkage(pdist(signed_profiles, metric="euclidean"), method="average")
    order = ordered_heatmap_indices(linkage_mat)
    np.save(profile_dir / f"{bundle.segment}_profile_matrix.npy", profile_matrix)
    np.save(profile_dir / f"{bundle.segment}_profile_matrix_signed.npy", signed_profiles)
    write_json(profile_dir / f"{bundle.segment}_cluster_order.json", {"order": order})
    plot_dendrogram(
        profile_dir / f"{bundle.segment}_auroc_profile_dendrogram.png",
        linkage_mat,
        f"{bundle.segment}: AUROC-profile clustering (signed profiles)",
    )
    target_short_labels = [short_name(ds) for ds in bundle.cols]
    plot_large_heatmap(
        profile_dir / f"{bundle.segment}_auroc_profile_heatmap.png",
        signed_profiles[order],
        f"{bundle.segment}: signed AUROC profiles",
        cmap="RdBu_r",
        colorbar_label="Signed AUROC (AUC - 0.5)",
        xticklabels=target_short_labels,
        yticklabels=None,
        center=0.0,
    )
    summary["auroc_profile"] = {
        "probe_manifest_csv": str(profile_dir / f"{bundle.segment}_probe_manifest.csv"),
        "profile_matrix_npy": str(profile_dir / f"{bundle.segment}_profile_matrix.npy"),
    }

    # 5. Spectral clustering
    spectral_dir = seg_out_dir / "spectral"
    spectral_dir.mkdir(parents=True, exist_ok=True)
    probe_similarity = correlation_similarity(bundle.matrix)
    np.save(spectral_dir / f"{bundle.segment}_probe_similarity.npy", probe_similarity)
    spectral_summary: Dict[str, Any] = {}
    for mode in ["shifted", "squared", "signed"]:
        spec = spectral_decomposition(probe_similarity, mode=mode, max_clusters=8, random_state=0)
        mode_dir = spectral_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        np.save(mode_dir / "eigenvalues.npy", spec["eigenvalues"])
        np.save(mode_dir / "embedding_2d.npy", spec["embedding_2d"])
        np.save(mode_dir / "cluster_labels.npy", spec["labels"])
        plot_spectral_eigenvalues(
            mode_dir / f"{bundle.segment}_spectral_{mode}_eigenvalues.png",
            spec["eigenvalues"],
            f"{bundle.segment}: spectral eigenvalues ({mode})",
        )
        plot_spectral_embedding(
            mode_dir / f"{bundle.segment}_spectral_{mode}_embedding.png",
            bundle.probe_manifest,
            spec["embedding_2d"],
            spec["labels"],
            f"{bundle.segment}: spectral embedding ({mode}, k={spec['chosen_k']})",
        )
        assignments = bundle.probe_manifest.copy()
        assignments["cluster"] = spec["labels"]
        assignments["embed_x"] = spec["embedding_2d"][:, 0]
        assignments["embed_y"] = spec["embedding_2d"][:, 1]
        assignments.to_csv(mode_dir / "cluster_assignments.csv", index=False)
        write_json(
            mode_dir / "summary.json",
            {
                "segment": bundle.segment,
                "mode": mode,
                "chosen_k": spec["chosen_k"],
                "gap_scores": spec["gap_scores"],
            },
        )
        spectral_summary[mode] = {
            "chosen_k": spec["chosen_k"],
            "assignments_csv": str(mode_dir / "cluster_assignments.csv"),
        }
    summary["spectral"] = spectral_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probe similarity analysis suite from existing artifacts.")
    parser.add_argument("--ood_results_root", type=str, required=True)
    parser.add_argument("--principled_results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fixed_pooling", type=str, default="mean")
    parser.add_argument("--fixed_layer", type=int, default=15)
    parser.add_argument("--segments", type=str, default="completion,full")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--mirror_png_root", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()
    variant = f"fixed-{normalize_pooling(args.fixed_pooling)}-l{int(args.fixed_layer)}"
    _, ood_model_root = split_root_and_model(Path(args.ood_results_root), model_dir)
    principled_results_dir = Path(args.principled_results_dir)
    mirror_root = Path(args.mirror_png_root) if args.mirror_png_root else None

    if args.output_root:
        run_root = Path(args.output_root) / run_id
    else:
        run_root = (
            Path(args.artifact_root)
            / "runs"
            / "probe_similarity_suite"
            / model_dir
            / "all-segments"
            / "pair-logits"
            / variant
            / run_id
        )

    inputs_dir = run_root / "inputs"
    meta_dir = run_root / "meta"
    chk_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    logs_dir = run_root / "logs"
    for path in [inputs_dir, meta_dir, chk_dir, results_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = meta_dir / "run_manifest.json"
    status_path = meta_dir / "status.json"
    progress_path = chk_dir / "progress.json"
    results_json_path = results_dir / "results.json"

    progress = read_json(progress_path, default={"completed_steps": []})

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "variant": variant,
            "paths": {
                "ood_results_root": str(ood_model_root),
                "principled_results_dir": str(principled_results_dir),
                "run_root": str(run_root),
                "mirror_png_root": str(mirror_root) if mirror_root else None,
            },
            "fixed_pooling": normalize_pooling(args.fixed_pooling),
            "fixed_layer": int(args.fixed_layer),
        },
    )

    def mark(step: str) -> None:
        done = set(progress.get("completed_steps", []))
        done.add(step)
        progress["completed_steps"] = sorted(done)
        progress["updated_at"] = utc_now()
        write_json(progress_path, progress)

    if args.resume and "finished" in set(progress.get("completed_steps", [])) and results_json_path.exists():
        update_status(status_path, "completed", "resume: already finished")
        print(f"[resume] already complete: {run_root}")
        return 0

    update_status(status_path, "running", "loading segment bundles")
    rows_map, cols_map = stage_spec()
    segments = [s.strip() for s in args.segments.split(",") if s.strip()]

    bundle_cache_dir = inputs_dir / "probe_scores"
    segment_bundles: Dict[str, SegmentBundle] = {}
    for segment in segments:
        print(f"[segment] loading {segment}")
        segment_bundles[segment] = load_segment_probe_scores(
            model_root=ood_model_root,
            segment=segment,
            seg_rows=rows_map[segment],
            seg_cols=cols_map[segment],
            cache_dir=bundle_cache_dir,
            resume=args.resume,
        )
    mark("load_probe_scores")

    update_status(status_path, "running", "running analyses")
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now(),
        "model": args.model,
        "model_dir": model_dir,
        "variant": variant,
        "segments": {},
    }
    for segment in segments:
        print(f"[segment] analyzing {segment}")
        seg_out_dir = results_dir / segment
        seg_summary = analyze_segment(
            bundle=segment_bundles[segment],
            principled_results_dir=principled_results_dir,
            fixed_pooling=normalize_pooling(args.fixed_pooling),
            fixed_layer=int(args.fixed_layer),
            model_root=ood_model_root,
            seg_out_dir=seg_out_dir,
        )
        summary["segments"][segment] = seg_summary
    mark("analyze_segments")

    mirrored = mirror_pngs(results_dir, mirror_root)
    summary["mirrored_pngs"] = mirrored
    write_json(results_json_path, summary)
    write_json(results_dir / "summary.json", summary)
    update_status(status_path, "completed", "finished")
    mark("finished")
    print(f"[done] artifacts -> {run_root}")
    if mirror_root is not None:
        print(f"[done] mirrored pngs -> {mirror_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
