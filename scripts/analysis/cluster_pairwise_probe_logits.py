#!/usr/bin/env python3
"""
Assemble probe-by-sample logit matrices from pairwise OOD artifacts and run
hierarchical clustering over probes.

Expected input artifacts per pair:
  results/ood_evaluation/<model_dir>/from-<source>/to-<target>/
    - pair_logits.safetensors
    - pair_logits_manifest.json

Outputs per segment:
  - probe_manifest.csv
  - sample_manifest.csv
  - score_matrix.npy
  - similarity_matrix.npy
  - distance_matrix.npy
  - linkage.npy
  - probe_manifest_clustered.csv
  - dendrogram.png
  - similarity_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    m = {
        "ConvincingGame": "CG",
        "HarmPressureChoice": "HPC",
        "InstructedDeception": "ID",
        "Mask": "M",
        "AILiar": "AL",
        "InsiderTrading": "IT",
        "Roleplaying": "RP",
    }
    return f"{m.get(base, base)}-{'c' if seg == 'completion' else 'f'}"


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
            "Deception-InsiderTrading-SallyConcat-completion",
            "Deception-Roleplaying-completion",
        ],
        "full": [
            "Deception-ConvincingGame-full",
            "Deception-HarmPressureChoice-full",
            "Deception-InstructedDeception-full",
            "Deception-Mask-full",
            "Deception-AILiar-full",
            "Deception-InsiderTrading-SallyConcat-full",
            "Deception-Roleplaying-full",
        ],
    }
    return rows, cols


def pair_manifest_path(model_root: Path, source_dataset: str, target_dataset: str) -> Path:
    return model_root / f"from-{source_dataset}" / f"to-{target_dataset}" / "pair_logits_manifest.json"


def pair_tensor_path(model_root: Path, source_dataset: str, target_dataset: str) -> Path:
    return model_root / f"from-{source_dataset}" / f"to-{target_dataset}" / "pair_logits.safetensors"


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


def compute_similarity_and_distance(matrix: np.ndarray, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    if metric == "correlation":
        sim = correlation_similarity(matrix)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        return sim, dist

    if metric == "spearman":
        ranked = np.apply_along_axis(rankdata, 1, matrix)
        sim = correlation_similarity(ranked)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        return sim, dist

    condensed = pdist(matrix, metric=metric)
    dist = squareform(condensed)
    if metric == "cosine":
        sim = 1.0 - dist
    else:
        sim = np.full_like(dist, np.nan)
    np.fill_diagonal(dist, 0.0)
    return sim, dist


def plot_dendrogram(path: Path, linkage_mat: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(18, 6))
    dendrogram(linkage_mat, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_similarity_heatmap(path: Path, similarity: np.ndarray, order: Sequence[int], title: str) -> None:
    ordered = similarity[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(ordered, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster probes from pairwise per-sample logits.")
    parser.add_argument("--ood_results_root", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--segments", type=str, default="completion,full")
    parser.add_argument("--distance", type=str, default="correlation", choices=["correlation", "spearman", "euclidean", "cosine"])
    parser.add_argument("--linkage", type=str, default="average", choices=["average", "complete", "single", "ward"])
    parser.add_argument("--output_root", type=str, default="artifacts/runs/probe_logit_clustering")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.linkage == "ward" and args.distance not in {"euclidean"}:
        raise ValueError("Ward linkage requires Euclidean distance.")
    model_dir = args.model.replace("/", "_")
    _, ood_model_root = split_root_and_model(Path(args.ood_results_root), model_dir)
    run_id = args.run_id or utc_run_id()
    run_root = Path(args.output_root) / model_dir / run_id
    meta_dir = run_root / "meta"
    results_dir = run_root / "results"
    meta_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.json"
    if args.resume and summary_path.exists():
        print(f"[resume] summary already exists at {summary_path}")
        return 0

    rows_map, cols_map = stage_spec()
    segments = [s.strip() for s in args.segments.split(",") if s.strip()]

    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "ood_results_root": str(ood_model_root),
            "segments": segments,
            "distance": args.distance,
            "linkage": args.linkage,
        },
    )

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now(),
        "model": args.model,
        "distance": args.distance,
        "linkage": args.linkage,
        "segments": {},
    }

    for segment in segments:
        seg_rows = rows_map[segment]
        seg_cols = cols_map[segment]
        seg_dir = results_dir / segment
        seg_dir.mkdir(parents=True, exist_ok=True)
        print(f"[segment] building {segment}")

        sample_rows: List[Dict[str, Any]] = []
        sample_offsets: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for target_dataset in seg_cols:
            manifest_path, source_used = find_any_manifest_for_target(ood_model_root, seg_rows, target_dataset)
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
            manifest_path, target_used = find_any_manifest_for_source(ood_model_root, source_dataset, seg_cols)
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
                manifest_path = pair_manifest_path(ood_model_root, source_dataset, target_dataset)
                tensor_path = pair_tensor_path(ood_model_root, source_dataset, target_dataset)
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
                    score_key = str(rec["score_key"])
                    values = tensors[score_key].detach().cpu().numpy().astype(np.float32)
                    matrix[row_idx, start:end] = values

        if np.isnan(matrix).any():
            missing = int(np.isnan(matrix).sum())
            raise RuntimeError(f"Segment {segment} matrix still has {missing} missing cells")

        sim, dist = compute_similarity_and_distance(matrix, args.distance)
        condensed = squareform(dist, checks=False)
        linkage_mat = linkage(condensed, method=args.linkage)
        dendro = dendrogram(linkage_mat, no_plot=True)
        order = [int(x) for x in dendro["leaves"]]

        np.save(seg_dir / "score_matrix.npy", matrix)
        np.save(seg_dir / "similarity_matrix.npy", sim)
        np.save(seg_dir / "distance_matrix.npy", dist)
        np.save(seg_dir / "linkage.npy", linkage_mat)
        write_json(seg_dir / "cluster_order.json", {"order": order})

        write_csv_rows(
            seg_dir / "probe_manifest.csv",
            probe_rows,
            ["segment", "source_dataset", "source_short", "pooling", "layer", "score_key", "target_reference"],
        )
        write_csv_rows(
            seg_dir / "sample_manifest.csv",
            sample_rows,
            ["segment", "target_dataset", "target_short", "source_reference", "global_sample_index", "target_sample_index", "sample_id", "label"],
        )
        clustered_probe_rows = [dict(probe_rows[idx], cluster_rank=rank) for rank, idx in enumerate(order)]
        write_csv_rows(
            seg_dir / "probe_manifest_clustered.csv",
            clustered_probe_rows,
            ["segment", "source_dataset", "source_short", "pooling", "layer", "score_key", "target_reference", "cluster_rank"],
        )

        plot_dendrogram(seg_dir / "dendrogram.png", linkage_mat, f"{segment}: probe clustering ({args.distance}, {args.linkage})")
        plot_similarity_heatmap(seg_dir / "similarity_heatmap.png", sim, order, f"{segment}: reordered similarity")

        summary["segments"][segment] = {
            "num_probes": int(matrix.shape[0]),
            "num_samples": int(matrix.shape[1]),
            "probe_manifest_csv": str(seg_dir / "probe_manifest.csv"),
            "sample_manifest_csv": str(seg_dir / "sample_manifest.csv"),
            "score_matrix_npy": str(seg_dir / "score_matrix.npy"),
            "similarity_matrix_npy": str(seg_dir / "similarity_matrix.npy"),
            "distance_matrix_npy": str(seg_dir / "distance_matrix.npy"),
            "linkage_npy": str(seg_dir / "linkage.npy"),
            "clustered_probe_manifest_csv": str(seg_dir / "probe_manifest_clustered.csv"),
            "dendrogram_png": str(seg_dir / "dendrogram.png"),
            "similarity_heatmap_png": str(seg_dir / "similarity_heatmap.png"),
        }

    write_json(summary_path, summary)
    print(f"[done] wrote clustering run to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
