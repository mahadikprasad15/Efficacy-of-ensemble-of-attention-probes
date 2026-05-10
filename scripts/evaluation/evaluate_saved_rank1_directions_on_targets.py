#!/usr/bin/env python3
"""
Evaluate saved target-learned rank-1 directions on one or more target datasets.

This script is intended for the post-hoc transfer step after a completed
`run_target_rank1_source_sweep.py` run. It:

1. Loads saved learned directions from the source-sweep run.
2. Loads pooled target test activations for each requested evaluation target.
3. Evaluates baseline and transformed performance for:
   - single-source probes,
   - optional target self probe,
   - optional combined probes.
4. Writes per-target matrices, summaries, plots, and mirrored outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))

from combined_probe_pipeline import (  # noqa: E402
    default_run_id,
    ensure_dir,
    model_dir_name,
    read_json,
    utc_now_iso,
    write_json,
)
from evaluate_target_rank1_probe_transfer import (  # noqa: E402
    SUPPORTED_POOLINGS,
    apply_rank1_transform_np,
    build_evaluation_probe_specs,
    compute_metrics,
    copy_tree_subset,
    dataset_segment_name,
    get_git_commit,
    load_linear_probe,
    metrics_to_rows,
    mirror_png,
    parse_csv_list,
    slugify_dataset_base,
    target_self_probe_name,
    write_csv,
)


DEFAULT_DIRECTION_SOURCES = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
]
DEFAULT_TARGET_DATASETS = [
    "Deception-InsiderTrading-SallyConcat",
    "Deception-Roleplaying",
]
SHORT_LABELS = {
    "Deception-ConvincingGame": "CG",
    "Deception-HarmPressureChoice": "HPC",
    "Deception-InstructedDeception": "ID",
    "Deception-Mask": "Mask",
    "Deception-AILiar": "AL",
    "Deception-InsiderTrading-SallyConcat_self": "IT-self",
    "Deception-Roleplaying_self": "RP-self",
    "completion_all6": "all6",
    "completion_id_mask": "id_mask",
    "completion_id_mask_al": "id_mask_al",
    "completion_cg_id": "cg_id",
}


def append_log_line(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


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
    *,
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


def evaluate_probe_logits(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, float]:
    logits = x @ w + b
    metrics = compute_metrics(y, logits)
    metrics["logit_mean"] = float(np.mean(logits))
    metrics["logit_std"] = float(np.std(logits))
    return metrics


def evaluate_probe_collection(
    x: np.ndarray,
    y: np.ndarray,
    probe_specs: Sequence[Dict[str, Any]],
    weights: Dict[str, Tuple[np.ndarray, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for spec in probe_specs:
        w, b = weights[str(spec["probe_name"])]
        out[str(spec["probe_name"])] = evaluate_probe_logits(x, y, w, b)
    return out


def write_matrix_csv(path: Path, row_labels: Sequence[str], col_labels: Sequence[str], matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["direction_source_dataset"] + list(col_labels))
        for i, row_name in enumerate(row_labels):
            writer.writerow([row_name] + [("" if np.isnan(v) else float(v)) for v in matrix[i]])


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    output_path: Path,
    *,
    fmt: str = ".3f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "YlGnBu",
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(9.0, len(col_labels) * 0.95), max(4.8, len(row_labels) * 0.72)))
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
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_matched_delta_bar(rows: Sequence[Dict[str, Any]], target_label: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    labels = [SHORT_LABELS.get(str(r["direction_source_dataset"]), str(r["direction_source_dataset"])) for r in rows]
    values = [float(r["matched_delta_auc"]) if str(r["matched_probe_dataset"]) else 0.0 for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.15), 5))
    ax.bar(x, values, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta AUROC vs target baseline")
    ax.set_title(f"Matched-probe delta on {target_label}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def short_label(name: str) -> str:
    return SHORT_LABELS.get(name, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved learned rank-1 directions on one or more targets.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--single_probes_root", type=str, default="data/probes")
    parser.add_argument("--source_sweep_root", type=str, required=True)
    parser.add_argument("--target_datasets", type=str, default=",".join(DEFAULT_TARGET_DATASETS))
    parser.add_argument("--segment", type=str, choices=["completion", "full"], default="completion")
    parser.add_argument("--pooling", type=str, choices=SUPPORTED_POOLINGS, default="mean")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--direction_source_datasets", type=str, default=",".join(DEFAULT_DIRECTION_SOURCES))
    parser.add_argument("--evaluation_single_datasets", type=str, default="")
    parser.add_argument("--include_target_self", action="store_true")
    parser.add_argument("--combined_run_root", type=str, default="")
    parser.add_argument("--combined_probe_names", type=str, default="")
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
    target_datasets = parse_csv_list(args.target_datasets)
    if not target_datasets:
        raise ValueError("At least one evaluation target dataset is required")
    evaluation_single_datasets = parse_csv_list(args.evaluation_single_datasets) or list(direction_source_datasets)
    combined_probe_names = parse_csv_list(args.combined_probe_names)

    model_dir = model_dir_name(args.model)
    target_segments = [dataset_segment_name(ds, args.segment) for ds in target_datasets]
    targets_slug = "__".join(slugify_dataset_base(ds) for ds in target_datasets)
    run_id = args.run_id.strip() or f"{default_run_id()}-saved-rank1-target-transfer-v1"

    source_sweep_root = Path(args.source_sweep_root)
    source_sweep_manifest = read_json(source_sweep_root / "meta" / "run_manifest.json", default={})
    learned_target_segment = str(
        source_sweep_manifest.get("target_segment")
        or dataset_segment_name(str(source_sweep_manifest.get("target_dataset", "unknown-target")), args.segment)
    )
    learned_target_slug = slugify_dataset_base(learned_target_segment)

    artifact_root = Path(args.artifact_root)
    run_root = (
        artifact_root
        / "runs"
        / "saved_rank1_multitarget_eval"
        / model_dir
        / f"from-{learned_target_slug}"
        / f"to-{targets_slug}"
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

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "saved_rank1_multitarget_eval"
    mirror_root = Path(args.mirror_results_root) if args.mirror_results_root else default_mirror_root
    external_run_root = mirror_root / f"from-{learned_target_slug}" / f"to-{targets_slug}" / f"{args.pooling}_l{args.layer}" / run_id

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

    progress = read_json(progress_path, default={"completed_targets": [], "updated_at": utc_now_iso()})
    completed_targets = set(progress.get("completed_targets", []))

    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "git_commit": get_git_commit(),
            "model": args.model,
            "model_dir": model_dir,
            "segment": args.segment,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "source_sweep_root": str(source_sweep_root),
            "source_sweep_target_segment": learned_target_segment,
            "direction_source_datasets": direction_source_datasets,
            "target_datasets": target_datasets,
            "target_segments": target_segments,
            "evaluation_single_datasets": evaluation_single_datasets,
            "include_target_self": bool(args.include_target_self),
            "combined_run_root": str(Path(args.combined_run_root)) if args.combined_run_root else "",
            "combined_probe_names": combined_probe_names,
            "run_root": str(run_root),
            "mirror_results_root": str(external_run_root),
            "gallery_root": str(gallery_root),
        },
    )
    write_json(status_path, {"state": "running", "message": "starting run", "updated_at": utc_now_iso()})
    append_log_line(log_path, f"[start] run_id={run_id} source_sweep_target={learned_target_segment}")

    direction_manifest_rows: List[Dict[str, Any]] = []
    direction_payloads: Dict[str, Dict[str, Any]] = {}
    for direction_source_dataset in direction_source_datasets:
        source_slug = slugify_dataset_base(direction_source_dataset)
        learned_npz = source_sweep_root / "source_runs" / source_slug / "results" / "learned_direction.npz"
        selection_summary = source_sweep_root / "source_runs" / source_slug / "results" / "selection_summary.json"
        if not learned_npz.exists():
            raise FileNotFoundError(f"Missing learned direction for {direction_source_dataset}: {learned_npz}")
        arr = np.load(str(learned_npz))
        direction = np.asarray(arr["direction"], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError(f"Saved direction norm is zero for {direction_source_dataset}: {learned_npz}")
        direction = (direction / norm).astype(np.float32)
        alpha = float(np.asarray(arr["alpha"]).reshape(-1)[0])
        epoch = int(np.asarray(arr["epoch"]).reshape(-1)[0])
        direction_payloads[direction_source_dataset] = {
            "direction": direction,
            "alpha": alpha,
            "epoch": epoch,
            "npz_path": str(learned_npz),
            "selection_summary": str(selection_summary),
        }
        direction_manifest_rows.append(
            {
                "direction_source_dataset": direction_source_dataset,
                "direction_slug": source_slug,
                "learned_direction_npz": str(learned_npz),
                "selection_summary_json": str(selection_summary),
                "alpha": alpha,
                "epoch": epoch,
            }
        )
    write_csv(
        inputs_dir / "direction_manifest.csv",
        direction_manifest_rows,
        ["direction_source_dataset", "direction_slug", "learned_direction_npz", "selection_summary_json", "alpha", "epoch"],
    )
    write_json(inputs_dir / "source_sweep_manifest.json", source_sweep_manifest)

    metrics_rows: List[Dict[str, Any]] = []
    if metrics_jsonl_path.exists():
        with metrics_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics_rows.append(json.loads(line))

    activations_root = Path(args.activations_root)
    single_probes_root = Path(args.single_probes_root)

    target_summaries: Dict[str, Dict[str, Any]] = {}
    gallery_paths: List[str] = []

    for target_dataset, target_segment in zip(target_datasets, target_segments):
        target_results_dir = results_dir / "by_target" / target_segment
        target_plots_dir = plots_dir / target_segment
        if args.resume and target_segment in completed_targets and (target_results_dir / "summary_by_direction.csv").exists():
            target_summaries[target_segment] = {
                "target_dataset": target_segment,
                "artifacts": {
                    "baseline_metrics_csv": str(target_results_dir / "baseline_metrics.csv"),
                    "test_auc_matrix_csv": str(target_results_dir / "test_auc_matrix.csv"),
                    "test_delta_auc_matrix_csv": str(target_results_dir / "test_delta_auc_matrix.csv"),
                    "summary_by_direction_csv": str(target_results_dir / "summary_by_direction.csv"),
                    "summary_by_probe_csv": str(target_results_dir / "summary_by_probe.csv"),
                    "test_auc_heatmap_png": str(target_plots_dir / "test_auc_heatmap.png"),
                    "test_delta_auc_heatmap_png": str(target_plots_dir / "test_delta_auc_heatmap.png"),
                    "matched_probe_delta_bar_png": str(target_plots_dir / "matched_probe_delta_bar.png"),
                },
            }
            append_log_line(log_path, f"[skip] target {target_segment} already completed")
            continue

        append_log_line(log_path, f"[target] start {target_segment}")
        target_root = activations_root / model_dir / target_segment
        test_dir = target_root / "test"
        x_test, y_test, test_ids = load_pooled_split(
            test_dir,
            args.layer,
            args.pooling,
            f"{target_segment} test",
            use_tqdm=not args.no_tqdm,
        )

        target_inputs_dir = inputs_dir / "targets" / target_segment
        ensure_dir(target_inputs_dir)
        ensure_dir(target_results_dir)
        ensure_dir(target_plots_dir)

        write_json(
            target_inputs_dir / "target_stats.json",
            {
                "target_dataset": target_segment,
                "n_test": int(x_test.shape[0]),
                "dim": int(x_test.shape[1]),
                "test_label_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
                "sample_id_count": len(test_ids),
            },
        )

        probe_specs = build_evaluation_probe_specs(
            single_probes_root=single_probes_root,
            model_dir=model_dir,
            evaluation_single_datasets=evaluation_single_datasets,
            target_dataset=target_dataset,
            segment=args.segment,
            pooling=args.pooling,
            layer=args.layer,
            include_target_self=args.include_target_self,
            combined_run_root=args.combined_run_root,
            combined_probe_names=combined_probe_names,
        )
        self_probe_name = target_self_probe_name(probe_specs)

        probe_weights: Dict[str, Tuple[np.ndarray, float]] = {}
        probe_manifest_rows: List[Dict[str, Any]] = []
        for spec in probe_specs:
            probe_name = str(spec["probe_name"])
            if probe_name in probe_weights:
                continue
            probe_path = Path(str(spec["probe_path"]))
            w, b = load_linear_probe(probe_path)
            if int(w.shape[0]) != int(x_test.shape[1]):
                raise ValueError(
                    f"Probe dim mismatch for {probe_name} on {target_segment}: probe={w.shape[0]} features={x_test.shape[1]}"
                )
            probe_weights[probe_name] = (w, b)
            probe_manifest_rows.append(
                {
                    "probe_name": probe_name,
                    "probe_kind": str(spec["probe_kind"]),
                    "source_dataset": str(spec["source_dataset"]),
                    "probe_path": str(probe_path),
                    "dim": int(w.shape[0]),
                }
            )
        write_csv(
            target_inputs_dir / "probe_manifest.csv",
            probe_manifest_rows,
            ["probe_name", "probe_kind", "source_dataset", "probe_path", "dim"],
        )

        baseline_metrics = evaluate_probe_collection(x_test, y_test, probe_specs, probe_weights)
        baseline_by_probe_split = {(str(name), "test"): metrics for name, metrics in baseline_metrics.items()}
        baseline_rows = metrics_to_rows(
            baseline_metrics,
            probe_specs,
            split="test",
            method="baseline",
            baseline_by_probe_split=baseline_by_probe_split,
            optimization_probe_names=[],
            self_probe_name=self_probe_name,
            extra={
                "direction_source_dataset": "",
                "direction_slug": "",
                "direction_npz": "",
                "alpha": "",
                "epoch": "",
                "learned_target_dataset": learned_target_segment,
                "evaluation_target_dataset": target_segment,
            },
        )
        for row in baseline_rows:
            row["target_dataset"] = target_segment

        transformed_rows_for_target: List[Dict[str, Any]] = []
        for direction_source_dataset in direction_source_datasets:
            payload = direction_payloads[direction_source_dataset]
            x_test_trans = apply_rank1_transform_np(x_test, payload["direction"], float(payload["alpha"]))
            transformed_metrics = evaluate_probe_collection(x_test_trans, y_test, probe_specs, probe_weights)
            rows = metrics_to_rows(
                transformed_metrics,
                probe_specs,
                split="test",
                method="transferred_rank1",
                baseline_by_probe_split=baseline_by_probe_split,
                optimization_probe_names=[],
                self_probe_name=self_probe_name,
                extra={
                    "direction_source_dataset": direction_source_dataset,
                    "direction_slug": slugify_dataset_base(direction_source_dataset),
                    "direction_npz": payload["npz_path"],
                    "alpha": float(payload["alpha"]),
                    "epoch": int(payload["epoch"]),
                    "learned_target_dataset": learned_target_segment,
                    "evaluation_target_dataset": target_segment,
                },
            )
            for row in rows:
                row["target_dataset"] = target_segment
            transformed_rows_for_target.extend(rows)

        target_metrics_rows = baseline_rows + transformed_rows_for_target
        with (target_results_dir / "metrics_rows.jsonl").open("w", encoding="utf-8") as f:
            for row in target_metrics_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        probe_order = [str(spec["probe_name"]) for spec in probe_specs]
        auc_matrix = np.full((len(direction_source_datasets), len(probe_order)), np.nan, dtype=np.float64)
        delta_matrix = np.full((len(direction_source_datasets), len(probe_order)), np.nan, dtype=np.float64)
        baseline_auc_by_probe = {str(name): float(metrics["auc"]) for name, metrics in baseline_metrics.items()}

        for i, direction_source_dataset in enumerate(direction_source_datasets):
            for j, probe_name in enumerate(probe_order):
                rec = next(
                    (
                        r for r in transformed_rows_for_target
                        if str(r["direction_source_dataset"]) == direction_source_dataset
                        and str(r["probe_name"]) == probe_name
                    ),
                    None,
                )
                if rec is None:
                    continue
                auc_matrix[i, j] = float(rec["auc"])
                delta_matrix[i, j] = float(rec["delta_auc_vs_baseline"])

        write_matrix_csv(target_results_dir / "test_auc_matrix.csv", direction_source_datasets, probe_order, auc_matrix)
        write_matrix_csv(target_results_dir / "test_delta_auc_matrix.csv", direction_source_datasets, probe_order, delta_matrix)
        write_csv(
            target_results_dir / "baseline_metrics.csv",
            baseline_rows,
            [
                "timestamp",
                "direction_source_dataset",
                "direction_slug",
                "direction_npz",
                "alpha",
                "epoch",
                "learned_target_dataset",
                "evaluation_target_dataset",
                "target_dataset",
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
            ],
        )

        summary_by_direction_rows: List[Dict[str, Any]] = []
        for i, direction_source_dataset in enumerate(direction_source_datasets):
            direction_values = delta_matrix[i]
            valid_idx = [j for j in range(len(probe_order)) if not np.isnan(direction_values[j])]
            valid_pairs = [(probe_order[j], float(direction_values[j])) for j in valid_idx]
            single_values = [
                float(direction_values[j])
                for j, probe_name in enumerate(probe_order)
                if probe_name in evaluation_single_datasets and not np.isnan(direction_values[j])
            ]
            combined_values = [
                float(direction_values[j])
                for j, probe_name in enumerate(probe_order)
                if probe_name in combined_probe_names and not np.isnan(direction_values[j])
            ]
            self_delta = np.nan
            if self_probe_name and self_probe_name in probe_order:
                self_idx = probe_order.index(self_probe_name)
                if not np.isnan(direction_values[self_idx]):
                    self_delta = float(direction_values[self_idx])

            matched_probe_dataset = direction_source_dataset if direction_source_dataset in probe_order else ""
            matched_delta = np.nan
            matched_transformed_auc = np.nan
            matched_baseline_auc = np.nan
            if matched_probe_dataset:
                matched_idx = probe_order.index(matched_probe_dataset)
                if not np.isnan(direction_values[matched_idx]):
                    matched_delta = float(direction_values[matched_idx])
                    matched_transformed_auc = float(auc_matrix[i, matched_idx])
                    matched_baseline_auc = float(baseline_auc_by_probe[matched_probe_dataset])

            best_probe_name, best_probe_delta = ("", np.nan)
            worst_probe_name, worst_probe_delta = ("", np.nan)
            if valid_pairs:
                best_probe_name, best_probe_delta = max(valid_pairs, key=lambda x: x[1])
                worst_probe_name, worst_probe_delta = min(valid_pairs, key=lambda x: x[1])

            payload = direction_payloads[direction_source_dataset]
            summary_by_direction_rows.append(
                {
                    "direction_source_dataset": direction_source_dataset,
                    "matched_probe_dataset": matched_probe_dataset,
                    "matched_transformed_auc": matched_transformed_auc,
                    "matched_baseline_auc": matched_baseline_auc,
                    "matched_delta_auc": matched_delta,
                    "mean_delta_auc_all_probes": float(np.mean([v for _, v in valid_pairs])) if valid_pairs else np.nan,
                    "mean_delta_auc_single_probes": float(np.mean(single_values)) if single_values else np.nan,
                    "mean_delta_auc_combined_probes": float(np.mean(combined_values)) if combined_values else np.nan,
                    "target_self_delta_auc": self_delta,
                    "best_probe_name": best_probe_name,
                    "best_probe_delta_auc": best_probe_delta,
                    "worst_probe_name": worst_probe_name,
                    "worst_probe_delta_auc": worst_probe_delta,
                    "alpha": float(payload["alpha"]),
                    "epoch": int(payload["epoch"]),
                    "learned_direction_npz": payload["npz_path"],
                }
            )

        summary_by_probe_rows: List[Dict[str, Any]] = []
        for j, probe_name in enumerate(probe_order):
            probe_values = delta_matrix[:, j]
            valid = [(direction_source_datasets[i], float(probe_values[i])) for i in range(len(direction_source_datasets)) if not np.isnan(probe_values[i])]
            best_direction, best_delta = ("", np.nan)
            best_auc = np.nan
            if valid:
                best_direction, best_delta = max(valid, key=lambda x: x[1])
                best_auc = float(auc_matrix[direction_source_datasets.index(best_direction), j])
            spec = next(s for s in probe_specs if str(s["probe_name"]) == probe_name)
            summary_by_probe_rows.append(
                {
                    "probe_name": probe_name,
                    "probe_kind": str(spec["probe_kind"]),
                    "source_dataset": str(spec["source_dataset"]),
                    "baseline_auc": float(baseline_auc_by_probe[probe_name]),
                    "best_direction_source_dataset": best_direction,
                    "best_transformed_auc": best_auc,
                    "best_delta_auc": best_delta,
                }
            )

        write_csv(
            target_results_dir / "summary_by_direction.csv",
            summary_by_direction_rows,
            [
                "direction_source_dataset",
                "matched_probe_dataset",
                "matched_transformed_auc",
                "matched_baseline_auc",
                "matched_delta_auc",
                "mean_delta_auc_all_probes",
                "mean_delta_auc_single_probes",
                "mean_delta_auc_combined_probes",
                "target_self_delta_auc",
                "best_probe_name",
                "best_probe_delta_auc",
                "worst_probe_name",
                "worst_probe_delta_auc",
                "alpha",
                "epoch",
                "learned_direction_npz",
            ],
        )
        write_csv(
            target_results_dir / "summary_by_probe.csv",
            summary_by_probe_rows,
            [
                "probe_name",
                "probe_kind",
                "source_dataset",
                "baseline_auc",
                "best_direction_source_dataset",
                "best_transformed_auc",
                "best_delta_auc",
            ],
        )

        row_labels_short = [short_label(x) for x in direction_source_datasets]
        col_labels_short = [short_label(x) for x in probe_order]
        auc_heatmap_path = target_plots_dir / "test_auc_heatmap.png"
        delta_heatmap_path = target_plots_dir / "test_delta_auc_heatmap.png"
        matched_bar_path = target_plots_dir / "matched_probe_delta_bar.png"
        plot_heatmap(
            auc_matrix,
            row_labels_short,
            col_labels_short,
            f"{target_segment} transformed test AUROC",
            auc_heatmap_path,
            vmin=0.0,
            vmax=1.0,
            cmap="YlGnBu",
        )
        delta_abs = float(np.nanmax(np.abs(delta_matrix))) if np.any(~np.isnan(delta_matrix)) else 0.05
        delta_abs = max(delta_abs, 0.05)
        plot_heatmap(
            delta_matrix,
            row_labels_short,
            col_labels_short,
            f"{target_segment} delta AUROC vs baseline",
            delta_heatmap_path,
            vmin=-delta_abs,
            vmax=delta_abs,
            cmap="RdBu_r",
        )
        plot_matched_delta_bar(summary_by_direction_rows, target_segment, matched_bar_path)

        if gallery_root:
            gallery_paths.extend(
                p for p in [
                    mirror_png(
                        auc_heatmap_path,
                        gallery_root / f"{slugify_dataset_base(target_dataset)}_{args.segment}_{args.pooling}_l{args.layer}_saved_rank1_test_auc_heatmap.png",
                    ),
                    mirror_png(
                        delta_heatmap_path,
                        gallery_root / f"{slugify_dataset_base(target_dataset)}_{args.segment}_{args.pooling}_l{args.layer}_saved_rank1_test_delta_auc_heatmap.png",
                    ),
                    mirror_png(
                        matched_bar_path,
                        gallery_root / f"{slugify_dataset_base(target_dataset)}_{args.segment}_{args.pooling}_l{args.layer}_saved_rank1_matched_probe_delta_bar.png",
                    ),
                ] if p
            )

        target_summaries[target_segment] = {
            "target_dataset": target_segment,
            "n_test_samples": int(x_test.shape[0]),
            "probe_names": probe_order,
            "artifacts": {
                "baseline_metrics_csv": str(target_results_dir / "baseline_metrics.csv"),
                "test_auc_matrix_csv": str(target_results_dir / "test_auc_matrix.csv"),
                "test_delta_auc_matrix_csv": str(target_results_dir / "test_delta_auc_matrix.csv"),
                "summary_by_direction_csv": str(target_results_dir / "summary_by_direction.csv"),
                "summary_by_probe_csv": str(target_results_dir / "summary_by_probe.csv"),
                "test_auc_heatmap_png": str(auc_heatmap_path),
                "test_delta_auc_heatmap_png": str(delta_heatmap_path),
                "matched_probe_delta_bar_png": str(matched_bar_path),
            },
        }

        completed_targets.add(target_segment)
        progress["completed_targets"] = sorted(completed_targets)
        progress["updated_at"] = utc_now_iso()
        write_json(progress_path, progress)
        append_log_line(log_path, f"[target] completed {target_segment}")

    deduped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in metrics_rows:
        key = (
            str(row.get("evaluation_target_dataset", row.get("target_dataset", ""))),
            str(row.get("direction_source_dataset", "")),
            str(row["probe_name"]),
            str(row["method"]),
        )
        deduped[key] = row

    for target_dataset, target_segment in zip(target_datasets, target_segments):
        target_results_path = results_dir / "by_target" / target_segment / "metrics_rows.jsonl"
        if not target_results_path.exists():
            continue
        with target_results_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = (
                    str(row.get("evaluation_target_dataset", row.get("target_dataset", ""))),
                    str(row.get("direction_source_dataset", "")),
                    str(row["probe_name"]),
                    str(row["method"]),
                )
                deduped[key] = row

    all_metrics_rows = list(deduped.values())
    all_metrics_rows.sort(
        key=lambda r: (
            str(r.get("evaluation_target_dataset", r.get("target_dataset", ""))),
            str(r.get("direction_source_dataset", "")),
            str(r["probe_name"]),
            str(r["method"]),
        )
    )
    with metrics_jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_metrics_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    write_csv(
        metrics_long_path,
        all_metrics_rows,
        [
            "timestamp",
            "direction_source_dataset",
            "direction_slug",
            "direction_npz",
            "alpha",
            "epoch",
            "learned_target_dataset",
            "evaluation_target_dataset",
            "target_dataset",
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
        ],
    )

    summary = {
        "run_id": run_id,
        "completed_at": utc_now_iso(),
        "model": args.model,
        "segment": args.segment,
        "pooling": args.pooling,
        "layer": int(args.layer),
        "source_sweep_root": str(source_sweep_root),
        "source_sweep_target_segment": learned_target_segment,
        "direction_source_datasets": direction_source_datasets,
        "target_datasets": target_datasets,
        "evaluation_single_datasets": evaluation_single_datasets,
        "include_target_self": bool(args.include_target_self),
        "combined_probe_names": combined_probe_names,
        "artifacts": {
            "direction_manifest_csv": str(inputs_dir / "direction_manifest.csv"),
            "metrics_long_csv": str(metrics_long_path),
        },
        "target_summaries": target_summaries,
        "gallery_paths": gallery_paths,
    }
    write_json(summary_path, summary)

    if external_run_root:
        copy_tree_subset(run_root, external_run_root, ["meta", "checkpoints", "inputs", "results", "logs"])

    write_json(status_path, {"state": "completed", "message": "completed successfully", "updated_at": utc_now_iso()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
