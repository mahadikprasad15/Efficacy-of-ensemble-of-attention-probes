#!/usr/bin/env python3
"""
Extend the fixed completion mean@L15 AUROC matrix with 4 truth_spec claims datasets.

This script reuses the existing fixed completion matrix for old-old cells and
computes only the missing claims-related cells directly from frozen probes and
cached test activations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT := SCRIPT_DIR.parents[1]))
sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
sys.path.append(str(REPO_ROOT / "scripts" / "evaluation"))

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
    build_evaluation_probe_specs,
    copy_tree_subset,
    dataset_segment_name,
    evaluate_probe_collection,
    get_git_commit,
    load_linear_probe,
    load_pooled_split,
    mirror_png,
    parse_csv_list,
    write_csv,
)


CLAIMS_DATASETS = [
    "Deception-ClaimsDefinitional",
    "Deception-ClaimsEvidential",
    "Deception-ClaimsFictional",
    "Deception-ClaimsLogical",
]

SHORT_LABELS = {
    "Deception-ConvincingGame": "CG",
    "Deception-HarmPressureChoice": "HPC",
    "Deception-InstructedDeception": "ID",
    "Deception-Mask": "Mask",
    "Deception-AILiar": "AL",
    "Deception-InsiderTrading": "IT",
    "Deception-Roleplaying": "RP",
    "Deception-ClaimsDefinitional": "ClaimsDef",
    "Deception-ClaimsEvidential": "ClaimsEvid",
    "Deception-ClaimsFictional": "ClaimsFict",
    "Deception-ClaimsLogical": "ClaimsLogic",
}


def short_label(dataset_name: str) -> str:
    base = dataset_name
    if dataset_name.endswith("-completion"):
        base = dataset_name[: -len("-completion")]
    return SHORT_LABELS.get(base, base.replace("Deception-", ""))


def read_matrix_csv(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = list(header[1:])
        rows: List[str] = []
        matrix_rows: List[List[float]] = []
        for row in reader:
            rows.append(row[0])
            vals: List[float] = []
            for v in row[1:]:
                vals.append(np.nan if v == "" else float(v))
            matrix_rows.append(vals)
    return rows, cols, np.asarray(matrix_rows, dtype=np.float64)


def write_matrix_csv(path: Path, rows: Sequence[str], cols: Sequence[str], matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_dataset"] + list(cols))
        for i, row_name in enumerate(rows):
            out = [row_name]
            for j in range(len(cols)):
                value = matrix[i, j]
                out.append("" if np.isnan(value) else float(value))
            writer.writerow(out)


def plot_heatmap(output_path: Path, rows: Sequence[str], cols: Sequence[str], matrix: np.ndarray, title: str) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(14.0, 8.5))
    im = ax.imshow(matrix, aspect="auto", vmin=0.5, vmax=1.0, cmap="YlGnBu")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([short_label(x) for x in cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([short_label(x) for x in rows])
    ax.set_xlabel("Eval dataset")
    ax.set_ylabel("Probe source dataset")
    ax.set_title(title, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = "NA" if np.isnan(v) else f"{v:.2f}"
            color = "#4a4a4a" if np.isnan(v) or v < 0.75 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend the fixed completion mean@L15 AUROC matrix with claims datasets.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--single_probes_root", type=str, default="data/probes")
    parser.add_argument(
        "--existing_results_dir",
        type=str,
        required=True,
        help="Results dir containing completion/matrix_fixed_mean_L15_auc.csv",
    )
    parser.add_argument("--segment", type=str, choices=["completion"], default="completion")
    parser.add_argument("--pooling", type=str, choices=SUPPORTED_POOLINGS, default="mean")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--claims_datasets", type=str, default=",".join(CLAIMS_DATASETS))
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--mirror_results_root", type=str, default="")
    parser.add_argument("--gallery_root", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def target_job_key(target_dataset_segment: str, evaluation_source_count: int) -> str:
    return f"{target_dataset_segment}__n{evaluation_source_count}"


def main() -> int:
    args = parse_args()
    if args.segment != "completion":
        raise ValueError("This extension script only supports completion for v1.")

    claims_datasets = parse_csv_list(args.claims_datasets)
    model_dir = model_dir_name(args.model)
    run_id = args.run_id.strip() or f"{default_run_id()}-claims-fixed-mean-l15-extension"

    artifact_root = Path(args.artifact_root)
    run_root = (
        artifact_root
        / "runs"
        / "claims_fixed_completion_matrix_extension"
        / model_dir
        / f"{args.pooling}_l{args.layer}"
        / run_id
    )
    inputs_dir = run_root / "inputs"
    results_dir = run_root / "results"
    plots_dir = results_dir / "plots"
    meta_dir = run_root / "meta"
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    for d in [inputs_dir, results_dir, plots_dir, meta_dir, checkpoints_dir, logs_dir]:
        ensure_dir(d)

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "claims_fixed_completion_matrix_extension"
    mirror_root = Path(args.mirror_results_root) if args.mirror_results_root else default_mirror_root
    external_run_root = mirror_root / run_id

    default_gallery_root = (
        Path("results")
        / "ood_evaluation"
        / model_dir
        / "all_pairwise_results_final"
        / "generated_plots_after_id_fix"
    )
    gallery_root = Path(args.gallery_root) if args.gallery_root else default_gallery_root

    status_path = meta_dir / "status.json"
    summary_path = results_dir / "summary.json"
    if args.resume and read_json(status_path, default={}).get("state") == "completed" and summary_path.exists():
        print(f"[skip] run already completed -> {run_root}")
        return 0

    existing_results_dir = Path(args.existing_results_dir)
    existing_auc_path = existing_results_dir / "completion" / f"matrix_fixed_{args.pooling}_L{args.layer}_auc.csv"
    if not existing_auc_path.exists():
        raise FileNotFoundError(f"Missing existing fixed matrix CSV: {existing_auc_path}")

    old_rows, old_cols, old_auc = read_matrix_csv(existing_auc_path)
    old_source_bases = [x[: -len("-completion")] if x.endswith("-completion") else x for x in old_rows]
    old_target_bases = [x[: -len("-completion")] if x.endswith("-completion") else x for x in old_cols]
    all_source_bases = old_source_bases + claims_datasets
    new_row_names = [dataset_segment_name(x, "completion") for x in claims_datasets]
    final_rows = old_rows + new_row_names
    final_cols = old_cols + new_row_names

    write_matrix_csv(results_dir / "existing_fixed_mean_l15_auc_snapshot.csv", old_rows, old_cols, old_auc)
    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "git_commit": get_git_commit(),
            "model": args.model,
            "model_dir": model_dir,
            "existing_results_dir": str(existing_results_dir),
            "existing_auc_path": str(existing_auc_path),
            "claims_datasets": claims_datasets,
            "pooling": args.pooling,
            "layer": int(args.layer),
            "run_root": str(run_root),
        },
    )
    write_json(status_path, {"state": "running", "message": "starting run", "updated_at": utc_now_iso()})

    progress_path = checkpoints_dir / "progress.json"
    progress = read_json(progress_path, default={"completed_targets": []})
    completed_targets = set(str(x) for x in progress.get("completed_targets", []))

    activations_root = Path(args.activations_root)
    probe_cache: Dict[str, Tuple[np.ndarray, float]] = {}
    computed_rows: List[Dict[str, Any]] = []
    per_target_dir = results_dir / "per_target"
    ensure_dir(per_target_dir)

    def load_probe_weights(specs: Sequence[Dict[str, Any]], feature_dim: int) -> Dict[str, Tuple[np.ndarray, float]]:
        weights: Dict[str, Tuple[np.ndarray, float]] = {}
        for spec in specs:
            probe_name = str(spec["probe_name"])
            if probe_name not in probe_cache:
                w, b = load_linear_probe(Path(str(spec["probe_path"])))
                if int(w.shape[0]) != int(feature_dim):
                    raise ValueError(f"Probe dim mismatch for {probe_name}: probe={w.shape[0]} features={feature_dim}")
                probe_cache[probe_name] = (w, b)
            weights[probe_name] = probe_cache[probe_name]
        return weights

    def evaluate_target(target_base: str, evaluation_source_bases: Sequence[str]) -> List[Dict[str, Any]]:
        target_segment = dataset_segment_name(target_base, args.segment)
        x_test, y_test, _ = load_pooled_split(
            activations_root / model_dir / target_segment / "test",
            args.layer,
            args.pooling,
            f"{target_segment} test",
        )
        specs = build_evaluation_probe_specs(
            single_probes_root=Path(args.single_probes_root),
            model_dir=model_dir,
            evaluation_single_datasets=evaluation_source_bases,
            target_dataset=target_base,
            segment=args.segment,
            pooling=args.pooling,
            layer=args.layer,
            include_target_self=False,
            combined_run_root="",
            combined_probe_names=[],
        )
        weights = load_probe_weights(specs, feature_dim=int(x_test.shape[1]))
        metrics_by_probe = evaluate_probe_collection(x_test, y_test, specs, weights)
        rows: List[Dict[str, Any]] = []
        for spec in specs:
            probe_name = str(spec["probe_name"])
            metrics = metrics_by_probe[probe_name]
            rows.append(
                {
                    "timestamp": utc_now_iso(),
                    "row_dataset": dataset_segment_name(str(spec["source_dataset"]), args.segment),
                    "source_dataset": str(spec["source_dataset"]),
                    "col_dataset": target_segment,
                    "target_dataset": target_base,
                    "probe_path": str(spec["probe_path"]),
                    "probe_kind": str(spec["probe_kind"]),
                    "auc": float(metrics["auc"]),
                    "accuracy": float(metrics["accuracy"]),
                    "f1": float(metrics["f1"]),
                    "count": int(metrics["count"]),
                    "cell_source": "evaluated",
                }
            )
        return rows

    jobs: List[Tuple[str, List[str]]] = []
    for target_base in claims_datasets:
        jobs.append((target_base, list(all_source_bases)))
    for target_base in old_target_bases:
        jobs.append((target_base, list(claims_datasets)))

    for target_base, eval_sources in jobs:
        key = target_job_key(dataset_segment_name(target_base, args.segment), len(eval_sources))
        target_csv = per_target_dir / f"{key}.csv"
        if args.resume and key in completed_targets and target_csv.exists():
            with target_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                computed_rows.extend([dict(r) for r in reader])
            continue

        rows = evaluate_target(target_base, eval_sources)
        write_csv(
            target_csv,
            rows,
            ["timestamp", "row_dataset", "source_dataset", "col_dataset", "target_dataset", "probe_path", "probe_kind", "auc", "accuracy", "f1", "count", "cell_source"],
        )
        computed_rows.extend(rows)
        completed_targets.add(key)
        progress["completed_targets"] = sorted(completed_targets)
        progress["updated_at"] = utc_now_iso()
        write_json(progress_path, progress)

    write_csv(
        results_dir / "metrics_long.csv",
        computed_rows,
        ["timestamp", "row_dataset", "source_dataset", "col_dataset", "target_dataset", "probe_path", "probe_kind", "auc", "accuracy", "f1", "count", "cell_source"],
    )

    final_auc = np.full((len(final_rows), len(final_cols)), np.nan, dtype=np.float64)
    provenance_rows: List[Dict[str, Any]] = []

    old_row_index = {name: i for i, name in enumerate(old_rows)}
    old_col_index = {name: j for j, name in enumerate(old_cols)}
    final_row_index = {name: i for i, name in enumerate(final_rows)}
    final_col_index = {name: j for j, name in enumerate(final_cols)}

    for row_name in old_rows:
        for col_name in old_cols:
            i_old = old_row_index[row_name]
            j_old = old_col_index[col_name]
            i_new = final_row_index[row_name]
            j_new = final_col_index[col_name]
            final_auc[i_new, j_new] = old_auc[i_old, j_old]
            provenance_rows.append(
                {
                    "row_dataset": row_name,
                    "col_dataset": col_name,
                    "cell_source": "existing_fixed_matrix",
                    "auc": "" if np.isnan(old_auc[i_old, j_old]) else float(old_auc[i_old, j_old]),
                }
            )

    for row in computed_rows:
        row_name = str(row["row_dataset"])
        col_name = str(row["col_dataset"])
        i = final_row_index[row_name]
        j = final_col_index[col_name]
        final_auc[i, j] = float(row["auc"])
        provenance_rows.append(
            {
                "row_dataset": row_name,
                "col_dataset": col_name,
                "cell_source": "evaluated",
                "auc": float(row["auc"]),
            }
        )

    write_matrix_csv(results_dir / "matrix_fixed_mean_L15_auc.csv", final_rows, final_cols, final_auc)
    write_csv(results_dir / "cell_provenance.csv", provenance_rows, ["row_dataset", "col_dataset", "cell_source", "auc"])

    plot_path = plots_dir / "heatmap_fixed_mean_L15_auc.png"
    plot_heatmap(
        plot_path,
        final_rows,
        final_cols,
        final_auc,
        title="Fixed Config AUROC | Completion | mean L15 | Claims Extended",
    )

    gallery_paths: List[str] = []
    if gallery_root:
        gallery_png = gallery_root / "completion_fixed_mean_l15_auc_claims_extended.png"
        mirrored = mirror_png(plot_path, gallery_png)
        if mirrored:
            gallery_paths.append(mirrored)

    summary = {
        "run_id": run_id,
        "completed_at": utc_now_iso(),
        "model": args.model,
        "pooling": args.pooling,
        "layer": int(args.layer),
        "claims_datasets": claims_datasets,
        "existing_auc_path": str(existing_auc_path),
        "final_shape": [int(final_auc.shape[0]), int(final_auc.shape[1])],
        "num_existing_cells": int(len(old_rows) * len(old_cols)),
        "num_evaluated_cells": int(len(computed_rows)),
        "artifacts": {
            "metrics_long_csv": str(results_dir / "metrics_long.csv"),
            "matrix_fixed_mean_l15_auc_csv": str(results_dir / "matrix_fixed_mean_L15_auc.csv"),
            "cell_provenance_csv": str(results_dir / "cell_provenance.csv"),
            "heatmap_png": str(plot_path),
        },
        "gallery_paths": gallery_paths,
    }
    write_json(summary_path, summary)

    copy_tree_subset(run_root, external_run_root, ["meta", "checkpoints", "inputs", "results", "logs"])
    write_json(status_path, {"state": "completed", "message": "completed successfully", "updated_at": utc_now_iso()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
