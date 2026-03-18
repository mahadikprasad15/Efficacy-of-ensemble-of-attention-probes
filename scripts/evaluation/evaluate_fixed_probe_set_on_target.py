#!/usr/bin/env python3
"""
Lightweight fixed-probe evaluation on a single target dataset/split.

Evaluates frozen single-source probes, optional target self, and optional
combined probes on a target test split for a fixed pooling/layer config.
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
    build_evaluation_probe_specs,
    copy_tree_subset,
    dataset_segment_name,
    evaluate_probe_collection,
    get_git_commit,
    load_linear_probe,
    load_pooled_split,
    mirror_png,
    parse_csv_list,
    target_self_probe_name,
    write_csv,
)


DEFAULT_SINGLE_DATASETS = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
    "Deception-Roleplaying",
]

SHORT_LABELS = {
    "Deception-ConvincingGame": "CG",
    "Deception-HarmPressureChoice": "HPC",
    "Deception-InstructedDeception": "ID",
    "Deception-Mask": "Mask",
    "Deception-AILiar": "AL",
    "Deception-Roleplaying": "RP",
    "Deception-HarmPressureChoice_self": "HPC self",
    "Deception-Roleplaying_self": "RP self",
    "completion_all6": "all6",
    "completion_id_mask": "id_mask",
    "completion_id_mask_al": "id_mask_al",
    "completion_cg_id": "cg_id",
}


def short_label(name: str) -> str:
    return SHORT_LABELS.get(name, name)


def append_log_line(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


def plot_sorted_auc_bar(rows: Sequence[Dict[str, Any]], title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    ordered = sorted(rows, key=lambda r: float(r["auc"]), reverse=True)
    labels = [short_label(str(r["probe_name"])) for r in ordered]
    values = [float(r["auc"]) for r in ordered]
    colors = []
    for row in ordered:
        kind = str(row["probe_kind"])
        if kind == "combined":
            colors.append("#59a14f")
        elif kind == "target_self":
            colors.append("#f28e2b")
        else:
            colors.append("#4c78a8")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9.5, len(labels) * 1.0), 5.6))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed probe set on one target dataset.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--single_probes_root", type=str, default="data/probes")
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--segment", type=str, choices=["completion", "full"], default="completion")
    parser.add_argument("--pooling", type=str, choices=SUPPORTED_POOLINGS, default="mean")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--evaluation_single_datasets", type=str, default=",".join(DEFAULT_SINGLE_DATASETS))
    parser.add_argument("--include_target_self", action="store_true")
    parser.add_argument("--combined_run_root", type=str, default="")
    parser.add_argument("--combined_probe_names", type=str, default="")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--mirror_results_root", type=str, default="")
    parser.add_argument("--gallery_root", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_single_datasets = parse_csv_list(args.evaluation_single_datasets)
    combined_probe_names = parse_csv_list(args.combined_probe_names)
    if not evaluation_single_datasets and not combined_probe_names and not args.include_target_self:
        raise ValueError("At least one probe must be requested")

    model_dir = model_dir_name(args.model)
    target_segment = dataset_segment_name(args.target_dataset, args.segment)
    run_id = args.run_id.strip() or f"{default_run_id()}-{args.target_dataset.split('-')[-1].lower()}-{args.segment}-{args.pooling}-l{args.layer}-fixed-eval-v1"

    artifact_root = Path(args.artifact_root)
    run_root = (
        artifact_root
        / "runs"
        / "fixed_probe_set_eval"
        / model_dir
        / target_segment
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

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "fixed_probe_set_eval"
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
    summary_path = results_dir / "summary.json"
    if args.resume and read_json(status_path, default={}).get("state") == "completed" and summary_path.exists():
        print(f"[skip] run already completed -> {run_root}")
        return 0

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
    append_log_line(log_path, f"[start] run_id={run_id} target={target_segment}")

    activations_root = Path(args.activations_root)
    test_dir = activations_root / model_dir / target_segment / "test"
    x_test, y_test, test_ids = load_pooled_split(test_dir, args.layer, args.pooling, f"{target_segment} test")
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

    probe_specs = build_evaluation_probe_specs(
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
    self_probe_name = target_self_probe_name(probe_specs)

    weights: Dict[str, Tuple[np.ndarray, float]] = {}
    probe_manifest_rows: List[Dict[str, Any]] = []
    for spec in probe_specs:
        probe_name = str(spec["probe_name"])
        if probe_name in weights:
            continue
        probe_path = Path(str(spec["probe_path"]))
        w, b = load_linear_probe(probe_path)
        if int(w.shape[0]) != int(x_test.shape[1]):
            raise ValueError(
                f"Probe dim mismatch for {probe_name}: probe={w.shape[0]} features={x_test.shape[1]}"
            )
        weights[probe_name] = (w, b)
        probe_manifest_rows.append(
            {
                "probe_name": probe_name,
                "probe_kind": str(spec["probe_kind"]),
                "source_dataset": str(spec["source_dataset"]),
                "probe_path": str(probe_path),
                "dim": int(w.shape[0]),
                "is_target_self_probe": bool(self_probe_name is not None and probe_name == self_probe_name),
            }
        )
    write_csv(
        inputs_dir / "probe_manifest.csv",
        probe_manifest_rows,
        ["probe_name", "probe_kind", "source_dataset", "probe_path", "dim", "is_target_self_probe"],
    )

    metrics_by_probe = evaluate_probe_collection(x_test, y_test, probe_specs, weights)
    rows: List[Dict[str, Any]] = []
    for spec in probe_specs:
        probe_name = str(spec["probe_name"])
        metrics = metrics_by_probe[probe_name]
        rows.append(
            {
                "timestamp": utc_now_iso(),
                "target_dataset": target_segment,
                "probe_name": probe_name,
                "probe_kind": str(spec["probe_kind"]),
                "source_dataset": str(spec["source_dataset"]),
                "probe_path": str(spec["probe_path"]),
                "auc": float(metrics["auc"]),
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "count": int(metrics["count"]),
                "logit_mean": float(metrics["logit_mean"]),
                "logit_std": float(metrics["logit_std"]),
                "is_target_self_probe": bool(self_probe_name is not None and probe_name == self_probe_name),
                "is_combined_probe": bool(str(spec["probe_kind"]) == "combined"),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["auc"]), reverse=True)
    write_csv(
        results_dir / "test_auc_by_probe.csv",
        rows_sorted,
        [
            "timestamp",
            "target_dataset",
            "probe_name",
            "probe_kind",
            "source_dataset",
            "probe_path",
            "auc",
            "accuracy",
            "f1",
            "count",
            "logit_mean",
            "logit_std",
            "is_target_self_probe",
            "is_combined_probe",
        ],
    )

    plot_path = plots_dir / "test_auc_sorted_bar.png"
    plot_sorted_auc_bar(rows_sorted, f"{target_segment}: fixed probes on test", plot_path)

    gallery_paths: List[str] = []
    if gallery_root:
        gallery_png = gallery_root / f"{args.target_dataset.replace('Deception-', '').lower()}_{args.segment}_{args.pooling}_l{args.layer}_fixed_probe_set_test_auc_sorted_bar.png"
        mirrored = mirror_png(plot_path, gallery_png)
        if mirrored:
            gallery_paths.append(mirrored)

    summary = {
        "run_id": run_id,
        "completed_at": utc_now_iso(),
        "model": args.model,
        "target_dataset": target_segment,
        "segment": args.segment,
        "pooling": args.pooling,
        "layer": int(args.layer),
        "evaluation_single_datasets": evaluation_single_datasets,
        "include_target_self": bool(args.include_target_self),
        "combined_probe_names": combined_probe_names,
        "n_test_samples": int(x_test.shape[0]),
        "best_probe": rows_sorted[0]["probe_name"] if rows_sorted else "",
        "best_auc": float(rows_sorted[0]["auc"]) if rows_sorted else np.nan,
        "artifacts": {
            "target_stats_json": str(inputs_dir / "target_stats.json"),
            "probe_manifest_csv": str(inputs_dir / "probe_manifest.csv"),
            "test_auc_by_probe_csv": str(results_dir / "test_auc_by_probe.csv"),
            "test_auc_sorted_bar_png": str(plot_path),
        },
        "gallery_paths": gallery_paths,
    }
    write_json(summary_path, summary)

    copy_tree_subset(run_root, external_run_root, ["meta", "checkpoints", "inputs", "results", "logs"])
    write_json(status_path, {"state": "completed", "message": "completed successfully", "updated_at": utc_now_iso()})
    append_log_line(log_path, f"[done] run root -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
