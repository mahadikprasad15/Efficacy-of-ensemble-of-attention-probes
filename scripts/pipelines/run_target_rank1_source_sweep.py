#!/usr/bin/env python3
"""
Run a held-out source sweep for learned target-side rank-1 direction removal.

For a fixed target dataset/segment/pooling/layer, this wrapper:
1. Trains one rank-1 direction per optimization source dataset.
2. Evaluates each learned direction on a shared evaluation probe set.
3. Aggregates learned test AUROC and delta-vs-baseline matrices.
4. Mirrors wrapper artifacts and selected plots into results/ood_evaluation.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "scripts", "utils"))
from combined_probe_pipeline import (  # noqa: E402
    default_run_id,
    ensure_dir,
    model_dir_name,
    read_json,
    utc_now_iso,
    write_json,
)


DEFAULT_SOURCE_BASES = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
]

DEFAULT_COMBINED_PROBES = [
    "completion_all6",
    "completion_id_mask",
    "completion_id_mask_al",
    "completion_cg_id",
]

PROBE_SHORT_LABELS = {
    "Deception-ConvincingGame": "CG",
    "Deception-HarmPressureChoice": "HPC",
    "Deception-InstructedDeception": "ID",
    "Deception-Mask": "Mask",
    "Deception-AILiar": "AL",
    "Deception-Roleplaying_self": "RP self",
    "completion_all6": "all6",
    "completion_id_mask": "id_mask",
    "completion_id_mask_al": "id_mask_al",
    "completion_cg_id": "cg_id",
}


def parse_csv_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def dataset_segment_name(dataset_base: str, segment: str) -> str:
    if dataset_base.endswith("-completion") or dataset_base.endswith("-full"):
        return dataset_base
    return f"{dataset_base}-{segment}"


def slugify_dataset_base(dataset_base: str) -> str:
    return dataset_base.replace("Deception-", "").replace("_", "-").lower()


def child_probe_set_slug(source_base: str, include_target_self: bool) -> str:
    return f"frozen-{slugify_dataset_base(source_base)}" + ("-self" if include_target_self else "")


def child_run_root(
    artifact_root: Path,
    model_dir: str,
    target_segment: str,
    source_base: str,
    include_target_self: bool,
    pooling: str,
    layer: int,
    child_run_id: str,
) -> Path:
    return (
        artifact_root
        / "runs"
        / "target_rank1_probe_transfer"
        / model_dir
        / target_segment
        / child_probe_set_slug(source_base, include_target_self)
        / f"{pooling}_l{layer}"
        / child_run_id
    )


def append_log_line(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip("\n") + "\n")


def run_stream(label: str, cmd: Sequence[str], log_path: Path | None = None) -> None:
    print(f"[start] {label}")
    print("[cmd]", " ".join(str(x) for x in cmd))
    if log_path is not None:
        append_log_line(log_path, f"[start] {label}")
        append_log_line(log_path, "[cmd] " + " ".join(str(x) for x in cmd))
    t0 = time.time()
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if log_path is not None:
            append_log_line(log_path, line.rstrip("\n"))
    rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        if log_path is not None:
            append_log_line(log_path, f"[failed] {label} exit_code={rc} elapsed={elapsed:.1f}s")
        raise RuntimeError(f"{label} failed with exit code {rc} after {elapsed:.1f}s")
    print(f"[done] {label} elapsed={elapsed:.1f}s")
    if log_path is not None:
        append_log_line(log_path, f"[done] {label} elapsed={elapsed:.1f}s")


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


def copy_png(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def write_matrix_csv(path: Path, row_key: str, row_labels: Sequence[str], col_labels: Sequence[str], matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_key] + list(col_labels))
        for i, label in enumerate(row_labels):
            row = [label]
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                row.append("" if np.isnan(val) else str(val))
            writer.writerow(row)


def make_heatmap(
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
    fig, ax = plt.subplots(figsize=(max(8.5, len(col_labels) * 1.0), max(4.6, len(row_labels) * 0.7)))
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
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_grouped_bar(
    categories: Sequence[str],
    left_vals: Sequence[float],
    right_vals: Sequence[float],
    left_label: str,
    right_label: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8.0, len(categories) * 1.5), 5.2))
    ax.bar(x - width / 2.0, left_vals, width=width, label=left_label)
    ax.bar(x + width / 2.0, right_vals, width=width, label=right_label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def child_done(child_root: Path) -> bool:
    status = read_json(child_root / "meta" / "status.json", default={})
    return status.get("state") == "completed" and (child_root / "results" / "summary.json").exists()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def probe_order_labels(
    evaluation_single_datasets: Sequence[str],
    target_dataset: str,
    include_target_self: bool,
    combined_probe_names: Sequence[str],
) -> List[str]:
    cols = list(evaluation_single_datasets)
    if include_target_self:
        cols.append(f"{target_dataset}_self")
    cols.extend(combined_probe_names)
    deduped: List[str] = []
    seen: set[str] = set()
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def short_label(name: str) -> str:
    return PROBE_SHORT_LABELS.get(name, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a held-out RP rank-1 source sweep")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--single_probes_root", type=str, default="data/probes")
    parser.add_argument("--target_dataset", type=str, default="Deception-Roleplaying")
    parser.add_argument("--segment", type=str, choices=["completion", "full"], default="completion")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--source_datasets", type=str, default=",".join(DEFAULT_SOURCE_BASES))
    parser.add_argument("--evaluation_single_datasets", type=str, default="")
    parser.add_argument("--combined_run_root", type=str, required=True)
    parser.add_argument("--combined_probe_names", type=str, default=",".join(DEFAULT_COMBINED_PROBES))
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--mirror_results_root", type=str, default=None)
    parser.add_argument("--gallery_root", type=str, default=None)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--alpha_init", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--lambda_self", type=float, default=1.0)
    parser.add_argument("--lambda_dist", type=float, default=1e-3)
    parser.add_argument("--self_auc_drop_tolerance", type=float, default=0.01)
    parser.add_argument("--random_directions", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_datasets = parse_csv_list(args.source_datasets)
    if not source_datasets:
        raise ValueError("At least one source dataset is required")
    evaluation_single_datasets = parse_csv_list(args.evaluation_single_datasets) or list(source_datasets)
    combined_probe_names = parse_csv_list(args.combined_probe_names)
    include_target_self = True

    model_dir = model_dir_name(args.model)
    target_segment = dataset_segment_name(args.target_dataset, args.segment)
    run_id = args.run_id.strip() or f"{default_run_id()}-{args.segment}-rp-rank1-source-sweep-v1"
    cfg_slug = f"fixed-{args.pooling}-l{args.layer}"

    artifact_root = Path(args.artifact_root)
    canonical_run_root = (
        artifact_root
        / "runs"
        / "target_rank1_source_sweep"
        / model_dir
        / target_segment
        / cfg_slug
        / run_id
    )
    meta_dir = canonical_run_root / "meta"
    checkpoints_dir = canonical_run_root / "checkpoints"
    logs_dir = canonical_run_root / "logs"
    results_dir = canonical_run_root / "results"
    source_runs_dir = canonical_run_root / "source_runs"
    ensure_dir(meta_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)
    ensure_dir(results_dir)
    ensure_dir(source_runs_dir)
    wrapper_log_path = logs_dir / "run.log"

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "target_rank1_source_sweep"
    mirror_root = Path(args.mirror_results_root) if args.mirror_results_root else default_mirror_root
    external_run_root = mirror_root / target_segment / cfg_slug / run_id

    default_gallery_root = (
        Path("results")
        / "ood_evaluation"
        / model_dir
        / "all_pairwise_results_final"
        / "generated_plots_after_id_fix"
    )
    gallery_root = Path(args.gallery_root) if args.gallery_root else default_gallery_root

    progress_path = checkpoints_dir / "progress.json"
    status_path = meta_dir / "status.json"
    progress: MutableMapping[str, object] = read_json(progress_path, default={"completed_sources": [], "updated_at": utc_now_iso()})
    completed_sources = set(progress.get("completed_sources", []))

    write_json(meta_dir / "run_manifest.json", {
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
        "evaluation_single_datasets": evaluation_single_datasets,
        "combined_run_root": str(Path(args.combined_run_root)),
        "combined_probe_names": combined_probe_names,
        "canonical_run_root": str(canonical_run_root),
        "mirror_results_root": str(external_run_root),
        "gallery_root": str(gallery_root),
    })
    write_json(status_path, {
        "state": "running",
        "stage": "child_runs",
        "updated_at": utc_now_iso(),
        "message": "running child learned-direction source sweeps",
    })

    for idx, source_base in enumerate(source_datasets, start=1):
        source_slug = slugify_dataset_base(source_base)
        child_run_id = f"{run_id}-{source_slug}"
        child_root = child_run_root(
            artifact_root=artifact_root,
            model_dir=model_dir,
            target_segment=target_segment,
            source_base=source_base,
            include_target_self=include_target_self,
            pooling=args.pooling,
            layer=args.layer,
            child_run_id=child_run_id,
        )

        if args.resume and source_slug in completed_sources and child_done(child_root):
            msg = f"[skip] source {idx}/{len(source_datasets)} {source_base} already completed"
            print(msg)
            append_log_line(wrapper_log_path, msg)
        else:
            cmd = [
                sys.executable,
                "-u",
                "scripts/evaluation/evaluate_target_rank1_probe_transfer.py",
                "--model", args.model,
                "--activations_root", str(Path(args.activations_root)),
                "--single_probes_root", str(Path(args.single_probes_root)),
                "--target_dataset", args.target_dataset,
                "--segment", args.segment,
                "--pooling", args.pooling,
                "--layer", str(args.layer),
                "--optimization_source_datasets", source_base,
                "--evaluation_single_datasets", ",".join(evaluation_single_datasets),
                "--include_target_self",
                "--combined_run_root", str(Path(args.combined_run_root)),
                "--combined_probe_names", ",".join(combined_probe_names),
                "--artifact_root", str(artifact_root),
                "--run_id", child_run_id,
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--patience", str(args.patience),
                "--alpha_init", str(args.alpha_init),
                "--alpha_max", str(args.alpha_max),
                "--lambda_self", str(args.lambda_self),
                "--lambda_dist", str(args.lambda_dist),
                "--self_auc_drop_tolerance", str(args.self_auc_drop_tolerance),
                "--random_directions", str(args.random_directions),
                "--seed", str(args.seed),
            ]
            if args.resume:
                cmd.append("--resume")
            run_stream(f"[source {idx}/{len(source_datasets)}] {source_base}", cmd, log_path=wrapper_log_path)
            completed_sources.add(source_slug)
            progress["completed_sources"] = sorted(completed_sources)
            progress["updated_at"] = utc_now_iso()
            write_json(progress_path, dict(progress))

        if not child_done(child_root):
            raise FileNotFoundError(f"Missing completed child run for {source_base}: {child_root}")

        copy_tree_subset(
            child_root,
            source_runs_dir / source_slug,
            ["meta", "checkpoints", "results", "logs"],
        )

    write_json(status_path, {
        "state": "running",
        "stage": "aggregate",
        "updated_at": utc_now_iso(),
        "message": "aggregating source-sweep outputs",
    })

    probe_order = probe_order_labels(
        evaluation_single_datasets=evaluation_single_datasets,
        target_dataset=args.target_dataset,
        include_target_self=include_target_self,
        combined_probe_names=combined_probe_names,
    )
    full_auc_matrix = np.full((len(source_datasets), len(probe_order)), np.nan, dtype=np.float64)
    full_delta_matrix = np.full((len(source_datasets), len(probe_order)), np.nan, dtype=np.float64)
    heldout_delta_matrix = np.full((len(source_datasets), len(probe_order)), np.nan, dtype=np.float64)
    long_rows: List[Dict[str, object]] = []
    source_summary_rows: List[Dict[str, object]] = []
    gallery_paths: List[str] = []
    child_summaries: Dict[str, Dict[str, object]] = {}

    for row_idx, source_base in enumerate(source_datasets):
        source_slug = slugify_dataset_base(source_base)
        child_run_id = f"{run_id}-{source_slug}"
        child_root = child_run_root(
            artifact_root=artifact_root,
            model_dir=model_dir,
            target_segment=target_segment,
            source_base=source_base,
            include_target_self=include_target_self,
            pooling=args.pooling,
            layer=args.layer,
            child_run_id=child_run_id,
        )
        child_summary = read_json(child_root / "results" / "summary.json", default={})
        child_summaries[source_base] = child_summary
        child_rows = read_csv_rows(child_root / "results" / "metrics_long.csv")
        learned_test_rows = [
            r for r in child_rows
            if r["method"] == "learned" and r["split"] == "test"
        ]

        heldout_single_deltas: List[float] = []
        heldout_combined_deltas: List[float] = []
        self_delta: float | None = None
        opt_delta: float | None = None

        for row in learned_test_rows:
            probe_name = row["probe_name"]
            if probe_name not in probe_order:
                continue
            col_idx = probe_order.index(probe_name)
            auc = float(row["auc"])
            delta = float(row["delta_auc_vs_baseline"])
            full_auc_matrix[row_idx, col_idx] = auc
            full_delta_matrix[row_idx, col_idx] = delta
            is_heldout = str(row.get("is_heldout_probe", "")).lower() == "true"
            if is_heldout:
                heldout_delta_matrix[row_idx, col_idx] = delta

            long_rows.append({
                "optimization_source_dataset": source_base,
                "target_dataset": target_segment,
                "probe_name": probe_name,
                "probe_kind": row["probe_kind"],
                "is_optimization_probe": row.get("is_optimization_probe", ""),
                "is_target_self_probe": row.get("is_target_self_probe", ""),
                "is_combined_probe": row.get("is_combined_probe", ""),
                "is_heldout_probe": row.get("is_heldout_probe", ""),
                "auc": auc,
                "delta_auc_vs_baseline": delta,
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
                "selected_epoch": int(child_summary.get("selected_epoch", -1)),
                "selected_alpha": float(child_summary.get("selected_alpha", np.nan)),
                "child_run_root": str(child_root),
                "child_snapshot_root": str(source_runs_dir / source_slug),
            })

            if str(row.get("is_combined_probe", "")).lower() == "true" and is_heldout:
                heldout_combined_deltas.append(delta)
            elif is_heldout:
                heldout_single_deltas.append(delta)
            if probe_name == f"{args.target_dataset}_self":
                self_delta = delta
            if probe_name == source_base:
                opt_delta = delta

        source_summary_rows.append({
            "optimization_source_dataset": source_base,
            "heldout_single_test_delta_avg": float(np.mean(heldout_single_deltas)) if heldout_single_deltas else np.nan,
            "heldout_combined_test_delta_avg": float(np.mean(heldout_combined_deltas)) if heldout_combined_deltas else np.nan,
            "target_self_test_delta": np.nan if self_delta is None else float(self_delta),
            "optimization_probe_test_delta": np.nan if opt_delta is None else float(opt_delta),
            "selected_epoch": int(child_summary.get("selected_epoch", -1)),
            "selected_alpha": float(child_summary.get("selected_alpha", np.nan)),
            "child_run_root": str(child_root),
        })

    row_labels = [short_label(x) for x in source_datasets]
    col_labels = [short_label(x) for x in probe_order]

    full_auc_csv = results_dir / "full_test_auc_matrix.csv"
    full_delta_csv = results_dir / "full_test_delta_matrix.csv"
    heldout_delta_csv = results_dir / "heldout_test_delta_matrix.csv"
    write_matrix_csv(full_auc_csv, "optimization_source_dataset", source_datasets, probe_order, full_auc_matrix)
    write_matrix_csv(full_delta_csv, "optimization_source_dataset", source_datasets, probe_order, full_delta_matrix)
    write_matrix_csv(heldout_delta_csv, "optimization_source_dataset", source_datasets, probe_order, heldout_delta_matrix)

    with (results_dir / "summary_long.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "optimization_source_dataset",
            "target_dataset",
            "probe_name",
            "probe_kind",
            "is_optimization_probe",
            "is_target_self_probe",
            "is_combined_probe",
            "is_heldout_probe",
            "auc",
            "delta_auc_vs_baseline",
            "accuracy",
            "f1",
            "selected_epoch",
            "selected_alpha",
            "child_run_root",
            "child_snapshot_root",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    with (results_dir / "source_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "optimization_source_dataset",
            "heldout_single_test_delta_avg",
            "heldout_combined_test_delta_avg",
            "target_self_test_delta",
            "optimization_probe_test_delta",
            "selected_epoch",
            "selected_alpha",
            "child_run_root",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_summary_rows)

    full_auc_png = results_dir / "full_test_auc_heatmap.png"
    full_delta_png = results_dir / "full_test_delta_heatmap.png"
    heldout_delta_png = results_dir / "heldout_test_delta_heatmap.png"
    heldout_avg_png = results_dir / "heldout_group_delta_summary.png"
    self_delta_png = results_dir / "self_delta_summary.png"

    make_heatmap(
        full_auc_matrix,
        row_labels,
        col_labels,
        title=f"{target_segment}: learned direction test AUROC",
        output_path=full_auc_png,
        fmt=".3f",
        vmin=0.0,
        vmax=1.0,
        cmap="YlGnBu",
    )
    max_abs_delta = float(np.nanmax(np.abs(full_delta_matrix))) if np.any(~np.isnan(full_delta_matrix)) else 0.05
    make_heatmap(
        full_delta_matrix,
        row_labels,
        col_labels,
        title=f"{target_segment}: learned direction test delta vs baseline",
        output_path=full_delta_png,
        fmt=".3f",
        vmin=-max_abs_delta,
        vmax=max_abs_delta,
        cmap="RdBu_r",
    )
    heldout_abs = float(np.nanmax(np.abs(heldout_delta_matrix))) if np.any(~np.isnan(heldout_delta_matrix)) else 0.05
    make_heatmap(
        heldout_delta_matrix,
        row_labels,
        col_labels,
        title=f"{target_segment}: held-out learned direction test delta",
        output_path=heldout_delta_png,
        fmt=".3f",
        vmin=-heldout_abs,
        vmax=heldout_abs,
        cmap="RdBu_r",
    )

    make_grouped_bar(
        categories=row_labels,
        left_vals=[float(r["heldout_single_test_delta_avg"]) for r in source_summary_rows],
        right_vals=[float(r["heldout_combined_test_delta_avg"]) for r in source_summary_rows],
        left_label="Held-out singles",
        right_label="Held-out combined",
        title=f"{target_segment}: average held-out delta by source-trained direction",
        ylabel="Delta AUROC",
        output_path=heldout_avg_png,
    )
    make_grouped_bar(
        categories=row_labels,
        left_vals=[float(r["target_self_test_delta"]) for r in source_summary_rows],
        right_vals=[float(r["optimization_probe_test_delta"]) for r in source_summary_rows],
        left_label="RP self",
        right_label="Optimization probe",
        title=f"{target_segment}: self stability vs optimized-source gain",
        ylabel="Delta AUROC",
        output_path=self_delta_png,
    )

    gallery_prefix = f"{slugify_dataset_base(args.target_dataset)}_{args.segment}_{args.pooling}_l{args.layer}_rank1_source_sweep"
    for src, name in [
        (full_auc_png, f"{gallery_prefix}_full_test_auc_heatmap.png"),
        (full_delta_png, f"{gallery_prefix}_full_test_delta_heatmap.png"),
        (heldout_delta_png, f"{gallery_prefix}_heldout_test_delta_heatmap.png"),
        (heldout_avg_png, f"{gallery_prefix}_heldout_group_delta_summary.png"),
        (self_delta_png, f"{gallery_prefix}_self_delta_summary.png"),
    ]:
        gallery_paths.append(copy_png(src, gallery_root / name))

    comparison_summary = {
        "run_id": run_id,
        "updated_at": utc_now_iso(),
        "model": args.model,
        "target_dataset": args.target_dataset,
        "target_segment": target_segment,
        "pooling": args.pooling,
        "layer": int(args.layer),
        "source_datasets": source_datasets,
        "evaluation_single_datasets": evaluation_single_datasets,
        "combined_probe_names": combined_probe_names,
        "canonical_run_root": str(canonical_run_root),
        "mirror_results_root": str(external_run_root),
        "gallery_root": str(gallery_root),
        "outputs": {
            "full_test_auc_matrix_csv": str(full_auc_csv),
            "full_test_delta_matrix_csv": str(full_delta_csv),
            "heldout_test_delta_matrix_csv": str(heldout_delta_csv),
            "summary_long_csv": str(results_dir / "summary_long.csv"),
            "source_summary_csv": str(results_dir / "source_summary.csv"),
            "full_test_auc_heatmap_png": str(full_auc_png),
            "full_test_delta_heatmap_png": str(full_delta_png),
            "heldout_test_delta_heatmap_png": str(heldout_delta_png),
            "heldout_group_delta_summary_png": str(heldout_avg_png),
            "self_delta_summary_png": str(self_delta_png),
        },
        "child_runs": {
            source_base: {
                "child_run_root": str(
                    child_run_root(
                        artifact_root=artifact_root,
                        model_dir=model_dir,
                        target_segment=target_segment,
                        source_base=source_base,
                        include_target_self=include_target_self,
                        pooling=args.pooling,
                        layer=args.layer,
                        child_run_id=f"{run_id}-{slugify_dataset_base(source_base)}",
                    )
                ),
                "snapshot_root": str(source_runs_dir / slugify_dataset_base(source_base)),
                "selected_epoch": int(child_summaries[source_base].get("selected_epoch", -1)),
                "selected_alpha": float(child_summaries[source_base].get("selected_alpha", np.nan)),
            }
            for source_base in source_datasets
        },
        "gallery_pngs": gallery_paths,
    }
    write_json(results_dir / "summary.json", comparison_summary)

    copy_tree_subset(canonical_run_root, external_run_root, ["meta", "checkpoints", "results", "logs", "source_runs"])

    write_json(status_path, {
        "state": "completed",
        "stage": "done",
        "updated_at": utc_now_iso(),
        "message": "completed successfully",
    })
    progress["completed_sources"] = sorted(completed_sources)
    progress["updated_at"] = utc_now_iso()
    write_json(progress_path, dict(progress))

    print(f"[done] canonical run root -> {canonical_run_root}")
    print(f"[done] mirrored results -> {external_run_root}")
    print(f"[done] mirrored gallery pngs -> {gallery_root}")
    append_log_line(wrapper_log_path, f"[done] canonical run root -> {canonical_run_root}")
    append_log_line(wrapper_log_path, f"[done] mirrored results -> {external_run_root}")
    append_log_line(wrapper_log_path, f"[done] mirrored gallery pngs -> {gallery_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
