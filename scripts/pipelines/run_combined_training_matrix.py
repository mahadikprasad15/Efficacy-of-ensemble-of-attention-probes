#!/usr/bin/env python3
"""
Run fixed-config combined-training experiments across named dataset combos.

This wrapper orchestrates repeated runs of
`scripts/pipelines/run_combined_deception_segment_probes.py`, then aggregates
their outputs into a combo-by-target comparison matrix. Canonical artifacts live
under `artifacts/runs/...`; results are optionally mirrored into
`results/ood_evaluation/.../combined_training` and selected PNGs are mirrored
into the flat gallery folder used by the notebook analysis flow.
"""

from __future__ import annotations

import argparse
import csv
import json
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


SOURCE_BASES = [
    "Deception-ConvincingGame",
    "Deception-HarmPressureChoice",
    "Deception-InstructedDeception",
    "Deception-Mask",
    "Deception-AILiar",
    "Deception-Roleplaying",
]
TARGET_ONLY_BASE = "Deception-InsiderTrading-SallyConcat"
ALL_TARGET_BASES = SOURCE_BASES + [TARGET_ONLY_BASE]

COMBO_REGISTRY: Mapping[str, Mapping[str, Sequence[str]]] = {
    "completion": {
        "completion_all6": SOURCE_BASES,
        "completion_id_mask": [
            "Deception-InstructedDeception",
            "Deception-Mask",
        ],
        "completion_id_mask_al": [
            "Deception-InstructedDeception",
            "Deception-Mask",
            "Deception-AILiar",
        ],
        "completion_cg_id": [
            "Deception-ConvincingGame",
            "Deception-InstructedDeception",
        ],
    },
    "full": {
        "full_all6": SOURCE_BASES,
        "full_id_mask": [
            "Deception-InstructedDeception",
            "Deception-Mask",
        ],
        "full_id_mask_al": [
            "Deception-InstructedDeception",
            "Deception-Mask",
            "Deception-AILiar",
        ],
        "full_cg_id": [
            "Deception-ConvincingGame",
            "Deception-InstructedDeception",
        ],
    },
}

SHORT_LABELS = {
    "Deception-ConvincingGame-completion": "CG-c",
    "Deception-HarmPressureChoice-completion": "HPC-c",
    "Deception-InstructedDeception-completion": "ID-c",
    "Deception-Mask-completion": "M-c",
    "Deception-AILiar-completion": "AL-c",
    "Deception-InsiderTrading-SallyConcat-completion": "IT-c",
    "Deception-Roleplaying-completion": "RP-c",
    "Deception-ConvincingGame-full": "CG-f",
    "Deception-HarmPressureChoice-full": "HPC-f",
    "Deception-InstructedDeception-full": "ID-f",
    "Deception-Mask-full": "M-f",
    "Deception-AILiar-full": "AL-f",
    "Deception-InsiderTrading-SallyConcat-full": "IT-f",
    "Deception-Roleplaying-full": "RP-f",
}


def parse_csv_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def dataset_segment_name(dataset_base: str, segment: str) -> str:
    return f"{dataset_base}-{segment}"


def config_slug(poolings: Sequence[str], layers: Sequence[str]) -> str:
    if len(poolings) == 1 and len(layers) == 1:
        return f"fixed-{poolings[0]}-l{layers[0]}"
    return f"pool-{'-'.join(poolings)}_layers-{'-'.join(layers)}"


def make_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    output_path: Path,
    *,
    fmt: str = ".3f",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(8.0, len(col_labels) * 1.35), max(4.5, len(row_labels) * 0.75)))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
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


def copy_tree_subset(src_root: Path, dst_root: Path, subdirs: Sequence[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
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


def combo_done(combo_results_dir: Path) -> bool:
    status = read_json(combo_results_dir / "meta" / "status.json", default={})
    summary_path = combo_results_dir / "results" / "summary.json"
    best_csv = combo_results_dir / "results" / "best_by_test_dataset.csv"
    return status.get("state") == "completed" and summary_path.exists() and best_csv.exists()


def target_order_for_segment(segment: str) -> List[str]:
    return [dataset_segment_name(base, segment) for base in ALL_TARGET_BASES]


def short_labels_for_targets(targets: Sequence[str]) -> List[str]:
    return [SHORT_LABELS.get(t, t) for t in targets]


def read_best_rows(best_csv: Path) -> List[Dict[str, str]]:
    with best_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_long_rows(
    *,
    combo_name: str,
    train_bases: Sequence[str],
    segment: str,
    best_rows: Sequence[Mapping[str, str]],
    child_summary: Mapping[str, object],
) -> List[Dict[str, object]]:
    counts = dict(child_summary.get("n_test_samples", {}))
    rows: List[Dict[str, object]] = []
    for row in best_rows:
        rows.append({
            "combo_name": combo_name,
            "segment": segment,
            "train_datasets": ",".join(train_bases),
            "test_dataset": row["test_dataset"],
            "test_dataset_base": row["test_dataset_base"],
            "pooling": row["pooling"],
            "layer": int(row["best_layer"]),
            "auc": float(row["best_auc"]),
            "accuracy": float(row["best_accuracy"]),
            "f1": float(row["best_f1"]),
            "count": int(counts.get(row["test_dataset"], 0)),
            "child_results_dir": str(child_summary.get("results_dir", "")),
            "child_probe_run_dir": str(child_summary.get("probe_run_dir", "")),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-config combined-training matrix experiments")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--activations_root", type=str, required=True)
    parser.add_argument("--segment", type=str, choices=["completion", "full"], default="completion")
    parser.add_argument("--combo_names", type=str, default="")
    parser.add_argument("--poolings", type=str, default="mean")
    parser.add_argument("--layers", type=str, default="15")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--mirror_results_root", type=str, default=None)
    parser.add_argument("--gallery_root", type=str, default=None)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    poolings = parse_csv_list(args.poolings)
    layers = parse_csv_list(args.layers)
    if not poolings:
        raise ValueError("At least one pooling is required")
    if not layers:
        raise ValueError("At least one layer is required")

    if args.segment not in COMBO_REGISTRY:
        raise ValueError(f"No combo registry for segment: {args.segment}")

    available_combo_names = list(COMBO_REGISTRY[args.segment].keys())
    combo_names = parse_csv_list(args.combo_names) or available_combo_names
    unknown = [name for name in combo_names if name not in COMBO_REGISTRY[args.segment]]
    if unknown:
        raise ValueError(f"Unknown combos for {args.segment}: {unknown}")

    run_id = args.run_id.strip() or f"{default_run_id()}-{args.segment}-combined-training-v1"
    model_dir = model_dir_name(args.model)
    cfg_slug = config_slug(poolings, layers)

    artifact_root = Path(args.artifact_root)
    canonical_run_root = artifact_root / "runs" / "combined_training_matrix" / model_dir / args.segment / cfg_slug / run_id
    meta_dir = canonical_run_root / "meta"
    checkpoints_dir = canonical_run_root / "checkpoints"
    logs_dir = canonical_run_root / "logs"
    results_dir = canonical_run_root / "results"
    comparison_dir = results_dir / "comparison"
    child_probe_root = canonical_run_root / "probes"
    child_results_root = canonical_run_root / "combo_runs"
    ensure_dir(meta_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)
    ensure_dir(comparison_dir)
    ensure_dir(child_probe_root)
    ensure_dir(child_results_root)
    wrapper_log_path = logs_dir / "run.log"

    default_mirror_root = Path("results") / "ood_evaluation" / model_dir / "combined_training"
    mirror_root = Path(args.mirror_results_root) if args.mirror_results_root else default_mirror_root
    external_run_root = mirror_root / args.segment / cfg_slug / run_id

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
    progress: MutableMapping[str, object] = read_json(progress_path, default={
        "completed_combos": [],
        "updated_at": utc_now_iso(),
    })
    completed_combos = set(progress.get("completed_combos", []))

    write_json(meta_dir / "run_manifest.json", {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "model": args.model,
        "model_dir": model_dir,
        "segment": args.segment,
        "combo_names": combo_names,
        "combo_registry": {
            name: list(COMBO_REGISTRY[args.segment][name])
            for name in combo_names
        },
        "poolings": poolings,
        "layers": layers,
        "activations_root": str(Path(args.activations_root)),
        "canonical_run_root": str(canonical_run_root),
        "mirror_results_root": str(external_run_root),
        "gallery_root": str(gallery_root),
        "child_probe_root": str(child_probe_root),
        "child_results_root": str(child_results_root),
    })
    write_json(status_path, {
        "state": "running",
        "stage": "combo_runs",
        "updated_at": utc_now_iso(),
        "message": "running child combined-training experiments",
    })

    for combo_idx, combo_name in enumerate(combo_names, start=1):
        train_bases = list(COMBO_REGISTRY[args.segment][combo_name])
        test_only_bases = [base for base in ALL_TARGET_BASES if base not in train_bases]
        combo_result_dir = child_results_root / combo_name

        if args.resume and combo_name in completed_combos and combo_done(combo_result_dir):
            message = f"[skip] combo {combo_idx}/{len(combo_names)} {combo_name} already completed"
            print(message)
            append_log_line(wrapper_log_path, message)
            continue

        cmd = [
            sys.executable,
            "-u",
            "scripts/pipelines/run_combined_deception_segment_probes.py",
            "--model", args.model,
            "--activations_root", str(Path(args.activations_root)),
            "--segment", args.segment,
            "--train_datasets", ",".join(train_bases),
            "--test_only_datasets", ",".join(test_only_bases),
            "--poolings", ",".join(poolings),
            "--layers", ",".join(layers),
            "--probe_output_root", str(child_probe_root),
            "--results_output_root", str(child_results_root),
            "--run_name", combo_name,
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.verbose:
            cmd.append("--verbose")

        run_stream(f"[combo {combo_idx}/{len(combo_names)}] {combo_name}", cmd, log_path=wrapper_log_path)
        completed_combos.add(combo_name)
        progress["completed_combos"] = sorted(completed_combos)
        progress["updated_at"] = utc_now_iso()
        write_json(progress_path, dict(progress))

    write_json(status_path, {
        "state": "running",
        "stage": "aggregate",
        "updated_at": utc_now_iso(),
        "message": "aggregating combo comparison outputs",
    })

    target_order = target_order_for_segment(args.segment)
    target_short_labels = short_labels_for_targets(target_order)
    combo_short_labels = combo_names
    auc_matrix = np.full((len(combo_names), len(target_order)), np.nan, dtype=np.float64)
    long_rows: List[Dict[str, object]] = []
    gallery_paths: List[str] = []
    child_summaries: Dict[str, Dict[str, object]] = {}

    for combo_idx, combo_name in enumerate(combo_names):
        combo_result_dir = child_results_root / combo_name
        child_summary = read_json(combo_result_dir / "results" / "summary.json", default={})
        if not child_summary:
            raise FileNotFoundError(f"Missing child summary for combo {combo_name}: {combo_result_dir / 'results' / 'summary.json'}")
        child_summaries[combo_name] = child_summary

        best_rows = read_best_rows(combo_result_dir / "results" / "best_by_test_dataset.csv")
        for row in best_rows:
            if row["pooling"] not in poolings:
                continue
            target_name = row["test_dataset"]
            if target_name not in target_order:
                continue
            target_idx = target_order.index(target_name)
            auc_matrix[combo_idx, target_idx] = float(row["best_auc"])

        long_rows.extend(build_long_rows(
            combo_name=combo_name,
            train_bases=list(COMBO_REGISTRY[args.segment][combo_name]),
            segment=args.segment,
            best_rows=best_rows,
            child_summary=child_summary,
        ))

        child_auc_png = combo_result_dir / "results" / "heatmap_pooling_vs_test_auc.png"
        if child_auc_png.exists():
            gallery_paths.append(copy_png(
                child_auc_png,
                gallery_root / f"combined_training_{combo_name}_auc_heatmap.png",
            ))

    matrix_csv = comparison_dir / "combined_auc_matrix.csv"
    write_matrix_csv(matrix_csv, "combo_name", combo_names, target_order, auc_matrix)

    long_csv = comparison_dir / "combined_summary_long.csv"
    with long_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "combo_name",
            "segment",
            "train_datasets",
            "test_dataset",
            "test_dataset_base",
            "pooling",
            "layer",
            "auc",
            "accuracy",
            "f1",
            "count",
            "child_results_dir",
            "child_probe_run_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    comparison_png = comparison_dir / "combined_auc_heatmap.png"
    make_heatmap(
        matrix=auc_matrix,
        row_labels=combo_short_labels,
        col_labels=target_short_labels,
        title=f"{args.segment}: Combined training AUROC | {cfg_slug}",
        output_path=comparison_png,
        fmt=".3f",
        vmin=0.0,
        vmax=1.0,
    )
    gallery_paths.append(copy_png(
        comparison_png,
        gallery_root / f"combined_training_{args.segment}_comparison_auc_heatmap.png",
    ))

    comparison_summary = {
        "run_id": run_id,
        "updated_at": utc_now_iso(),
        "segment": args.segment,
        "model": args.model,
        "config": {
            "poolings": poolings,
            "layers": layers,
            "config_slug": cfg_slug,
        },
        "combo_names": combo_names,
        "targets": target_order,
        "canonical_run_root": str(canonical_run_root),
        "mirror_results_root": str(external_run_root),
        "gallery_root": str(gallery_root),
        "comparison_outputs": {
            "combined_auc_matrix_csv": str(matrix_csv),
            "combined_summary_long_csv": str(long_csv),
            "combined_auc_heatmap_png": str(comparison_png),
            "mirrored_combined_auc_matrix_csv": str(external_run_root / "results" / "comparison" / "combined_auc_matrix.csv"),
            "mirrored_combined_summary_long_csv": str(external_run_root / "results" / "comparison" / "combined_summary_long.csv"),
            "mirrored_combined_auc_heatmap_png": str(external_run_root / "results" / "comparison" / "combined_auc_heatmap.png"),
        },
        "child_runs": {
            name: {
                "train_datasets": list(COMBO_REGISTRY[args.segment][name]),
                "test_only_datasets": [base for base in ALL_TARGET_BASES if base not in COMBO_REGISTRY[args.segment][name]],
                "combo_results_root": str(child_results_root / name),
                "mirrored_combo_results_root": str(external_run_root / "combo_runs" / name),
                "summary_json": str(child_results_root / name / "results" / "summary.json"),
                "mirrored_summary_json": str(external_run_root / "combo_runs" / name / "results" / "summary.json"),
            }
            for name in combo_names
        },
        "gallery_pngs": gallery_paths,
    }
    write_json(comparison_dir / "summary.json", comparison_summary)
    write_json(results_dir / "summary.json", comparison_summary)

    # Mirror canonical wrapper results/meta/checkpoints.
    copy_tree_subset(canonical_run_root, external_run_root, ["meta", "checkpoints", "results", "logs"])
    # Mirror child combo run results/meta/checkpoints.
    for combo_name in combo_names:
        copy_tree_subset(
            child_results_root / combo_name,
            external_run_root / "combo_runs" / combo_name,
            ["meta", "checkpoints", "results"],
        )

    write_json(status_path, {
        "state": "completed",
        "stage": "done",
        "updated_at": utc_now_iso(),
        "message": "completed successfully",
    })
    progress["completed_combos"] = sorted(completed_combos)
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
