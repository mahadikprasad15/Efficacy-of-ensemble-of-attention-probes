#!/usr/bin/env python3
"""
Build pairwise score matrices from existing probe/ood artifacts (no retraining).

For each segment (completion/full), this script exports:
  - 12 fixed matrices: poolings x fixed layers (default 4 x 3)
  - 1 source-val-selected matrix: one (pooling, layer) per source row selected
    from source validation metrics only, then evaluated on all OOD test targets.

Expected artifacts:
  - Source validation:
      data/probes/<model_dir>/<DatasetBase>_slices/<DatasetSegment>/<pooling>/layer_results.json
  - Pairwise OOD test:
      results/ood_evaluation/<model_dir>/from-<source>/to-<target>/pair_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


POOLING_ORDER = ["attn", "last", "max", "mean"]
ALL_POOLINGS = ["mean", "max", "last", "attn"]
SEGMENTS = ["completion", "full"]


@dataclass
class PairLookup:
    metrics: Dict[Tuple[str, int], Dict[str, Optional[float]]] = field(default_factory=dict)
    source_file: Optional[str] = None
    errors: List[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    """
    Supports either:
      - root = <base> (contains <base>/<model_dir>)
      - root = <base>/<model_dir>
    """
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


def normalize_pooling(pooling: Any) -> str:
    p = str(pooling).strip().lower()
    if p in ["none", "final", "final_token"]:
        return "last"
    return p


def parse_float(v: Any) -> Optional[float]:
    if v in [None, ""]:
        return None
    try:
        return float(v)
    except Exception:
        return None


def parse_int(v: Any) -> Optional[int]:
    if v in [None, ""]:
        return None
    try:
        return int(v)
    except Exception:
        return None


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for item in parse_csv_list(value):
        out.append(int(item))
    return out


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


def pooling_priority_score(pooling: str) -> int:
    p = normalize_pooling(pooling)
    try:
        return -POOLING_ORDER.index(p)
    except ValueError:
        return -999


def choose_source_best(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    def key(r: Dict[str, Any]) -> Tuple[float, float, int, int]:
        val_auc = parse_float(r.get("val_auc"))
        val_acc = parse_float(r.get("val_acc"))
        layer = parse_int(r.get("layer"))
        return (
            float(val_auc if val_auc is not None else -1.0),
            float(val_acc if val_acc is not None else -1.0),
            -int(layer if layer is not None else 10_000),
            pooling_priority_score(str(r.get("pooling", ""))),
        )

    return max(rows, key=key)


def read_source_val_rows(
    probes_model_root: Path,
    source_dataset: str,
    poolings: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    base = dataset_base(source_dataset)

    for pooling in poolings:
        p = normalize_pooling(pooling)
        layer_results_path = (
            probes_model_root
            / f"{base}_slices"
            / source_dataset
            / p
            / "layer_results.json"
        )
        if not layer_results_path.exists():
            missing.append(
                {
                    "source_dataset": source_dataset,
                    "pooling": p,
                    "reason": "missing_layer_results_json",
                    "path": str(layer_results_path),
                }
            )
            continue

        raw = read_json(layer_results_path, default={})
        if not isinstance(raw, list):
            missing.append(
                {
                    "source_dataset": source_dataset,
                    "pooling": p,
                    "reason": "invalid_layer_results_format",
                    "path": str(layer_results_path),
                }
            )
            continue

        for row in raw:
            if not isinstance(row, dict):
                continue
            layer = parse_int(row.get("layer"))
            val_auc = parse_float(row.get("val_auc"))
            if val_auc is None:
                # Legacy fallback
                val_auc = parse_float(row.get("auc"))
            if layer is None or val_auc is None:
                continue
            val_acc = parse_float(row.get("val_acc"))
            if val_acc is None:
                val_acc = parse_float(row.get("accuracy"))
            all_rows.append(
                {
                    "source_dataset": source_dataset,
                    "pooling": p,
                    "layer": int(layer),
                    "val_auc": float(val_auc),
                    "val_acc": float(val_acc if val_acc is not None else -1.0),
                    "source_file": str(layer_results_path),
                }
            )

    return all_rows, missing


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

        # Legacy / sparse case: only best exists
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


def parse_eval_ood_file(path: Path) -> Dict[Tuple[str, int], Dict[str, Optional[float]]]:
    out: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}
    data = read_json(path, default={})
    layer_rows = data.get("layer_results", [])
    if not isinstance(layer_rows, list):
        return out

    pooling = normalize_pooling(path.parent.name)
    for row in layer_rows:
        if not isinstance(row, dict):
            continue
        layer = parse_int(row.get("layer"))
        auc = parse_float(row.get("auc"))
        if layer is None or auc is None:
            continue
        out[(pooling, int(layer))] = {
            "auc": float(auc),
            "accuracy": parse_float(row.get("accuracy")),
            "f1": parse_float(row.get("f1")),
        }
    return out


def load_pair_lookup(pair_dir: Path, poolings: Sequence[str], split: str = "test") -> PairLookup:
    lookup = PairLookup()

    pair_summary_path = pair_dir / "pair_summary.json"
    if pair_summary_path.exists():
        data = read_json(pair_summary_path, default={})
        metrics = parse_pair_summary_metrics(data)
        if metrics:
            lookup.metrics = metrics
            lookup.source_file = str(pair_summary_path)
            return lookup
        lookup.errors.append("pair_summary_missing_layer_metrics")
    else:
        lookup.errors.append("pair_summary_missing")

    # Fallback: pooling-specific eval files from eval_ood.py
    metrics: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}
    for pooling in poolings:
        p = normalize_pooling(pooling)
        pdir = pair_dir / p
        if not pdir.exists():
            continue
        candidates = sorted(pdir.glob(f"eval_ood_*_{split}.json"))
        if split == "test":
            # Extra tolerance for odd naming
            candidates += sorted(pdir.glob("eval_ood_*_test.json"))
        if not candidates:
            continue
        for c in candidates:
            m = parse_eval_ood_file(c)
            metrics.update(m)
        if metrics:
            lookup.source_file = str(pdir)
    if metrics:
        lookup.metrics = metrics
        if not lookup.source_file:
            lookup.source_file = str(pair_dir)
        return lookup

    # Fallback: ood_results_all_pooling.json (best-only per pooling)
    ood_all = pair_dir / "ood_results_all_pooling.json"
    if ood_all.exists():
        data = read_json(ood_all, default={})
        for pooling in ALL_POOLINGS:
            payload = data.get(pooling)
            if not isinstance(payload, dict):
                continue
            layer = parse_int(payload.get("best_layer"))
            auc = parse_float(payload.get("best_auc"))
            if layer is None or auc is None:
                continue
            lookup.metrics[(normalize_pooling(pooling), int(layer))] = {
                "auc": float(auc),
                "accuracy": parse_float(payload.get("best_accuracy")),
                "f1": parse_float(payload.get("best_f1")),
            }
        if lookup.metrics:
            lookup.source_file = str(ood_all)
            return lookup
        lookup.errors.append("ood_results_all_pooling_missing_metrics")

    lookup.errors.append("no_pair_metrics_found")
    return lookup


def write_matrix_csv(
    path: Path,
    rows: Sequence[str],
    cols: Sequence[str],
    matrix: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_dataset"] + list(cols))
        for i, r in enumerate(rows):
            out_row: List[Any] = [r]
            for j in range(len(cols)):
                v = matrix[i, j]
                out_row.append("" if np.isnan(v) else float(v))
            writer.writerow(out_row)


def plot_matrix_heatmap(
    output_path: Path,
    matrix: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str],
    title: str,
    segment: str,
    vmin: float,
    vmax: float,
    cmap: str,
    dpi: int,
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([dataset_tick_label(c) for c in cols], rotation=30, ha="right")
    ax.set_yticklabels([dataset_tick_label(r) for r in rows])

    ax.set_xlabel("Eval dataset (probe tested on)")
    if segment == "full":
        ax.set_ylabel("Full-prompt probes")
    else:
        ax.set_ylabel("Completion probes")
    ax.set_title(title, fontweight="bold", fontsize=14)

    # White grid lines between cells for readability.
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell annotations.
    mid = (vmin + vmax) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = "NA" if np.isnan(v) else f"{v:.2f}"
            if np.isnan(v):
                color = "#4a4a4a"
            else:
                color = "white" if v >= mid else "#4a4a4a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    # Highlight true diagonal cells.
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if r == c:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=2.4)
                ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pairwise score matrices from existing artifacts.")
    parser.add_argument(
        "--probes_root",
        type=str,
        default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/probes",
    )
    parser.add_argument(
        "--ood_results_root",
        type=str,
        default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/ood_evaluation",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional direct run output root; outputs go under <output_root>/<run_id>/...",
    )
    parser.add_argument("--segments", type=str, default="completion,full")
    parser.add_argument("--poolings", type=str, default="mean,max,last,attn")
    parser.add_argument("--fixed_layers", type=str, default="10,12,15")
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", type=str, default="YlGnBu")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()
    segments = parse_csv_list(args.segments)
    for seg in segments:
        if seg not in SEGMENTS:
            raise ValueError(f"Unsupported segment '{seg}'. Choose from {SEGMENTS}.")

    poolings = [normalize_pooling(p) for p in parse_csv_list(args.poolings)]
    for p in poolings:
        if p not in ALL_POOLINGS:
            raise ValueError(f"Unsupported pooling '{p}'. Choose from {ALL_POOLINGS}.")

    fixed_layers = parse_int_csv(args.fixed_layers)
    if not fixed_layers:
        raise ValueError("--fixed_layers must contain at least one layer.")

    probes_base, probes_model_root = split_root_and_model(Path(args.probes_root), model_dir)
    ood_base, ood_model_root = split_root_and_model(Path(args.ood_results_root), model_dir)

    if args.output_root:
        run_root = Path(args.output_root) / run_id
    else:
        run_root = Path(args.artifact_root) / "runs" / "pairwise_score_matrix_from_artifacts" / model_dir / run_id
    meta_dir = run_root / "meta"
    out_dir = run_root / "results"
    ensure_dir(meta_dir)
    ensure_dir(out_dir)

    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    summary_path = out_dir / "summary.json"

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "model_dir": model_dir,
            "paths": {
                "probes_base": str(probes_base),
                "probes_model_root": str(probes_model_root),
                "ood_base": str(ood_base),
                "ood_model_root": str(ood_model_root),
                "run_root": str(run_root),
            },
            "segments": segments,
            "poolings": poolings,
            "fixed_layers": fixed_layers,
            "plot": {
                "vmin": args.vmin,
                "vmax": args.vmax,
                "cmap": args.cmap,
                "dpi": args.dpi,
            },
            "selection_rule": "max(val_auc), tie val_acc, then lower layer, then pooling attn>last>max>mean",
            "resume": bool(args.resume),
        },
    )

    if args.resume and summary_path.exists():
        write_json(
            status_path,
            {
                "state": "completed",
                "message": "resume requested and summary already exists",
                "updated_at": utc_now(),
            },
        )
        print(f"[resume] summary exists: {summary_path}")
        return 0

    write_json(status_path, {"state": "running", "message": "building matrices", "updated_at": utc_now()})
    print(f"[start] run_id={run_id}")
    print(f"[start] probes_model_root={probes_model_root}")
    print(f"[start] ood_model_root={ood_model_root}")

    rows_map, cols_map = stage_spec()
    global_summary: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now(),
        "segments": {},
    }

    for segment in segments:
        print(f"[segment] {segment}")
        rows = rows_map[segment]
        cols = cols_map[segment]
        seg_out = out_dir / segment
        ensure_dir(seg_out)

        # Preload pair lookups once for this segment.
        pair_lookup_map: Dict[Tuple[str, str], PairLookup] = {}
        for src in rows:
            for tgt in cols:
                pair_dir = ood_model_root / f"from-{src}" / f"to-{tgt}"
                lookup = load_pair_lookup(pair_dir=pair_dir, poolings=poolings, split="test")
                pair_lookup_map[(src, tgt)] = lookup

        missing_rows: List[Dict[str, Any]] = []
        fixed_long_rows: List[Dict[str, Any]] = []
        fixed_outputs: List[Dict[str, Any]] = []

        # A) 12 fixed matrices
        for pooling in poolings:
            for layer in fixed_layers:
                mat = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
                for i, src in enumerate(rows):
                    for j, tgt in enumerate(cols):
                        lookup = pair_lookup_map[(src, tgt)]
                        metric = lookup.metrics.get((pooling, int(layer)))
                        if metric is None:
                            missing_rows.append(
                                {
                                    "segment": segment,
                                    "variant": "fixed",
                                    "pooling": pooling,
                                    "layer": int(layer),
                                    "row_dataset": src,
                                    "col_dataset": tgt,
                                    "reason": ";".join(lookup.errors) if lookup.errors else "config_not_found_in_pair_metrics",
                                    "pair_source_file": lookup.source_file,
                                }
                            )
                            fixed_long_rows.append(
                                {
                                    "segment": segment,
                                    "variant": "fixed",
                                    "row_dataset": src,
                                    "col_dataset": tgt,
                                    "pooling": pooling,
                                    "layer": int(layer),
                                    "auc": "",
                                    "status": "missing",
                                    "pair_source_file": lookup.source_file or "",
                                }
                            )
                            continue
                        auc = metric.get("auc")
                        if auc is None:
                            missing_rows.append(
                                {
                                    "segment": segment,
                                    "variant": "fixed",
                                    "pooling": pooling,
                                    "layer": int(layer),
                                    "row_dataset": src,
                                    "col_dataset": tgt,
                                    "reason": "auc_missing_for_config",
                                    "pair_source_file": lookup.source_file,
                                }
                            )
                            fixed_long_rows.append(
                                {
                                    "segment": segment,
                                    "variant": "fixed",
                                    "row_dataset": src,
                                    "col_dataset": tgt,
                                    "pooling": pooling,
                                    "layer": int(layer),
                                    "auc": "",
                                    "status": "missing",
                                    "pair_source_file": lookup.source_file or "",
                                }
                            )
                            continue
                        mat[i, j] = float(auc)
                        fixed_long_rows.append(
                            {
                                "segment": segment,
                                "variant": "fixed",
                                "row_dataset": src,
                                "col_dataset": tgt,
                                "pooling": pooling,
                                "layer": int(layer),
                                "auc": float(auc),
                                "status": "ok",
                                "pair_source_file": lookup.source_file or "",
                            }
                        )

                csv_path = seg_out / f"matrix_fixed_{pooling}_L{layer}_auc.csv"
                png_path = seg_out / f"heatmap_fixed_{pooling}_L{layer}_auc.png"
                write_matrix_csv(csv_path, rows, cols, mat)
                plot_matrix_heatmap(
                    output_path=png_path,
                    matrix=mat,
                    rows=rows,
                    cols=cols,
                    title=f"AUROC | {segment} | fixed {pooling} L{layer}",
                    segment=segment,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    cmap=args.cmap,
                    dpi=args.dpi,
                )
                fixed_outputs.append(
                    {
                        "pooling": pooling,
                        "layer": int(layer),
                        "csv": str(csv_path),
                        "png": str(png_path),
                        "num_filled": int(np.sum(~np.isnan(mat))),
                        "num_total": int(mat.size),
                    }
                )

        # B) source-val-selected matrix
        selected_rows: List[Dict[str, Any]] = []
        source_selection: Dict[str, Tuple[str, int]] = {}
        for src in rows:
            val_rows, source_missing = read_source_val_rows(
                probes_model_root=probes_model_root,
                source_dataset=src,
                poolings=poolings,
            )
            for miss in source_missing:
                missing_rows.append(
                    {
                        "segment": segment,
                        "variant": "source_val_selected",
                        "pooling": miss.get("pooling"),
                        "layer": "",
                        "row_dataset": src,
                        "col_dataset": "",
                        "reason": miss.get("reason", "missing_source_validation"),
                        "pair_source_file": miss.get("path"),
                    }
                )

            best = choose_source_best(val_rows)
            if best is None:
                selected_rows.append(
                    {
                        "segment": segment,
                        "source_dataset": src,
                        "selected_pooling": "",
                        "selected_layer": "",
                        "val_auc": "",
                        "val_acc": "",
                        "status": "missing_source_validation",
                        "source_file": "",
                    }
                )
                continue

            sp = normalize_pooling(best["pooling"])
            sl = int(best["layer"])
            source_selection[src] = (sp, sl)
            selected_rows.append(
                {
                    "segment": segment,
                    "source_dataset": src,
                    "selected_pooling": sp,
                    "selected_layer": sl,
                    "val_auc": float(best["val_auc"]),
                    "val_acc": float(best["val_acc"]),
                    "status": "ok",
                    "source_file": str(best["source_file"]),
                }
            )

        source_sel_mat = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
        source_sel_long: List[Dict[str, Any]] = []

        for i, src in enumerate(rows):
            selected = source_selection.get(src)
            for j, tgt in enumerate(cols):
                if selected is None:
                    source_sel_long.append(
                        {
                            "segment": segment,
                            "variant": "source_val_selected",
                            "row_dataset": src,
                            "col_dataset": tgt,
                            "pooling": "",
                            "layer": "",
                            "auc": "",
                            "status": "missing_source_selection",
                            "pair_source_file": "",
                        }
                    )
                    continue

                pooling, layer = selected
                lookup = pair_lookup_map[(src, tgt)]
                metric = lookup.metrics.get((pooling, int(layer)))
                if metric is None or metric.get("auc") is None:
                    missing_rows.append(
                        {
                            "segment": segment,
                            "variant": "source_val_selected",
                            "pooling": pooling,
                            "layer": int(layer),
                            "row_dataset": src,
                            "col_dataset": tgt,
                            "reason": "selected_config_missing_in_pair_test_metrics",
                            "pair_source_file": lookup.source_file,
                        }
                    )
                    source_sel_long.append(
                        {
                            "segment": segment,
                            "variant": "source_val_selected",
                            "row_dataset": src,
                            "col_dataset": tgt,
                            "pooling": pooling,
                            "layer": int(layer),
                            "auc": "",
                            "status": "missing",
                            "pair_source_file": lookup.source_file or "",
                        }
                    )
                    continue

                auc = float(metric["auc"])
                source_sel_mat[i, j] = auc
                source_sel_long.append(
                    {
                        "segment": segment,
                        "variant": "source_val_selected",
                        "row_dataset": src,
                        "col_dataset": tgt,
                        "pooling": pooling,
                        "layer": int(layer),
                        "auc": auc,
                        "status": "ok",
                        "pair_source_file": lookup.source_file or "",
                    }
                )

        source_sel_csv = seg_out / "matrix_source_val_selected_auc.csv"
        source_sel_png = seg_out / "heatmap_source_val_selected_auc.png"
        write_matrix_csv(source_sel_csv, rows, cols, source_sel_mat)
        plot_matrix_heatmap(
            output_path=source_sel_png,
            matrix=source_sel_mat,
            rows=rows,
            cols=cols,
            title=f"AUROC | {segment} | source-val-selected",
            segment=segment,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            dpi=args.dpi,
        )

        # Structured exports
        write_csv_rows(
            seg_out / "source_selected_configs.csv",
            selected_rows,
            [
                "segment",
                "source_dataset",
                "selected_pooling",
                "selected_layer",
                "val_auc",
                "val_acc",
                "status",
                "source_file",
            ],
        )
        write_csv_rows(
            seg_out / "matrix_long_fixed_configs.csv",
            fixed_long_rows,
            ["segment", "variant", "row_dataset", "col_dataset", "pooling", "layer", "auc", "status", "pair_source_file"],
        )
        write_csv_rows(
            seg_out / "matrix_long_source_val_selected.csv",
            source_sel_long,
            ["segment", "variant", "row_dataset", "col_dataset", "pooling", "layer", "auc", "status", "pair_source_file"],
        )
        write_csv_rows(
            seg_out / "missing_report.csv",
            missing_rows,
            ["segment", "variant", "pooling", "layer", "row_dataset", "col_dataset", "reason", "pair_source_file"],
        )

        global_summary["segments"][segment] = {
            "rows": rows,
            "cols": cols,
            "fixed_outputs": fixed_outputs,
            "source_val_selected": {
                "csv": str(source_sel_csv),
                "png": str(source_sel_png),
                "num_filled": int(np.sum(~np.isnan(source_sel_mat))),
                "num_total": int(source_sel_mat.size),
            },
            "tables": {
                "selected_configs_csv": str(seg_out / "source_selected_configs.csv"),
                "fixed_long_csv": str(seg_out / "matrix_long_fixed_configs.csv"),
                "source_val_selected_long_csv": str(seg_out / "matrix_long_source_val_selected.csv"),
                "missing_report_csv": str(seg_out / "missing_report.csv"),
            },
            "missing_count": len(missing_rows),
        }

    write_json(summary_path, global_summary)
    write_json(status_path, {"state": "completed", "message": "matrices generated", "updated_at": utc_now()})
    print(f"[done] outputs -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
