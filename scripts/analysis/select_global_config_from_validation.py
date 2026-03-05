#!/usr/bin/env python3
"""
Select one global (pooling, layer) config from validation across all pair cells,
then report its test matrix.

This script does not retrain probes. It reads existing pairwise OOD artifacts.
Selection objective defaults to minimax regret on validation AUROC.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


POOLING_ORDER = ["attn", "last", "max", "mean"]
ALL_POOLINGS = ["mean", "max", "last", "attn"]
SEGMENTS = ["completion", "full"]


@dataclass(frozen=True)
class Config:
    pooling: str
    layer: int

    @property
    def name(self) -> str:
        return f"{self.pooling}_L{self.layer}"


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
    if root.name == model_dir:
        return root.parent, root
    candidate = root / model_dir
    if candidate.exists():
        return root, candidate
    return root.parent, root


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


def parse_layers(value: str, num_layers: int) -> List[int]:
    if not value.strip():
        return list(range(num_layers))
    out: List[int] = []
    for item in parse_csv_list(value):
        if "-" in item:
            a, b = item.split("-", 1)
            start = int(a)
            end = int(b)
            if end < start:
                raise ValueError(f"Invalid layer range: {item}")
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(item))
    uniq = sorted(set(out))
    for layer in uniq:
        if layer < 0:
            raise ValueError(f"Negative layer in --layers: {layer}")
    if uniq and max(uniq) >= num_layers:
        raise ValueError(f"--layers has {max(uniq)} but num_layers is {num_layers}")
    return uniq


def normalize_pooling(pooling: Any) -> str:
    p = str(pooling).strip().lower()
    if p in ["none", "final_token", "final"]:
        return "last"
    return p


def pooling_rank(pooling: str) -> int:
    p = normalize_pooling(pooling)
    try:
        return POOLING_ORDER.index(p)
    except ValueError:
        return 999


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


def dataset_base(dataset_name: str) -> str:
    if dataset_name.endswith("-completion"):
        return dataset_name[: -len("-completion")]
    if dataset_name.endswith("-full"):
        return dataset_name[: -len("-full")]
    return dataset_name


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
    name = base_label_map().get(base, base.replace("Deception-", ""))
    return f"{name}\n({seg})"


def parse_pair_summary_metrics(data: Dict[str, Any]) -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    poolings = data.get("poolings", {})
    if not isinstance(poolings, dict):
        return out
    for pooling, payload in poolings.items():
        if not isinstance(payload, dict):
            continue
        p = normalize_pooling(pooling)
        layers = payload.get("layers", [])
        if isinstance(layers, list):
            for row in layers:
                if not isinstance(row, dict):
                    continue
                layer = parse_int(row.get("layer"))
                auc = parse_float(row.get("auc"))
                if layer is None or auc is None:
                    continue
                out[(p, int(layer))] = float(auc)
    return out


def parse_eval_ood_metrics(path: Path, pooling: str) -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    data = read_json(path, default={})
    layer_rows = data.get("layer_results", [])
    if not isinstance(layer_rows, list):
        return out
    p = normalize_pooling(pooling)
    for row in layer_rows:
        if not isinstance(row, dict):
            continue
        layer = parse_int(row.get("layer"))
        auc = parse_float(row.get("auc"))
        if layer is None or auc is None:
            continue
        out[(p, int(layer))] = float(auc)
    return out


def load_pair_split_metrics(
    pair_dir: Path,
    poolings: Sequence[str],
    split: str,
) -> Tuple[Dict[Tuple[str, int], float], List[str], Optional[str]]:
    """
    Returns:
      metrics[(pooling,layer)] = auc
      errors
      source path string
    """
    errors: List[str] = []

    # For test split, pair_summary.json is the preferred source if it has full layer metrics.
    if split == "test":
        p = pair_dir / "pair_summary.json"
        if p.exists():
            data = read_json(p, default={})
            metrics = parse_pair_summary_metrics(data)
            if metrics:
                return metrics, errors, str(p)
            errors.append("pair_summary_no_layer_metrics")

    metrics: Dict[Tuple[str, int], float] = {}
    source: Optional[str] = None

    if split in ["validation", "val"]:
        split_patterns = ["*validation.json", "*_val.json"]
    elif split == "test":
        split_patterns = ["*_test.json"]
    else:
        split_patterns = [f"*_{split}.json"]

    for pooling in poolings:
        p = normalize_pooling(pooling)
        pdir = pair_dir / p
        if not pdir.exists():
            continue
        files: List[Path] = []
        for pat in split_patterns:
            files.extend(sorted(pdir.glob(f"eval_ood_{pat}")))
        if not files:
            continue
        for f in files:
            metrics.update(parse_eval_ood_metrics(f, p))
        source = str(pdir)

    if metrics:
        return metrics, errors, source

    # Last fallback for test only: best-only file.
    if split == "test":
        p = pair_dir / "ood_results_all_pooling.json"
        if p.exists():
            data = read_json(p, default={})
            for pooling in ALL_POOLINGS:
                payload = data.get(pooling)
                if not isinstance(payload, dict):
                    continue
                layer = parse_int(payload.get("best_layer"))
                auc = parse_float(payload.get("best_auc"))
                if layer is None or auc is None:
                    continue
                metrics[(normalize_pooling(pooling), int(layer))] = float(auc)
            if metrics:
                return metrics, errors, str(p)
            errors.append("ood_results_all_pooling_missing_metrics")

    errors.append(f"no_metrics_found_for_split_{split}")
    return metrics, errors, source


def write_matrix_csv(path: Path, rows: Sequence[str], cols: Sequence[str], matrix: np.ndarray) -> None:
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


def plot_heatmap(
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
    ax.set_ylabel("Full-prompt probes" if segment == "full" else "Completion probes")
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    mid = (vmin + vmax) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = "NA" if np.isnan(v) else f"{v:.2f}"
            color = "#4a4a4a" if np.isnan(v) or v < mid else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

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
    p = argparse.ArgumentParser(description="Select global config from validation and report test matrix.")
    p.add_argument(
        "--ood_results_root",
        type=str,
        default="/content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/ood_evaluation",
    )
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--artifact_root", type=str, default="artifacts")
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--segments", type=str, default="completion,full")
    p.add_argument("--poolings", type=str, default="mean,max,last,attn")
    p.add_argument("--num_layers", type=int, default=15)
    p.add_argument(
        "--layers",
        type=str,
        default="",
        help="Comma list/ranges (e.g. 0-14 or 2,4,6). Empty => all [0,num_layers-1].",
    )
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--test_split", type=str, default="test")
    p.add_argument(
        "--objective",
        type=str,
        default="minimax_regret",
        choices=["minimax_regret", "maximin", "mean"],
    )
    p.add_argument("--exclude_diagonal", action="store_true", default=False)
    p.add_argument(
        "--missing_score",
        type=float,
        default=0.5,
        help="Score used for missing cells in objective computation.",
    )
    p.add_argument("--min_coverage", type=float, default=0.95)
    p.add_argument("--vmin", type=float, default=0.5)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--cmap", type=str, default="YlGnBu")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    run_id = args.run_id or utc_run_id()
    segments = parse_csv_list(args.segments)
    for seg in segments:
        if seg not in SEGMENTS:
            raise ValueError(f"Unsupported segment: {seg}")

    poolings = [normalize_pooling(p) for p in parse_csv_list(args.poolings)]
    for p in poolings:
        if p not in ALL_POOLINGS:
            raise ValueError(f"Unsupported pooling: {p}")

    layers = parse_layers(args.layers, num_layers=args.num_layers)
    configs = [Config(pooling=p, layer=l) for p in poolings for l in layers]
    if not configs:
        raise RuntimeError("No candidate configs.")

    ood_base, ood_model_root = split_root_and_model(Path(args.ood_results_root), model_dir)
    if args.output_root:
        run_root = Path(args.output_root) / run_id
    else:
        run_root = Path(args.artifact_root) / "runs" / "global_config_selection_from_validation" / model_dir / run_id
    meta_dir = run_root / "meta"
    out_dir = run_root / "results"
    ensure_dir(meta_dir)
    ensure_dir(out_dir)

    status_path = meta_dir / "status.json"
    write_json(status_path, {"state": "running", "message": "selecting global config", "updated_at": utc_now()})
    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": args.model,
            "paths": {
                "ood_base": str(ood_base),
                "ood_model_root": str(ood_model_root),
                "run_root": str(run_root),
            },
            "segments": segments,
            "poolings": poolings,
            "layers": layers,
            "objective": args.objective,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "exclude_diagonal": bool(args.exclude_diagonal),
            "missing_score": float(args.missing_score),
            "min_coverage": float(args.min_coverage),
        },
    )

    rows_map, cols_map = stage_spec()
    summary: Dict[str, Any] = {"run_id": run_id, "segments": {}}

    for segment in segments:
        rows = rows_map[segment]
        cols = cols_map[segment]
        seg_out = out_dir / segment
        ensure_dir(seg_out)
        print(f"[segment] {segment} rows={len(rows)} cols={len(cols)} configs={len(configs)}")

        # Build val/test score tensors: C x R x K
        val_tensor = np.full((len(configs), len(rows), len(cols)), np.nan, dtype=np.float64)
        test_tensor = np.full((len(configs), len(rows), len(cols)), np.nan, dtype=np.float64)
        missing_rows: List[Dict[str, Any]] = []

        for i, src in enumerate(rows):
            for j, tgt in enumerate(cols):
                pair_dir = ood_model_root / f"from-{src}" / f"to-{tgt}"
                val_metrics, val_errors, val_source = load_pair_split_metrics(pair_dir, poolings=poolings, split=args.val_split)
                test_metrics, test_errors, test_source = load_pair_split_metrics(pair_dir, poolings=poolings, split=args.test_split)
                for ci, cfg in enumerate(configs):
                    v = val_metrics.get((cfg.pooling, cfg.layer))
                    t = test_metrics.get((cfg.pooling, cfg.layer))
                    if v is not None:
                        val_tensor[ci, i, j] = float(v)
                    if t is not None:
                        test_tensor[ci, i, j] = float(t)
                if not val_metrics:
                    missing_rows.append(
                        {
                            "segment": segment,
                            "row_dataset": src,
                            "col_dataset": tgt,
                            "split": args.val_split,
                            "reason": ";".join(val_errors) if val_errors else "no_validation_metrics",
                            "source_path": val_source or str(pair_dir),
                        }
                    )
                if not test_metrics:
                    missing_rows.append(
                        {
                            "segment": segment,
                            "row_dataset": src,
                            "col_dataset": tgt,
                            "split": args.test_split,
                            "reason": ";".join(test_errors) if test_errors else "no_test_metrics",
                            "source_path": test_source or str(pair_dir),
                        }
                    )

        # Flatten cells for objective.
        cell_indices: List[Tuple[int, int]] = []
        for i, src in enumerate(rows):
            for j, tgt in enumerate(cols):
                if args.exclude_diagonal and src == tgt:
                    continue
                cell_indices.append((i, j))
        n_cells = len(cell_indices)
        if n_cells == 0:
            raise RuntimeError(f"No objective cells for {segment}.")

        val_flat = np.full((len(configs), n_cells), np.nan, dtype=np.float64)
        test_flat = np.full((len(configs), n_cells), np.nan, dtype=np.float64)
        for ci in range(len(configs)):
            for pi, (i, j) in enumerate(cell_indices):
                val_flat[ci, pi] = val_tensor[ci, i, j]
                test_flat[ci, pi] = test_tensor[ci, i, j]

        # Coverage and objective matrices.
        coverage = np.mean(~np.isnan(val_flat), axis=1)
        val_filled = np.where(np.isnan(val_flat), args.missing_score, val_flat)
        oracle = np.max(val_filled, axis=0)  # per-cell best among configs
        regrets = oracle[None, :] - val_filled

        ranking_rows: List[Dict[str, Any]] = []
        for ci, cfg in enumerate(configs):
            reg = regrets[ci]
            row = {
                "segment": segment,
                "config": cfg.name,
                "pooling": cfg.pooling,
                "layer": cfg.layer,
                "coverage": float(coverage[ci]),
                "max_regret": float(np.max(reg)),
                "q90_regret": float(np.quantile(reg, 0.90)),
                "mean_regret": float(np.mean(reg)),
                "min_val_auc": float(np.min(val_filled[ci])),
                "q10_val_auc": float(np.quantile(val_filled[ci], 0.10)),
                "mean_val_auc": float(np.mean(val_filled[ci])),
                "min_test_auc": float(np.min(np.where(np.isnan(test_flat[ci]), args.missing_score, test_flat[ci]))),
                "mean_test_auc": float(np.mean(np.where(np.isnan(test_flat[ci]), args.missing_score, test_flat[ci]))),
            }
            ranking_rows.append(row)

        # Keep candidates with enough coverage; if none, fall back to all.
        eligible = [r for r in ranking_rows if r["coverage"] >= float(args.min_coverage)]
        if not eligible:
            eligible = ranking_rows

        def sort_key(r: Dict[str, Any]) -> Tuple[float, float, float, float, int, int]:
            if args.objective == "maximin":
                primary = -float(r["min_val_auc"])
                secondary = -float(r["q10_val_auc"])
                tertiary = -float(r["mean_val_auc"])
            elif args.objective == "mean":
                primary = -float(r["mean_val_auc"])
                secondary = -float(r["q10_val_auc"])
                tertiary = -float(r["min_val_auc"])
            else:
                primary = float(r["max_regret"])
                secondary = float(r["q90_regret"])
                tertiary = -float(r["mean_val_auc"])
            # final tie-breaks
            return (
                primary,
                secondary,
                tertiary,
                -float(r["coverage"]),
                pooling_rank(str(r["pooling"])),
                int(r["layer"]),
            )

        eligible_sorted = sorted(eligible, key=sort_key)
        best = eligible_sorted[0]
        best_cfg = Config(pooling=str(best["pooling"]), layer=int(best["layer"]))
        best_ci = configs.index(best_cfg)

        # Export ranking and selected config.
        ranking_rows_sorted = sorted(ranking_rows, key=sort_key)
        write_csv_rows(
            seg_out / "global_config_ranking_validation.csv",
            ranking_rows_sorted,
            [
                "segment",
                "config",
                "pooling",
                "layer",
                "coverage",
                "max_regret",
                "q90_regret",
                "mean_regret",
                "min_val_auc",
                "q10_val_auc",
                "mean_val_auc",
                "min_test_auc",
                "mean_test_auc",
            ],
        )
        write_csv_rows(
            seg_out / "missing_pair_metrics.csv",
            missing_rows,
            ["segment", "row_dataset", "col_dataset", "split", "reason", "source_path"],
        )

        selected_payload = {
            "segment": segment,
            "selected_config": {"pooling": best_cfg.pooling, "layer": best_cfg.layer, "name": best_cfg.name},
            "objective": args.objective,
            "exclude_diagonal": bool(args.exclude_diagonal),
            "missing_score": float(args.missing_score),
            "min_coverage": float(args.min_coverage),
            "coverage_selected": float(best["coverage"]),
            "validation_metrics_selected": {
                "max_regret": float(best["max_regret"]),
                "q90_regret": float(best["q90_regret"]),
                "mean_regret": float(best["mean_regret"]),
                "min_val_auc": float(best["min_val_auc"]),
                "q10_val_auc": float(best["q10_val_auc"]),
                "mean_val_auc": float(best["mean_val_auc"]),
            },
            "created_at": utc_now(),
        }
        write_json(seg_out / "selected_global_config.json", selected_payload)

        # Selected config matrices.
        val_sel = val_tensor[best_ci]
        test_sel = test_tensor[best_ci]
        val_csv = seg_out / "matrix_validation_selected_config_auc.csv"
        test_csv = seg_out / "matrix_test_selected_config_auc.csv"
        val_png = seg_out / "heatmap_validation_selected_config_auc.png"
        test_png = seg_out / "heatmap_test_selected_config_auc.png"
        write_matrix_csv(val_csv, rows, cols, val_sel)
        write_matrix_csv(test_csv, rows, cols, test_sel)
        plot_heatmap(
            output_path=val_png,
            matrix=val_sel,
            rows=rows,
            cols=cols,
            title=f"AUROC | {segment} | validation selected {best_cfg.name}",
            segment=segment,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            dpi=args.dpi,
        )
        plot_heatmap(
            output_path=test_png,
            matrix=test_sel,
            rows=rows,
            cols=cols,
            title=f"AUROC | {segment} | test for selected {best_cfg.name}",
            segment=segment,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            dpi=args.dpi,
        )

        summary["segments"][segment] = {
            "selected_config": best_cfg.name,
            "selected_pooling": best_cfg.pooling,
            "selected_layer": best_cfg.layer,
            "outputs": {
                "ranking_csv": str(seg_out / "global_config_ranking_validation.csv"),
                "selected_json": str(seg_out / "selected_global_config.json"),
                "validation_matrix_csv": str(val_csv),
                "test_matrix_csv": str(test_csv),
                "validation_heatmap_png": str(val_png),
                "test_heatmap_png": str(test_png),
                "missing_csv": str(seg_out / "missing_pair_metrics.csv"),
            },
        }
        print(
            f"[selected] {segment}: {best_cfg.name} "
            f"(cov={best['coverage']:.3f}, max_regret={best['max_regret']:.4f}, mean_val_auc={best['mean_val_auc']:.4f})"
        )

    write_json(out_dir / "summary.json", summary)
    write_json(status_path, {"state": "completed", "message": "done", "updated_at": utc_now()})
    print(f"[done] outputs -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
