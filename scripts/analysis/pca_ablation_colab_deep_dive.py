#!/usr/bin/env python3
"""
Colab-first deep-dive analysis for PCA ablation sweeps.

This script aggregates pooling-wise PCA ablation JSON outputs, computes least-k OOD gains,
builds ranked analysis tables, creates convenience plots, and prepares phase-2 PCA direction
catalog artifacts for SAE/oracle workflows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_BASELINE_KEYS = (
    "id_val_auc",
    "id_val_acc",
    "ood_test_auc",
    "ood_test_acc",
    "generalization_gap",
)

REQUIRED_SWEEP_KEYS = (
    "id_val_auc",
    "id_val_acc",
    "ood_test_auc",
    "ood_test_acc",
    "generalization_gap",
)

DEFAULT_POOLINGS = ("mean", "max", "last")
STEP_DISCOVER = "discover_inputs"
STEP_BUILD_TABLES = "build_tables"
STEP_PLOTS = "build_plots"
STEP_PHASE2 = "phase2_prep"
STEP_FINALIZE = "finalize"


def utc_now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def to_float(value: object, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def parse_poolings(raw: str) -> List[str]:
    if not raw.strip():
        return list(DEFAULT_POOLINGS)
    out: List[str] = []
    seen = set()
    for token in raw.split(","):
        pooling = token.strip().lower()
        if not pooling or pooling in seen:
            continue
        out.append(pooling)
        seen.add(pooling)
    return out


def try_get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def init_status(meta_dir: Path, run_id: str, state: str, current_step: Optional[str]) -> None:
    status = {
        "run_id": run_id,
        "state": state,
        "current_step": current_step,
        "last_updated_utc": utc_now_iso(),
    }
    write_json(meta_dir / "status.json", status)


def load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {
            "completed_steps": [],
            "updated_at_utc": utc_now_iso(),
            "step_outputs": {},
        }
    try:
        payload = read_json(progress_path)
    except Exception:
        payload = {
            "completed_steps": [],
            "updated_at_utc": utc_now_iso(),
            "step_outputs": {},
        }
    payload.setdefault("completed_steps", [])
    payload.setdefault("step_outputs", {})
    return payload


def save_progress(progress_path: Path, progress: dict) -> None:
    progress["updated_at_utc"] = utc_now_iso()
    write_json(progress_path, progress)


def mark_step_completed(progress: dict, step: str, outputs: Iterable[Path]) -> None:
    if step not in progress["completed_steps"]:
        progress["completed_steps"].append(step)
    progress["step_outputs"][step] = [str(p) for p in outputs]


def outputs_exist(paths: Iterable[Path]) -> bool:
    for p in paths:
        if not p.exists():
            return False
    return True


def normalize_result_maps(results: dict) -> Tuple[Dict[int, dict], Dict[int, Dict[int, dict]]]:
    baseline_raw = results.get("baseline", {})
    sweep_raw = results.get("sweep", {})

    baseline: Dict[int, dict] = {}
    for layer, metrics in baseline_raw.items():
        baseline[int(layer)] = metrics

    sweep: Dict[int, Dict[int, dict]] = {}
    for k, layer_map in sweep_raw.items():
        sweep[int(k)] = {int(layer): metrics for layer, metrics in layer_map.items()}
    return baseline, sweep


def validate_keys(metrics: dict, required: Iterable[str], context: str) -> None:
    missing = [k for k in required if k not in metrics]
    if missing:
        raise KeyError(f"Missing required keys {missing} in {context}")


def discover_inputs(
    input_root: Path,
    poolings: List[str],
    results_filename: str,
    strict: bool,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    available: Dict[str, dict] = {}
    missing: Dict[str, str] = {}

    for pooling in poolings:
        pooling_dir = input_root / pooling
        results_path = pooling_dir / results_filename
        pca_artifacts_dir = pooling_dir / "pca_artifacts"
        if not results_path.exists():
            missing[pooling] = f"missing {results_filename}"
            continue
        available[pooling] = {
            "pooling_dir": str(pooling_dir),
            "results_path": str(results_path),
            "pca_artifacts_dir": str(pca_artifacts_dir),
        }

    if strict and missing:
        missing_msg = ", ".join([f"{k}: {v}" for k, v in sorted(missing.items())])
        raise FileNotFoundError(f"Missing pooling inputs under {input_root}: {missing_msg}")
    if not available:
        raise FileNotFoundError(f"No valid pooling inputs found under {input_root}")
    return available, missing


def build_pooling_frames(pooling: str, results: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline, sweep = normalize_result_maps(results)

    baseline_rows: List[dict] = []
    baseline_map: Dict[int, dict] = {}
    for layer in sorted(baseline):
        metrics = baseline[layer]
        validate_keys(metrics, REQUIRED_BASELINE_KEYS, f"baseline[{pooling}][{layer}]")

        b = {
            "pooling": pooling,
            "layer": int(layer),
            "baseline_id_val_auc": to_float(metrics.get("id_val_auc")),
            "baseline_id_val_acc": to_float(metrics.get("id_val_acc")),
            "baseline_ood_test_auc": to_float(metrics.get("ood_test_auc")),
            "baseline_ood_test_acc": to_float(metrics.get("ood_test_acc")),
            "baseline_gap": to_float(metrics.get("generalization_gap")),
        }
        baseline_rows.append(b)
        baseline_map[layer] = b

    long_rows: List[dict] = []
    for k in sorted(sweep):
        layer_map = sweep[k]
        for layer in sorted(layer_map):
            metrics = layer_map[layer]
            validate_keys(metrics, REQUIRED_SWEEP_KEYS, f"sweep[{pooling}][{k}][{layer}]")

            baseline_for_layer = baseline_map.get(
                layer,
                {
                    "baseline_id_val_auc": np.nan,
                    "baseline_id_val_acc": np.nan,
                    "baseline_ood_test_auc": np.nan,
                    "baseline_ood_test_acc": np.nan,
                    "baseline_gap": np.nan,
                },
            )

            id_val_auc = to_float(metrics.get("id_val_auc"))
            id_val_acc = to_float(metrics.get("id_val_acc"))
            ood_test_auc = to_float(metrics.get("ood_test_auc"))
            ood_test_acc = to_float(metrics.get("ood_test_acc"))
            generalization_gap = to_float(metrics.get("generalization_gap"))

            base_id_auc = baseline_for_layer["baseline_id_val_auc"]
            base_ood_auc = baseline_for_layer["baseline_ood_test_auc"]
            base_gap = baseline_for_layer["baseline_gap"]

            delta_id = to_float(
                metrics.get("delta_id_auc_vs_baseline"),
                default=id_val_auc - base_id_auc,
            )
            delta_ood = to_float(
                metrics.get("delta_ood_auc_vs_baseline"),
                default=ood_test_auc - base_ood_auc,
            )
            delta_gap = to_float(
                metrics.get("delta_gap_vs_baseline"),
                default=generalization_gap - base_gap,
            )

            long_rows.append(
                {
                    "pooling": pooling,
                    "k": int(k),
                    "layer": int(layer),
                    "id_val_auc": id_val_auc,
                    "id_val_acc": id_val_acc,
                    "ood_test_auc": ood_test_auc,
                    "ood_test_acc": ood_test_acc,
                    "generalization_gap": generalization_gap,
                    "delta_id_auc_vs_baseline": delta_id,
                    "delta_ood_auc_vs_baseline": delta_ood,
                    "delta_gap_vs_baseline": delta_gap,
                    "baseline_id_val_auc": base_id_auc,
                    "baseline_id_val_acc": baseline_for_layer["baseline_id_val_acc"],
                    "baseline_ood_test_auc": base_ood_auc,
                    "baseline_ood_test_acc": baseline_for_layer["baseline_ood_test_acc"],
                    "baseline_gap": base_gap,
                }
            )

    baseline_df = pd.DataFrame(baseline_rows).sort_values(["pooling", "layer"]).reset_index(drop=True)
    long_df = pd.DataFrame(long_rows).sort_values(["pooling", "layer", "k"]).reset_index(drop=True)
    return baseline_df, long_df


def validate_delta_consistency(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()

    checks = df_long.copy()
    checks["calc_delta_id"] = checks["id_val_auc"] - checks["baseline_id_val_auc"]
    checks["calc_delta_ood"] = checks["ood_test_auc"] - checks["baseline_ood_test_auc"]
    checks["calc_delta_gap"] = checks["generalization_gap"] - checks["baseline_gap"]
    checks["abs_err_delta_id"] = (checks["delta_id_auc_vs_baseline"] - checks["calc_delta_id"]).abs()
    checks["abs_err_delta_ood"] = (checks["delta_ood_auc_vs_baseline"] - checks["calc_delta_ood"]).abs()
    checks["abs_err_delta_gap"] = (checks["delta_gap_vs_baseline"] - checks["calc_delta_gap"]).abs()

    grouped = (
        checks.groupby("pooling", as_index=False)[
            ["abs_err_delta_id", "abs_err_delta_ood", "abs_err_delta_gap"]
        ]
        .max()
        .rename(
            columns={
                "abs_err_delta_id": "max_abs_err_delta_id",
                "abs_err_delta_ood": "max_abs_err_delta_ood",
                "abs_err_delta_gap": "max_abs_err_delta_gap",
            }
        )
    )
    return grouped


def extract_least_k_ood_gain(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    gain = df_long[df_long["delta_ood_auc_vs_baseline"] > 0].copy()
    if gain.empty:
        all_pairs = df_long[["pooling", "layer"]].drop_duplicates().sort_values(["pooling", "layer"])
        return pd.DataFrame(), all_pairs.reset_index(drop=True)

    gain = gain.sort_values(
        ["pooling", "layer", "k", "delta_ood_auc_vs_baseline"],
        ascending=[True, True, True, False],
    )
    first = gain.drop_duplicates(["pooling", "layer"], keep="first").copy()
    first = first.sort_values(["pooling", "layer"]).reset_index(drop=True)

    least = first.rename(
        columns={
            "k": "k_min_ood_gain",
            "ood_test_auc": "ood_test_auc_at_k",
            "id_val_auc": "id_val_auc_at_k",
            "id_val_acc": "id_val_acc_at_k",
            "ood_test_acc": "ood_test_acc_at_k",
            "generalization_gap": "gap_at_k",
        }
    )[
        [
            "pooling",
            "layer",
            "k_min_ood_gain",
            "baseline_ood_test_auc",
            "ood_test_auc_at_k",
            "delta_ood_auc_vs_baseline",
            "baseline_id_val_auc",
            "id_val_auc_at_k",
            "delta_id_auc_vs_baseline",
            "baseline_id_val_acc",
            "id_val_acc_at_k",
            "baseline_ood_test_acc",
            "ood_test_acc_at_k",
            "baseline_gap",
            "gap_at_k",
            "delta_gap_vs_baseline",
        ]
    ]

    all_pairs = df_long[["pooling", "layer"]].drop_duplicates()
    gained_pairs = least[["pooling", "layer"]].drop_duplicates()
    no_gain = (
        all_pairs.merge(gained_pairs.assign(_has_gain=1), on=["pooling", "layer"], how="left")
        .query("_has_gain != 1")
        .drop(columns="_has_gain")
        .sort_values(["pooling", "layer"])
        .reset_index(drop=True)
    )

    return least, no_gain


def build_ranking_tables(df_leastk: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df_leastk.empty:
        return {
            "rank_by_k_then_ood_gain": df_leastk.copy(),
            "rank_by_ood_gain_ascending": df_leastk.copy(),
            "rank_by_ood_gain_desc": df_leastk.copy(),
        }

    by_k_then_gain = df_leastk.sort_values(
        ["k_min_ood_gain", "delta_ood_auc_vs_baseline"],
        ascending=[True, False],
    ).reset_index(drop=True)
    by_gain_asc = df_leastk.sort_values(
        ["delta_ood_auc_vs_baseline", "k_min_ood_gain"],
        ascending=[True, True],
    ).reset_index(drop=True)
    by_gain_desc = df_leastk.sort_values(
        ["delta_ood_auc_vs_baseline", "k_min_ood_gain"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return {
        "rank_by_k_then_ood_gain": by_k_then_gain,
        "rank_by_ood_gain_ascending": by_gain_asc,
        "rank_by_ood_gain_desc": by_gain_desc,
    }


def build_all_point_rankings(df_long: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cols = [
        "pooling",
        "layer",
        "k",
        "baseline_ood_test_auc",
        "ood_test_auc",
        "delta_ood_auc_vs_baseline",
        "baseline_id_val_auc",
        "id_val_auc",
        "delta_id_auc_vs_baseline",
        "baseline_gap",
        "generalization_gap",
        "delta_gap_vs_baseline",
    ]
    if df_long.empty:
        empty = pd.DataFrame(columns=cols)
        return {
            "rank_all_points_by_ood_gain_ascending": empty.copy(),
            "rank_all_points_by_ood_gain_desc": empty.copy(),
        }

    view = df_long[cols].copy()
    asc = view.sort_values(
        ["delta_ood_auc_vs_baseline", "k", "pooling", "layer"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    desc = view.sort_values(
        ["delta_ood_auc_vs_baseline", "k", "pooling", "layer"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return {
        "rank_all_points_by_ood_gain_ascending": asc,
        "rank_all_points_by_ood_gain_desc": desc,
    }


def build_k_aggregate_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(
            columns=[
                "pooling",
                "k",
                "num_layers",
                "num_positive_ood_gain_layers",
                "mean_delta_ood_auc",
                "median_delta_ood_auc",
                "max_delta_ood_auc",
                "min_delta_ood_auc",
                "mean_delta_id_auc",
                "mean_ood_test_auc",
                "mean_id_val_auc",
            ]
        )

    tmp = df_long.copy()
    tmp["ood_positive"] = (tmp["delta_ood_auc_vs_baseline"] > 0).astype(int)

    grouped = (
        tmp.groupby(["pooling", "k"], as_index=False)
        .agg(
            num_layers=("layer", "nunique"),
            num_positive_ood_gain_layers=("ood_positive", "sum"),
            mean_delta_ood_auc=("delta_ood_auc_vs_baseline", "mean"),
            median_delta_ood_auc=("delta_ood_auc_vs_baseline", "median"),
            max_delta_ood_auc=("delta_ood_auc_vs_baseline", "max"),
            min_delta_ood_auc=("delta_ood_auc_vs_baseline", "min"),
            mean_delta_id_auc=("delta_id_auc_vs_baseline", "mean"),
            mean_ood_test_auc=("ood_test_auc", "mean"),
            mean_id_val_auc=("id_val_auc", "mean"),
        )
        .sort_values(["pooling", "k"])
        .reset_index(drop=True)
    )
    return grouped


def save_table(df: pd.DataFrame, csv_path: Path, parquet_path: Optional[Path] = None) -> None:
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    if parquet_path is not None:
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            # Parquet engine may be unavailable in some colab kernels.
            pass


def plot_heatmap_by_pooling(
    df_long: pd.DataFrame,
    metric_col: str,
    out_dir: Path,
    filename_prefix: str,
    title_prefix: str,
) -> List[Path]:
    ensure_dir(out_dir)
    outputs: List[Path] = []
    if df_long.empty:
        return outputs

    for pooling in sorted(df_long["pooling"].unique()):
        sub = df_long[df_long["pooling"] == pooling].copy()
        if sub.empty:
            continue

        pivot = (
            sub.pivot_table(index="k", columns="layer", values=metric_col, aggfunc="mean")
            .sort_index()
            .sort_index(axis=1)
        )
        if pivot.empty:
            continue

        values = pivot.values.astype(np.float64)
        finite = np.isfinite(values)
        vmax = float(np.nanmax(np.abs(values[finite]))) if finite.any() else 0.01
        vmax = max(vmax, 1e-4)

        plt.figure(figsize=(max(8, 0.45 * len(pivot.columns)), max(5, 0.4 * len(pivot.index))))
        draw = np.where(np.isfinite(values), values, 0.0)
        plt.imshow(draw, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(label=metric_col)
        plt.xticks(range(len(pivot.columns)), pivot.columns.tolist(), rotation=45)
        plt.yticks(range(len(pivot.index)), pivot.index.tolist())
        plt.xlabel("Layer")
        plt.ylabel("k removed PCs")
        plt.title(f"{title_prefix} | {pooling.upper()}")
        plt.tight_layout()
        out_path = out_dir / f"{filename_prefix}_{pooling}.png"
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        outputs.append(out_path)

    return outputs


def plot_delta_scatter(df_long: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if df_long.empty:
        return None

    plt.figure(figsize=(8.5, 6.5))
    colors = {
        "mean": "#1f77b4",
        "max": "#d62728",
        "last": "#2ca02c",
        "attn": "#9467bd",
    }

    for pooling in sorted(df_long["pooling"].unique()):
        sub = df_long[df_long["pooling"] == pooling]
        if sub.empty:
            continue
        sizes = 30.0 + 9.0 * np.log1p(sub["k"].to_numpy(dtype=np.float64))
        plt.scatter(
            sub["delta_id_auc_vs_baseline"],
            sub["delta_ood_auc_vs_baseline"],
            s=sizes,
            alpha=0.75,
            label=pooling,
            color=colors.get(pooling, "#333333"),
            edgecolor="black",
            linewidth=0.4,
        )

    plt.axvline(0.0, linestyle="--", color="#555555", linewidth=1.0, alpha=0.7)
    plt.axhline(0.0, linestyle="--", color="#555555", linewidth=1.0, alpha=0.7)
    plt.xlabel("Delta ID AUC vs baseline")
    plt.ylabel("Delta OOD AUC vs baseline")
    plt.title("ID vs OOD Delta Scatter (size ~ k)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return out_path


def pareto_nondominated(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def plot_pareto_front(df_long: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if df_long.empty:
        return None

    clean = df_long[["delta_id_auc_vs_baseline", "delta_ood_auc_vs_baseline", "pooling", "layer", "k"]].dropna()
    if clean.empty:
        return None

    x = clean["delta_id_auc_vs_baseline"].to_numpy(dtype=np.float64)
    y = clean["delta_ood_auc_vs_baseline"].to_numpy(dtype=np.float64)
    mask = pareto_nondominated(x, y)

    plt.figure(figsize=(8.5, 6.5))
    plt.scatter(x, y, s=28, alpha=0.3, color="#7f7f7f", label="All points")
    plt.scatter(
        x[mask],
        y[mask],
        s=70,
        alpha=0.95,
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
        label="Pareto frontier",
    )
    plt.axvline(0.0, linestyle="--", color="#555555", linewidth=1.0, alpha=0.7)
    plt.axhline(0.0, linestyle="--", color="#555555", linewidth=1.0, alpha=0.7)
    plt.xlabel("Delta ID AUC vs baseline")
    plt.ylabel("Delta OOD AUC vs baseline")
    plt.title("Pareto Frontier: ID/OOD Delta Tradeoff")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return out_path


def plot_best_ood_gain_by_pooling(df_leastk: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if df_leastk.empty:
        return None

    rows = []
    for pooling, sub in df_leastk.groupby("pooling"):
        best = sub.sort_values("delta_ood_auc_vs_baseline", ascending=False).iloc[0]
        rows.append(
            {
                "pooling": pooling,
                "best_delta_ood": float(best["delta_ood_auc_vs_baseline"]),
                "layer": int(best["layer"]),
                "k": int(best["k_min_ood_gain"]),
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("best_delta_ood", ascending=False)
    if plot_df.empty:
        return None

    plt.figure(figsize=(7.5, 5.0))
    bars = plt.bar(plot_df["pooling"], plot_df["best_delta_ood"], color="#1f77b4", alpha=0.85)
    for bar, (_, r) in zip(bars, plot_df.iterrows()):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"L{r['layer']} @ k={r['k']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.axhline(0.0, linestyle="--", color="#555555", linewidth=1.0, alpha=0.7)
    plt.ylabel("Best delta OOD AUC vs baseline")
    plt.xlabel("Pooling")
    plt.title("Best Least-k OOD Gain by Pooling")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return out_path


def parse_layer_from_npz_name(path: Path) -> Optional[int]:
    m = re.match(r"layer_(\d+)\.npz$", path.name)
    if not m:
        return None
    return int(m.group(1))


def build_pca_direction_catalog(
    pooling: str,
    pca_artifacts_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not pca_artifacts_dir.exists():
        return pd.DataFrame(), pd.DataFrame(), {"pooling": pooling, "reason": "missing pca_artifacts_dir"}

    rows: List[dict] = []
    inventory_rows: List[dict] = []
    files = sorted(pca_artifacts_dir.glob("layer_*.npz"))

    for path in files:
        layer = parse_layer_from_npz_name(path)
        if layer is None:
            continue
        with np.load(path, allow_pickle=False) as data:
            components = data.get("components")
            evr = data.get("explained_variance_ratio")
            sign_flips = data.get("sign_flips")
            mean = data.get("mean")
            n_train_samples_raw = data.get("n_train_samples")
            dim_raw = data.get("dim")

            if components is None or evr is None or mean is None:
                continue

            n_comp = int(components.shape[0])
            dim = int(components.shape[1]) if components.ndim == 2 else int(mean.shape[-1])
            n_train_samples = (
                int(np.asarray(n_train_samples_raw).reshape(-1)[0])
                if n_train_samples_raw is not None
                else np.nan
            )
            dim_value = int(np.asarray(dim_raw).reshape(-1)[0]) if dim_raw is not None else dim

            if sign_flips is None:
                sign_flips = np.zeros((n_comp,), dtype=np.int8)
            sign_flips = np.asarray(sign_flips).reshape(-1)
            evr = np.asarray(evr).reshape(-1)
            cumsum = np.cumsum(evr)

            for idx in range(n_comp):
                rows.append(
                    {
                        "pooling": pooling,
                        "layer": layer,
                        "pc_index": int(idx),
                        "pc_rank": int(idx + 1),
                        "explained_variance_ratio": float(evr[idx]),
                        "cumulative_explained_variance_ratio": float(cumsum[idx]),
                        "sign_flip": int(sign_flips[idx]) if idx < sign_flips.shape[0] else 0,
                        "n_components_saved": n_comp,
                        "dim": dim_value,
                        "n_train_samples": n_train_samples,
                        "artifact_file": path.name,
                        "artifact_relpath": str(path),
                    }
                )

            inventory_rows.append(
                {
                    "pooling": pooling,
                    "layer": layer,
                    "artifact_file": path.name,
                    "artifact_path": str(path),
                    "n_components_saved": n_comp,
                    "dim": dim_value,
                    "n_train_samples": n_train_samples,
                }
            )

    manifest = {
        "pooling": pooling,
        "pca_artifacts_dir": str(pca_artifacts_dir),
        "num_layer_npz_files": int(len(files)),
        "num_catalog_rows": int(len(rows)),
        "generated_at_utc": utc_now_iso(),
    }
    return pd.DataFrame(rows), pd.DataFrame(inventory_rows), manifest


def write_oracle_interface_templates(phase2_dir: Path) -> List[Path]:
    ensure_dir(phase2_dir)

    columns = [
        {
            "name": "sample_id",
            "dtype": "string",
            "required": True,
            "description": "Stable sample identifier from activation manifest.",
        },
        {
            "name": "split",
            "dtype": "string",
            "required": True,
            "description": "Dataset split name (id_train/id_val/ood_test/etc.).",
        },
        {
            "name": "label",
            "dtype": "int",
            "required": True,
            "description": "Binary class label for deception task.",
        },
        {
            "name": "pooling",
            "dtype": "string",
            "required": True,
            "description": "Pooling strategy used in probe/PCA analysis.",
        },
        {
            "name": "layer",
            "dtype": "int",
            "required": True,
            "description": "Transformer layer index for the vector.",
        },
        {
            "name": "k_removed",
            "dtype": "int",
            "required": True,
            "description": "Number of removed PCs applied before exporting vector.",
        },
        {
            "name": "vector_type",
            "dtype": "string",
            "required": True,
            "description": "Feature flavor (pooled, residualized, pca_projection, sae_latent, etc.).",
        },
        {
            "name": "vector_ref",
            "dtype": "string",
            "required": True,
            "description": "Pointer to vector payload (npz key, parquet row group, or file path).",
        },
        {
            "name": "source_activation_dir",
            "dtype": "string",
            "required": False,
            "description": "Path to activation source directory.",
        },
        {
            "name": "source_run_id",
            "dtype": "string",
            "required": False,
            "description": "Run provenance tag linking to run metadata.",
        },
    ]

    schema = {
        "name": "activation_oracle_input_manifest",
        "version": "1.0.0",
        "generated_at_utc": utc_now_iso(),
        "row_granularity": "one row per sample/layer/k vector",
        "description": (
            "Template contract for preparing vectorized activation records that can "
            "be consumed by SAE analysis or activation-oracle training pipelines."
        ),
        "columns": columns,
    }

    schema_path = phase2_dir / "oracle_input_schema.json"
    write_json(schema_path, schema)

    template_cols = [c["name"] for c in columns]
    template_df = pd.DataFrame(columns=template_cols)
    template_path = phase2_dir / "oracle_input_manifest_template.csv"
    template_df.to_csv(template_path, index=False)

    return [schema_path, template_path]


def run_analysis(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (input_root / args.analysis_subdir).resolve()
    )

    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    phase2_dir = output_root / "phase2_prep"
    meta_dir = output_root / "meta"
    checkpoints_dir = output_root / "checkpoints"
    progress_path = checkpoints_dir / "progress.json"

    for p in (tables_dir, figures_dir, phase2_dir, meta_dir, checkpoints_dir):
        ensure_dir(p)

    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    progress = load_progress(progress_path)
    completed_steps = set(progress["completed_steps"])

    status_path = meta_dir / "status.json"
    if status_path.exists() and not args.force_rebuild:
        prior_status = read_json(status_path)
        if prior_status.get("state") == "completed":
            print(f"Status already completed at {status_path}; use --force_rebuild to rerun.")
            return 0

    manifest = {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "script": "scripts/analysis/pca_ablation_colab_deep_dive.py",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "poolings_requested": parse_poolings(args.poolings),
        "results_filename": args.results_filename,
        "analysis_subdir": args.analysis_subdir,
        "force_rebuild": bool(args.force_rebuild),
        "strict": bool(args.strict),
        "git_commit": try_get_git_commit(Path(args.repo_root).resolve()),
    }
    write_json(meta_dir / "run_manifest.json", manifest)

    init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_DISCOVER)

    try:
        poolings = parse_poolings(args.poolings)
        if not poolings:
            raise ValueError("No poolings selected.")

        discover_outputs = [meta_dir / "discovered_inputs.json"]
        if args.force_rebuild or STEP_DISCOVER not in completed_steps or not outputs_exist(discover_outputs):
            available_inputs, missing_inputs = discover_inputs(
                input_root=input_root,
                poolings=poolings,
                results_filename=args.results_filename,
                strict=args.strict,
            )
            discovered_payload = {
                "generated_at_utc": utc_now_iso(),
                "input_root": str(input_root),
                "available_poolings": available_inputs,
                "missing_poolings": missing_inputs,
            }
            write_json(discover_outputs[0], discovered_payload)
            mark_step_completed(progress, STEP_DISCOVER, discover_outputs)
            save_progress(progress_path, progress)
            completed_steps.add(STEP_DISCOVER)
        else:
            discovered_payload = read_json(discover_outputs[0])
            available_inputs = discovered_payload.get("available_poolings", {})
            if not available_inputs:
                raise RuntimeError("No available poolings found in discovered_inputs.json")

        init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_BUILD_TABLES)

        tables_outputs = [
            tables_dir / "pca_sweep_long.csv",
            tables_dir / "pca_baseline_by_layer.csv",
            tables_dir / "least_k_ood_gain_per_pooling_layer.csv",
            tables_dir / "no_ood_gain_layers.csv",
            tables_dir / "delta_consistency_summary.csv",
            tables_dir / "rank_by_k_then_ood_gain.csv",
            tables_dir / "rank_by_ood_gain_ascending.csv",
            tables_dir / "rank_by_ood_gain_desc.csv",
            tables_dir / "rank_all_points_by_ood_gain_ascending.csv",
            tables_dir / "rank_all_points_by_ood_gain_desc.csv",
            tables_dir / "ood_gain_summary_by_pooling_k.csv",
        ]
        if args.force_rebuild or STEP_BUILD_TABLES not in completed_steps or not outputs_exist(tables_outputs):
            baseline_frames: List[pd.DataFrame] = []
            long_frames: List[pd.DataFrame] = []

            for pooling, info in sorted(available_inputs.items()):
                results_path = Path(info["results_path"])
                raw = read_json(results_path)
                bdf, ldf = build_pooling_frames(pooling=pooling, results=raw)
                baseline_frames.append(bdf)
                long_frames.append(ldf)

            baseline_df = (
                pd.concat(baseline_frames, ignore_index=True)
                if baseline_frames
                else pd.DataFrame(columns=["pooling", "layer"])
            )
            long_df = (
                pd.concat(long_frames, ignore_index=True)
                if long_frames
                else pd.DataFrame(columns=["pooling", "k", "layer"])
            )

            leastk_df, no_gain_df = extract_least_k_ood_gain(long_df)
            delta_summary_df = validate_delta_consistency(long_df)
            ranking = build_ranking_tables(leastk_df)
            all_point_ranking = build_all_point_rankings(long_df)
            k_summary_df = build_k_aggregate_summary(long_df)

            save_table(
                long_df,
                tables_dir / "pca_sweep_long.csv",
                tables_dir / "pca_sweep_long.parquet",
            )
            save_table(
                baseline_df,
                tables_dir / "pca_baseline_by_layer.csv",
                tables_dir / "pca_baseline_by_layer.parquet",
            )
            save_table(
                leastk_df,
                tables_dir / "least_k_ood_gain_per_pooling_layer.csv",
                tables_dir / "least_k_ood_gain_per_pooling_layer.parquet",
            )
            save_table(no_gain_df, tables_dir / "no_ood_gain_layers.csv", None)
            save_table(delta_summary_df, tables_dir / "delta_consistency_summary.csv", None)
            save_table(ranking["rank_by_k_then_ood_gain"], tables_dir / "rank_by_k_then_ood_gain.csv", None)
            save_table(ranking["rank_by_ood_gain_ascending"], tables_dir / "rank_by_ood_gain_ascending.csv", None)
            save_table(ranking["rank_by_ood_gain_desc"], tables_dir / "rank_by_ood_gain_desc.csv", None)
            save_table(
                all_point_ranking["rank_all_points_by_ood_gain_ascending"],
                tables_dir / "rank_all_points_by_ood_gain_ascending.csv",
                None,
            )
            save_table(
                all_point_ranking["rank_all_points_by_ood_gain_desc"],
                tables_dir / "rank_all_points_by_ood_gain_desc.csv",
                None,
            )
            save_table(k_summary_df, tables_dir / "ood_gain_summary_by_pooling_k.csv", None)

            mark_step_completed(progress, STEP_BUILD_TABLES, tables_outputs)
            save_progress(progress_path, progress)
            completed_steps.add(STEP_BUILD_TABLES)
        else:
            long_df = pd.read_csv(tables_dir / "pca_sweep_long.csv")
            leastk_df = pd.read_csv(tables_dir / "least_k_ood_gain_per_pooling_layer.csv")

        init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_PLOTS)

        plot_outputs = [
            figures_dir / "id_vs_ood_delta_scatter.png",
            figures_dir / "id_vs_ood_delta_pareto.png",
            figures_dir / "best_leastk_ood_gain_by_pooling.png",
        ]
        for pooling in sorted(available_inputs.keys()):
            plot_outputs.append(figures_dir / f"delta_ood_heatmap_{pooling}.png")
            plot_outputs.append(figures_dir / f"delta_id_heatmap_{pooling}.png")

        if args.force_rebuild or STEP_PLOTS not in completed_steps or not outputs_exist(plot_outputs):
            output_paths = []
            output_paths.extend(
                plot_heatmap_by_pooling(
                    df_long=long_df,
                    metric_col="delta_ood_auc_vs_baseline",
                    out_dir=figures_dir,
                    filename_prefix="delta_ood_heatmap",
                    title_prefix="Delta OOD AUC vs baseline",
                )
            )
            output_paths.extend(
                plot_heatmap_by_pooling(
                    df_long=long_df,
                    metric_col="delta_id_auc_vs_baseline",
                    out_dir=figures_dir,
                    filename_prefix="delta_id_heatmap",
                    title_prefix="Delta ID AUC vs baseline",
                )
            )
            scatter = plot_delta_scatter(long_df, figures_dir / "id_vs_ood_delta_scatter.png")
            if scatter:
                output_paths.append(scatter)
            pareto = plot_pareto_front(long_df, figures_dir / "id_vs_ood_delta_pareto.png")
            if pareto:
                output_paths.append(pareto)
            bar = plot_best_ood_gain_by_pooling(leastk_df, figures_dir / "best_leastk_ood_gain_by_pooling.png")
            if bar:
                output_paths.append(bar)

            mark_step_completed(progress, STEP_PLOTS, output_paths)
            save_progress(progress_path, progress)
            completed_steps.add(STEP_PLOTS)

        init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_PHASE2)

        phase2_outputs = [
            phase2_dir / "pca_direction_catalog.csv",
            phase2_dir / "pca_layer_inventory.csv",
            phase2_dir / "pca_artifact_manifest_resolved.json",
            phase2_dir / "oracle_input_schema.json",
            phase2_dir / "oracle_input_manifest_template.csv",
        ]
        if args.force_rebuild or STEP_PHASE2 not in completed_steps or not outputs_exist(phase2_outputs):
            direction_frames: List[pd.DataFrame] = []
            inventory_frames: List[pd.DataFrame] = []
            manifest_rows: List[dict] = []

            for pooling, info in sorted(available_inputs.items()):
                pca_dir = Path(info["pca_artifacts_dir"])
                direction_df, inventory_df, manifest_row = build_pca_direction_catalog(pooling, pca_dir)
                direction_frames.append(direction_df)
                inventory_frames.append(inventory_df)
                manifest_rows.append(manifest_row)

            catalog_df = (
                pd.concat(direction_frames, ignore_index=True)
                if direction_frames and any(not df.empty for df in direction_frames)
                else pd.DataFrame(
                    columns=[
                        "pooling",
                        "layer",
                        "pc_index",
                        "pc_rank",
                        "explained_variance_ratio",
                        "cumulative_explained_variance_ratio",
                        "sign_flip",
                        "n_components_saved",
                        "dim",
                        "n_train_samples",
                        "artifact_file",
                        "artifact_relpath",
                    ]
                )
            )
            inventory_df = (
                pd.concat(inventory_frames, ignore_index=True)
                if inventory_frames and any(not df.empty for df in inventory_frames)
                else pd.DataFrame(
                    columns=[
                        "pooling",
                        "layer",
                        "artifact_file",
                        "artifact_path",
                        "n_components_saved",
                        "dim",
                        "n_train_samples",
                    ]
                )
            )

            save_table(
                catalog_df,
                phase2_dir / "pca_direction_catalog.csv",
                phase2_dir / "pca_direction_catalog.parquet",
            )
            save_table(inventory_df, phase2_dir / "pca_layer_inventory.csv", None)
            write_json(
                phase2_dir / "pca_artifact_manifest_resolved.json",
                {
                    "generated_at_utc": utc_now_iso(),
                    "rows": manifest_rows,
                },
            )
            oracle_paths = write_oracle_interface_templates(phase2_dir)

            outputs_for_step = [
                phase2_dir / "pca_direction_catalog.csv",
                phase2_dir / "pca_layer_inventory.csv",
                phase2_dir / "pca_artifact_manifest_resolved.json",
            ] + oracle_paths
            mark_step_completed(progress, STEP_PHASE2, outputs_for_step)
            save_progress(progress_path, progress)
            completed_steps.add(STEP_PHASE2)

        init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_FINALIZE)

        summary_path = output_root / "results.json"
        summary = {
            "generated_at_utc": utc_now_iso(),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "poolings_analyzed": sorted(list(available_inputs.keys())),
            "counts": {
                "long_rows": int(pd.read_csv(tables_dir / "pca_sweep_long.csv").shape[0]),
                "baseline_rows": int(pd.read_csv(tables_dir / "pca_baseline_by_layer.csv").shape[0]),
                "least_k_rows": int(pd.read_csv(tables_dir / "least_k_ood_gain_per_pooling_layer.csv").shape[0]),
                "direction_catalog_rows": int(pd.read_csv(phase2_dir / "pca_direction_catalog.csv").shape[0]),
            },
            "table_paths": [str(p) for p in sorted(tables_dir.glob("*")) if p.is_file()],
            "figure_paths": [str(p) for p in sorted(figures_dir.glob("*")) if p.is_file()],
            "phase2_paths": [str(p) for p in sorted(phase2_dir.glob("*")) if p.is_file()],
        }
        write_json(summary_path, summary)

        mark_step_completed(progress, STEP_FINALIZE, [summary_path])
        save_progress(progress_path, progress)
        init_status(meta_dir, run_id=run_id, state="completed", current_step=None)

        print("=" * 88)
        print("PCA ablation deep-dive analysis complete.")
        print(f"Input root:  {input_root}")
        print(f"Output root: {output_root}")
        print(f"Summary:     {summary_path}")
        print("=" * 88)
        return 0

    except Exception:
        init_status(meta_dir, run_id=run_id, state="failed", current_step=None)
        write_json(
            meta_dir / "last_error.json",
            {
                "generated_at_utc": utc_now_iso(),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab-first deep-dive PCA ablation analysis")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=(
            "Directory containing pooling subdirs. Example: "
            "/content/drive/MyDrive/.../results/pca_ablation/<model>/<dataset>"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional explicit output root. Defaults to <input_root>/<analysis_subdir>.",
    )
    parser.add_argument(
        "--analysis_subdir",
        type=str,
        default="cross_pooling_analysis_notebook",
        help="Subdirectory under input_root for outputs when output_root is not set.",
    )
    parser.add_argument(
        "--poolings",
        type=str,
        default="mean,max,last",
        help="Comma-separated pooling list to analyze.",
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default="pca_ablation_results.json",
        help="Per-pooling results JSON file name.",
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Repo root used for git commit metadata.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested pooling is missing.",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Recompute all outputs even if status/checkpoints exist.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
