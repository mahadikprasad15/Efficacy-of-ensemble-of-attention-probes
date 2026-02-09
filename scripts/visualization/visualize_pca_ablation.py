#!/usr/bin/env python3
"""
Post-hoc visualization for PCA ablation results.

Reads:
  Single-run mode:
    - pca_ablation_results.json
    - config.json (optional)

  Cross-pooling mode:
    - {compare_poolings_root}/{pooling}/pca_ablation_results.json
    - {compare_poolings_root}/{pooling}/config.json (optional)

Writes:
  Single-run mode:
    - delta_heatmap_id.png
    - delta_heatmap_ood.png
    - id_vs_ood_tradeoff.png
    - layerwise_id_ood_vs_k_grid.png
    - layerwise_best_ood_gain_ranked.png
    - best_k_per_layer_by_ood.png
    - id_drop_vs_ood_gain_pareto.png
    - summary_best_layers.csv
    - top_candidates_by_ood.csv

  Cross-pooling mode:
    - cross_pooling_best_ood_auc_heatmap.png
    - cross_pooling_best_k_by_layer_heatmap.png
    - cross_pooling_k_curves_grid_ood.png
    - cross_pooling_summary.csv
    - cross_pooling_overview.json
"""

import argparse
import csv
import json
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

POOLING_ORDER_DEFAULT = ["mean", "max", "last", "attn"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_poolings(raw: str) -> List[str]:
    if not raw:
        return POOLING_ORDER_DEFAULT.copy()
    poolings = []
    seen = set()
    for token in raw.split(","):
        p = token.strip().lower()
        if not p or p in seen:
            continue
        poolings.append(p)
        seen.add(p)
    return poolings


def load_inputs(input_dir: str) -> Tuple[dict, dict]:
    results_path = os.path.join(input_dir, "pca_ablation_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing required file: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    config_path = os.path.join(input_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    return results, config


def normalize_result_maps(results: dict) -> Tuple[Dict[int, dict], Dict[int, Dict[int, dict]]]:
    baseline_raw = results.get("baseline", {})
    sweep_raw = results.get("sweep", {})

    baseline = {int(layer): metrics for layer, metrics in baseline_raw.items()}

    sweep: Dict[int, Dict[int, dict]] = {}
    for k, layer_map in sweep_raw.items():
        k_int = int(k)
        sweep[k_int] = {int(layer): metrics for layer, metrics in layer_map.items()}

    return baseline, sweep


def build_axes(baseline: Dict[int, dict], sweep: Dict[int, Dict[int, dict]]) -> Tuple[List[int], List[int]]:
    layers = sorted(
        set(baseline.keys()).union(*[set(layer_map.keys()) for layer_map in sweep.values()])
        if sweep
        else set(baseline.keys())
    )
    k_values = sorted(sweep.keys())
    return layers, k_values


def build_metric_matrices(
    baseline: Dict[int, dict],
    sweep: Dict[int, Dict[int, dict]],
    layers: List[int],
    k_values: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    id_auc = np.full((len(k_values), len(layers)), np.nan, dtype=np.float32)
    ood_auc = np.full((len(k_values), len(layers)), np.nan, dtype=np.float32)
    id_delta = np.full((len(k_values), len(layers)), np.nan, dtype=np.float32)
    ood_delta = np.full((len(k_values), len(layers)), np.nan, dtype=np.float32)
    base_id = np.full((len(layers),), np.nan, dtype=np.float32)
    base_ood = np.full((len(layers),), np.nan, dtype=np.float32)

    for j, layer in enumerate(layers):
        if layer in baseline:
            base_id[j] = float(baseline[layer].get("id_val_auc", np.nan))
            base_ood[j] = float(baseline[layer].get("ood_test_auc", np.nan))

    for i, k in enumerate(k_values):
        layer_map = sweep.get(k, {})
        for j, layer in enumerate(layers):
            if layer not in layer_map:
                continue
            m = layer_map[layer]
            id_auc[i, j] = float(m.get("id_val_auc", np.nan))
            ood_auc[i, j] = float(m.get("ood_test_auc", np.nan))
            id_delta[i, j] = float(m.get("delta_id_auc_vs_baseline", np.nan))
            ood_delta[i, j] = float(m.get("delta_ood_auc_vs_baseline", np.nan))

    return id_auc, ood_auc, id_delta, ood_delta, base_id, base_ood


def safe_nanmean(arr: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            return np.nanmean(arr, axis=axis)


def plot_delta_heatmap(
    matrix: np.ndarray,
    layers: List[int],
    k_values: List[int],
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(max(8, len(layers) * 0.35), max(5, len(k_values) * 0.35)))
    finite = np.isfinite(matrix)
    vmax = float(np.nanmax(np.abs(matrix[finite]))) if finite.any() else 0.01
    vmax = max(vmax, 1e-3)

    draw = np.where(np.isfinite(matrix), matrix, 0.0)
    plt.imshow(draw, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="Delta AUC vs baseline")
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.yticks(range(len(k_values)), k_values)
    plt.xlabel("Layer")
    plt.ylabel("k removed PCs")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_tradeoff(
    k_values: List[int],
    mean_id_auc: np.ndarray,
    mean_ood_auc: np.ndarray,
    baseline_id_mean: float,
    baseline_ood_mean: float,
    out_path: str,
    title_prefix: str = "",
) -> None:
    title = "ID vs OOD Tradeoff Across PCA Direction Removal"
    if title_prefix:
        title = f"{title_prefix} | {title}"

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, mean_id_auc, marker="o", linewidth=2.0, label="Mean ID Val AUC")
    plt.plot(k_values, mean_ood_auc, marker="s", linewidth=2.0, label="Mean OOD Test AUC")
    plt.axhline(baseline_id_mean, linestyle="--", alpha=0.5, color="#1f77b4", label="Baseline ID mean")
    plt.axhline(baseline_ood_mean, linestyle="--", alpha=0.5, color="#ff7f0e", label="Baseline OOD mean")
    plt.xlabel("k removed PCs")
    plt.ylabel("AUC")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_layerwise_grid(
    layers: List[int],
    k_values: List[int],
    id_auc: np.ndarray,
    ood_auc: np.ndarray,
    base_id: np.ndarray,
    base_ood: np.ndarray,
    out_path: str,
    grid_cols: int,
    title_prefix: str = "",
) -> None:
    n_layers = len(layers)
    cols = max(1, grid_cols)
    rows = int(math.ceil(n_layers / cols))

    y_vals = []
    y_vals.extend(id_auc[np.isfinite(id_auc)].tolist())
    y_vals.extend(ood_auc[np.isfinite(ood_auc)].tolist())
    y_vals.extend(base_id[np.isfinite(base_id)].tolist())
    y_vals.extend(base_ood[np.isfinite(base_ood)].tolist())
    y_min = max(0.0, min(y_vals) - 0.02) if y_vals else 0.0
    y_max = min(1.0, max(y_vals) + 0.02) if y_vals else 1.0

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.8), sharex=True, sharey=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, layer in enumerate(layers):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        ax.plot(k_values, id_auc[:, idx], marker="o", color="#1f77b4", linewidth=1.8, label="ID")
        ax.plot(k_values, ood_auc[:, idx], marker="s", color="#ff7f0e", linewidth=1.8, label="OOD")
        if np.isfinite(base_id[idx]):
            ax.axhline(base_id[idx], linestyle="--", color="#1f77b4", alpha=0.45, linewidth=1.2)
        if np.isfinite(base_ood[idx]):
            ax.axhline(base_ood[idx], linestyle="--", color="#ff7f0e", alpha=0.45, linewidth=1.2)
        ax.set_title(f"Layer {layer}", fontsize=9, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25)

    # Hide empty panels
    for idx in range(n_layers, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    for c in range(cols):
        axes[rows - 1, c].set_xlabel("k")
    for r in range(rows):
        axes[r, 0].set_ylabel("AUC")

    title = "Layerwise ID vs OOD Across PCA Removal (Dashed=Baseline)"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    fig.suptitle(title, y=1.01, fontsize=14, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_ranked_best_ood_gain(
    layers: List[int],
    k_values: List[int],
    ood_delta: np.ndarray,
    out_path: str,
    title_prefix: str = "",
) -> None:
    best_gain = []
    best_k = []
    for j in range(len(layers)):
        col = ood_delta[:, j]
        if not np.isfinite(col).any():
            best_gain.append(np.nan)
            best_k.append(np.nan)
            continue
        idx = int(np.nanargmax(col))
        best_gain.append(float(col[idx]))
        best_k.append(k_values[idx])

    records = [
        (layers[j], best_gain[j], best_k[j])
        for j in range(len(layers))
        if np.isfinite(best_gain[j])
    ]
    records.sort(key=lambda x: x[1], reverse=True)
    if not records:
        return

    names = [f"L{r[0]}" for r in records]
    vals = [r[1] for r in records]
    ks = [int(r[2]) for r in records]

    plt.figure(figsize=(max(8, len(records) * 0.6), 5.5))
    bars = plt.bar(range(len(records)), vals, color=["#2ca02c" if v >= 0 else "#d62728" for v in vals], alpha=0.85)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(range(len(records)), names, rotation=45)
    plt.ylabel("Best OOD Delta AUC")
    plt.xlabel("Layer (sorted by best OOD gain)")
    title = "Layerwise Best OOD Gain vs Baseline"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    plt.title(title)
    plt.grid(alpha=0.2, axis="y")

    for b, k in zip(bars, ks):
        h = b.get_height()
        y = h + 0.005 if h >= 0 else h - 0.02
        plt.text(b.get_x() + b.get_width() / 2, y, f"k={k}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_best_k_map_by_layer(
    layers: List[int],
    k_values: List[int],
    ood_auc: np.ndarray,
    out_path: str,
    title_prefix: str = "",
) -> None:
    best_k = np.full((len(layers),), np.nan, dtype=np.float32)
    for j in range(len(layers)):
        col = ood_auc[:, j]
        if not np.isfinite(col).any():
            continue
        idx = int(np.nanargmax(col))
        best_k[j] = float(k_values[idx])

    draw = best_k.reshape(1, -1)
    finite = np.isfinite(draw)
    vmin = float(np.nanmin(draw[finite])) if finite.any() else min(k_values)
    vmax = float(np.nanmax(draw[finite])) if finite.any() else max(k_values)
    if vmin == vmax:
        vmax = vmin + 1.0

    plt.figure(figsize=(max(8, len(layers) * 0.5), 2.8))
    img = np.where(np.isfinite(draw), draw, vmin)
    plt.imshow(img, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label("Best k by OOD AUC")
    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.yticks([0], ["best-k"])
    title = "Best k per Layer (Selected by OOD AUC)"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    plt.title(title)
    plt.xlabel("Layer")
    for j in range(len(layers)):
        txt = "NA" if not np.isfinite(best_k[j]) else str(int(best_k[j]))
        plt.text(j, 0, txt, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_pareto(
    layers: List[int],
    k_values: List[int],
    id_delta: np.ndarray,
    ood_delta: np.ndarray,
    out_path: str,
    title_prefix: str = "",
) -> None:
    points = []
    for i, k in enumerate(k_values):
        for j, layer in enumerate(layers):
            x = float(id_delta[i, j])
            y = float(ood_delta[i, j])
            if np.isfinite(x) and np.isfinite(y):
                points.append((layer, k, x, y))
    if not points:
        return

    # non-dominated frontier for maximizing both x and y
    frontier = []
    for a in points:
        dominated = False
        for b in points:
            if (b[2] >= a[2] and b[3] >= a[3]) and (b[2] > a[2] or b[3] > a[3]):
                dominated = True
                break
        if not dominated:
            frontier.append(a)

    plt.figure(figsize=(8.5, 6.5))
    sc = plt.scatter(
        [p[2] for p in points],
        [p[3] for p in points],
        c=[p[1] for p in points],
        cmap="plasma",
        alpha=0.75,
        s=28,
        edgecolors="none",
    )
    plt.colorbar(sc, label="k removed PCs")
    if frontier:
        plt.scatter(
            [p[2] for p in frontier],
            [p[3] for p in frontier],
            s=64,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Pareto frontier",
        )
    plt.axvline(0.0, linestyle="--", color="gray", alpha=0.6)
    plt.axhline(0.0, linestyle="--", color="gray", alpha=0.6)
    plt.xlabel("ID Delta AUC vs Baseline")
    plt.ylabel("OOD Delta AUC vs Baseline")
    title = "ID Drop vs OOD Gain Pareto View"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    plt.title(title)
    plt.grid(alpha=0.2)
    if frontier:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def write_summary_csv(
    layers: List[int],
    k_values: List[int],
    id_auc: np.ndarray,
    ood_auc: np.ndarray,
    out_path: str,
) -> None:
    rows = []
    for i, k in enumerate(k_values):
        id_row = id_auc[i, :]
        ood_row = ood_auc[i, :]

        valid_id = np.isfinite(id_row)
        valid_ood = np.isfinite(ood_row)

        best_layer_id = int(layers[int(np.nanargmax(id_row))]) if valid_id.any() else -1
        best_layer_ood = int(layers[int(np.nanargmax(ood_row))]) if valid_ood.any() else -1
        best_id_val = float(np.nanmax(id_row)) if valid_id.any() else np.nan
        best_ood_val = float(np.nanmax(ood_row)) if valid_ood.any() else np.nan

        ood_at_best_id = float(id_row[0] * np.nan)  # nan sentinel
        id_at_best_ood = float(id_row[0] * np.nan)  # nan sentinel
        if best_layer_id >= 0:
            j = layers.index(best_layer_id)
            ood_at_best_id = float(ood_row[j]) if np.isfinite(ood_row[j]) else np.nan
        if best_layer_ood >= 0:
            j = layers.index(best_layer_ood)
            id_at_best_ood = float(id_row[j]) if np.isfinite(id_row[j]) else np.nan

        rows.append(
            {
                "k": int(k),
                "best_layer_id": best_layer_id,
                "best_id_val_auc": best_id_val,
                "ood_at_best_id_layer": ood_at_best_id,
                "best_layer_ood": best_layer_ood,
                "best_ood_test_auc": best_ood_val,
                "id_at_best_ood_layer": id_at_best_ood,
                "mean_id_val_auc": float(np.nanmean(id_row)) if valid_id.any() else np.nan,
                "mean_ood_test_auc": float(np.nanmean(ood_row)) if valid_ood.any() else np.nan,
            }
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["k"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def candidate_score(main_metric: str, id_auc: float, ood_auc: float, id_delta: float, ood_delta: float) -> float:
    if main_metric == "ood_auc":
        return ood_auc
    if main_metric == "ood_delta":
        return ood_delta
    if main_metric == "harmonic":
        denom = id_auc + ood_auc
        return (2 * id_auc * ood_auc / denom) if denom > 0 else np.nan
    raise ValueError(f"Unknown main_metric: {main_metric}")


def write_top_candidates_csv(
    layers: List[int],
    k_values: List[int],
    id_auc: np.ndarray,
    ood_auc: np.ndarray,
    id_delta: np.ndarray,
    ood_delta: np.ndarray,
    out_path: str,
    top_n: int,
    main_metric: str,
) -> None:
    rows = []
    for i, k in enumerate(k_values):
        for j, layer in enumerate(layers):
            if not (np.isfinite(id_auc[i, j]) and np.isfinite(ood_auc[i, j])):
                continue
            ida = float(id_auc[i, j])
            ooda = float(ood_auc[i, j])
            idd = float(id_delta[i, j]) if np.isfinite(id_delta[i, j]) else np.nan
            oodd = float(ood_delta[i, j]) if np.isfinite(ood_delta[i, j]) else np.nan
            gap = ida - ooda
            score = candidate_score(main_metric, ida, ooda, idd, oodd)
            rows.append(
                {
                    "layer": int(layer),
                    "k": int(k),
                    "id_auc": ida,
                    "ood_auc": ooda,
                    "id_delta": idd,
                    "ood_delta": oodd,
                    "gap": gap,
                    "metric_score": score,
                }
            )

    rows.sort(key=lambda r: (r["metric_score"], r["id_delta"]), reverse=True)
    rows = rows[:top_n] if top_n > 0 else rows

    with open(out_path, "w", newline="") as f:
        fieldnames = [
            "layer",
            "k",
            "id_auc",
            "ood_auc",
            "id_delta",
            "ood_delta",
            "gap",
            "metric_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pooling_bundle(compare_root: str, pooling: str) -> Tuple[Optional[dict], Optional[str]]:
    input_dir = os.path.join(compare_root, pooling)
    if not os.path.isdir(input_dir):
        return None, f"missing directory: {input_dir}"

    try:
        results, config = load_inputs(input_dir)
    except FileNotFoundError as e:
        return None, str(e)

    baseline, sweep = normalize_result_maps(results)
    layers, k_values = build_axes(baseline, sweep)
    if not layers or not k_values:
        return None, f"no layers/k-values found in {input_dir}"

    id_auc, ood_auc, id_delta, ood_delta, base_id, base_ood = build_metric_matrices(
        baseline, sweep, layers, k_values
    )

    config_pooling = str(config.get("pooling", "")).strip().lower() if config else ""
    if config_pooling and config_pooling != pooling:
        print(
            f"[WARN] Pooling mismatch in {input_dir}: config pooling={config_pooling}, expected={pooling}. "
            "Continuing with directory name."
        )

    return {
        "pooling": pooling,
        "input_dir": input_dir,
        "baseline": baseline,
        "sweep": sweep,
        "layers": layers,
        "k_values": k_values,
        "id_auc": id_auc,
        "ood_auc": ood_auc,
        "id_delta": id_delta,
        "ood_delta": ood_delta,
        "base_id": base_id,
        "base_ood": base_ood,
    }, None


def align_pooling_bundles(
    pooling_bundles: Dict[str, dict]
) -> Tuple[List[int], List[int], Dict[str, dict]]:
    all_layers = sorted({layer for b in pooling_bundles.values() for layer in b["layers"]})
    all_k_values = sorted({k for b in pooling_bundles.values() for k in b["k_values"]})

    aligned: Dict[str, dict] = {}
    for pooling, bundle in pooling_bundles.items():
        layer_to_local = {layer: j for j, layer in enumerate(bundle["layers"])}
        k_to_local = {k: i for i, k in enumerate(bundle["k_values"])}

        shape = (len(all_k_values), len(all_layers))
        id_auc_aligned = np.full(shape, np.nan, dtype=np.float32)
        ood_auc_aligned = np.full(shape, np.nan, dtype=np.float32)
        id_delta_aligned = np.full(shape, np.nan, dtype=np.float32)
        ood_delta_aligned = np.full(shape, np.nan, dtype=np.float32)
        base_id_aligned = np.full((len(all_layers),), np.nan, dtype=np.float32)
        base_ood_aligned = np.full((len(all_layers),), np.nan, dtype=np.float32)

        for gi, k in enumerate(all_k_values):
            li = k_to_local.get(k)
            if li is None:
                continue
            for gj, layer in enumerate(all_layers):
                lj = layer_to_local.get(layer)
                if lj is None:
                    continue
                id_auc_aligned[gi, gj] = bundle["id_auc"][li, lj]
                ood_auc_aligned[gi, gj] = bundle["ood_auc"][li, lj]
                id_delta_aligned[gi, gj] = bundle["id_delta"][li, lj]
                ood_delta_aligned[gi, gj] = bundle["ood_delta"][li, lj]

        for gj, layer in enumerate(all_layers):
            lj = layer_to_local.get(layer)
            if lj is None:
                continue
            base_id_aligned[gj] = bundle["base_id"][lj]
            base_ood_aligned[gj] = bundle["base_ood"][lj]

        merged = dict(bundle)
        merged["id_auc_aligned"] = id_auc_aligned
        merged["ood_auc_aligned"] = ood_auc_aligned
        merged["id_delta_aligned"] = id_delta_aligned
        merged["ood_delta_aligned"] = ood_delta_aligned
        merged["base_id_aligned"] = base_id_aligned
        merged["base_ood_aligned"] = base_ood_aligned
        aligned[pooling] = merged

    return all_layers, all_k_values, aligned


def plot_cross_pooling_best_ood_auc_heatmap(
    poolings: List[str],
    layers: List[int],
    best_ood_auc: np.ndarray,
    out_path: str,
    title_prefix: str = "",
) -> None:
    plt.figure(figsize=(max(9, len(layers) * 0.35), max(4, len(poolings) * 0.9)))
    finite = np.isfinite(best_ood_auc)
    if finite.any():
        vmin = float(np.nanmin(best_ood_auc[finite]))
        vmax = float(np.nanmax(best_ood_auc[finite]))
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-3

    masked = np.ma.masked_invalid(best_ood_auc)
    im = plt.imshow(masked, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label("Best OOD AUC (max over k)")

    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.yticks(range(len(poolings)), [p.upper() for p in poolings])
    plt.xlabel("Layer")
    plt.ylabel("Pooling")
    title = "Cross-Pooling Best OOD AUC by Layer"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    plt.title(title)

    if len(layers) <= 36:
        for i in range(len(poolings)):
            for j in range(len(layers)):
                val = best_ood_auc[i, j]
                txt = "NA" if not np.isfinite(val) else f"{val:.3f}"
                color = "black" if not np.isfinite(val) or val < (vmin + vmax) / 2 else "white"
                plt.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cross_pooling_best_k_heatmap(
    poolings: List[str],
    layers: List[int],
    best_k_map: np.ndarray,
    all_k_values: List[int],
    out_path: str,
    title_prefix: str = "",
) -> None:
    plt.figure(figsize=(max(9, len(layers) * 0.35), max(4, len(poolings) * 0.9)))

    finite = np.isfinite(best_k_map)
    if finite.any():
        vmin = float(np.nanmin(best_k_map[finite]))
        vmax = float(np.nanmax(best_k_map[finite]))
    else:
        vmin = float(min(all_k_values)) if all_k_values else 0.0
        vmax = float(max(all_k_values)) if all_k_values else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    masked = np.ma.masked_invalid(best_k_map)
    im = plt.imshow(masked, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label("Best k (by OOD AUC)")

    plt.xticks(range(len(layers)), layers, rotation=45)
    plt.yticks(range(len(poolings)), [p.upper() for p in poolings])
    plt.xlabel("Layer")
    plt.ylabel("Pooling")
    title = "Cross-Pooling Best k by Layer"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    plt.title(title)

    if len(layers) <= 36:
        for i in range(len(poolings)):
            for j in range(len(layers)):
                val = best_k_map[i, j]
                txt = "NA" if not np.isfinite(val) else str(int(val))
                color = "white" if np.isfinite(val) else "black"
                plt.text(j, i, txt, ha="center", va="center", fontsize=7, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cross_pooling_k_curves_grid(
    poolings: List[str],
    layers: List[int],
    all_k_values: List[int],
    aligned: Dict[str, dict],
    out_path: str,
    title_prefix: str = "",
) -> None:
    n = len(poolings)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.0, rows * 4.6), sharex=True, sharey=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, pooling in enumerate(poolings):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        ood_mat = aligned[pooling]["ood_auc_aligned"]
        for j in range(len(layers)):
            y = ood_mat[:, j]
            if np.isfinite(y).any():
                ax.plot(all_k_values, y, color="#9aa5b1", alpha=0.35, linewidth=0.9)

        mean_curve = safe_nanmean(ood_mat, axis=1)
        if np.isfinite(mean_curve).any():
            ax.plot(all_k_values, mean_curve, color="#d62728", linewidth=2.2, label="Mean over layers")
            best_idx = int(np.nanargmax(mean_curve))
            ax.scatter(
                [all_k_values[best_idx]],
                [mean_curve[best_idx]],
                marker="*",
                s=130,
                color="#d62728",
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
                label=f"Best mean-k={all_k_values[best_idx]}",
            )

        baseline_mean = float(safe_nanmean(aligned[pooling]["base_ood_aligned"]))
        if np.isfinite(baseline_mean):
            ax.axhline(baseline_mean, linestyle="--", color="#1f77b4", alpha=0.65, linewidth=1.4, label="Baseline mean")

        ax.set_title(pooling.upper(), fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    for c in range(cols):
        axes[rows - 1, c].set_xlabel("k removed PCs")
    for r in range(rows):
        axes[r, 0].set_ylabel("OOD Test AUC")

    title = "Cross-Pooling k Behavior (OOD AUC vs k)"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    fig.suptitle(title, y=1.01, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def write_cross_pooling_summary_csv(
    poolings: List[str],
    layers: List[int],
    all_k_values: List[int],
    aligned: Dict[str, dict],
    out_path: str,
) -> None:
    rows = []
    for pooling in poolings:
        ood_mat = aligned[pooling]["ood_auc_aligned"]
        id_mat = aligned[pooling]["id_auc_aligned"]
        baseline_ood = aligned[pooling]["base_ood_aligned"]

        for j, layer in enumerate(layers):
            col = ood_mat[:, j]
            best_k = np.nan
            best_ood = np.nan
            best_id = np.nan
            gap = np.nan
            delta = np.nan

            if np.isfinite(col).any():
                i = int(np.nanargmax(col))
                best_k = float(all_k_values[i])
                best_ood = float(col[i])
                best_id = float(id_mat[i, j]) if np.isfinite(id_mat[i, j]) else np.nan
                if np.isfinite(best_id):
                    gap = best_id - best_ood
                if np.isfinite(baseline_ood[j]):
                    delta = best_ood - float(baseline_ood[j])

            rows.append(
                {
                    "pooling": pooling,
                    "layer": int(layer),
                    "best_k": best_k,
                    "best_ood_auc": best_ood,
                    "baseline_ood_auc": float(baseline_ood[j]) if np.isfinite(baseline_ood[j]) else np.nan,
                    "delta_ood_vs_baseline_at_best_k": delta,
                    "best_id_auc_at_best_k": best_id,
                    "generalization_gap_at_best_k": gap,
                }
            )

    with open(out_path, "w", newline="") as f:
        fieldnames = [
            "pooling",
            "layer",
            "best_k",
            "best_ood_auc",
            "baseline_ood_auc",
            "delta_ood_vs_baseline_at_best_k",
            "best_id_auc_at_best_k",
            "generalization_gap_at_best_k",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cross_pooling_overview_json(
    poolings: List[str],
    layers: List[int],
    all_k_values: List[int],
    aligned: Dict[str, dict],
    skipped: Dict[str, str],
    out_path: str,
) -> None:
    overview = {
        "valid_poolings": poolings,
        "skipped_poolings": [{"pooling": k, "reason": v} for k, v in skipped.items()],
        "global_layers": layers,
        "global_k_values": all_k_values,
        "per_pooling": {},
    }

    for pooling in poolings:
        ood_mat = aligned[pooling]["ood_auc_aligned"]
        mean_curve = safe_nanmean(ood_mat, axis=1)
        valid_mean = np.isfinite(mean_curve)
        if valid_mean.any():
            best_idx = int(np.nanargmax(mean_curve))
            best_mean_k = int(all_k_values[best_idx])
            best_mean_ood_auc = float(mean_curve[best_idx])
        else:
            best_mean_k = -1
            best_mean_ood_auc = float("nan")

        missing_cells = int(np.isnan(ood_mat).sum())
        total_cells = int(np.prod(ood_mat.shape))

        overview["per_pooling"][pooling] = {
            "input_dir": aligned[pooling]["input_dir"],
            "available_layers": aligned[pooling]["layers"],
            "available_k_values": aligned[pooling]["k_values"],
            "best_mean_k": best_mean_k,
            "best_mean_ood_auc": best_mean_ood_auc,
            "baseline_mean_ood_auc": float(safe_nanmean(aligned[pooling]["base_ood_aligned"])),
            "missing_cells": missing_cells,
            "total_cells": total_cells,
        }

    with open(out_path, "w") as f:
        json.dump(overview, f, indent=2)


def run_single_mode(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or args.input_dir
    ensure_dir(output_dir)

    results, _config = load_inputs(args.input_dir)
    baseline, sweep = normalize_result_maps(results)
    layers, k_values = build_axes(baseline, sweep)
    if not layers or not k_values:
        raise ValueError("No layers/k-values found in results.")

    id_auc, ood_auc, id_delta, ood_delta, base_id, base_ood = build_metric_matrices(
        baseline, sweep, layers, k_values
    )
    baseline_mean_id = float(np.nanmean(base_id))
    baseline_mean_ood = float(np.nanmean(base_ood))
    mean_id_auc = safe_nanmean(id_auc, axis=1)
    mean_ood_auc = safe_nanmean(ood_auc, axis=1)

    plot_delta_heatmap(
        id_delta,
        layers,
        k_values,
        "ID Validation AUC Delta vs Baseline",
        os.path.join(output_dir, "delta_heatmap_id.png"),
    )
    plot_delta_heatmap(
        ood_delta,
        layers,
        k_values,
        "OOD Test AUC Delta vs Baseline",
        os.path.join(output_dir, "delta_heatmap_ood.png"),
    )
    plot_tradeoff(
        k_values,
        mean_id_auc,
        mean_ood_auc,
        baseline_mean_id,
        baseline_mean_ood,
        os.path.join(output_dir, "id_vs_ood_tradeoff.png"),
        title_prefix=args.title_prefix,
    )

    plot_layerwise_grid(
        layers,
        k_values,
        id_auc,
        ood_auc,
        base_id,
        base_ood,
        os.path.join(output_dir, "layerwise_id_ood_vs_k_grid.png"),
        grid_cols=args.grid_cols,
        title_prefix=args.title_prefix,
    )
    plot_ranked_best_ood_gain(
        layers,
        k_values,
        ood_delta,
        os.path.join(output_dir, "layerwise_best_ood_gain_ranked.png"),
        title_prefix=args.title_prefix,
    )
    plot_best_k_map_by_layer(
        layers,
        k_values,
        ood_auc,
        os.path.join(output_dir, "best_k_per_layer_by_ood.png"),
        title_prefix=args.title_prefix,
    )
    plot_pareto(
        layers,
        k_values,
        id_delta,
        ood_delta,
        os.path.join(output_dir, "id_drop_vs_ood_gain_pareto.png"),
        title_prefix=args.title_prefix,
    )

    write_summary_csv(
        layers,
        k_values,
        id_auc,
        ood_auc,
        os.path.join(output_dir, "summary_best_layers.csv"),
    )
    write_top_candidates_csv(
        layers,
        k_values,
        id_auc,
        ood_auc,
        id_delta,
        ood_delta,
        os.path.join(output_dir, "top_candidates_by_ood.csv"),
        top_n=args.top_n_candidates,
        main_metric=args.main_metric,
    )

    print("=" * 90)
    print("PCA ABLATION VISUALIZATION COMPLETE (single-run mode)")
    print("=" * 90)
    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Layers: {layers}")
    print(f"k values: {k_values}")
    return 0


def run_cross_pooling_mode(args: argparse.Namespace) -> int:
    if args.primary_metric != "ood_auc":
        raise ValueError(f"Unsupported --primary_metric={args.primary_metric}. Only 'ood_auc' is supported.")

    poolings = parse_poolings(args.poolings)
    if not poolings:
        raise ValueError("No poolings resolved from --poolings")

    output_dir = args.output_dir or args.compare_poolings_root
    ensure_dir(output_dir)

    loaded: Dict[str, dict] = {}
    skipped: Dict[str, str] = {}
    for pooling in poolings:
        bundle, reason = load_pooling_bundle(args.compare_poolings_root, pooling)
        if bundle is None:
            skipped[pooling] = reason or "unknown error"
            print(f"[WARN] Skipping pooling '{pooling}': {skipped[pooling]}")
            continue
        loaded[pooling] = bundle

    valid_poolings = [p for p in poolings if p in loaded]
    if not valid_poolings:
        raise ValueError("No valid pooling bundles found under --compare_poolings_root")

    all_layers, all_k_values, aligned = align_pooling_bundles(loaded)
    if not all_layers or not all_k_values:
        raise ValueError("No global layers/k-values found after loading pooling bundles.")

    best_ood_auc = np.full((len(valid_poolings), len(all_layers)), np.nan, dtype=np.float32)
    best_k_map = np.full((len(valid_poolings), len(all_layers)), np.nan, dtype=np.float32)

    for i, pooling in enumerate(valid_poolings):
        ood_mat = aligned[pooling]["ood_auc_aligned"]
        for j in range(len(all_layers)):
            col = ood_mat[:, j]
            if not np.isfinite(col).any():
                continue
            best_idx = int(np.nanargmax(col))
            best_ood_auc[i, j] = float(col[best_idx])
            best_k_map[i, j] = float(all_k_values[best_idx])

    plot_cross_pooling_best_ood_auc_heatmap(
        valid_poolings,
        all_layers,
        best_ood_auc,
        os.path.join(output_dir, "cross_pooling_best_ood_auc_heatmap.png"),
        title_prefix=args.title_prefix,
    )
    plot_cross_pooling_best_k_heatmap(
        valid_poolings,
        all_layers,
        best_k_map,
        all_k_values,
        os.path.join(output_dir, "cross_pooling_best_k_by_layer_heatmap.png"),
        title_prefix=args.title_prefix,
    )
    plot_cross_pooling_k_curves_grid(
        valid_poolings,
        all_layers,
        all_k_values,
        aligned,
        os.path.join(output_dir, "cross_pooling_k_curves_grid_ood.png"),
        title_prefix=args.title_prefix,
    )

    write_cross_pooling_summary_csv(
        valid_poolings,
        all_layers,
        all_k_values,
        aligned,
        os.path.join(output_dir, "cross_pooling_summary.csv"),
    )
    write_cross_pooling_overview_json(
        valid_poolings,
        all_layers,
        all_k_values,
        aligned,
        skipped,
        os.path.join(output_dir, "cross_pooling_overview.json"),
    )

    print("=" * 90)
    print("PCA ABLATION VISUALIZATION COMPLETE (cross-pooling mode)")
    print("=" * 90)
    print(f"Compare root: {args.compare_poolings_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Valid poolings: {valid_poolings}")
    print(f"Skipped poolings: {skipped}")
    print(f"Global layers: {all_layers}")
    print(f"Global k values: {all_k_values}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize PCA ablation result bundle")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Single-run mode: directory with pca_ablation_results.json",
    )
    input_group.add_argument(
        "--compare_poolings_root",
        type=str,
        default=None,
        help="Cross-pooling mode: root containing {pooling}/pca_ablation_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: input_dir or compare_poolings_root)",
    )
    parser.add_argument("--title_prefix", type=str, default="", help="Optional title prefix")
    parser.add_argument("--grid_cols", type=int, default=5, help="Columns for layer grid")
    parser.add_argument("--top_n_candidates", type=int, default=20, help="Rows in top-candidates CSV")
    parser.add_argument(
        "--poolings",
        type=str,
        default="mean,max,last,attn",
        help="Cross-pooling mode: comma-separated pooling order",
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="ood_auc",
        choices=["ood_auc"],
        help="Cross-pooling mode: primary metric for best-k/heatmaps",
    )
    parser.add_argument(
        "--main_metric",
        type=str,
        default="ood_auc",
        choices=["ood_auc", "ood_delta", "harmonic"],
        help="Primary metric for candidate ranking",
    )
    args = parser.parse_args()

    if args.input_dir:
        return run_single_mode(args)
    return run_cross_pooling_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
