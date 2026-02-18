#!/usr/bin/env python3
"""Analyze per-probe PCA consensus overlap and PC reuse patterns.

Inputs:
- consensus_summary.csv from train_pca_consensus_probes.py
- per-probe selections: data/probes_pca_consensus/<model>/<dataset>/<pooling>/K_<K>/layer_<L>/selections/selected_r*.npy

Outputs (in output_dir):
- overlap_by_layer_k.csv
- overlap_by_k.csv
- overlap_by_layer.csv
- pc_reuse_frequency_all.csv
- pc_reuse_frequency_topN.csv
- overlap_heatmap_layer_k.png
- overlap_vs_k.png
- overlap_vs_layer.png
- pc_reuse_topN.png
- consensus_ratio_vs_ood.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRID_ALPHA = 0.2
OOD_COLOR = "#c26d3f"


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "#f6f5f2",
            "axes.facecolor": "#f6f5f2",
            "axes.edgecolor": "#9ea3a8",
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "-",
            "savefig.facecolor": "#f6f5f2",
        }
    )


def load_consensus_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "K",
        "layer",
        "threshold",
        "ood_test_auc",
        "id_val_auc",
        "num_consensus_pcs",
        "consensus_pc_indices",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    return df


def parse_pc_indices(raw: object) -> List[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        vals = json.loads(text)
    except Exception:
        return []
    if not isinstance(vals, (list, tuple)):
        return []
    out: List[int] = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def load_probe_selections(
    probes_root: Path, model_dir: str, dataset: str, pooling: str, k: int, layer: int
) -> List[np.ndarray]:
    sel_dir = probes_root / model_dir / dataset / pooling / f"K_{k}" / f"layer_{layer}" / "selections"
    if not sel_dir.exists():
        return []
    masks: List[np.ndarray] = []
    for path in sorted(sel_dir.glob("selected_r*.npy")):
        try:
            idx = np.load(path)
        except Exception:
            continue
        idx = np.asarray(idx).reshape(-1)
        if idx.size == 0:
            mask = np.zeros(k, dtype=np.int8)
        else:
            mask = np.zeros(k, dtype=np.int8)
            valid = idx[(idx >= 0) & (idx < k)].astype(np.int64)
            mask[valid] = 1
        masks.append(mask)
    return masks


def pairwise_jaccard(masks: List[np.ndarray]) -> Tuple[float, float, int]:
    if len(masks) < 2:
        return float("nan"), float("nan"), 0
    overlaps: List[float] = []
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            a = masks[i].astype(bool)
            b = masks[j].astype(bool)
            union = np.logical_or(a, b).sum()
            if union == 0:
                overlaps.append(1.0)
            else:
                inter = np.logical_and(a, b).sum()
                overlaps.append(float(inter) / float(union))
    if not overlaps:
        return float("nan"), float("nan"), 0
    return float(np.mean(overlaps)), float(np.std(overlaps)), len(overlaps)


def extract_pc_counts(rows: Iterable[object]) -> Dict[int, int]:
    pc_counts: Dict[int, int] = {}
    for raw in rows:
        for pc in parse_pc_indices(raw):
            pc_counts[pc] = pc_counts.get(pc, 0) + 1
    return pc_counts


def plot_overlap_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    layers = sorted(df["layer"].unique())
    ks = sorted(df["K"].unique())
    mat = np.full((len(layers), len(ks)), np.nan, dtype=float)
    for _, r in df.iterrows():
        li = layers.index(int(r["layer"]))
        ki = ks.index(int(r["K"]))
        mat[li, ki] = float(r["mean_jaccard"])

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    vmax = np.nanmax(mat) if not np.isnan(mat).all() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.6, vmax))
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("K")
    ax.set_ylabel("Layer")
    ax.set_title("Mean Probe Jaccard Overlap by Layer/K")
    for i in range(len(layers)):
        for j in range(len(ks)):
            if np.isnan(mat[i, j]):
                continue
            color = "#111" if mat[i, j] < 0.45 else "white"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Jaccard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_vs_k(df_k: pd.DataFrame, out_path: Path) -> None:
    if df_k.empty:
        return
    d = df_k.sort_values("K")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.errorbar(d["K"], d["mean_jaccard"], yerr=d["std_jaccard"], fmt="-o", color="#2c5f9e")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean Jaccard")
    ax.set_title("Probe Overlap vs K")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_vs_layer(df_l: pd.DataFrame, out_path: Path) -> None:
    if df_l.empty:
        return
    d = df_l.sort_values("layer")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.errorbar(d["layer"], d["mean_jaccard"], yerr=d["std_jaccard"], fmt="-o", color="#5b8a72")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Jaccard")
    ax.set_title("Probe Overlap vs Layer")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pc_reuse(df_top: pd.DataFrame, out_path: Path, top_n: int) -> None:
    if df_top.empty:
        return
    counts = extract_pc_counts(df_top["consensus_pc_indices"].tolist())
    if not counts:
        return
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max(10, min(30, top_n))]
    pcs = [pc for pc, _ in items]
    freq = [counts[pc] for pc in pcs]

    fig, ax = plt.subplots(figsize=(10.5, max(5.5, len(pcs) * 0.3)))
    y = np.arange(len(pcs))[::-1]
    ax.barh(y, list(reversed(freq)), color=OOD_COLOR, edgecolor="#3a3a3a", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"PC {pc}" for pc in pcs[::-1]])
    ax.set_xlabel("Count in Top-N configs")
    ax.set_title(f"PC Reuse Frequency (Top {top_n} by OOD AUC)")
    ax.grid(True, axis="x", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_ratio_vs_ood(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    d = df.copy()
    d["consensus_ratio"] = d["num_consensus_pcs"] / d["K"]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    sc = ax.scatter(
        d["consensus_ratio"],
        d["ood_test_auc"],
        c=d["layer"],
        cmap="viridis",
        s=28 + 2.0 * d["num_consensus_pcs"].to_numpy(),
        alpha=0.7,
        edgecolor="black",
        linewidth=0.2,
    )
    ax.set_xlabel("Consensus Ratio (selected/K)")
    ax.set_ylabel("OOD Test AUC")
    ax.set_title("Consensus Ratio vs OOD AUC")
    ax.axhline(0.5, linestyle="--", color="#666", lw=1.0, alpha=0.75)
    ax.grid(True, alpha=GRID_ALPHA)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA consensus overlap analysis")
    parser.add_argument("--consensus_csv", type=str, required=True)
    parser.add_argument("--probes_root", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pooling", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    apply_style()

    consensus_csv = Path(args.consensus_csv)
    probes_root = Path(args.probes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_consensus_df(consensus_csv)
    df = df.dropna(subset=["ood_test_auc", "id_val_auc"]).copy()

    # Overlap metrics per (layer, K)
    rows = []
    missing = 0
    for (layer, k), _ in df.groupby(["layer", "K"]):
        masks = load_probe_selections(probes_root, args.model_dir, args.dataset, args.pooling, int(k), int(layer))
        if not masks:
            missing += 1
            continue
        mean_j, std_j, pairs = pairwise_jaccard(masks)
        rows.append(
            {
                "layer": int(layer),
                "K": int(k),
                "num_probes": int(len(masks)),
                "num_pairs": int(pairs),
                "mean_jaccard": float(mean_j),
                "std_jaccard": float(std_j),
            }
        )

    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(output_dir / "overlap_by_layer_k.csv", index=False)

    # Aggregate by K and by layer
    if not overlap_df.empty:
        overlap_by_k = overlap_df.groupby("K", as_index=False).agg(
            mean_jaccard=("mean_jaccard", "mean"),
            std_jaccard=("mean_jaccard", "std"),
            count=("mean_jaccard", "count"),
        )
        overlap_by_layer = overlap_df.groupby("layer", as_index=False).agg(
            mean_jaccard=("mean_jaccard", "mean"),
            std_jaccard=("mean_jaccard", "std"),
            count=("mean_jaccard", "count"),
        )
        overlap_by_k.to_csv(output_dir / "overlap_by_k.csv", index=False)
        overlap_by_layer.to_csv(output_dir / "overlap_by_layer.csv", index=False)
    else:
        overlap_by_k = pd.DataFrame()
        overlap_by_layer = pd.DataFrame()

    # PC reuse frequency
    df_sorted = df.sort_values("ood_test_auc", ascending=False).copy()
    df_top = df_sorted.head(args.top_n)
    counts_all = extract_pc_counts(df_sorted["consensus_pc_indices"].tolist())
    counts_top = extract_pc_counts(df_top["consensus_pc_indices"].tolist())

    def counts_to_df(counts: Dict[int, int]) -> pd.DataFrame:
        if not counts:
            return pd.DataFrame(columns=["pc_index", "count"])
        items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return pd.DataFrame(items, columns=["pc_index", "count"])

    counts_to_df(counts_all).to_csv(output_dir / "pc_reuse_frequency_all.csv", index=False)
    counts_to_df(counts_top).to_csv(output_dir / f"pc_reuse_frequency_top{args.top_n}.csv", index=False)

    # Plots
    plot_overlap_heatmap(overlap_df, output_dir / "overlap_heatmap_layer_k.png")
    plot_overlap_vs_k(overlap_by_k, output_dir / "overlap_vs_k.png")
    plot_overlap_vs_layer(overlap_by_layer, output_dir / "overlap_vs_layer.png")
    plot_pc_reuse(df_top, output_dir / "pc_reuse_topN.png", args.top_n)
    plot_consensus_ratio_vs_ood(df, output_dir / "consensus_ratio_vs_ood.png")

    print("Saved outputs to:", output_dir)
    if missing:
        print(f"Warning: missing selection files for {missing} (layer,K) pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
