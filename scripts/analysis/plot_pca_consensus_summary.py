#!/usr/bin/env python3
"""
Analyze consensus PCA probe results and generate tables + plots.

Inputs:
- consensus_summary.csv produced by train_pca_consensus_probes.py

Outputs:
- top5_ood_auc.csv
- top20_ood_auc.csv
- top20_summary_by_k.csv
- top20_summary_by_layer.csv
- top20_summary_by_threshold.csv
- a set of plots matching the existing style palette
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OOD_COLOR = "#d08c60"  # muted orange
ID_COLOR = "#4c72b0"   # muted blue
GRID_ALPHA = 0.25


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names if needed
    for col in ["K", "layer", "threshold", "ood_test_auc", "id_val_auc", "num_consensus_pcs"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    return df


def save_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def plot_top5_bars(df_top5: pd.DataFrame, out_path: Path) -> None:
    layers = df_top5["layer"].astype(int).tolist()
    x = np.arange(len(layers))

    ood = df_top5["ood_test_auc"].astype(float).to_numpy()
    idv = df_top5["id_val_auc"].astype(float).to_numpy()
    ks = df_top5["K"].astype(int).tolist()

    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, ood, width, color=OOD_COLOR, label="OOD Test AUC")
    plt.bar(x + width/2, idv, width, color=ID_COLOR, label="ID Val AUC")

    for i, k in enumerate(ks):
        plt.text(x[i], max(ood[i], idv[i]) + 0.01, f"K={k}", ha="center", va="bottom", fontsize=9)

    plt.axhline(0.5, linestyle="--", color="#777", linewidth=1, alpha=0.6, label="Random (0.5)")
    plt.xticks(x, [f"L{l}" for l in layers])
    plt.ylabel("AUC")
    plt.title("Top 5 OOD AUC (best across K, threshold) — OOD vs ID")
    plt.grid(alpha=GRID_ALPHA)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top20_scatter(df_top20: pd.DataFrame, out_path: Path) -> None:
    x = df_top20["id_val_auc"].astype(float).to_numpy()
    y = df_top20["ood_test_auc"].astype(float).to_numpy()
    layers = df_top20["layer"].astype(int).to_numpy()
    ks = df_top20["K"].astype(int).to_numpy()

    sizes = 30 + 10 * (ks / ks.max())
    plt.figure(figsize=(8.5, 6.5))
    sc = plt.scatter(x, y, c=layers, cmap="viridis", s=sizes, alpha=0.85, edgecolors="black", linewidth=0.3)
    plt.axline((0, 0), slope=1, linestyle="--", color="#777", linewidth=1, alpha=0.6)
    plt.xlabel("ID Val AUC")
    plt.ylabel("OOD Test AUC")
    plt.title("Top 20 OOD AUC — ID vs OOD (color=layer, size=K)")
    plt.grid(alpha=GRID_ALPHA)
    plt.colorbar(sc, label="Layer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_consensus_threshold_vs_ood(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for k in sorted(df["K"].unique()):
        sub = df[df["K"] == k]
        sub = sub.groupby("threshold", as_index=False)["ood_test_auc"].mean()
        plt.plot(sub["threshold"], sub["ood_test_auc"], marker="o", label=f"K={k}")
    plt.xlabel("Consensus Threshold")
    plt.ylabel("OOD Test AUC")
    plt.title("OOD AUC vs Consensus Threshold (mean across layers)")
    plt.grid(alpha=GRID_ALPHA)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_consensus_size_vs_ood(df: pd.DataFrame, out_path: Path) -> None:
    x = df["num_consensus_pcs"].astype(float).to_numpy()
    y = df["ood_test_auc"].astype(float).to_numpy()
    layers = df["layer"].astype(int).to_numpy()

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=layers, cmap="viridis", alpha=0.8, edgecolors="black", linewidth=0.3)
    plt.xlabel("# Consensus PCs")
    plt.ylabel("OOD Test AUC")
    plt.title("Consensus Size vs OOD AUC (color=layer)")
    plt.grid(alpha=GRID_ALPHA)
    plt.colorbar(sc, label="Layer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    pivot = df.pivot_table(index="layer", columns="K", values=value_col, aggfunc="mean")
    if pivot.empty:
        return
    plt.figure(figsize=(10, 6))
    data = pivot.to_numpy()
    plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns.tolist())
    plt.yticks(range(len(pivot.index)), [f"L{l}" for l in pivot.index.tolist()])
    plt.xlabel("K")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gap_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    df["gap"] = df["id_val_auc"] - df["ood_test_auc"]
    plot_heatmap(df, "gap", out_path, "ID–OOD AUC Gap Heatmap (Layer × K)")


def plot_k_vs_consensus(df: pd.DataFrame, out_path: Path) -> None:
    agg = df.groupby("K", as_index=False)["num_consensus_pcs"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 5))
    plt.errorbar(agg["K"], agg["mean"], yerr=agg["std"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Avg # Consensus PCs")
    plt.title("K vs Avg Consensus Size")
    plt.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ood_auc_vs_k(df: pd.DataFrame, out_path: Path) -> None:
    agg = df.groupby("K", as_index=False)["ood_test_auc"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 5))
    plt.errorbar(agg["K"], agg["mean"], yerr=agg["std"], marker="o", color=OOD_COLOR)
    plt.xlabel("K")
    plt.ylabel("OOD Test AUC")
    plt.title("OOD AUC vs K (mean ± std across layers)")
    plt.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_layer_frequency(df_top20: pd.DataFrame, out_path: Path) -> None:
    counts = df_top20["layer"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    plt.bar([f"L{l}" for l in counts.index], counts.values, color=ID_COLOR)
    plt.xlabel("Layer")
    plt.ylabel("Count in Top-20")
    plt.title("Layer Frequency in Top-20 OOD")
    plt.grid(alpha=GRID_ALPHA, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pc_frequency(df: pd.DataFrame, out_path_hist: Path, out_path_top: Path, top_n: int = 20) -> None:
    # Expand consensus_pc_indices (stored as string list)
    pc_counts = {}
    for idxs in df["consensus_pc_indices"].dropna().tolist():
        try:
            if isinstance(idxs, str):
                vals = json.loads(idxs)
            else:
                vals = idxs
            for v in vals:
                pc_counts[int(v)] = pc_counts.get(int(v), 0) + 1
        except Exception:
            continue

    if not pc_counts:
        return

    # Histogram of frequencies
    freq = np.array(list(pc_counts.values()))
    plt.figure(figsize=(7, 5))
    plt.hist(freq, bins=10, color=OOD_COLOR, edgecolor="black")
    plt.xlabel("Selection Frequency")
    plt.ylabel("# PCs")
    plt.title("PC Selection Frequency Distribution")
    plt.grid(alpha=GRID_ALPHA, axis="y")
    plt.tight_layout()
    plt.savefig(out_path_hist, dpi=200)
    plt.close()

    # Top PCs by frequency
    top = sorted(pc_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    plt.figure(figsize=(10, 4))
    plt.bar([str(k) for k, _ in top], [v for _, v in top], color=ID_COLOR)
    plt.xlabel("PC Index")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} PCs by Frequency")
    plt.grid(alpha=GRID_ALPHA, axis="y")
    plt.tight_layout()
    plt.savefig(out_path_top, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Consensus PCA summary analysis")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(input_csv)
    df = df.dropna(subset=["ood_test_auc", "id_val_auc"])

    # Top 5 by OOD AUC
    df_top5 = df.sort_values("ood_test_auc", ascending=False).head(5)
    save_table(df_top5, output_dir / "top5_ood_auc.csv")

    # Top 20 by OOD AUC
    df_top20 = df.sort_values("ood_test_auc", ascending=False).head(args.top_n)
    save_table(df_top20, output_dir / "top20_ood_auc.csv")

    # Summaries for top-20
    top20_by_k = df_top20.groupby("K", as_index=False).agg(
        count=("ood_test_auc", "count"),
        mean_ood_auc=("ood_test_auc", "mean"),
        mean_id_auc=("id_val_auc", "mean"),
    )
    save_table(top20_by_k, output_dir / "top20_summary_by_k.csv")

    top20_by_layer = df_top20.groupby("layer", as_index=False).agg(
        count=("ood_test_auc", "count"),
        mean_ood_auc=("ood_test_auc", "mean"),
        mean_id_auc=("id_val_auc", "mean"),
    )
    save_table(top20_by_layer, output_dir / "top20_summary_by_layer.csv")

    top20_by_threshold = df_top20.groupby("threshold", as_index=False).agg(
        count=("ood_test_auc", "count"),
        mean_ood_auc=("ood_test_auc", "mean"),
        mean_id_auc=("id_val_auc", "mean"),
    )
    save_table(top20_by_threshold, output_dir / "top20_summary_by_threshold.csv")

    # Plots
    plot_top5_bars(df_top5, output_dir / "top5_ood_auc_bar.png")
    plot_top20_scatter(df_top20, output_dir / "top20_ood_auc_scatter.png")
    plot_consensus_threshold_vs_ood(df, output_dir / "consensus_threshold_vs_ood_auc.png")
    plot_consensus_size_vs_ood(df, output_dir / "consensus_size_vs_ood_auc.png")
    plot_heatmap(df, "ood_test_auc", output_dir / "ood_auc_heatmap.png", "OOD Test AUC Heatmap (Layer × K)")
    plot_heatmap(df, "id_val_auc", output_dir / "id_auc_heatmap.png", "ID Val AUC Heatmap (Layer × K)")
    plot_gap_heatmap(df, output_dir / "gap_heatmap.png")
    plot_k_vs_consensus(df, output_dir / "k_vs_consensus_size.png")
    plot_ood_auc_vs_k(df, output_dir / "ood_auc_vs_k_mean_std.png")
    plot_layer_frequency(df_top20, output_dir / "top20_layer_frequency.png")
    plot_pc_frequency(df, output_dir / "pc_frequency_histogram.png", output_dir / "top_pc_frequency.png")

    print("Saved outputs to:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
