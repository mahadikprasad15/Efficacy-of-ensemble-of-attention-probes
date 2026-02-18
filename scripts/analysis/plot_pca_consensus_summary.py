#!/usr/bin/env python3
"""Analyze consensus PCA probe results and generate tables + v2 plots."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

# Palette: muted paper style, distinct from previous plots.
OOD_COLOR = "#c26d3f"
ID_COLOR = "#355f9e"
POS_GAP_COLOR = "#3f8f6a"
NEG_GAP_COLOR = "#b44b4b"
GRID_ALPHA = 0.18


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.facecolor": "#f6f5f2",
            "axes.facecolor": "#f6f5f2",
            "axes.edgecolor": "#9ea3a8",
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "-",
            "savefig.facecolor": "#f6f5f2",
        }
    )


def load_df(path: Path) -> pd.DataFrame:
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


def save_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


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
        try:
            vals = ast.literal_eval(text)
        except Exception:
            return []
    if not isinstance(vals, (list, tuple)):
        return []
    out = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def extract_pc_counts(rows: Iterable[object]) -> Dict[int, int]:
    pc_counts: Dict[int, int] = {}
    for raw in rows:
        for pc in parse_pc_indices(raw):
            pc_counts[pc] = pc_counts.get(pc, 0) + 1
    return pc_counts


def plot_top5_lollipop(df_top5: pd.DataFrame, out_path: Path) -> None:
    if df_top5.empty:
        return
    d = df_top5.sort_values("ood_test_auc", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, row in d.iterrows():
        x_id = float(row["id_val_auc"])
        x_ood = float(row["ood_test_auc"])
        gap_color = POS_GAP_COLOR if x_ood >= x_id else NEG_GAP_COLOR
        ax.hlines(i, min(x_id, x_ood), max(x_id, x_ood), color=gap_color, lw=3.5, alpha=0.85)

    ax.scatter(d["id_val_auc"], y, color=ID_COLOR, s=85, label="ID Val AUC", zorder=3, edgecolor="white", linewidth=0.8)
    ax.scatter(d["ood_test_auc"], y, color=OOD_COLOR, s=85, label="OOD Test AUC", zorder=3, edgecolor="white", linewidth=0.8)

    labels = [
        f"L{int(r.layer)} K{int(r.K)} t={float(r.threshold):.1f} pcs={int(r.num_consensus_pcs)}"
        for _, r in d.iterrows()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUC")
    ax.set_title("Top-5 OOD Runs (Lollipop ID vs OOD)")
    ax.axvline(0.5, linestyle="--", color="#666", lw=1.0, alpha=0.75)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=GRID_ALPHA)

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top20_rank_traces(df_top20: pd.DataFrame, out_path: Path) -> None:
    if df_top20.empty:
        return
    df = df_top20.sort_values("ood_test_auc", ascending=False).reset_index(drop=True)
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(x, df["ood_test_auc"], color=OOD_COLOR, marker="o", lw=2.1, label="OOD Test AUC")
    ax.plot(x, df["id_val_auc"], color=ID_COLOR, marker="o", lw=2.1, label="ID Val AUC")

    above = df["ood_test_auc"] >= df["id_val_auc"]
    ax.fill_between(x, df["ood_test_auc"], df["id_val_auc"], where=above, color=POS_GAP_COLOR, alpha=0.12, interpolate=True)
    ax.fill_between(x, df["ood_test_auc"], df["id_val_auc"], where=~above, color=NEG_GAP_COLOR, alpha=0.10, interpolate=True)

    for i in range(min(4, len(df))):
        r = df.iloc[i]
        ax.text(
            x[i] + 0.08,
            float(r["ood_test_auc"]) + 0.008,
            f"L{int(r['layer'])},K{int(r['K'])},t{float(r['threshold']):.1f}",
            fontsize=9,
            color="#333",
        )

    ax.set_xlabel("Rank (sorted by OOD AUC)")
    ax.set_ylabel("AUC")
    ax.set_title("Top-20 OOD Runs: Rank Traces")
    ax.axhline(0.5, linestyle="--", color="#666", lw=1.0, alpha=0.75)
    ax.set_xticks(x)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_k_layer_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    # Pick best threshold for each (layer, K) by OOD AUC.
    idx = df.groupby(["layer", "K"])["ood_test_auc"].idxmax()
    best = df.loc[idx].copy()

    layers = sorted(best["layer"].unique())
    ks = sorted(best["K"].unique())

    auc_mat = np.full((len(layers), len(ks)), np.nan, dtype=float)
    txt_mat: List[List[str]] = [["" for _ in ks] for _ in layers]
    layer_pos = {l: i for i, l in enumerate(layers)}
    k_pos = {k: i for i, k in enumerate(ks)}

    for _, r in best.iterrows():
        li = layer_pos[int(r["layer"])]
        ki = k_pos[int(r["K"])]
        auc_mat[li, ki] = float(r["ood_test_auc"])
        txt_mat[li][ki] = f"{float(r['ood_test_auc']):.3f}\nt={float(r['threshold']):.1f}"

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    im = ax.imshow(auc_mat, aspect="auto", cmap="YlOrBr", vmin=0.35, vmax=max(0.75, np.nanmax(auc_mat)))
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("K")
    ax.set_ylabel("Layer")
    ax.set_title("Best OOD AUC per (Layer, K) with Winning Threshold")

    for i in range(len(layers)):
        for j in range(len(ks)):
            if np.isnan(auc_mat[i, j]):
                continue
            color = "#111" if auc_mat[i, j] < 0.62 else "white"
            ax.text(j, i, txt_mat[i][j], ha="center", va="center", fontsize=8.5, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("OOD Test AUC (best threshold)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_gap_box(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    d = df.copy()
    d["gap"] = d["ood_test_auc"] - d["id_val_auc"]

    thresholds = sorted(d["threshold"].unique())
    groups = [d[d["threshold"] == t]["gap"].to_numpy() for t in thresholds]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bp = ax.boxplot(groups, patch_artist=True, labels=[f"t={t:.1f}" for t in thresholds], showfliers=False)
    box_colors = ["#c3d1e6", "#f0d4c0", "#cfe7de", "#e3c4c4"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(box_colors[i % len(box_colors)])
        patch.set_edgecolor("#555")
        patch.set_alpha(0.9)

    norm = plt.Normalize(vmin=float(d["K"].min()), vmax=float(d["K"].max()))
    cmap = plt.cm.viridis
    rng = np.random.default_rng(123)
    for i, t in enumerate(thresholds):
        sub = d[d["threshold"] == t]
        xj = rng.normal(loc=i + 1, scale=0.06, size=len(sub))
        ax.scatter(xj, sub["gap"], c=sub["K"], cmap=cmap, norm=norm, s=15, alpha=0.45, linewidth=0)

    ax.axhline(0.0, linestyle="--", color="#666", lw=1.0, alpha=0.8)
    ax.set_ylabel("OOD AUC - ID AUC")
    ax.set_title("Generalization Gap by Threshold (colored by K)")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("K")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_efficiency(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    d = df.copy()
    d["consensus_ratio"] = d["num_consensus_pcs"] / d["K"]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    markers = ["o", "s", "^", "D", "P", "X"]
    thresholds = sorted(d["threshold"].unique())
    norm = plt.Normalize(vmin=float(d["layer"].min()), vmax=float(d["layer"].max()))
    cmap = plt.cm.cividis

    for i, t in enumerate(thresholds):
        sub = d[d["threshold"] == t]
        if sub.empty:
            continue
        sc = ax.scatter(
            sub["consensus_ratio"],
            sub["ood_test_auc"],
            c=sub["layer"],
            cmap=cmap,
            norm=norm,
            marker=markers[i % len(markers)],
            s=28 + 2.2 * sub["num_consensus_pcs"],
            alpha=0.78,
            edgecolors="black",
            linewidth=0.2,
            label=f"t={t:.1f}",
        )

    ax.set_xlabel("Consensus Ratio (selected PCs / K)")
    ax.set_ylabel("OOD Test AUC")
    ax.set_title("Consensus Efficiency Map")
    ax.axhline(0.5, linestyle="--", color="#666", lw=1.0, alpha=0.75)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(title="Threshold", loc="lower right")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Layer")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_pc_frequency(df_top20: pd.DataFrame, out_path: Path, max_pcs: int = 22) -> None:
    if df_top20.empty:
        return

    counts = extract_pc_counts(df_top20["consensus_pc_indices"].tolist())
    if not counts:
        return
    # Frequency plus mean OOD among rows that include that PC.
    top_pc_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_pcs]
    pcs = [pc for pc, _ in top_pc_items]

    mean_ood = []
    for pc in pcs:
        mask = df_top20["consensus_pc_indices"].apply(lambda raw: pc in set(parse_pc_indices(raw)))
        if mask.any():
            mean_ood.append(float(df_top20.loc[mask, "ood_test_auc"].mean()))
        else:
            mean_ood.append(np.nan)

    freq = np.array([counts[pc] for pc in pcs], dtype=float)
    y = np.arange(len(pcs))[::-1]

    fig, ax = plt.subplots(figsize=(10.5, max(6.0, len(pcs) * 0.3)))
    norm = plt.Normalize(vmin=np.nanmin(mean_ood), vmax=np.nanmax(mean_ood))
    cmap = plt.cm.magma
    colors = cmap(norm(mean_ood))
    ax.barh(y, freq[::-1], color=colors[::-1], edgecolor="#3a3a3a", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"PC {pc}" for pc in pcs[::-1]])
    ax.set_xlabel("Count in Top-N OOD Configs")
    ax.set_title("Most Reused Consensus PCs (bar color = mean OOD AUC)")
    ax.grid(True, axis="x", alpha=GRID_ALPHA)

    for i, c in enumerate(freq[::-1]):
        ax.text(c + 0.1, y[i], f"{int(c)}", va="center", fontsize=9, color="#333")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Mean OOD AUC when PC selected")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_frontier(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    d = df.copy()
    x = d["id_val_auc"].to_numpy()
    y = d["ood_test_auc"].to_numpy()

    # Non-dominated points under maximize(x), maximize(y).
    dominated = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        if dominated[i]:
            continue
        better = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        better[i] = False
        if np.any(better):
            dominated[i] = True
    frontier = d.loc[~dominated].sort_values("id_val_auc")

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    sizes = 22 + 2.0 * d["num_consensus_pcs"].to_numpy()
    sc = ax.scatter(
        d["id_val_auc"],
        d["ood_test_auc"],
        c=d["K"],
        cmap="plasma",
        s=sizes,
        alpha=0.62,
        linewidth=0.2,
        edgecolor="#1f1f1f",
    )
    ax.plot(frontier["id_val_auc"], frontier["ood_test_auc"], color="#232323", lw=2.2, label="Pareto Frontier")

    top_frontier = frontier.sort_values("ood_test_auc", ascending=False).head(4)
    for _, r in top_frontier.iterrows():
        ax.text(
            float(r["id_val_auc"]) + 0.004,
            float(r["ood_test_auc"]) + 0.004,
            f"L{int(r['layer'])},K{int(r['K'])},t{float(r['threshold']):.1f}",
            fontsize=8.5,
            color="#222",
        )

    ax.axline((0.3, 0.3), (0.8, 0.8), color="#888", linestyle="--", lw=1.0, alpha=0.7)
    ax.set_xlim(0.3, 0.8)
    ax.set_ylim(0.3, 0.8)
    ax.set_xlabel("ID Val AUC")
    ax.set_ylabel("OOD Test AUC")
    ax.set_title("Pareto View: Keep High ID While Maximizing OOD")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="lower right")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("K")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA consensus summary analysis (v2 visual pack)")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    apply_style()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(input_csv)
    df = df.dropna(subset=["ood_test_auc", "id_val_auc"]).copy()

    # Tables (unchanged contract)
    df_top5 = df.sort_values("ood_test_auc", ascending=False).head(5).copy()
    save_table(df_top5, output_dir / "top5_ood_auc.csv")

    df_top20 = df.sort_values("ood_test_auc", ascending=False).head(args.top_n).copy()
    save_table(df_top20, output_dir / "top20_ood_auc.csv")

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

    # Visual set v2.
    plot_top5_lollipop(df_top5, output_dir / "v2_top5_lollipop.png")
    plot_top20_rank_traces(df_top20, output_dir / "v2_top20_rank_traces.png")
    plot_best_k_layer_heatmap(df, output_dir / "v2_best_k_layer_heatmap.png")
    plot_threshold_gap_box(df, output_dir / "v2_threshold_gap_box.png")
    plot_consensus_efficiency(df, output_dir / "v2_consensus_efficiency.png")
    plot_top_pc_frequency(df_top20, output_dir / "v2_top_pc_frequency.png")
    plot_pareto_frontier(df, output_dir / "v2_pareto_frontier.png")

    print("Saved outputs to:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
