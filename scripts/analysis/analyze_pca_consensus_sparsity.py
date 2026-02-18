#!/usr/bin/env python3
"""Sparsity-first analysis for PCA consensus probe runs.

This script prioritizes:
1) learned sparsity per (layer, K) from individual probe weights,
2) consensus sparsity and OOD performance for winning thresholds,
3) top-N layer/K/PC attribution tables and plots.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRID_ALPHA = 0.20


@dataclass(frozen=True)
class ProbeStats:
    sparsity_count: int
    sparsity_ratio: float
    val_auc: float
    val_acc: float
    val_epoch: int


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze PCA consensus runs with sparsity-first outputs")
    p.add_argument("--consensus_csv", type=str, required=True)
    p.add_argument("--probes_root", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--pooling", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument("--rank_metric", type=str, default="ood_test_auc", choices=["ood_test_auc", "id_val_auc"])
    p.add_argument(
        "--collapse_threshold",
        type=str,
        default="best_per_layer_k",
        choices=["best_per_layer_k", "none"],
    )
    return p.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "layer",
        "K",
        "threshold",
        "eps",
        "num_consensus_pcs",
        "consensus_pc_indices",
        "ood_test_auc",
        "id_val_auc",
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
        try:
            vals = ast.literal_eval(text)
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


def select_winners(df: pd.DataFrame, rank_metric: str, collapse_threshold: str) -> pd.DataFrame:
    if collapse_threshold == "none":
        out = df.sort_values(
            ["layer", "K", rank_metric, "id_val_auc", "threshold"],
            ascending=[True, True, False, False, True],
        ).reset_index(drop=True)
        return out

    sort_df = df.sort_values(
        ["layer", "K", rank_metric, "id_val_auc", "threshold"],
        ascending=[True, True, False, False, True],
    )
    winners = sort_df.drop_duplicates(subset=["layer", "K"], keep="first").reset_index(drop=True)
    return winners


def load_probe_stats_for_layer_k(
    probes_root: Path,
    model_dir: str,
    dataset: str,
    pooling: str,
    layer: int,
    k: int,
    eps: float,
) -> Tuple[List[ProbeStats], int]:
    probe_dir = probes_root / model_dir / dataset / pooling / f"K_{k}" / f"layer_{layer}" / "probes"
    if not probe_dir.exists():
        return [], 0

    stats: List[ProbeStats] = []
    parse_errors = 0
    for npz_path in sorted(probe_dir.glob("probe_r*.npz")):
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "w" not in data:
                    parse_errors += 1
                    continue
                w = np.asarray(data["w"], dtype=np.float32).reshape(-1)
                if w.shape[0] != int(k):
                    parse_errors += 1
                    continue
                sparsity_count = int((np.abs(w) > eps).sum())
                sparsity_ratio = float(sparsity_count / float(k)) if k > 0 else np.nan

                val_auc = float(np.asarray(data["val_auc"]).reshape(-1)[0]) if "val_auc" in data else np.nan
                val_acc = float(np.asarray(data["val_acc"]).reshape(-1)[0]) if "val_acc" in data else np.nan
                val_epoch = (
                    int(np.asarray(data["val_epoch"]).reshape(-1)[0]) if "val_epoch" in data else -1
                )
                stats.append(
                    ProbeStats(
                        sparsity_count=sparsity_count,
                        sparsity_ratio=sparsity_ratio,
                        val_auc=val_auc,
                        val_acc=val_acc,
                        val_epoch=val_epoch,
                    )
                )
        except Exception:
            parse_errors += 1
            continue
    return stats, parse_errors


def aggregate_probe_stats(stats: Sequence[ProbeStats]) -> Dict[str, float]:
    if not stats:
        return {
            "num_loaded_probes": 0,
            "probe_sparsity_count_mean": np.nan,
            "probe_sparsity_count_std": np.nan,
            "probe_sparsity_ratio_mean": np.nan,
            "probe_sparsity_ratio_std": np.nan,
            "probe_sparsity_ratio_min": np.nan,
            "probe_sparsity_ratio_max": np.nan,
            "probe_val_auc_mean": np.nan,
            "probe_val_auc_std": np.nan,
            "probe_val_auc_best": np.nan,
            "probe_val_acc_mean": np.nan,
            "probe_val_acc_std": np.nan,
            "probe_val_epoch_mean": np.nan,
        }

    sparsity_counts = np.asarray([s.sparsity_count for s in stats], dtype=float)
    sparsity_ratios = np.asarray([s.sparsity_ratio for s in stats], dtype=float)
    val_auc = np.asarray([s.val_auc for s in stats], dtype=float)
    val_acc = np.asarray([s.val_acc for s in stats], dtype=float)
    val_epoch = np.asarray([s.val_epoch for s in stats], dtype=float)

    return {
        "num_loaded_probes": int(len(stats)),
        "probe_sparsity_count_mean": float(np.nanmean(sparsity_counts)),
        "probe_sparsity_count_std": float(np.nanstd(sparsity_counts)),
        "probe_sparsity_ratio_mean": float(np.nanmean(sparsity_ratios)),
        "probe_sparsity_ratio_std": float(np.nanstd(sparsity_ratios)),
        "probe_sparsity_ratio_min": float(np.nanmin(sparsity_ratios)),
        "probe_sparsity_ratio_max": float(np.nanmax(sparsity_ratios)),
        "probe_val_auc_mean": float(np.nanmean(val_auc)),
        "probe_val_auc_std": float(np.nanstd(val_auc)),
        "probe_val_auc_best": float(np.nanmax(val_auc)),
        "probe_val_acc_mean": float(np.nanmean(val_acc)),
        "probe_val_acc_std": float(np.nanstd(val_acc)),
        "probe_val_epoch_mean": float(np.nanmean(val_epoch)),
    }


def build_winner_tables(
    winners_df: pd.DataFrame,
    probes_root: Path,
    model_dir: str,
    dataset: str,
    pooling: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    winner_rows: List[dict] = []
    probe_rows: List[dict] = []
    total_parse_errors = 0

    for _, row in winners_df.iterrows():
        layer = int(row["layer"])
        k = int(row["K"])
        eps = float(row["eps"])
        stats, parse_errors = load_probe_stats_for_layer_k(
            probes_root=probes_root,
            model_dir=model_dir,
            dataset=dataset,
            pooling=pooling,
            layer=layer,
            k=k,
            eps=eps,
        )
        total_parse_errors += parse_errors
        agg = aggregate_probe_stats(stats)
        consensus_ratio = float(row["num_consensus_pcs"] / k) if k > 0 else np.nan

        winner_out = {
            "layer": layer,
            "K": k,
            "threshold": float(row["threshold"]),
            "eps": eps,
            "ood_test_auc": float(row["ood_test_auc"]),
            "id_val_auc": float(row["id_val_auc"]),
            "num_consensus_pcs": int(row["num_consensus_pcs"]),
            "consensus_ratio": consensus_ratio,
            "consensus_pc_indices": row["consensus_pc_indices"],
        }
        winner_out.update(agg)
        winner_rows.append(winner_out)

        probe_row = {
            "layer": layer,
            "K": k,
            "threshold": float(row["threshold"]),
            "eps": eps,
            "ood_test_auc": float(row["ood_test_auc"]),
            "id_val_auc": float(row["id_val_auc"]),
            "num_consensus_pcs": int(row["num_consensus_pcs"]),
            "consensus_ratio": consensus_ratio,
        }
        probe_row.update(agg)
        probe_rows.append(probe_row)

    winner_tbl = pd.DataFrame(winner_rows).sort_values(["layer", "K"]).reset_index(drop=True)
    probe_tbl = pd.DataFrame(probe_rows).sort_values(["layer", "K"]).reset_index(drop=True)
    return winner_tbl, probe_tbl, total_parse_errors


def extract_pc_counts(rows: Iterable[object]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for raw in rows:
        for pc in parse_pc_indices(raw):
            counts[pc] = counts.get(pc, 0) + 1
    return counts


def pc_frequency_df(df: pd.DataFrame) -> pd.DataFrame:
    counts = extract_pc_counts(df["consensus_pc_indices"].tolist())
    n = len(df)
    rows = []
    for pc, c in counts.items():
        rows.append(
            {
                "pc_index": int(pc),
                "count": int(c),
                "frequency": float(c / n) if n > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "pc_index"], ascending=[False, True]).reset_index(drop=True)


def pc_enrichment_df(all_df: pd.DataFrame, top_df: pd.DataFrame) -> pd.DataFrame:
    all_counts = extract_pc_counts(all_df["consensus_pc_indices"].tolist())
    top_counts = extract_pc_counts(top_df["consensus_pc_indices"].tolist())
    all_n = len(all_df)
    top_n = len(top_df)

    union_pcs = sorted(set(all_counts.keys()) | set(top_counts.keys()))
    rows = []
    for pc in union_pcs:
        all_c = int(all_counts.get(pc, 0))
        top_c = int(top_counts.get(pc, 0))
        all_freq = float(all_c / all_n) if all_n > 0 else np.nan
        top_freq = float(top_c / top_n) if top_n > 0 else np.nan
        enrichment = np.nan
        if all_freq and all_freq > 0:
            enrichment = float(top_freq / all_freq)

        present_mask = all_df["consensus_pc_indices"].apply(lambda raw: pc in set(parse_pc_indices(raw)))
        if present_mask.any():
            mean_present = float(all_df.loc[present_mask, "ood_test_auc"].mean())
        else:
            mean_present = np.nan
        if (~present_mask).any():
            mean_absent = float(all_df.loc[~present_mask, "ood_test_auc"].mean())
        else:
            mean_absent = np.nan
        delta = mean_present - mean_absent if not np.isnan(mean_present) and not np.isnan(mean_absent) else np.nan

        rows.append(
            {
                "pc_index": int(pc),
                "count_all": all_c,
                "count_top": top_c,
                "freq_all": all_freq,
                "freq_top": top_freq,
                "enrichment_top_vs_all": enrichment,
                "mean_ood_present": mean_present,
                "mean_ood_absent": mean_absent,
                "delta_present_minus_absent": delta,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["enrichment_top_vs_all", "count_top", "pc_index"],
        ascending=[False, False, True],
        na_position="last",
    )
    return out.reset_index(drop=True)


def corr_rows_for_scope(df: pd.DataFrame, scope: str, x_col: str, y_col: str) -> List[dict]:
    d = df[[x_col, y_col]].dropna().copy()
    if len(d) < 3:
        return [
            {
                "scope": scope,
                "x_metric": x_col,
                "y_metric": y_col,
                "method": "pearson",
                "n": int(len(d)),
                "corr": np.nan,
            },
            {
                "scope": scope,
                "x_metric": x_col,
                "y_metric": y_col,
                "method": "spearman",
                "n": int(len(d)),
                "corr": np.nan,
            },
        ]

    pearson = float(d[x_col].corr(d[y_col], method="pearson"))
    spearman = float(d[x_col].corr(d[y_col], method="spearman"))
    return [
        {"scope": scope, "x_metric": x_col, "y_metric": y_col, "method": "pearson", "n": int(len(d)), "corr": pearson},
        {
            "scope": scope,
            "x_metric": x_col,
            "y_metric": y_col,
            "method": "spearman",
            "n": int(len(d)),
            "corr": spearman,
        },
    ]


def build_correlation_table(winner_tbl: pd.DataFrame, rank_metric: str) -> pd.DataFrame:
    rows: List[dict] = []
    pairs = [
        ("probe_sparsity_ratio_mean", rank_metric),
        ("consensus_ratio", rank_metric),
        ("probe_val_auc_mean", rank_metric),
    ]
    for x_col, y_col in pairs:
        rows.extend(corr_rows_for_scope(winner_tbl, "global", x_col, y_col))
        for layer, sub in winner_tbl.groupby("layer"):
            rows.extend(corr_rows_for_scope(sub, f"layer_{int(layer)}", x_col, y_col))
    out = pd.DataFrame(rows)
    return out


def pivot_heatmap(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, List[int], List[int]]:
    layers = sorted(df["layer"].unique())
    ks = sorted(df["K"].unique())
    mat = np.full((len(layers), len(ks)), np.nan, dtype=float)
    layer_pos = {int(v): i for i, v in enumerate(layers)}
    k_pos = {int(v): i for i, v in enumerate(ks)}
    for _, row in df.iterrows():
        li = layer_pos[int(row["layer"])]
        ki = k_pos[int(row["K"])]
        mat[li, ki] = float(row[value_col]) if not pd.isna(row[value_col]) else np.nan
    return mat, layers, ks


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "YlGnBu",
) -> None:
    if df.empty:
        return
    mat, layers, ks = pivot_heatmap(df, value_col)
    vmax = np.nanmax(mat) if not np.isnan(mat).all() else 1.0
    vmin = np.nanmin(mat) if not np.isnan(mat).all() else 0.0

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([f"L{int(l)}" for l in layers])
    ax.set_xlabel("K")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    for i in range(len(layers)):
        for j in range(len(ks)):
            if np.isnan(mat[i, j]):
                continue
            txt = f"{mat[i, j]:.3f}" if value_col.endswith("auc") or "ratio" in value_col else f"{mat[i, j]:.2f}"
            color = "white" if (not np.isnan(vmax) and vmax > vmin and mat[i, j] > (vmin + vmax) / 2.0) else "#111"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.2, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    if df.empty:
        return
    d = df[[x_col, y_col, "K", "layer", "num_consensus_pcs"]].dropna()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    size = 30 + 2.0 * d["num_consensus_pcs"].to_numpy()
    sc = ax.scatter(
        d[x_col],
        d[y_col],
        c=d["K"],
        cmap="plasma",
        s=size,
        alpha=0.74,
        edgecolor="black",
        linewidth=0.2,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=GRID_ALPHA)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("K")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_topn_ranked(top_df: pd.DataFrame, out_path: Path, rank_metric: str) -> None:
    if top_df.empty:
        return
    d = top_df.copy().reset_index(drop=True)
    x = np.arange(1, len(d) + 1)
    labels = [f"L{int(l)}-K{int(k)}-t{float(t):.1f}" for l, k, t in zip(d["layer"], d["K"], d["threshold"])]

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.plot(x, d[rank_metric], color="#c26d3f", marker="o", lw=2.0, label=rank_metric)
    ax.plot(
        x,
        d["probe_sparsity_ratio_mean"],
        color="#2f5f9e",
        marker="s",
        lw=1.8,
        label="probe_sparsity_ratio_mean",
    )
    ax.plot(
        x,
        d["consensus_ratio"],
        color="#3f8f6a",
        marker="^",
        lw=1.8,
        label="consensus_ratio",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Top-N rank")
    ax.set_title("Top-N configs: performance with probe+consensus sparsity")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_count_bar(df: pd.DataFrame, key_col: str, out_path: Path, title: str, x_label: str) -> None:
    if df.empty:
        return
    d = df.groupby(key_col, as_index=False).size().sort_values("size", ascending=False)
    d[key_col] = d[key_col].astype(str)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.bar(d[key_col], d["size"], color="#355f9e", edgecolor="#222", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count in Top-N")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pc_enrichment(pc_df: pd.DataFrame, out_path: Path, n_show: int = 20) -> None:
    if pc_df.empty:
        return
    d = pc_df.dropna(subset=["enrichment_top_vs_all"]).head(n_show).copy()
    if d.empty:
        return
    d = d.sort_values("enrichment_top_vs_all", ascending=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(10.0, max(5.5, len(d) * 0.28)))
    ax.barh(y, d["enrichment_top_vs_all"], color="#c26d3f", edgecolor="#222", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"PC {int(pc)}" for pc in d["pc_index"]])
    ax.set_xlabel("Enrichment (top freq / all freq)")
    ax.set_title("PC enrichment in Top-N runs")
    ax.axvline(1.0, linestyle="--", color="#666", lw=1.0, alpha=0.75)
    ax.grid(True, axis="x", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    apply_style()

    consensus_csv = Path(args.consensus_csv)
    probes_root = Path(args.probes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_df(consensus_csv)
    df_raw = df_raw.dropna(subset=["ood_test_auc", "id_val_auc"]).copy()
    if df_raw.empty:
        raise ValueError("No rows with valid ood_test_auc/id_val_auc in consensus CSV.")
    df_raw = df_raw.sort_values(
        ["ood_test_auc", "id_val_auc", "layer", "K", "threshold"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)

    winners_df = select_winners(df_raw, rank_metric=args.rank_metric, collapse_threshold=args.collapse_threshold)
    winners_tbl, probe_tbl, parse_errors = build_winner_tables(
        winners_df=winners_df,
        probes_root=probes_root,
        model_dir=args.model_dir,
        dataset=args.dataset,
        pooling=args.pooling,
    )

    # Persist winner/probe summary tables.
    winners_tbl.to_csv(output_dir / "layer_k_winners.csv", index=False)
    probe_tbl.to_csv(output_dir / "layer_k_probe_sparsity_summary.csv", index=False)

    # Top-N table uses full ranking, then enrich with per-(layer,K) probe aggregates.
    top_n = max(1, int(args.top_n))
    top_df = df_raw.head(top_n).copy()
    join_cols = [
        "layer",
        "K",
        "probe_sparsity_count_mean",
        "probe_sparsity_count_std",
        "probe_sparsity_ratio_mean",
        "probe_sparsity_ratio_std",
        "probe_sparsity_ratio_min",
        "probe_sparsity_ratio_max",
        "probe_val_auc_mean",
        "probe_val_auc_std",
        "probe_val_auc_best",
        "num_loaded_probes",
    ]
    top_df = top_df.merge(probe_tbl[join_cols].drop_duplicates(["layer", "K"]), on=["layer", "K"], how="left")
    top_df["consensus_ratio"] = top_df["num_consensus_pcs"] / top_df["K"]
    top_df.to_csv(output_dir / f"top{top_n}_layer_k_sparsity.csv", index=False)

    # PC frequency and enrichment.
    pc_all = pc_frequency_df(df_raw)
    pc_top = pc_frequency_df(top_df)
    pc_enrich = pc_enrichment_df(df_raw, top_df)
    pc_all.to_csv(output_dir / "pc_frequency_all.csv", index=False)
    pc_top.to_csv(output_dir / f"pc_frequency_top{top_n}.csv", index=False)
    pc_enrich.to_csv(output_dir / f"pc_enrichment_top{top_n}_vs_all.csv", index=False)

    # Correlations.
    corr_tbl = build_correlation_table(winners_tbl, rank_metric=args.rank_metric)
    corr_tbl.to_csv(output_dir / "sparsity_auc_correlation.csv", index=False)

    # Plots.
    plot_heatmap(
        winners_tbl,
        value_col=args.rank_metric,
        out_path=output_dir / "heatmap_ood_by_layer_k_winner.png",
        title=f"Winner {args.rank_metric} by Layer/K",
        cbar_label=args.rank_metric,
        cmap="YlOrBr",
    )
    plot_heatmap(
        probe_tbl,
        value_col="probe_sparsity_ratio_mean",
        out_path=output_dir / "heatmap_probe_sparsity_by_layer_k.png",
        title="Mean probe sparsity ratio by Layer/K",
        cbar_label="Probe sparsity ratio (|w| > eps)",
        cmap="YlGnBu",
    )
    plot_scatter(
        winners_tbl,
        x_col="probe_sparsity_ratio_mean",
        y_col=args.rank_metric,
        out_path=output_dir / "scatter_probe_sparsity_vs_ood.png",
        title=f"Probe sparsity vs {args.rank_metric}",
        x_label="Mean probe sparsity ratio",
        y_label=args.rank_metric,
    )
    plot_scatter(
        winners_tbl,
        x_col="consensus_ratio",
        y_col=args.rank_metric,
        out_path=output_dir / "scatter_consensus_ratio_vs_ood.png",
        title=f"Consensus ratio vs {args.rank_metric}",
        x_label="Consensus ratio (num_consensus_pcs / K)",
        y_label=args.rank_metric,
    )
    plot_topn_ranked(
        top_df,
        output_dir / f"top{top_n}_ranked_lk_with_sparsity.png",
        rank_metric=args.rank_metric,
    )
    plot_count_bar(
        top_df,
        key_col="layer",
        out_path=output_dir / f"bar_top{top_n}_layer_contribution.png",
        title=f"Top-{top_n} contribution by layer",
        x_label="Layer",
    )
    plot_count_bar(
        top_df,
        key_col="K",
        out_path=output_dir / f"bar_top{top_n}_k_contribution.png",
        title=f"Top-{top_n} contribution by K",
        x_label="K",
    )
    plot_pc_enrichment(
        pc_enrich,
        output_dir / f"bar_pc_enrichment_top{top_n}.png",
        n_show=min(20, top_n),
    )

    # Metadata.
    metadata = {
        "script": "scripts/analysis/analyze_pca_consensus_sparsity.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "consensus_csv": str(consensus_csv.resolve()),
            "probes_root": str(probes_root.resolve()),
            "model_dir": args.model_dir,
            "dataset": args.dataset,
            "pooling": args.pooling,
        },
        "config": {
            "top_n": int(top_n),
            "rank_metric": args.rank_metric,
            "collapse_threshold": args.collapse_threshold,
        },
        "summary": {
            "rows_raw_valid": int(len(df_raw)),
            "rows_winner": int(len(winners_tbl)),
            "rows_top_n": int(len(top_df)),
            "probe_parse_errors": int(parse_errors),
        },
        "outputs": {
            "tables": [
                "layer_k_winners.csv",
                "layer_k_probe_sparsity_summary.csv",
                f"top{top_n}_layer_k_sparsity.csv",
                "pc_frequency_all.csv",
                f"pc_frequency_top{top_n}.csv",
                f"pc_enrichment_top{top_n}_vs_all.csv",
                "sparsity_auc_correlation.csv",
            ],
            "plots": [
                "heatmap_ood_by_layer_k_winner.png",
                "heatmap_probe_sparsity_by_layer_k.png",
                "scatter_probe_sparsity_vs_ood.png",
                "scatter_consensus_ratio_vs_ood.png",
                f"top{top_n}_ranked_lk_with_sparsity.png",
                f"bar_top{top_n}_layer_contribution.png",
                f"bar_top{top_n}_k_contribution.png",
                f"bar_pc_enrichment_top{top_n}.png",
            ],
        },
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved outputs to: {output_dir}")
    if parse_errors:
        print(f"Warning: skipped {parse_errors} malformed probe files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
