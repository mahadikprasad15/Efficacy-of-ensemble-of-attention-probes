#!/usr/bin/env python3
"""Generate corrected PCA-consensus visualizations from one run directory.

Expected inputs under --run_dir:
- consensus_summary.csv
- consensus_retrain_summary.csv (optional)

This script exposes reusable plotting functions and a CLI that writes the 8-plot
bundle similar to the dashboard family shown in the user-provided PNGs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_idx_list(raw) -> List[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    s = str(raw).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
    except Exception:
        return []
    if isinstance(parsed, (list, tuple)):
        return [int(x) for x in parsed]
    return []


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _grid_axes(n: int, ncols: int = 4, figsize_scale: Tuple[float, float] = (4.2, 3.2)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    flat = axes.ravel().tolist()
    for i in range(n, len(flat)):
        flat[i].axis("off")
    return fig, flat


@dataclass
class RunData:
    consensus_raw: pd.DataFrame
    consensus: pd.DataFrame
    retrain: pd.DataFrame
    layers: List[int]
    k_values: List[int]
    thresholds_kept: List[float]
    thresholds_dropped: List[float]
    meta: Dict[str, str]


def _thresholds_to_drop(consensus: pd.DataFrame, atol: float = 1e-12) -> List[float]:
    """Drop thresholds that are exactly redundant over (layer,K). Keep lower threshold."""
    thresholds = sorted(float(x) for x in consensus["threshold"].unique())
    if len(thresholds) <= 1:
        return []

    drops: List[float] = []
    key_cols = ["layer", "K"]
    cmp_cols = ["num_consensus_pcs", "id_val_auc", "ood_test_auc", "id_val_acc", "ood_test_acc"]

    def _snap(df: pd.DataFrame, t: float) -> pd.DataFrame:
        out = df[np.isclose(df["threshold"].astype(float), t, atol=atol)].copy()
        out = out[key_cols + cmp_cols].sort_values(key_cols).reset_index(drop=True)
        return out

    for i, t0 in enumerate(thresholds):
        if t0 in drops:
            continue
        a = _snap(consensus, t0)
        for t1 in thresholds[i + 1 :]:
            if t1 in drops:
                continue
            b = _snap(consensus, t1)
            if len(a) != len(b):
                continue
            equal = True
            for c in cmp_cols:
                av = a[c].to_numpy()
                bv = b[c].to_numpy()
                mask_nan = np.isnan(av) & np.isnan(bv)
                mask_num = ~mask_nan
                if mask_num.any() and not np.allclose(av[mask_num], bv[mask_num], equal_nan=True, atol=1e-12):
                    equal = False
                    break
                if np.any(np.isnan(av[mask_num])) or np.any(np.isnan(bv[mask_num])):
                    equal = False
                    break
            if equal:
                drops.append(float(t1))
    return sorted(set(drops))


def load_run_data(
    run_dir: str,
    npc_min: int = 10,
    baseline_auc: float = 0.5,
    drop_redundant_thresholds: bool = True,
    use_label_flip: bool = False,
) -> RunData:
    consensus_path = os.path.join(run_dir, "consensus_summary.csv")
    if not os.path.exists(consensus_path):
        raise FileNotFoundError(f"Missing {consensus_path}")
    consensus = pd.read_csv(consensus_path)
    consensus_raw = consensus.copy()

    retrain_path = os.path.join(run_dir, "consensus_retrain_summary.csv")
    retrain = pd.read_csv(retrain_path) if os.path.exists(retrain_path) else pd.DataFrame()

    if "consensus_pc_indices" in consensus.columns:
        consensus["consensus_pc_indices"] = consensus["consensus_pc_indices"].apply(_parse_idx_list)
    else:
        consensus["consensus_pc_indices"] = [[] for _ in range(len(consensus))]
    if "dropped_pc_indices" in consensus.columns:
        consensus["dropped_pc_indices"] = consensus["dropped_pc_indices"].apply(_parse_idx_list)
    else:
        consensus["dropped_pc_indices"] = [[] for _ in range(len(consensus))]

    if not retrain.empty:
        if "consensus_pc_indices" in retrain.columns:
            retrain["consensus_pc_indices"] = retrain["consensus_pc_indices"].apply(_parse_idx_list)
        if "dropped_pc_indices" in retrain.columns:
            retrain["dropped_pc_indices"] = retrain["dropped_pc_indices"].apply(_parse_idx_list)

    drops: List[float] = []
    if drop_redundant_thresholds:
        drops = _thresholds_to_drop(consensus)
        if drops:
            consensus = consensus[~consensus["threshold"].astype(float).isin(drops)].copy()
            if not retrain.empty and "source_threshold" in retrain.columns:
                retrain = retrain[~retrain["source_threshold"].astype(float).isin(drops)].copy()

    # Join retrain scores onto consensus rows by (layer,K,threshold).
    retrain_map: Dict[Tuple[int, int, float], Dict[str, float]] = {}
    if not retrain.empty:
        th_col = "source_threshold" if "source_threshold" in retrain.columns else "threshold"
        for _, r in retrain.iterrows():
            key = (int(r["layer"]), int(r["K"]), float(r[th_col]))
            retrain_map[key] = {
                "retrain_ood_test_auc": _safe_float(r.get("ood_test_auc")),
                "retrain_id_val_auc": _safe_float(r.get("id_val_auc")),
                "retrain_num_consensus_pcs": _safe_float(r.get("num_consensus_pcs")),
                "retrain_probe_path": str(r.get("probe_path", "")),
            }

    def _get_map(row, field: str):
        key = (int(row["layer"]), int(row["K"]), float(row["threshold"]))
        info = retrain_map.get(key)
        return np.nan if info is None else info.get(field, np.nan)

    for c in ["retrain_ood_test_auc", "retrain_id_val_auc", "retrain_num_consensus_pcs", "retrain_probe_path"]:
        consensus[c] = consensus.apply(lambda r: _get_map(r, c), axis=1)

    consensus["has_consensus"] = (consensus["num_consensus_pcs"] > 0) & consensus["ood_test_auc"].notna()
    consensus["small_npc"] = consensus["has_consensus"] & (consensus["num_consensus_pcs"] < npc_min)
    consensus["has_retrain"] = consensus["retrain_ood_test_auc"].notna()
    consensus["label_flip"] = (
        consensus["has_consensus"]
        & consensus["has_retrain"]
        & ((consensus["ood_test_auc"] - baseline_auc) * (consensus["retrain_ood_test_auc"] - baseline_auc) < 0)
    )
    if not use_label_flip:
        consensus["label_flip"] = False
    consensus["reliable"] = consensus["has_consensus"] & (~consensus["small_npc"]) & (~consensus["label_flip"])

    layers = sorted(int(x) for x in consensus["layer"].unique())
    k_values = sorted(int(x) for x in consensus["K"].unique())
    thresholds_kept = sorted(float(x) for x in consensus["threshold"].unique())
    thresholds_dropped = sorted(float(x) for x in drops)

    first = consensus.iloc[0]
    meta = {
        "model": str(first.get("model", "")),
        "dataset": str(first.get("dataset", "")),
        "pooling": str(first.get("pooling", "")),
        "run_dir": os.path.abspath(run_dir),
    }

    return RunData(
        consensus_raw=consensus_raw,
        consensus=consensus.reset_index(drop=True),
        retrain=retrain.reset_index(drop=True),
        layers=layers,
        k_values=k_values,
        thresholds_kept=thresholds_kept,
        thresholds_dropped=thresholds_dropped,
        meta=meta,
    )


def _pivot_matrix(df: pd.DataFrame, layers: Sequence[int], k_values: Sequence[int], value_col: str) -> np.ndarray:
    m = np.full((len(layers), len(k_values)), np.nan, dtype=float)
    row_map = {l: i for i, l in enumerate(layers)}
    col_map = {k: j for j, k in enumerate(k_values)}
    for _, r in df.iterrows():
        i = row_map[int(r["layer"])]
        j = col_map[int(r["K"])]
        m[i, j] = _safe_float(r.get(value_col))
    return m


def _annotate_matrix_cells(
    ax: plt.Axes,
    df: pd.DataFrame,
    layers: Sequence[int],
    k_values: Sequence[int],
    value_col: str,
    npc_col: str = "num_consensus_pcs",
    hatch_mask_col: str = "has_consensus",
    border_mask_col: str = "small_npc",
    flip_mask_col: str = "label_flip",
    show_npc: bool = True,
    npc_mode: str = "small",
):
    idx = {(int(r["layer"]), int(r["K"])): r for _, r in df.iterrows()}
    for i, layer in enumerate(layers):
        for j, k in enumerate(k_values):
            r = idx.get((int(layer), int(k)))
            if r is None:
                ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#d8d8d8", alpha=0.7))
                continue

            has_cons = bool(r.get(hatch_mask_col, False))
            val = _safe_float(r.get(value_col))
            npc = int(r.get(npc_col, 0)) if not np.isnan(_safe_float(r.get(npc_col))) else 0
            is_small = bool(r.get(border_mask_col, False))
            is_flip = bool(r.get(flip_mask_col, False))

            if not has_cons or np.isnan(val):
                ax.add_patch(
                    mpl.patches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#cccccc",
                        hatch="////",
                        edgecolor="#666666",
                        lw=0.0,
                        alpha=0.9,
                    )
                )
                continue

            txt_color = "black" if val > 0.42 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8.5, color=txt_color, weight="bold")
            show_npc_here = show_npc and (
                (npc_mode == "all") or (npc_mode == "small" and is_small)
            )
            if show_npc_here:
                ax.text(j, i + 0.20, f"npc={npc}", ha="center", va="center", fontsize=6.5, color="#ff9f1c")

            if is_small:
                ax.add_patch(
                    mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#ff7f0e", lw=2.0)
                )
            if is_flip:
                ax.text(j - 0.36, i - 0.30, "x", fontsize=10, color="#b22222", fontweight="bold")


def plot_ood_auc_heatmaps(data: RunData, out_path: str, vmin: float = 0.25, vmax: float = 0.7) -> None:
    thresholds = data.thresholds_kept[:2]
    if len(thresholds) == 1:
        thresholds = [thresholds[0], thresholds[0]]

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.035])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    cax = fig.add_subplot(gs[:, 2])
    cmap = plt.get_cmap("RdYlGn")

    for col, thr in enumerate(thresholds):
        d_thr = data.consensus[np.isclose(data.consensus["threshold"].astype(float), thr)].copy()
        m_org = _pivot_matrix(d_thr, data.layers, data.k_values, "ood_test_auc")
        im = axes[0, col].imshow(m_org, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        _annotate_matrix_cells(axes[0, col], d_thr, data.layers, data.k_values, "ood_test_auc")
        axes[0, col].set_title(f"threshold={thr:.1f}  ·  Original Probe", fontsize=11, weight="bold")
        axes[0, col].set_xticks(range(len(data.k_values)))
        axes[0, col].set_xticklabels([f"K={k}" for k in data.k_values], fontsize=8)
        axes[0, col].set_yticks(range(len(data.layers)))
        axes[0, col].set_yticklabels([f"L{l}" for l in data.layers], fontsize=9)
        axes[0, col].grid(color="white", lw=1.2, alpha=0.8)

        m_ret = _pivot_matrix(d_thr, data.layers, data.k_values, "retrain_ood_test_auc")
        axes[1, col].imshow(m_ret, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        _annotate_matrix_cells(
            axes[1, col],
            d_thr,
            data.layers,
            data.k_values,
            "retrain_ood_test_auc",
            npc_col="retrain_num_consensus_pcs",
            show_npc=True,
            npc_mode="small",
        )
        axes[1, col].set_title(f"threshold={thr:.1f}  ·  Retrained Probe", fontsize=11, weight="bold")
        axes[1, col].set_xticks(range(len(data.k_values)))
        axes[1, col].set_xticklabels([f"K={k}" for k in data.k_values], fontsize=8)
        axes[1, col].set_yticks(range(len(data.layers)))
        axes[1, col].set_yticklabels([f"L{l}" for l in data.layers], fontsize=9)
        axes[1, col].grid(color="white", lw=1.2, alpha=0.8)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("OOD AUC")

    dropped = (
        f" [thr={','.join(f'{t:.1f}' for t in data.thresholds_dropped)} dropped]"
        if data.thresholds_dropped
        else ""
    )
    fig.suptitle(
        "OOD AUC Heatmap (Layer × K)  ·  CORRECTED"
        + dropped
        + "\n"
        + f"pooling={data.meta.get('pooling')} · dataset={data.meta.get('dataset')} · npc<10 orange border",
        fontsize=12,
        weight="bold",
    )
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _layer_color(layer: int) -> str:
    colors = {
        8: "#e53935",
        9: "#fb8c00",
        10: "#43a047",
        11: "#1e88e5",
        12: "#8e24aa",
    }
    return colors.get(int(layer), "#546e7a")


def plot_top10_ranking(data: RunData, out_path: str, top_n: int = 10, baseline_auc: float = 0.5) -> None:
    df = data.consensus.copy()
    candidates = df.sort_values("ood_test_auc", ascending=False).reset_index(drop=True)
    reliable_desc = candidates[candidates["reliable"]].copy().head(top_n)
    reliable_desc["rank"] = np.arange(1, len(reliable_desc) + 1)
    reliable = reliable_desc.sort_values("ood_test_auc", ascending=True).reset_index(drop=True)
    excluded = candidates[(~candidates["reliable"])].head(12)

    fig, (ax, ax_right) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [3.6, 1.4]}, constrained_layout=True
    )
    if reliable.empty:
        ax.text(0.5, 0.5, "No reliable configs after filters", ha="center", va="center", fontsize=13)
        ax.axis("off")
        ax_right.axis("off")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    reliable = reliable.sort_values("ood_test_auc", ascending=True).reset_index(drop=True)
    y = np.arange(len(reliable))
    colors = [_layer_color(int(l)) for l in reliable["layer"]]
    bars = ax.barh(y, reliable["ood_test_auc"], color=colors, alpha=0.90, height=0.68, label="Original")

    has_rt = reliable["retrain_ood_test_auc"].notna().to_numpy()
    if has_rt.any():
        ax.scatter(
            reliable.loc[has_rt, "retrain_ood_test_auc"],
            y[has_rt],
            marker="s",
            s=58,
            facecolors="none",
            edgecolors="#616161",
            linewidths=1.8,
            label="Retrained",
            zorder=5,
        )

    for yi, b, (_, r) in zip(y, bars, reliable.iterrows()):
        ax.text(float(r["ood_test_auc"]) + 0.006, yi, f"{float(r['ood_test_auc']):.3f}", va="center", ha="left", fontsize=10, weight="bold")
        ax.text(
            max(0.11, float(r["ood_test_auc"]) - 0.09),
            yi,
            f"ratio={100.0*float(r['consensus_ratio']):.1f}%  avg={int(round(float(r['avg_selected_per_probe'])))}",
            va="center",
            ha="left",
            fontsize=8.2,
            color="#37474f",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.12"),
        )

    ylabels = [
        f"#{int(r.rank)} L{int(r.layer)} K={int(r.K)} t={float(r.threshold):.1f} npc={int(r.num_consensus_pcs)}"
        for r in reliable.itertuples(index=False)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.axvline(baseline_auc, ls="--", lw=1.6, color="#555555")
    ax.set_ylabel("Ranked configuration", fontsize=12)
    ax.set_xlabel("OOD AUC", fontsize=13)
    ax.set_xlim(0.10, max(0.75, float(np.nanmax(reliable["ood_test_auc"]) + 0.09)))
    ax.grid(axis="x", alpha=0.22)
    ax.set_title(
        f"Top-{top_n} Configurations by OOD AUC  ·  CORRECTED",
        fontsize=16,
        weight="bold",
    )

    legend_elems = [
        mpl.patches.Patch(facecolor=_layer_color(8), label="Layer 8"),
        mpl.patches.Patch(facecolor=_layer_color(9), label="Layer 9"),
        mpl.patches.Patch(facecolor=_layer_color(10), label="Layer 10"),
        mpl.patches.Patch(facecolor=_layer_color(11), label="Layer 11"),
        mpl.patches.Patch(facecolor=_layer_color(12), label="Layer 12"),
        mpl.patches.Patch(facecolor="#9e9e9e", edgecolor="#9e9e9e", label="Original (solid bar)"),
        mpl.patches.Patch(facecolor="none", edgecolor="#9e9e9e", linewidth=2, label="Retrained (outline bar)"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="lower right", ncol=2, framealpha=0.95)

    ax_right.axis("off")
    ax_right.set_title("Excluded From Ranking", fontsize=12, weight="bold")
    if not excluded.empty:
        lines = []
        for r in excluded.itertuples(index=False):
            reasons = []
            if bool(r.small_npc):
                reasons.append("unreliable_small")
            if bool(r.label_flip):
                reasons.append("label_flip")
            reason = "+".join(reasons) if reasons else "filtered"
            lines.append(
                f"✗ L{int(r.layer)} K={int(r.K)} thr={float(r.threshold):.1f} OOD={float(r.ood_test_auc):.3f} [{reason}]"
            )
        ax_right.text(
            0.0,
            1.0,
            "\n".join(lines[:12]),
            transform=ax_right.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="#fff8e1", edgecolor="#ff7f0e", boxstyle="round,pad=0.45"),
        )

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _heatmap_panel(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    layers: Sequence[int],
    k_values: Sequence[int],
    cmap: str = "YlOrRd",
    fmt: str = "{:.0f}",
    pct: bool = False,
):
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"K={k}" for k in k_values], fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                txt = "na"
            else:
                txt = f"{v:.1f}%" if pct else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    ax.grid(color="white", lw=1.0, alpha=0.8)
    return im


def plot_sparsity_analysis(data: RunData, out_path: str) -> None:
    ths = sorted(data.thresholds_kept)
    t0 = ths[0]
    t1 = ths[min(1, len(ths) - 1)]
    d0 = data.consensus[np.isclose(data.consensus["threshold"], t0)].copy()
    d1 = data.consensus[np.isclose(data.consensus["threshold"], t1)].copy()

    def _mat(df: pd.DataFrame, col: str):
        return _pivot_matrix(df, data.layers, data.k_values, col)

    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)

    def _add_small_npc_border(ax: plt.Axes, sub_df: pd.DataFrame):
        row_map = {l: i for i, l in enumerate(data.layers)}
        col_map = {k: j for j, k in enumerate(data.k_values)}
        for _, rr in sub_df[sub_df["small_npc"]].iterrows():
            i = row_map[int(rr["layer"])]
            j = col_map[int(rr["K"])]
            ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#ff7f0e", lw=2.0))

    im = _heatmap_panel(axes[0, 0], _mat(d0, "num_consensus_pcs"), f"thr={t0:.1f} · # Consensus PCs", data.layers, data.k_values)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    _add_small_npc_border(axes[0, 0], d0)

    im = _heatmap_panel(axes[0, 1], 100.0 * _mat(d0, "consensus_ratio"), f"thr={t0:.1f} · Consensus Ratio (%)", data.layers, data.k_values, pct=True)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    _add_small_npc_border(axes[0, 1], d0)

    im = _heatmap_panel(
        axes[0, 2],
        100.0 * (_mat(d0, "avg_selected_per_probe") / np.asarray(data.k_values)[None, :]),
        f"thr={t0:.1f} · Probe Sel. Density (avg/K)",
        data.layers,
        data.k_values,
        cmap="Purples",
        pct=True,
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    _add_small_npc_border(axes[0, 2], d0)

    im = _heatmap_panel(axes[1, 0], _mat(d1, "num_consensus_pcs"), f"thr={t1:.1f} · # Consensus PCs", data.layers, data.k_values)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    _add_small_npc_border(axes[1, 0], d1)

    im = _heatmap_panel(axes[1, 1], 100.0 * _mat(d1, "consensus_ratio"), f"thr={t1:.1f} · Consensus Ratio (%)", data.layers, data.k_values, pct=True)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    _add_small_npc_border(axes[1, 1], d1)

    im = _heatmap_panel(
        axes[1, 2],
        100.0 * (_mat(d1, "avg_selected_per_probe") / np.asarray(data.k_values)[None, :]),
        f"thr={t1:.1f} · Probe Sel. Density (avg/K)",
        data.layers,
        data.k_values,
        cmap="Purples",
        pct=True,
    )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    _add_small_npc_border(axes[1, 2], d1)

    # Shared stats on bottom row.
    avg_sel_mat = _mat(d0, "avg_selected_per_probe")
    im = _heatmap_panel(axes[2, 0], avg_sel_mat, f"Avg PCs/Probe (thr={t0:.1f}, shared)", data.layers, data.k_values, cmap="Blues")
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)

    count_thresh = np.zeros((len(data.layers), len(data.k_values)))
    reliable_count = np.zeros((len(data.layers), len(data.k_values)))
    for i, l in enumerate(data.layers):
        for j, k in enumerate(data.k_values):
            sub = data.consensus[(data.consensus["layer"] == l) & (data.consensus["K"] == k)]
            count_thresh[i, j] = float((sub["num_consensus_pcs"] > 0).sum())
            reliable_count[i, j] = float(sub["reliable"].sum())
    im = _heatmap_panel(axes[2, 1], count_thresh, "# Thresholds Yielding Consensus", data.layers, data.k_values, cmap="YlGn")
    fig.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)
    im = _heatmap_panel(axes[2, 2], reliable_count, "# Reliable Configs (npc>=10)", data.layers, data.k_values, cmap="BuGn")
    fig.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Sparsity Analysis  ·  CORRECTED\nGold=valid consensus · Orange border=npc<10 (unreliable)",
        fontsize=12,
        weight="bold",
    )
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _top_reliable_rows(data: RunData, top_n: int = 8) -> pd.DataFrame:
    df = data.consensus[data.consensus["reliable"]].sort_values("ood_test_auc", ascending=False).head(top_n).copy()
    return df


def plot_selected_pcs_strips(data: RunData, out_path: str, top_n: int = 8) -> None:
    top = _top_reliable_rows(data, top_n=top_n)
    if top.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No reliable configs", ha="center", va="center")
        ax.axis("off")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(top), 1, figsize=(18, 1.35 * len(top)), sharex=False)
    if len(top) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, top.iterrows()):
        K = int(r["K"])
        sel = set(int(x) for x in r["consensus_pc_indices"])
        arr = np.zeros((1, K))
        for i in range(K):
            arr[0, i] = 1.0 if i in sel else 0.0
        ax.imshow(arr, aspect="auto", cmap=mpl.colors.ListedColormap(["#cfcfcf", _layer_color(int(r["layer"]))]))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(f"L{int(r['layer'])}  K={K}  thr={float(r['threshold']):.1f}  npc={int(r['num_consensus_pcs'])}", fontsize=12, rotation=0, ha="right", va="center", labelpad=38)
        label = f"OOD AUC={float(r['ood_test_auc']):.3f}  |  {int(r['num_consensus_pcs'])}/{K} ({100.0*float(r['consensus_ratio']):.1f}%)"
        ax.text(
            0.99,
            0.5,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor=_layer_color(int(r["layer"])), alpha=0.85, boxstyle="round,pad=0.15"),
        )
    fig.suptitle(
        f"Selected Consensus PC Indices  — Top-{len(top)} Reliable Configs\nColour = selected  ·  Grey = dropped",
        fontsize=18,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.92])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_pcs_density(data: RunData, out_path: str, top_n: int = 8) -> None:
    top = _top_reliable_rows(data, top_n=top_n)
    if top.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No reliable configs", ha="center", va="center")
        ax.axis("off")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    fig, flat_axes = _grid_axes(len(top), ncols=4, figsize_scale=(4.4, 3.3))
    for ax, (_, r) in zip(flat_axes, top.iterrows()):
        K = int(r["K"])
        sel = np.asarray([int(x) for x in r["consensus_pc_indices"]], dtype=float)
        norm = sel / float(max(K - 1, 1))
        ax.hist(norm, bins=20, color=_layer_color(int(r["layer"])), alpha=0.75, edgecolor="white")
        mu = float(np.mean(norm)) if norm.size else np.nan
        med = float(np.median(norm)) if norm.size else np.nan
        ax.axvline(mu, color="black", ls="--", lw=2, label=f"μ={mu:.2f}" if not np.isnan(mu) else "μ=na")
        ax.axvline(med, color="gray", ls=":", lw=2, label=f"med={med:.2f}" if not np.isnan(med) else "med=na")
        ax.set_xlim(0, 1)
        ax.set_title(
            f"L{int(r['layer'])}  K={K}  thr={float(r['threshold']):.1f}  npc={int(r['num_consensus_pcs'])}\nOOD={float(r['ood_test_auc']):.3f}",
            fontsize=12,
            weight="bold",
        )
        ax.set_xlabel("Norm. PC idx", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Consensus PC Position Density  (normalised idx / K)  ·  Top-{len(top)} Reliable Configs",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_deltas_contributions(data: RunData, out_path: str, baseline_auc: float = 0.5) -> None:
    df = data.consensus.copy()
    # Focus on strongest original OOD rows for readability.
    top = df.sort_values("ood_test_auc", ascending=False).head(14).copy()
    top["cfg"] = top.apply(lambda r: f"L{int(r.layer)}\nK={int(r.K)}", axis=1)
    top["delta_rt"] = top["retrain_ood_test_auc"] - top["ood_test_auc"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # A: ID vs OOD + retrain overlay for top rows
    ax = axes[0, 0]
    x = np.arange(len(top))
    w = 0.30
    ax.bar(x - w / 2, top["id_val_auc"], width=w, alpha=0.8, color="#90caf9", label="ID val")
    ax.bar(x + w / 2, top["ood_test_auc"], width=w, alpha=0.85, color="#66bb6a", label="OOD test")
    has_rt = top["retrain_ood_test_auc"].notna().to_numpy()
    if has_rt.any():
        ax.bar(x[has_rt] + w * 1.3, top.loc[has_rt, "retrain_ood_test_auc"], width=w * 0.8, alpha=0.8, color="#ffcc80", label="OOD retrain")
    for i, r in top.reset_index(drop=True).iterrows():
        if bool(r["small_npc"]):
            ax.text(i, max(r["ood_test_auc"], 0.2) + 0.015, "npc<10", color="#ff7f0e", fontsize=7, ha="center")
        if bool(r["label_flip"]):
            ax.text(i - 0.15, max(r["ood_test_auc"], 0.2) + 0.03, "xFLIP", color="#b22222", fontsize=8, ha="center", weight="bold")
    ax.axhline(baseline_auc, color="#555555", ls="--", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(top["cfg"], fontsize=8)
    ax.set_ylim(0.10, max(0.85, float(np.nanmax(top[["id_val_auc", "ood_test_auc"]].to_numpy()) + 0.06)))
    ax.set_title("A · ID Val vs OOD Test", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # B: Retrain deltas sorted
    ax = axes[0, 1]
    rt = df[df["retrain_ood_test_auc"].notna()].copy()
    rt["delta_rt"] = rt["retrain_ood_test_auc"] - rt["ood_test_auc"]
    rt = rt.sort_values("delta_rt", ascending=True).head(14)
    xx = np.arange(len(rt))
    colors = np.where(rt["delta_rt"] >= 0, "#43a047", "#d32f2f")
    ax.bar(xx, rt["delta_rt"], color=colors, alpha=0.9)
    for i, r in rt.reset_index(drop=True).iterrows():
        lbl = f"L{int(r.layer)}\nK={int(r.K)}"
        ax.text(i, -0.02 if r["delta_rt"] >= 0 else r["delta_rt"] - 0.02, lbl, fontsize=8, ha="center", va="top")
    ax.axhline(0.0, color="#555555", lw=1.4)
    ax.set_title("B · Retrain Δ (OOD AUC retrain - original)", fontsize=11, weight="bold")
    ax.set_ylabel("Δ OOD AUC")
    ax.grid(axis="y", alpha=0.25)

    # C: Sparsity vs OOD with flags
    ax = axes[1, 0]
    for layer, sub in df.groupby("layer"):
        ax.scatter(100.0 * sub["consensus_ratio"], sub["ood_test_auc"], s=48, color=_layer_color(int(layer)), alpha=0.8, label=f"L{int(layer)}")
    small = df[df["small_npc"]]
    if not small.empty:
        ax.scatter(100.0 * small["consensus_ratio"], small["ood_test_auc"], marker="^", s=60, facecolors="none", edgecolors="#ff7f0e", linewidths=1.8, label="npc<10")
    flips = df[df["label_flip"]]
    if not flips.empty:
        ax.scatter(100.0 * flips["consensus_ratio"], flips["ood_test_auc"], marker="x", s=50, c="#b22222", label="label flip")
    ax.axhline(baseline_auc, color="#555555", ls="--", lw=1.2)
    ax.set_xlabel("Consensus Ratio (%)")
    ax.set_ylabel("OOD AUC")
    ax.set_title("C · Sparsity vs OOD AUC", fontsize=11, weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    # D: per-layer distributions and medians
    ax = axes[1, 1]
    x0 = np.arange(len(data.layers))
    for i, layer in enumerate(data.layers):
        sub = df[df["layer"] == layer]
        ax.scatter(np.full(len(sub), i), sub["ood_test_auc"], color=_layer_color(layer), alpha=0.55)
        if len(sub) > 0:
            vals = sub["ood_test_auc"].dropna().to_numpy()
            if len(vals) > 0:
                med = float(np.median(vals))
                ax.plot([i - 0.25, i + 0.25], [med, med], color=_layer_color(layer), lw=2.4)
    ax.axhline(baseline_auc, color="#555555", ls="--", lw=1.2)
    ax.set_xticks(x0)
    ax.set_xticklabels([f"Layer {l}" for l in data.layers])
    ax.set_ylabel("OOD AUC")
    ax.set_title("D · OOD AUC per Layer", fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=0.25)

    dropped = (
        f" [thr={','.join(f'{t:.1f}' for t in data.thresholds_dropped)} dropped]"
        if data.thresholds_dropped
        else ""
    )
    fig.suptitle(
        "Contributions & Deltas  ·  CORRECTED" + dropped,
        fontsize=13,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layer_k_profiles(data: RunData, out_path: str, baseline_auc: float = 0.5) -> None:
    df = data.consensus.copy()
    ths = sorted(data.thresholds_kept)
    t0 = ths[0]
    t1 = ths[min(1, len(ths) - 1)]
    d0 = df[np.isclose(df["threshold"], t0)].copy()
    d1 = df[np.isclose(df["threshold"], t1)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # A: Avg per-probe selections vs K per layer
    ax = axes[0, 0]
    for layer, sub in d0.groupby("layer"):
        sub = sub.sort_values("K")
        ax.plot(sub["K"], sub["avg_selected_per_probe"], marker="o", label=f"Layer {int(layer)}", color=_layer_color(int(layer)))
    ax.set_title(f"A · Avg Per-Probe Selection vs K (thr={t0:.1f})", fontsize=11, weight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Avg sel/probe")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # B: Consensus ratio vs layer (solid t0, dashed t1)
    ax = axes[0, 1]
    for K, sub in d0.groupby("K"):
        sub = sub.sort_values("layer")
        ax.plot(sub["layer"], sub["consensus_ratio"], marker="o", label=f"K={int(K)}", lw=1.7)
    for K, sub in d1.groupby("K"):
        sub = sub.sort_values("layer")
        ax.plot(sub["layer"], sub["consensus_ratio"], marker="o", ls="--", alpha=0.55, lw=1.2)
    ax.set_title("B · Consensus Ratio vs Layer", fontsize=11, weight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Consensus ratio")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    # C: OOD AUC vs #consensus PCs
    ax = axes[1, 0]
    for layer, sub in df.groupby("layer"):
        ax.scatter(sub["num_consensus_pcs"], sub["ood_test_auc"], color=_layer_color(int(layer)), label=f"L{int(layer)}", alpha=0.85)
    ax.axhline(baseline_auc, color="#555555", ls="--", lw=1.2)
    ax.set_title("C · OOD AUC vs # Consensus PCs", fontsize=11, weight="bold")
    ax.set_xlabel("# Consensus PCs")
    ax.set_ylabel("OOD AUC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # D: Mean OOD AUC vs K with std bars
    ax = axes[1, 1]
    for thr, sub in [(t0, d0), (t1, d1)]:
        stats = sub[sub["reliable"]].groupby("K")["ood_test_auc"].agg(["mean", "std"]).reset_index()
        if stats.empty:
            continue
        ax.errorbar(
            stats["K"],
            stats["mean"],
            yerr=stats["std"].fillna(0.0),
            marker="o",
            capsize=3,
            lw=1.8,
            ls="-" if thr == t0 else "--",
            label=f"thr={thr:.1f} (n={len(sub[sub['reliable']])})",
        )
    ax.axhline(baseline_auc, color="#777777", ls="--", lw=1.1, label="Random")
    ax.set_title("D · Mean OOD AUC vs K (reliable only)", fontsize=11, weight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("OOD AUC (mean ± std)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Layer & K Profiles  ·  CORRECTED", fontsize=13, weight="bold")
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(data: RunData, out_path: str, baseline_auc: float = 0.5) -> None:
    """Compact dashboard mirroring key diagnostics in one canvas."""
    df = data.consensus.copy()
    top = _top_reliable_rows(data, top_n=6)
    ths = sorted(data.thresholds_kept)
    t0 = ths[0]
    t1 = ths[min(1, len(ths) - 1)]
    d0 = df[np.isclose(df["threshold"], t0)].copy()
    d1 = df[np.isclose(df["threshold"], t1)].copy()

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 6, height_ratios=[1.1, 1.1, 1.1, 1.4], hspace=0.5, wspace=0.45)

    # Mini heatmaps
    ax = fig.add_subplot(gs[0, 0:2])
    m = _pivot_matrix(d0, data.layers, data.k_values, "ood_test_auc")
    im = ax.imshow(m, cmap="RdYlGn", vmin=0.25, vmax=0.7, aspect="auto")
    _annotate_matrix_cells(ax, d0, data.layers, data.k_values, "ood_test_auc", show_npc=False)
    ax.set_title(f"OOD AUC · thr={t0:.1f}", fontsize=10, weight="bold")
    ax.set_xticks(range(len(data.k_values)))
    ax.set_xticklabels(data.k_values, fontsize=8)
    ax.set_yticks(range(len(data.layers)))
    ax.set_yticklabels([f"L{l}" for l in data.layers], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 2:4])
    m = _pivot_matrix(d1, data.layers, data.k_values, "ood_test_auc")
    im = ax.imshow(m, cmap="RdYlGn", vmin=0.25, vmax=0.7, aspect="auto")
    _annotate_matrix_cells(ax, d1, data.layers, data.k_values, "ood_test_auc", show_npc=False)
    ax.set_title(f"OOD AUC · thr={t1:.1f}", fontsize=10, weight="bold")
    ax.set_xticks(range(len(data.k_values)))
    ax.set_xticklabels(data.k_values, fontsize=8)
    ax.set_yticks(range(len(data.layers)))
    ax.set_yticklabels([f"L{l}" for l in data.layers], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 4:5])
    top_rel = top.sort_values("ood_test_auc", ascending=False)
    ax.barh(np.arange(len(top_rel)), top_rel["ood_test_auc"], color=[_layer_color(int(x)) for x in top_rel["layer"]])
    ax.axvline(baseline_auc, ls="--", color="#666666")
    ax.set_yticks(np.arange(len(top_rel)))
    ax.set_yticklabels([f"L{int(r.layer)} K={int(r.K)}" for r in top_rel.itertuples(index=False)], fontsize=7)
    ax.set_title("Top Reliable", fontsize=9, weight="bold")

    # Delta bars
    ax = fig.add_subplot(gs[1, 0:2])
    rt = df[df["retrain_ood_test_auc"].notna()].copy()
    rt["delta"] = rt["retrain_ood_test_auc"] - rt["ood_test_auc"]
    rt = rt.sort_values("delta").head(10)
    ax.bar(np.arange(len(rt)), rt["delta"], color=np.where(rt["delta"] >= 0, "#43a047", "#d32f2f"))
    ax.axhline(0, color="#444444", lw=1.2)
    ax.set_xticks(np.arange(len(rt)))
    ax.set_xticklabels([f"L{int(r.layer)}\nK={int(r.K)}" for r in rt.itertuples(index=False)], fontsize=7)
    ax.set_title("Δ OOD retrain-orig (top 10 by delta)", fontsize=9, weight="bold")
    ax.grid(axis="y", alpha=0.25)

    # Scatter sparsity vs OOD
    ax = fig.add_subplot(gs[1, 2:4])
    for layer, sub in df.groupby("layer"):
        ax.scatter(100.0 * sub["consensus_ratio"], sub["ood_test_auc"], s=24, alpha=0.8, color=_layer_color(layer), label=f"L{layer}")
    ax.axhline(baseline_auc, ls="--", color="#555555")
    ax.set_title("Sparsity vs OOD", fontsize=9, weight="bold")
    ax.set_xlabel("Consensus ratio (%)", fontsize=8)
    ax.set_ylabel("OOD AUC", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    # Curves
    ax = fig.add_subplot(gs[1, 4:6])
    for K, sub in d1.groupby("K"):
        sub = sub.sort_values("layer")
        ax.plot(sub["layer"], sub["consensus_ratio"], marker="o", lw=1.4, label=f"K={int(K)}")
    ax.set_title(f"Consensus ratio vs layer (thr={t1:.1f})", fontsize=9, weight="bold")
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("ratio", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    # Strips for top reliable
    ax = fig.add_subplot(gs[2:, :])
    if top.empty:
        ax.text(0.5, 0.5, "No reliable configs", ha="center", va="center")
        ax.axis("off")
    else:
        y = 0
        ytick = []
        ylbl = []
        for _, r in top.iterrows():
            K = int(r["K"])
            sel = set(int(x) for x in r["consensus_pc_indices"])
            xs = np.arange(K)
            ys = np.full(K, y)
            ax.scatter(xs, ys, s=8, c=["#2e7d32" if x in sel else "#bdbdbd" for x in xs], marker="s", linewidths=0)
            ytick.append(y)
            ylbl.append(f"L{int(r['layer'])} K={K} thr={float(r['threshold']):.1f}")
            y += 1
        ax.set_yticks(ytick)
        ax.set_yticklabels(ylbl, fontsize=8)
        ax.set_xlabel("PC index")
        ax.set_xlim(-1, max(data.k_values) + 1)
        ax.set_title("PC Selection Strips — Top Reliable Configs", fontsize=10, weight="bold")
        ax.grid(axis="x", alpha=0.18)

    dropped = (
        f" [thr={','.join(f'{t:.1f}' for t in data.thresholds_dropped)} dropped]"
        if data.thresholds_dropped
        else ""
    )
    fig.suptitle("PCA Consensus Probe · OOD Summary · CORRECTED" + dropped, fontsize=14, weight="bold")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    run_dir: str,
    out_dir: str,
    npc_min: int = 10,
    baseline_auc: float = 0.5,
    top_n: int = 10,
    use_label_flip: bool = False,
) -> Dict[str, str]:
    _ensure_dir(out_dir)
    data = load_run_data(
        run_dir,
        npc_min=npc_min,
        baseline_auc=baseline_auc,
        drop_redundant_thresholds=True,
        use_label_flip=use_label_flip,
    )

    outputs = {
        "01_ood_auc_heatmaps_corrected.png": os.path.join(out_dir, "01_ood_auc_heatmaps_corrected.png"),
        "02_top10_ranking_corrected.png": os.path.join(out_dir, "02_top10_ranking_corrected.png"),
        "03_sparsity_analysis_corrected.png": os.path.join(out_dir, "03_sparsity_analysis_corrected.png"),
        "04a_selected_pcs_strips_corrected.png": os.path.join(out_dir, "04a_selected_pcs_strips_corrected.png"),
        "04b_selected_pcs_density_corrected.png": os.path.join(out_dir, "04b_selected_pcs_density_corrected.png"),
        "05_deltas_contributions_corrected.png": os.path.join(out_dir, "05_deltas_contributions_corrected.png"),
        "06_layer_k_profiles_corrected.png": os.path.join(out_dir, "06_layer_k_profiles_corrected.png"),
        "07_summary_dashboard_corrected.png": os.path.join(out_dir, "07_summary_dashboard_corrected.png"),
    }

    plot_ood_auc_heatmaps(data, outputs["01_ood_auc_heatmaps_corrected.png"])
    plot_top10_ranking(data, outputs["02_top10_ranking_corrected.png"], top_n=top_n, baseline_auc=baseline_auc)
    plot_sparsity_analysis(data, outputs["03_sparsity_analysis_corrected.png"])
    plot_selected_pcs_strips(data, outputs["04a_selected_pcs_strips_corrected.png"], top_n=8)
    plot_selected_pcs_density(data, outputs["04b_selected_pcs_density_corrected.png"], top_n=8)
    plot_deltas_contributions(data, outputs["05_deltas_contributions_corrected.png"], baseline_auc=baseline_auc)
    plot_layer_k_profiles(data, outputs["06_layer_k_profiles_corrected.png"], baseline_auc=baseline_auc)
    plot_summary_dashboard(data, outputs["07_summary_dashboard_corrected.png"], baseline_auc=baseline_auc)

    manifest = {
        "run_dir": os.path.abspath(run_dir),
        "out_dir": os.path.abspath(out_dir),
        "model": data.meta.get("model"),
        "dataset": data.meta.get("dataset"),
        "pooling": data.meta.get("pooling"),
        "thresholds_kept": data.thresholds_kept,
        "thresholds_dropped": data.thresholds_dropped,
        "npc_min": int(npc_min),
        "baseline_auc": float(baseline_auc),
        "top_n_ranking": int(top_n),
        "use_label_flip": bool(use_label_flip),
        "outputs": outputs,
    }
    manifest_path = os.path.join(out_dir, "viz_manifest_corrected.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    outputs["viz_manifest_corrected.json"] = manifest_path
    return outputs


def main() -> int:
    p = argparse.ArgumentParser(description="Generate corrected PCA-consensus visualization bundle")
    p.add_argument("--run_dir", type=str, required=True, help="Directory with consensus_summary.csv")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for png bundle")
    p.add_argument("--npc_min", type=int, default=10, help="Minimum #consensus PCs to be considered reliable")
    p.add_argument("--baseline_auc", type=float, default=0.5, help="Random baseline AUC")
    p.add_argument("--top_n", type=int, default=10, help="Top-N for ranking chart")
    p.add_argument(
        "--use_label_flip",
        action="store_true",
        help="Enable label-flip filtering/marking (disabled by default).",
    )
    args = p.parse_args()

    outputs = generate_all_figures(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        npc_min=args.npc_min,
        baseline_auc=args.baseline_auc,
        top_n=args.top_n,
        use_label_flip=args.use_label_flip,
    )
    print("Generated files:")
    for _, v in outputs.items():
        print(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
