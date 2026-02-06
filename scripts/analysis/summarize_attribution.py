#!/usr/bin/env python3
"""
Create publication-style attribution dashboards from probe attribution outputs.

Inputs (default):
  results/probe_attribution/<model>/<dataset>/<pooling>/
  results/layer_agnostic_attribution/<model>/<dataset>/<pooling>/
  results/probes_layer_agnostic/<model>/<dataset>/<pooling>/layer_results.json (optional)

Outputs:
  results/attribution_summary/<model>/<dataset>/
    pooled/<pooling>/tables/*.csv
    pooled/<pooling>/figures/*.(png|pdf)
    layer_agnostic/<pooling>/tables/*.csv
    layer_agnostic/<pooling>/figures/*.(png|pdf)
    combined_summary.csv
    missing_report.csv
    figure_manifest.csv
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
ID_COLOR = "#4FA3C7"
OOD_COLOR = "#B04A8B"
POS_COLOR = "#1FA77A"
NEG_COLOR = "#E45756"
GRID_COLOR = "#D9D9D9"
NEUTRAL_COLOR = "#8C8C8C"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.4,
            "lines.markersize": 5,
        }
    )
    if sns is not None:
        sns.set_style("whitegrid")


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if df is None:
        return
    df.to_csv(path, index=False)


def parse_layer_from_filename(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"_layer_(\d+)\.csv$", name)
    if not m:
        return None
    return int(m.group(1))


def infer_behavior(sample_id: str) -> str:
    s = str(sample_id).lower()
    if "deceptive" in s:
        return "deceptive"
    if "honest" in s:
        return "honest"
    return "unknown"


# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
def collect_pooled_tables(base_dir: str, strict_schema: bool) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, str]]]:
    missing = []
    tables: Dict[str, pd.DataFrame] = {}

    ckpt_files = sorted(glob.glob(os.path.join(base_dir, "checkpoint_metrics_layer_*.csv")))
    dyn_files = sorted(glob.glob(os.path.join(base_dir, "training_dynamics_layer_*.csv")))
    inf_files = sorted(glob.glob(os.path.join(base_dir, "layer_influence_layer_*.csv")))
    prog_files = sorted(glob.glob(os.path.join(base_dir, "layer_progress_layer_*.csv")))
    sinf_files = sorted(glob.glob(os.path.join(base_dir, "sample_influence_top*_layer_*.csv")))
    sprog_files = sorted(glob.glob(os.path.join(base_dir, "sample_progress_top*_layer_*.csv")))

    if not ckpt_files:
        missing.append({"scope": "pooled", "kind": "checkpoint_metrics", "path": os.path.join(base_dir, "checkpoint_metrics_layer_*.csv")})
    if not dyn_files:
        missing.append({"scope": "pooled", "kind": "training_dynamics", "path": os.path.join(base_dir, "training_dynamics_layer_*.csv")})

    def load_layered(files: List[str], kind: str) -> pd.DataFrame:
        rows = []
        for p in files:
            try:
                df = pd.read_csv(p)
            except Exception as e:
                missing.append({"scope": "pooled", "kind": f"{kind}_read_error", "path": p, "message": str(e)})
                continue
            layer = parse_layer_from_filename(p)
            if layer is None:
                missing.append({"scope": "pooled", "kind": f"{kind}_layer_parse_error", "path": p})
                continue
            df["layer"] = layer
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    tables["checkpoint_metrics_long"] = load_layered(ckpt_files, "checkpoint_metrics")
    tables["training_dynamics_long"] = load_layered(dyn_files, "training_dynamics")
    tables["layer_influence_long"] = load_layered(inf_files, "layer_influence")
    tables["layer_progress_long"] = load_layered(prog_files, "layer_progress")
    tables["sample_influence_long"] = load_layered(sinf_files, "sample_influence")
    tables["sample_progress_long"] = load_layered(sprog_files, "sample_progress")

    if not tables["sample_influence_long"].empty and "sample_id" in tables["sample_influence_long"].columns:
        tables["sample_influence_long"]["behavior_label"] = tables["sample_influence_long"]["sample_id"].map(infer_behavior)
    if not tables["sample_progress_long"].empty and "sample_id" in tables["sample_progress_long"].columns:
        tables["sample_progress_long"]["behavior_label"] = tables["sample_progress_long"]["sample_id"].map(infer_behavior)

    if strict_schema:
        required = {
            "checkpoint_metrics_long": {"epoch", "split", "auc", "acc", "best_so_far", "layer"},
            "training_dynamics_long": {"epoch", "cos_to_final", "w_norm", "layer"},
        }
        for key, cols in required.items():
            df = tables.get(key, pd.DataFrame())
            if df.empty:
                raise ValueError(f"Missing required table: {key} in {base_dir}")
            missing_cols = cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Table {key} missing columns: {sorted(missing_cols)}")

    return tables, missing


def collect_layer_agnostic_tables(base_dir: str) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, str]]]:
    missing = []
    tables: Dict[str, pd.DataFrame] = {}

    def load_if_exists(name: str) -> pd.DataFrame:
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            missing.append({"scope": "layer_agnostic", "kind": name, "path": path})
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as e:
            missing.append({"scope": "layer_agnostic", "kind": f"{name}_read_error", "path": path, "message": str(e)})
            return pd.DataFrame()

    tables["checkpoint_metrics_long"] = load_if_exists("checkpoint_metrics.csv")
    tables["training_dynamics_long"] = load_if_exists("training_dynamics.csv")
    tables["layer_influence_long"] = load_if_exists("layer_influence.csv")
    tables["layer_progress_long"] = load_if_exists("layer_progress.csv")

    sample_inf = glob.glob(os.path.join(base_dir, "sample_influence_top*.csv"))
    sample_prog = glob.glob(os.path.join(base_dir, "sample_progress_top*.csv"))

    if sample_inf:
        tables["sample_influence_long"] = pd.read_csv(sample_inf[0])
        if "sample_id" in tables["sample_influence_long"].columns:
            tables["sample_influence_long"]["behavior_label"] = tables["sample_influence_long"]["sample_id"].map(infer_behavior)
    else:
        tables["sample_influence_long"] = pd.DataFrame()
        missing.append({"scope": "layer_agnostic", "kind": "sample_influence", "path": os.path.join(base_dir, "sample_influence_top*.csv")})

    if sample_prog:
        tables["sample_progress_long"] = pd.read_csv(sample_prog[0])
        if "sample_id" in tables["sample_progress_long"].columns:
            tables["sample_progress_long"]["behavior_label"] = tables["sample_progress_long"]["sample_id"].map(infer_behavior)
    else:
        tables["sample_progress_long"] = pd.DataFrame()
        missing.append({"scope": "layer_agnostic", "kind": "sample_progress", "path": os.path.join(base_dir, "sample_progress_top*.csv")})

    return tables, missing


# -----------------------------------------------------------------------------
# Derived metrics
# -----------------------------------------------------------------------------
def _safe_std(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=0))


def _safe_cv(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(arr, ddof=0) / abs(mean))


def _convergence_rate(dyn: pd.DataFrame) -> float:
    if dyn.empty:
        return 0.0
    d = dyn.sort_values("epoch")
    if len(d) < 2:
        return 0.0
    first = d.iloc[0]
    k = min(5, len(d) - 1)
    kth = d.iloc[k]
    de = float(kth["epoch"] - first["epoch"])
    if de <= 0:
        return 0.0
    return float((kth["cos_to_final"] - first["cos_to_final"]) / de)


def compute_derived_layer_metrics(ckpt: pd.DataFrame, dyn: pd.DataFrame) -> pd.DataFrame:
    if ckpt.empty:
        return pd.DataFrame()

    metrics = []
    for layer, g in ckpt.groupby("layer"):
        g = g.sort_values("epoch")
        id_rows = g[g["split"] == "id"].sort_values("epoch")
        ood_rows = g[g["split"] == "ood"].sort_values("epoch")

        final_id_auc = float(id_rows.iloc[-1]["auc"]) if not id_rows.empty else np.nan
        final_ood_auc = float(ood_rows.iloc[-1]["auc"]) if not ood_rows.empty else np.nan
        final_id_acc = float(id_rows.iloc[-1]["acc"]) if not id_rows.empty else np.nan
        final_ood_acc = float(ood_rows.iloc[-1]["acc"]) if not ood_rows.empty else np.nan

        if not ood_rows.empty:
            best_ood_idx = ood_rows["auc"].idxmax()
            best_ood_auc = float(ood_rows.loc[best_ood_idx, "auc"])
            best_ood_epoch = int(ood_rows.loc[best_ood_idx, "epoch"])
            best_ood_acc = float(ood_rows.loc[best_ood_idx, "acc"])
        else:
            best_ood_auc = np.nan
            best_ood_epoch = np.nan
            best_ood_acc = np.nan

        if not id_rows.empty:
            best_id_idx = id_rows["auc"].idxmax()
            best_id_auc = float(id_rows.loc[best_id_idx, "auc"])
            best_id_epoch = int(id_rows.loc[best_id_idx, "epoch"])
            best_id_acc = float(id_rows.loc[best_id_idx, "acc"])
        else:
            best_id_auc = np.nan
            best_id_epoch = np.nan
            best_id_acc = np.nan

        id_auc_arr = id_rows["auc"].to_numpy(dtype=float) if not id_rows.empty else np.array([])
        ood_auc_arr = ood_rows["auc"].to_numpy(dtype=float) if not ood_rows.empty else np.array([])

        id_auc_std = _safe_std(id_auc_arr)
        ood_auc_std = _safe_std(ood_auc_arr)
        id_auc_range = float(np.ptp(id_auc_arr)) if id_auc_arr.size > 0 else 0.0
        ood_auc_range = float(np.ptp(ood_auc_arr)) if ood_auc_arr.size > 0 else 0.0
        id_auc_cv = _safe_cv(id_auc_arr)
        ood_auc_cv = _safe_cv(ood_auc_arr)

        gen_gap = float(final_id_auc - final_ood_auc) if np.isfinite(final_id_auc) and np.isfinite(final_ood_auc) else np.nan
        ood_drop = float(best_ood_auc - final_ood_auc) if np.isfinite(best_ood_auc) and np.isfinite(final_ood_auc) else np.nan

        dyn_l = dyn[dyn["layer"] == layer].sort_values("epoch") if not dyn.empty else pd.DataFrame()
        if not dyn_l.empty:
            cos_final = float(dyn_l.iloc[-1]["cos_to_final"])
            w_norm_final = float(dyn_l.iloc[-1]["w_norm"])
            conv_rate = _convergence_rate(dyn_l)
        else:
            cos_final = np.nan
            w_norm_final = np.nan
            conv_rate = np.nan

        metrics.append(
            {
                "layer": int(layer),
                "final_id_auc": final_id_auc,
                "final_ood_auc": final_ood_auc,
                "best_id_auc": best_id_auc,
                "best_ood_auc": best_ood_auc,
                "final_id_acc": final_id_acc,
                "final_ood_acc": final_ood_acc,
                "best_id_acc": best_id_acc,
                "best_ood_acc": best_ood_acc,
                "best_id_epoch": best_id_epoch,
                "best_ood_epoch": best_ood_epoch,
                "gen_gap": gen_gap,
                "ood_drop_best_to_final": ood_drop,
                "id_auc_std": id_auc_std,
                "ood_auc_std": ood_auc_std,
                "id_auc_range": id_auc_range,
                "ood_auc_range": ood_auc_range,
                "id_auc_cv": id_auc_cv,
                "ood_auc_cv": ood_auc_cv,
                "ood_stability": ood_auc_std,
                "cos_final": cos_final,
                "w_norm_final": w_norm_final,
                "convergence_rate": conv_rate,
            }
        )

    df = pd.DataFrame(metrics).sort_values("layer").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def _register_figure(manifest: List[Dict[str, str]], figure_id: str, status: str, paths: List[str], inputs: List[str], note: str = "") -> None:
    manifest.append(
        {
            "figure_id": figure_id,
            "status": status,
            "outputs": "|".join(paths),
            "inputs": "|".join(inputs),
            "note": note,
        }
    )


def _save_figure(fig: plt.Figure, out_base: str, formats: List[str], dpi: int) -> List[str]:
    saved = []
    for ext in formats:
        p = f"{out_base}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved


def _line_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = NEG_COLOR) -> Optional[float]:
    if len(x) < 2:
        return None
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    yy = m * xx + b
    ax.plot(xx, yy, color=color, linestyle="-", linewidth=2)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else np.nan
    return r


def _available_layers(df: pd.DataFrame) -> List[int]:
    if df is None or df.empty or "layer" not in df.columns:
        return []
    return sorted(df["layer"].dropna().astype(int).unique().tolist())


# -----------------------------------------------------------------------------
# Plot families (pooled)
# -----------------------------------------------------------------------------
def plot_performance_overview(derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "performance_overview_2x2"
    if derived.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["derived_layer_metrics.csv"], "empty derived metrics")
        return

    d = derived.sort_values("layer")
    layers = d["layer"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # A: Final epoch ID vs OOD AUC
    ax = axes[0, 0]
    width = 0.42
    ax.bar(layers - width / 2, d["final_id_auc"], width=width, color=ID_COLOR, alpha=0.92, label="ID AUC")
    ax.bar(layers + width / 2, d["final_ood_auc"], width=width, color=OOD_COLOR, alpha=0.88, label="OOD AUC")
    ax.set_title("Final Epoch: ID vs OOD AUC by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y")

    # B: Generalization gap
    ax = axes[0, 1]
    gap = d["gen_gap"].to_numpy(dtype=float)
    colors = [POS_COLOR if x <= 0 else NEG_COLOR for x in gap]
    ax.bar(layers, gap, color=colors, alpha=0.92)
    ax.axhline(0.0, color="#444444", linewidth=1.2)
    ax.set_title("Generalization Gap (ID AUC - OOD AUC)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gap (positive = overfitting)")
    ax.grid(True, axis="y")

    # C: Best vs Final OOD
    ax = axes[1, 0]
    ax.plot(layers, d["best_ood_auc"], color=ID_COLOR, marker="o", label="Best OOD AUC")
    ax.plot(layers, d["final_ood_auc"], color=OOD_COLOR, marker="s", linestyle="--", label="Final OOD AUC")
    ax.set_title("Best vs Final OOD Performance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("OOD AUC")
    ax.legend(loc="upper left")
    ax.grid(True)

    # D: OOD drop from best to final
    ax = axes[1, 1]
    drop = d["ood_drop_best_to_final"].to_numpy(dtype=float)
    colors = [POS_COLOR if x <= 0 else NEG_COLOR for x in drop]
    ax.bar(layers, drop, color=colors, alpha=0.92)
    ax.axhline(0.0, color="#444444", linewidth=1.2)
    ax.set_title("OOD Performance Drop from Best to Final")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC Drop")
    ax.grid(True, axis="y")

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["derived_layer_metrics.csv"])


def plot_final_auc_acc(derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "final_auc_and_acc_across_layers"
    if derived.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["derived_layer_metrics.csv"], "empty derived metrics")
        return

    d = derived.sort_values("layer")
    layers = d["layer"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    ax = axes[0]
    ax.plot(layers, d["final_id_auc"], marker="o", color=ID_COLOR, label="ID")
    ax.plot(layers, d["final_ood_auc"], marker="o", color="#B08C2B", label="OOD")
    ax.set_title("Final AUC Performance Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(layers, d["final_id_acc"], marker="s", color=ID_COLOR, label="ID")
    ax.plot(layers, d["final_ood_acc"], marker="s", color="#B08C2B", label="OOD")
    ax.set_title("Final Accuracy Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["derived_layer_metrics.csv"])


def plot_auc_evolution_grid(ckpt: pd.DataFrame, derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "auc_evolution_all_layers_grid"
    if ckpt.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["checkpoint_metrics_long.csv"], "empty checkpoint table")
        return

    layers = _available_layers(ckpt)
    if not layers:
        _register_figure(manifest, figure_id, "skipped", [], ["checkpoint_metrics_long.csv"], "no layers")
        return

    n = len(layers)
    ncols = 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)

    best_epoch_map = {}
    if not derived.empty and "best_ood_epoch" in derived.columns:
        best_epoch_map = dict(zip(derived["layer"].astype(int), derived["best_ood_epoch"]))

    for i, layer in enumerate(layers):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = ckpt[ckpt["layer"] == layer].sort_values("epoch")
        sid = sub[sub["split"] == "id"]
        sod = sub[sub["split"] == "ood"]
        if not sid.empty:
            ax.plot(sid["epoch"], sid["auc"], color=ID_COLOR, marker="o", label="ID")
        if not sod.empty:
            ax.plot(sod["epoch"], sod["auc"], color=OOD_COLOR, marker="s", label="OOD")
        be = best_epoch_map.get(layer)
        if pd.notna(be):
            ax.axvline(float(be), color=NEUTRAL_COLOR, linestyle="--", linewidth=1.0)
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.grid(True)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("AUC Evolution Across All Layers (ID vs OOD)\nDashed line = Best OOD epoch", y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()

    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["checkpoint_metrics_long.csv", "derived_layer_metrics.csv"])


def plot_learning_curves_topk(ckpt: pd.DataFrame, derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, top_k: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "learning_curves_topk_ood_layers"
    if ckpt.empty or derived.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["checkpoint_metrics_long.csv", "derived_layer_metrics.csv"], "missing tables")
        return

    pick = (
        derived.dropna(subset=["best_ood_auc"])
        .sort_values(["best_ood_auc", "layer"], ascending=[False, True])
        .head(top_k)
    )
    if pick.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["derived_layer_metrics.csv"], "no top layers")
        return

    layers = pick["layer"].astype(int).tolist()
    n = len(layers)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    for i, layer in enumerate(layers):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = ckpt[ckpt["layer"] == layer].sort_values("epoch")
        sid = sub[sub["split"] == "id"]
        sod = sub[sub["split"] == "ood"]

        if not sid.empty:
            ax.plot(sid["epoch"], sid["auc"], color="#FF6F91", marker="o", label="ID AUC")
            ax.plot(sid["epoch"], sid["acc"], color="#C7A33C", marker="s", label="ID Acc")
        if not sod.empty:
            ax.plot(sod["epoch"], sod["auc"], color="#5AAE61", marker="o", label="OOD AUC")
            ax.plot(sod["epoch"], sod["acc"], color="#46C2CB", marker="s", label="OOD Acc")

        ax.set_title(f"Layer {layer} Learning Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=9)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"Learning Curves for Top {top_k} Performing Layers (by OOD AUC)", y=1.01)
    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["checkpoint_metrics_long.csv", "derived_layer_metrics.csv"])


def plot_dynamics_select_layers(dyn: pd.DataFrame, derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "cosine_weightnorm_select_layers"
    if dyn.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["training_dynamics_long.csv"], "empty dynamics")
        return

    if not derived.empty and "best_ood_auc" in derived.columns:
        sel = (
            derived.sort_values(["best_ood_auc", "layer"], ascending=[False, True])
            .head(5)["layer"]
            .astype(int)
            .tolist()
        )
        sel = sorted(sel)
    else:
        layers = _available_layers(dyn)
        sel = layers[:5]

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    palette = plt.cm.magma(np.linspace(0.15, 0.85, max(len(sel), 1)))

    ax = axes[0]
    for i, layer in enumerate(sel):
        sub = dyn[dyn["layer"] == layer].sort_values("epoch")
        ax.plot(sub["epoch"], sub["cos_to_final"], color=palette[i], label=f"Layer {layer}")
    ax.set_title("Convergence to Final Weights\n(Select Layers)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity to Final")
    ax.grid(True)
    ax.legend()

    ax = axes[1]
    for i, layer in enumerate(sel):
        sub = dyn[dyn["layer"] == layer].sort_values("epoch")
        ax.plot(sub["epoch"], sub["w_norm"], color=palette[i], label=f"Layer {layer}")
    ax.set_title("Weight Norm Growth\n(Select Layers)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Norm")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["training_dynamics_long.csv", "derived_layer_metrics.csv"])


def plot_dynamics_heatmaps(dyn: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "cosine_weightnorm_heatmaps"
    if dyn.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["training_dynamics_long.csv"], "empty dynamics")
        return

    dd = dyn.copy()
    dd["layer"] = dd["layer"].astype(int)
    dd["epoch"] = dd["epoch"].astype(int)

    cos_piv = dd.pivot_table(index="layer", columns="epoch", values="cos_to_final", aggfunc="mean")
    norm_piv = dd.pivot_table(index="layer", columns="epoch", values="w_norm", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    im0 = axes[0].imshow(cos_piv.values, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title("Cosine Similarity to Final Weights\nAcross Epochs and Layers")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Layer")
    axes[0].set_yticks(range(len(cos_piv.index)))
    axes[0].set_yticklabels(cos_piv.index.tolist())
    axes[0].set_xticks(range(len(cos_piv.columns)))
    axes[0].set_xticklabels(cos_piv.columns.tolist(), rotation=90)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(norm_piv.values, aspect="auto", cmap="plasma", interpolation="nearest")
    axes[1].set_title("Weight Norm Evolution\nAcross Epochs and Layers")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Layer")
    axes[1].set_yticks(range(len(norm_piv.index)))
    axes[1].set_yticklabels(norm_piv.index.tolist())
    axes[1].set_xticks(range(len(norm_piv.columns)))
    axes[1].set_xticklabels(norm_piv.columns.tolist(), rotation=90)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["training_dynamics_long.csv"])


def plot_convergence_analysis(dyn: pd.DataFrame, derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "convergence_analysis_2x3"
    if dyn.empty or derived.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["training_dynamics_long.csv", "derived_layer_metrics.csv"], "missing tables")
        return

    d = dyn.copy()
    d["layer"] = d["layer"].astype(int)
    d["epoch"] = d["epoch"].astype(int)
    layers = sorted(d["layer"].unique().tolist())

    fig, axes = plt.subplots(2, 3, figsize=(24, 13))

    # (1) convergence trajectories all layers
    ax = axes[0, 0]
    for layer in layers:
        sub = d[d["layer"] == layer].sort_values("epoch")
        ax.plot(sub["epoch"], sub["cos_to_final"], alpha=0.75, label=f"L{layer}")
    ax.set_title("Convergence Trajectories: All Layers")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity to Final")
    ax.grid(True)

    # (2) weight norm growth all layers
    ax = axes[0, 1]
    for layer in layers:
        sub = d[d["layer"] == layer].sort_values("epoch")
        ax.plot(sub["epoch"], sub["w_norm"], alpha=0.75)
    ax.set_title("Weight Norm Growth: All Layers")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Norm")
    ax.grid(True)

    # (3) convergence rate vs performance
    ax = axes[0, 2]
    x = derived["convergence_rate"].to_numpy(dtype=float)
    y = derived["best_ood_auc"].to_numpy(dtype=float)
    layer_labels = derived["layer"].astype(int).to_numpy()
    sc = ax.scatter(x, y, c=layer_labels, cmap="tab20", s=90, alpha=0.9)
    for xi, yi, li in zip(x, y, layer_labels):
        ax.text(xi, yi, str(li), fontsize=9)
    ax.set_title("Convergence Rate vs Final Performance")
    ax.set_xlabel("Convergence Rate")
    ax.set_ylabel("Best OOD AUC")
    ax.grid(True)

    # (4-6) snapshots at epochs 5,10,15
    for j, ep in enumerate([5, 10, 15]):
        ax = axes[1, j]
        rows = []
        for layer in layers:
            sub = d[d["layer"] == layer].sort_values("epoch")
            cand = sub[sub["epoch"] <= ep]
            if cand.empty:
                continue
            rows.append((int(layer), float(cand.iloc[-1]["cos_to_final"])))
        if not rows:
            ax.set_title(f"Epoch {ep}: no data")
            ax.axis("off")
            continue

        tmp = pd.DataFrame(rows, columns=["layer", "cos"])
        merged = tmp.merge(derived[["layer", "best_ood_auc"]], on="layer", how="inner")
        xs = merged["cos"].to_numpy(dtype=float)
        ys = merged["best_ood_auc"].to_numpy(dtype=float)
        ls = merged["layer"].to_numpy(dtype=int)
        ax.scatter(xs, ys, c=ls, cmap="viridis", s=85)
        for xi, yi, li in zip(xs, ys, ls):
            ax.text(xi, yi, str(li), fontsize=9)
        r = _line_fit(ax, xs, ys, color="#FF6B6B")
        if r is not None and np.isfinite(r):
            ax.text(0.03, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11, bbox=dict(boxstyle="round", fc="#F7E6C4", ec="#B89B6D"))
        ax.set_title(f"Epoch {ep}: Convergence vs Final Performance")
        ax.set_xlabel("Cosine Similarity to Final")
        ax.set_ylabel("Best OOD AUC")
        ax.grid(True)

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["training_dynamics_long.csv", "derived_layer_metrics.csv"])


def plot_stability(derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "training_stability_std_cv_range"
    if derived.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["derived_layer_metrics.csv"], "empty derived metrics")
        return

    d = derived.sort_values("layer")
    layers = d["layer"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    ax = axes[0, 0]
    width = 0.4
    ax.bar(layers - width / 2, d["id_auc_std"], width=width, color=ID_COLOR, label="ID")
    ax.bar(layers + width / 2, d["ood_auc_std"], width=width, color=OOD_COLOR, label="OOD")
    ax.set_title("Training Stability: Std Dev of AUC Across Epochs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Standard Deviation")
    ax.legend()
    ax.grid(True, axis="y")

    ax = axes[0, 1]
    ax.bar(layers - width / 2, d["id_auc_range"], width=width, color=ID_COLOR, label="ID")
    ax.bar(layers + width / 2, d["ood_auc_range"], width=width, color=OOD_COLOR, label="OOD")
    ax.set_title("Performance Variability: AUC Range Across Epochs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Range (Max - Min)")
    ax.legend()
    ax.grid(True, axis="y")

    ax = axes[1, 0]
    ax.bar(layers, d["best_ood_epoch"].fillna(0), color="#2FC4A7")
    ax.set_title("Optimal Stopping Point: Epoch with Best OOD AUC")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Epoch Number")
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    ax.bar(layers - width / 2, d["id_auc_cv"], width=width, color=ID_COLOR, label="ID")
    ax.bar(layers + width / 2, d["ood_auc_cv"], width=width, color=OOD_COLOR, label="OOD")
    ax.set_title("Normalized Stability: Coefficient of Variation")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV (lower = more stable)")
    ax.legend()
    ax.grid(True, axis="y")

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["derived_layer_metrics.csv"])


def plot_layer_comparison_heatmap_table(derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "layer_comparison_heatmap_and_table"
    if derived.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["derived_layer_metrics.csv"], "empty derived metrics")
        return

    d = derived.sort_values("layer").reset_index(drop=True)
    metrics = [
        ("final_ood_auc", True, "Final OOD AUC"),
        ("best_ood_auc", True, "Best OOD AUC"),
        ("gen_gap", False, "Gen Gap"),
        ("ood_stability", False, "OOD Stability"),
        ("best_ood_epoch", False, "Best Epoch"),
        ("cos_final", True, "Cos Sim"),
        ("w_norm_final", True, "Weight Norm"),
    ]

    norm_rows = []
    labels = []
    for col, higher_better, label in metrics:
        vals = d[col].to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            norm = np.zeros_like(vals)
        else:
            norm = (vals - vmin) / (vmax - vmin)
        if not higher_better:
            norm = 1.0 - norm
        norm_rows.append(norm)
        labels.append(label)

    norm_mat = np.vstack(norm_rows)

    fig, axes = plt.subplots(1, 2, figsize=(24, 9), gridspec_kw={"width_ratios": [1.2, 1.8]})

    ax = axes[0]
    im = ax.imshow(norm_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Layer Comparison: Normalized Metrics\n(Green = Better)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["layer"].astype(int).tolist())
    ax.set_xlabel("Layer")
    ax.set_ylabel("Metric")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Score")

    ax = axes[1]
    ax.axis("off")
    show_cols = ["layer", "final_ood_auc", "best_ood_auc", "gen_gap", "ood_stability", "best_ood_epoch", "cos_final", "w_norm_final"]
    show = d[show_cols].copy()
    show = show.round(4)

    table = ax.table(
        cellText=show.values,
        colLabels=["Layer", "Final OOD AUC", "Best OOD AUC", "Gen Gap", "OOD Stability", "Best Epoch", "Cos Sim", "Weight Norm"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)
    ax.set_title("Layer Metrics Summary Table")

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["derived_layer_metrics.csv"])


def plot_influence_overview(sample_inf: pd.DataFrame, derived: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, top_k_samples: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "influence_overview_2x2"
    if sample_inf.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["sample_influence_long.csv"], "empty influence table")
        return

    s = sample_inf.copy()
    s["layer"] = s["layer"].astype(int)
    s["influence"] = s["influence"].astype(float)
    if "behavior_label" not in s.columns:
        s["behavior_label"] = s["sample_id"].map(infer_behavior)

    fig, axes = plt.subplots(2, 2, figsize=(22, 13))

    # A: distribution by layer + behavior
    ax = axes[0, 0]
    if sns is not None:
        sns.violinplot(data=s, x="layer", y="influence", hue="behavior_label", split=False, inner="quartile", palette={"honest": ID_COLOR, "deceptive": OOD_COLOR, "unknown": NEUTRAL_COLOR}, ax=ax)
    else:
        for beh, color in [("honest", ID_COLOR), ("deceptive", OOD_COLOR)]:
            sub = s[s["behavior_label"] == beh]
            if sub.empty:
                continue
            ax.scatter(sub["layer"], sub["influence"], alpha=0.35, color=color, label=beh)
        ax.legend()
    ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--")
    ax.set_title("Distribution of Influence Scores by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Influence Score")

    # B: top samples average across layers
    ax = axes[0, 1]
    avg = s.groupby("sample_id", as_index=False)["influence"].mean()
    avg["behavior_label"] = avg["sample_id"].map(infer_behavior)
    top = avg.iloc[avg["influence"].abs().sort_values(ascending=False).index].head(top_k_samples)
    top = top.sort_values("influence", ascending=True)
    bar_colors = [OOD_COLOR if b == "deceptive" else ID_COLOR if b == "honest" else NEUTRAL_COLOR for b in top["behavior_label"]]
    ax.barh(top["sample_id"], top["influence"], color=bar_colors, alpha=0.95)
    ax.axvline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
    ax.set_title(f"Top {top_k_samples} Most Influential Samples\n(Average Across Layers)")
    ax.set_xlabel("Mean Influence Score")

    # C/D: honest/deceptive mean influence vs performance
    perf = derived[["layer", "best_ood_auc"]].copy() if not derived.empty else pd.DataFrame(columns=["layer", "best_ood_auc"])
    by = s.groupby(["layer", "behavior_label"], as_index=False)["influence"].mean()

    for ax, beh, title in [
        (axes[1, 0], "honest", "Honest Sample Influence vs Performance"),
        (axes[1, 1], "deceptive", "Deceptive Sample Influence vs Performance"),
    ]:
        sub = by[by["behavior_label"] == beh].rename(columns={"influence": "mean_influence"})
        m = sub.merge(perf, on="layer", how="inner")
        if not m.empty:
            sc = ax.scatter(m["mean_influence"], m["best_ood_auc"], c=m["layer"], cmap="viridis", s=90)
            for _, row in m.iterrows():
                ax.text(row["mean_influence"], row["best_ood_auc"], str(int(row["layer"])), fontsize=9)
            plt.colorbar(sc, ax=ax, label="Layer")
        ax.set_title(title)
        ax.set_xlabel(f"Mean {beh.title()} Influence")
        ax.set_ylabel("Best OOD AUC")
        ax.grid(True)

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["sample_influence_long.csv", "derived_layer_metrics.csv"])


def plot_influence_behavior_panels(sample_inf: pd.DataFrame, fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    figure_id = "influence_behavior_2x2"
    if sample_inf.empty:
        _register_figure(manifest, figure_id, "skipped", [], ["sample_influence_long.csv"], "empty influence table")
        return

    s = sample_inf.copy()
    s["layer"] = s["layer"].astype(int)
    s["influence"] = s["influence"].astype(float)
    if "behavior_label" not in s.columns:
        s["behavior_label"] = s["sample_id"].map(infer_behavior)

    fig, axes = plt.subplots(2, 2, figsize=(22, 13))

    # Mean sample influence by behavior type
    ax = axes[0, 0]
    grp = s.groupby(["layer", "behavior_label"], as_index=False)["influence"].mean()
    for beh, color, marker in [("honest", ID_COLOR, "o"), ("deceptive", OOD_COLOR, "s")]:
        sub = grp[grp["behavior_label"] == beh]
        if not sub.empty:
            ax.plot(sub["layer"], sub["influence"], color=color, marker=marker, label=beh.title())
    ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--")
    ax.set_title("Mean Sample Influence by Behavior Type")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Influence")
    ax.legend()
    ax.grid(True)

    # Distribution boxplot by behavior
    ax = axes[0, 1]
    bsub = s[s["behavior_label"].isin(["honest", "deceptive"])]
    if sns is not None and not bsub.empty:
        sns.boxplot(data=bsub, x="behavior_label", y="influence", palette={"honest": ID_COLOR, "deceptive": OOD_COLOR}, ax=ax)
    elif not bsub.empty:
        honest = bsub[bsub["behavior_label"] == "honest"]["influence"].to_numpy()
        deceptive = bsub[bsub["behavior_label"] == "deceptive"]["influence"].to_numpy()
        ax.boxplot([deceptive, honest], labels=["deceptive", "honest"])
    ax.set_title("Distribution of Influence Scores\nby Behavior Type")
    ax.set_xlabel("Behavior")
    ax.set_ylabel("Influence")
    ax.grid(True, axis="y")

    # Positive vs negative counts
    ax = axes[1, 0]
    cnt = s.groupby("layer")["influence"].agg(pos=lambda x: int((x > 0).sum()), neg=lambda x: int((x <= 0).sum())).reset_index()
    ax.bar(cnt["layer"], cnt["neg"], color=NEG_COLOR, label="Negative")
    ax.bar(cnt["layer"], cnt["pos"], bottom=cnt["neg"], color=POS_COLOR, label="Positive")
    ax.set_title("Distribution of Positive vs Negative\nInfluence Samples Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Samples")
    ax.legend()
    ax.grid(True, axis="y")

    # Range of influence scores
    ax = axes[1, 1]
    r = s.groupby("layer")["influence"].agg(max_pos="max", min_neg="min").reset_index()
    ax.bar(r["layer"], r["max_pos"], color=POS_COLOR, label="Max (Positive)")
    ax.bar(r["layer"], r["min_neg"], color=NEG_COLOR, label="Min (Negative)")
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
    ax.set_title("Range of Influence Scores\nper Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Influence Score")
    ax.legend()
    ax.grid(True, axis="y")

    fig.tight_layout()
    out_base = os.path.join(fig_dir, figure_id)
    paths = _save_figure(fig, out_base, formats, dpi)
    _register_figure(manifest, figure_id, "ok", paths, ["sample_influence_long.csv"])


# -----------------------------------------------------------------------------
# Layer-agnostic plotting
# -----------------------------------------------------------------------------
def plot_layer_agnostic_bundle(tables: Dict[str, pd.DataFrame], fig_dir: str, formats: List[str], dpi: int, manifest: List[Dict[str, str]]) -> None:
    ckpt = tables.get("checkpoint_metrics_long", pd.DataFrame())
    dyn = tables.get("training_dynamics_long", pd.DataFrame())
    linf = tables.get("layer_influence_long", pd.DataFrame())
    lprog = tables.get("layer_progress_long", pd.DataFrame())

    # checkpoint curves
    fid = "layer_agnostic_checkpoint_curves"
    if ckpt.empty:
        _register_figure(manifest, fid, "skipped", [], ["checkpoint_metrics.csv"], "missing")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        idr = ckpt[ckpt["split"] == "id"].sort_values("epoch")
        odr = ckpt[ckpt["split"] == "ood"].sort_values("epoch")
        if not idr.empty:
            ax.plot(idr["epoch"], idr["auc"], color=ID_COLOR, marker="o", label="ID AUC")
        if not odr.empty:
            ax.plot(odr["epoch"], odr["auc"], color=OOD_COLOR, marker="s", label="OOD AUC")
        ax.set_title("Layer-Agnostic Checkpoint AUC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.legend()
        ax.grid(True)
        out_base = os.path.join(fig_dir, fid)
        paths = _save_figure(fig, out_base, formats, dpi)
        _register_figure(manifest, fid, "ok", paths, ["checkpoint_metrics.csv"])

    # dynamics
    fid = "layer_agnostic_dynamics"
    if dyn.empty:
        _register_figure(manifest, fid, "skipped", [], ["training_dynamics.csv"], "missing")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        d = dyn.sort_values("epoch")
        axes[0].plot(d["epoch"], d["cos_to_final"], color=ID_COLOR)
        axes[0].set_title("Convergence to Final")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cosine to Final")
        axes[0].grid(True)

        axes[1].plot(d["epoch"], d["w_norm"], color=OOD_COLOR)
        axes[1].set_title("Weight Norm Growth")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Weight Norm")
        axes[1].grid(True)

        fig.tight_layout()
        out_base = os.path.join(fig_dir, fid)
        paths = _save_figure(fig, out_base, formats, dpi)
        _register_figure(manifest, fid, "ok", paths, ["training_dynamics.csv"])

    # influence/progress bars
    fid = "layer_agnostic_layer_scores"
    if linf.empty and lprog.empty:
        _register_figure(manifest, fid, "skipped", [], ["layer_influence.csv", "layer_progress.csv"], "missing")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if not linf.empty:
            axes[0].bar(linf["layer"], linf["mean_influence"], color=OOD_COLOR)
            axes[0].set_title("Layer Influence")
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("Mean Influence")
            axes[0].grid(True, axis="y")
        else:
            axes[0].axis("off")

        if not lprog.empty:
            axes[1].bar(lprog["layer"], lprog["mean_grad_alignment"], color=ID_COLOR)
            axes[1].set_title("Layer Progress")
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Mean Grad Alignment")
            axes[1].grid(True, axis="y")
        else:
            axes[1].axis("off")

        fig.tight_layout()
        out_base = os.path.join(fig_dir, fid)
        paths = _save_figure(fig, out_base, formats, dpi)
        _register_figure(manifest, fid, "ok", paths, ["layer_influence.csv", "layer_progress.csv"])


# -----------------------------------------------------------------------------
# Run-level orchestration
# -----------------------------------------------------------------------------
def process_pooled_pooling(
    pooling: str,
    src_dir: str,
    run_out_dir: str,
    formats: List[str],
    dpi: int,
    strict_schema: bool,
    top_k_layers: int,
    top_k_samples: int,
    global_missing: List[Dict[str, str]],
    global_manifest: List[Dict[str, str]],
) -> Optional[pd.DataFrame]:
    if not os.path.exists(src_dir):
        global_missing.append({"scope": "pooled", "pooling": pooling, "kind": "pooling_dir", "path": src_dir, "message": "missing"})
        return None

    tables_dir = os.path.join(run_out_dir, "tables")
    fig_dir = os.path.join(run_out_dir, "figures")
    ensure_dir(tables_dir)
    ensure_dir(fig_dir)

    tables, missing = collect_pooled_tables(src_dir, strict_schema=strict_schema)
    for m in missing:
        m["pooling"] = pooling
        global_missing.append(m)

    # Cast numeric columns defensively
    if not tables["checkpoint_metrics_long"].empty:
        for col in ["epoch", "auc", "acc", "best_so_far", "layer"]:
            if col in tables["checkpoint_metrics_long"].columns:
                tables["checkpoint_metrics_long"][col] = pd.to_numeric(tables["checkpoint_metrics_long"][col], errors="coerce")
        tables["checkpoint_metrics_long"] = tables["checkpoint_metrics_long"].dropna(subset=["epoch", "layer", "auc", "acc"])

    if not tables["training_dynamics_long"].empty:
        for col in ["epoch", "cos_to_final", "w_norm", "layer"]:
            if col in tables["training_dynamics_long"].columns:
                tables["training_dynamics_long"][col] = pd.to_numeric(tables["training_dynamics_long"][col], errors="coerce")
        tables["training_dynamics_long"] = tables["training_dynamics_long"].dropna(subset=["epoch", "layer"])

    if not tables["sample_influence_long"].empty:
        if "influence" in tables["sample_influence_long"].columns:
            tables["sample_influence_long"]["influence"] = pd.to_numeric(tables["sample_influence_long"]["influence"], errors="coerce")
            tables["sample_influence_long"] = tables["sample_influence_long"].dropna(subset=["influence"]) 

    derived = compute_derived_layer_metrics(tables["checkpoint_metrics_long"], tables["training_dynamics_long"])

    # Save canonical tables
    maybe_write_csv(tables["checkpoint_metrics_long"], os.path.join(tables_dir, "checkpoint_metrics_long.csv"))
    maybe_write_csv(tables["training_dynamics_long"], os.path.join(tables_dir, "training_dynamics_long.csv"))
    maybe_write_csv(tables["layer_influence_long"], os.path.join(tables_dir, "layer_influence_long.csv"))
    maybe_write_csv(tables["layer_progress_long"], os.path.join(tables_dir, "layer_progress_long.csv"))
    maybe_write_csv(tables["sample_influence_long"], os.path.join(tables_dir, "sample_influence_long.csv"))
    maybe_write_csv(tables["sample_progress_long"], os.path.join(tables_dir, "sample_progress_long.csv"))
    maybe_write_csv(derived, os.path.join(tables_dir, "derived_layer_metrics.csv"))

    # Build figures
    plot_performance_overview(derived, fig_dir, formats, dpi, global_manifest)
    plot_final_auc_acc(derived, fig_dir, formats, dpi, global_manifest)
    plot_layer_comparison_heatmap_table(derived, fig_dir, formats, dpi, global_manifest)
    plot_auc_evolution_grid(tables["checkpoint_metrics_long"], derived, fig_dir, formats, dpi, global_manifest)
    plot_learning_curves_topk(tables["checkpoint_metrics_long"], derived, fig_dir, formats, dpi, top_k_layers, global_manifest)
    plot_dynamics_select_layers(tables["training_dynamics_long"], derived, fig_dir, formats, dpi, global_manifest)
    plot_dynamics_heatmaps(tables["training_dynamics_long"], fig_dir, formats, dpi, global_manifest)
    plot_stability(derived, fig_dir, formats, dpi, global_manifest)
    plot_convergence_analysis(tables["training_dynamics_long"], derived, fig_dir, formats, dpi, global_manifest)
    plot_influence_overview(tables["sample_influence_long"], derived, fig_dir, formats, dpi, top_k_samples, global_manifest)
    plot_influence_behavior_panels(tables["sample_influence_long"], fig_dir, formats, dpi, global_manifest)

    # Tag manifest rows with pooling path
    for row in global_manifest:
        if "pooling" not in row:
            row["pooling"] = pooling

    return derived


def process_layer_agnostic_pooling(
    pooling: str,
    src_dir: str,
    la_results_dir: str,
    run_out_dir: str,
    formats: List[str],
    dpi: int,
    global_missing: List[Dict[str, str]],
    global_manifest: List[Dict[str, str]],
) -> Optional[pd.DataFrame]:
    if not os.path.exists(src_dir):
        global_missing.append({"scope": "layer_agnostic", "pooling": pooling, "kind": "pooling_dir", "path": src_dir, "message": "missing"})
        return None

    tables_dir = os.path.join(run_out_dir, "tables")
    fig_dir = os.path.join(run_out_dir, "figures")
    ensure_dir(tables_dir)
    ensure_dir(fig_dir)

    tables, missing = collect_layer_agnostic_tables(src_dir)
    for m in missing:
        m["pooling"] = pooling
        global_missing.append(m)

    # Canonical tables
    for name, df in tables.items():
        maybe_write_csv(df, os.path.join(tables_dir, f"{name}.csv"))

    # Optional per-layer json results
    layer_results_path = os.path.join(la_results_dir, pooling, "layer_results.json")
    derived = pd.DataFrame()
    if os.path.exists(layer_results_path):
        try:
            with open(layer_results_path, "r") as f:
                jr = json.load(f)
            id_map = jr.get("id_per_layer", {}) or {}
            ood_map = jr.get("ood_per_layer", {}) or {}
            rows = []
            layers = sorted(set([int(k) for k in id_map.keys()] + [int(k) for k in ood_map.keys()]))
            for l in layers:
                idd = id_map.get(str(l), id_map.get(l, {})) or {}
                odd = ood_map.get(str(l), ood_map.get(l, {})) or {}
                rows.append(
                    {
                        "layer": l,
                        "final_id_auc": idd.get("auc", np.nan),
                        "final_id_acc": idd.get("accuracy", np.nan),
                        "final_ood_auc": odd.get("auc", np.nan),
                        "final_ood_acc": odd.get("accuracy", np.nan),
                        "best_ood_auc": odd.get("auc", np.nan),
                        "best_ood_epoch": np.nan,
                    }
                )
            derived = pd.DataFrame(rows).sort_values("layer")
        except Exception as e:
            global_missing.append({"scope": "layer_agnostic", "pooling": pooling, "kind": "layer_results_parse", "path": layer_results_path, "message": str(e)})
    else:
        global_missing.append({"scope": "layer_agnostic", "pooling": pooling, "kind": "layer_results_json", "path": layer_results_path, "message": "missing"})

    maybe_write_csv(derived, os.path.join(tables_dir, "derived_layer_metrics.csv"))

    # LA figures
    plot_layer_agnostic_bundle(tables, fig_dir, formats, dpi, global_manifest)
    if not derived.empty:
        plot_final_auc_acc(derived, fig_dir, formats, dpi, global_manifest)

    for row in global_manifest:
        if "pooling" not in row:
            row["pooling"] = pooling

    return derived


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize attribution outputs with publication-style figures")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--poolings", nargs="+", default=["mean", "max", "last", "attn"])
    parser.add_argument("--layer_agnostic_poolings", nargs="+", default=["mean"], help="Poolings to scan in layer_agnostic_attribution")

    parser.add_argument("--input_probe_dir", type=str, default="results/probe_attribution")
    parser.add_argument("--input_layer_agnostic_dir", type=str, default="results/layer_agnostic_attribution")
    parser.add_argument("--input_layer_agnostic_results_dir", type=str, default="results/probes_layer_agnostic")

    parser.add_argument("--out_dir", type=str, default="results/attribution_summary")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)

    parser.add_argument("--top_k_layers", type=int, default=5)
    parser.add_argument("--top_k_samples", type=int, default=15)

    parser.add_argument("--strict_schema", action="store_true", help="Fail if required pooled schema files/columns are missing")
    args = parser.parse_args()

    configure_style()

    model_dir = args.model.replace("/", "_")
    run_root = os.path.join(args.out_dir, model_dir, args.dataset)
    ensure_dir(run_root)

    combined_rows = []
    missing_report: List[Dict[str, str]] = []
    figure_manifest: List[Dict[str, str]] = []

    # ---------------- Pooled ----------------
    for pooling in args.poolings:
        src = os.path.join(args.input_probe_dir, model_dir, args.dataset, pooling)
        dst = os.path.join(run_root, "pooled", pooling)
        ensure_dir(dst)

        derived = process_pooled_pooling(
            pooling=pooling,
            src_dir=src,
            run_out_dir=dst,
            formats=args.formats,
            dpi=args.dpi,
            strict_schema=args.strict_schema,
            top_k_layers=args.top_k_layers,
            top_k_samples=args.top_k_samples,
            global_missing=missing_report,
            global_manifest=figure_manifest,
        )

        if derived is not None and not derived.empty:
            row = {
                "probe_type": "pooled",
                "pooling": pooling,
                "best_final_ood_auc": float(np.nanmax(derived["final_ood_auc"])) if "final_ood_auc" in derived else np.nan,
                "best_best_ood_auc": float(np.nanmax(derived["best_ood_auc"])) if "best_ood_auc" in derived else np.nan,
                "mean_gen_gap": float(np.nanmean(derived["gen_gap"])) if "gen_gap" in derived else np.nan,
            }
            combined_rows.append(row)

    # ---------------- Layer-agnostic ----------------
    for pooling in args.layer_agnostic_poolings:
        src = os.path.join(args.input_layer_agnostic_dir, model_dir, args.dataset, pooling)
        dst = os.path.join(run_root, "layer_agnostic", pooling)
        ensure_dir(dst)

        la_results_base = os.path.join(args.input_layer_agnostic_results_dir, model_dir, args.dataset)

        derived = process_layer_agnostic_pooling(
            pooling=pooling,
            src_dir=src,
            la_results_dir=la_results_base,
            run_out_dir=dst,
            formats=args.formats,
            dpi=args.dpi,
            global_missing=missing_report,
            global_manifest=figure_manifest,
        )

        if derived is not None and not derived.empty:
            row = {
                "probe_type": "layer_agnostic",
                "pooling": pooling,
                "best_final_ood_auc": float(np.nanmax(derived["final_ood_auc"])) if "final_ood_auc" in derived else np.nan,
                "best_best_ood_auc": float(np.nanmax(derived["best_ood_auc"])) if "best_ood_auc" in derived else np.nan,
                "mean_gen_gap": float(np.nanmean((derived["final_id_auc"] - derived["final_ood_auc"]).to_numpy(dtype=float))) if {"final_id_auc", "final_ood_auc"}.issubset(derived.columns) else np.nan,
            }
            combined_rows.append(row)

    # Save global reports
    combined_df = pd.DataFrame(combined_rows)
    maybe_write_csv(combined_df, os.path.join(run_root, "combined_summary.csv"))

    miss_df = pd.DataFrame(missing_report)
    maybe_write_csv(miss_df, os.path.join(run_root, "missing_report.csv"))

    fig_df = pd.DataFrame(figure_manifest)
    maybe_write_csv(fig_df, os.path.join(run_root, "figure_manifest.csv"))

    print("=" * 80)
    print("Attribution summary complete")
    print(f"Output root: {run_root}")
    print(f"Combined summary: {os.path.join(run_root, 'combined_summary.csv')}")
    print(f"Missing report: {os.path.join(run_root, 'missing_report.csv')}")
    print(f"Figure manifest: {os.path.join(run_root, 'figure_manifest.csv')}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
