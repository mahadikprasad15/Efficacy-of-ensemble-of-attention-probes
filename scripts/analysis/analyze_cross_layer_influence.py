#!/usr/bin/env python3
"""
Cross-layer sample influence analysis (post-hoc).

Supports both:
1) Attribution summary tables layout:
   <root>/<pooling>/tables/sample_influence_long.csv
2) Raw attribution files layout:
   <root>/sample_influence_top*_layer_*.csv
   <root>/sample_influence_full_layer_*.csv

This script is parameterized for reusable analysis across poolings and layers.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cross_layer_influence")


@dataclass
class PoolingData:
    pooling: str
    influence: pd.DataFrame
    checkpoint_metrics: pd.DataFrame


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_layers(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return sorted(set(int(x.strip()) for x in raw.split(",") if x.strip()))


def parse_layer_from_name(path: str, suffix: str = r"_layer_(\d+)\.csv$") -> Optional[int]:
    m = re.search(suffix, os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def discover_poolings(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []

    # attribution_summary/pooled layout: each pooling is a directory.
    candidates = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        candidates.append(name)
    return sorted(candidates)


def _load_summary_tables(pooling_dir: str, score_col: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    tables_dir = os.path.join(pooling_dir, "tables")
    influence_path = os.path.join(tables_dir, "sample_influence_long.csv")
    ckpt_path = os.path.join(tables_dir, "checkpoint_metrics_long.csv")
    if not os.path.exists(influence_path):
        return None

    inf = pd.read_csv(influence_path)
    if "influence" not in inf.columns:
        raise ValueError(f"Expected 'influence' column in {influence_path}")
    inf = inf.rename(columns={"influence": "score"})
    for col in ("sample_id", "layer", "score"):
        if col not in inf.columns:
            raise ValueError(f"Missing required column '{col}' in {influence_path}")
    inf["layer"] = pd.to_numeric(inf["layer"], errors="coerce")
    inf["score"] = pd.to_numeric(inf["score"], errors="coerce")
    inf = inf.dropna(subset=["layer", "score"]).copy()
    inf["layer"] = inf["layer"].astype(int)

    if score_col != "influence":
        prog_path = os.path.join(tables_dir, "sample_progress_long.csv")
        if os.path.exists(prog_path):
            prog = pd.read_csv(prog_path)
            if "grad_alignment" in prog.columns:
                prog["grad_alignment"] = pd.to_numeric(prog["grad_alignment"], errors="coerce")
                prog = prog.dropna(subset=["grad_alignment", "layer"])
                prog["layer"] = prog["layer"].astype(int)
                inf = prog[["sample_id", "layer", "grad_alignment"]].rename(columns={"grad_alignment": "score"})
        else:
            logger.warning("Requested score_col=%s but no sample_progress_long.csv found; using influence", score_col)

    ckpt = pd.DataFrame()
    if os.path.exists(ckpt_path):
        ckpt = pd.read_csv(ckpt_path)
        for col in ("layer", "epoch", "auc"):
            if col in ckpt.columns:
                ckpt[col] = pd.to_numeric(ckpt[col], errors="coerce")
        ckpt = ckpt.dropna(subset=["layer", "epoch", "auc"]).copy()
        ckpt["layer"] = ckpt["layer"].astype(int)

    return inf, ckpt


def _load_raw_tables(pooling_dir: str, score_col: str, prefer_full: bool) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    # score files
    score_files: List[str] = []
    if score_col == "influence":
        if prefer_full:
            score_files = sorted(glob.glob(os.path.join(pooling_dir, "sample_influence_full_layer_*.csv")))
        if not score_files:
            score_files = sorted(glob.glob(os.path.join(pooling_dir, "sample_influence_top*_layer_*.csv")))
    else:
        if prefer_full:
            score_files = sorted(glob.glob(os.path.join(pooling_dir, "sample_progress_full_layer_*.csv")))
        if not score_files:
            score_files = sorted(glob.glob(os.path.join(pooling_dir, "sample_progress_top*_layer_*.csv")))

    if not score_files:
        return None

    rows = []
    for p in score_files:
        layer = parse_layer_from_name(p)
        if layer is None:
            continue
        df = pd.read_csv(p)
        score_name = "influence" if score_col == "influence" else "grad_alignment"
        if score_name not in df.columns:
            # Support already-normalized files where score column is named score.
            if "score" not in df.columns:
                continue
            score_name = "score"
        if "sample_id" not in df.columns:
            continue
        sdf = df[["sample_id", score_name]].copy()
        sdf["layer"] = layer
        sdf = sdf.rename(columns={score_name: "score"})
        rows.append(sdf)

    if not rows:
        return None

    inf = pd.concat(rows, ignore_index=True)
    inf["score"] = pd.to_numeric(inf["score"], errors="coerce")
    inf = inf.dropna(subset=["score"]).copy()
    inf["layer"] = inf["layer"].astype(int)

    # checkpoint files
    ckpt_files = sorted(glob.glob(os.path.join(pooling_dir, "checkpoint_metrics_layer_*.csv")))
    ckpt_rows = []
    for p in ckpt_files:
        layer = parse_layer_from_name(p)
        if layer is None:
            continue
        df = pd.read_csv(p)
        if not {"epoch", "split", "auc"}.issubset(df.columns):
            continue
        df = df[["epoch", "split", "auc"]].copy()
        df["layer"] = layer
        ckpt_rows.append(df)
    ckpt = pd.concat(ckpt_rows, ignore_index=True) if ckpt_rows else pd.DataFrame()
    if not ckpt.empty:
        for col in ("layer", "epoch", "auc"):
            ckpt[col] = pd.to_numeric(ckpt[col], errors="coerce")
        ckpt = ckpt.dropna(subset=["layer", "epoch", "auc"]).copy()
        ckpt["layer"] = ckpt["layer"].astype(int)

    return inf, ckpt


def load_pooling_data(root: str, pooling: str, score_col: str, prefer_full: bool) -> PoolingData:
    # root can be pooled root with subdirs OR direct pooling dir
    direct = root
    with_subdir = os.path.join(root, pooling)

    candidate_dirs = [with_subdir, direct]
    loaded_inf = None
    loaded_ckpt = None

    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        out = _load_summary_tables(d, score_col)
        if out is not None:
            loaded_inf, loaded_ckpt = out
            break
        out = _load_raw_tables(d, score_col, prefer_full=prefer_full)
        if out is not None:
            loaded_inf, loaded_ckpt = out
            break

    if loaded_inf is None:
        raise FileNotFoundError(f"Could not find influence files for pooling '{pooling}' under {root}")

    loaded_inf["pooling"] = pooling
    if loaded_ckpt is None:
        loaded_ckpt = pd.DataFrame()
    elif not loaded_ckpt.empty:
        loaded_ckpt["pooling"] = pooling

    return PoolingData(pooling=pooling, influence=loaded_inf, checkpoint_metrics=loaded_ckpt)


def compute_layer_metrics(ckpt: pd.DataFrame) -> pd.DataFrame:
    if ckpt.empty:
        return pd.DataFrame(columns=["pooling", "layer", "best_ood_auc", "final_id_auc", "final_ood_auc", "gen_gap"])

    rows = []
    for (pooling, layer), g in ckpt.groupby(["pooling", "layer"]):
        g = g.sort_values("epoch")
        id_rows = g[g["split"] == "id"].sort_values("epoch")
        ood_rows = g[g["split"] == "ood"].sort_values("epoch")

        final_id_auc = float(id_rows.iloc[-1]["auc"]) if not id_rows.empty else np.nan
        final_ood_auc = float(ood_rows.iloc[-1]["auc"]) if not ood_rows.empty else np.nan
        best_ood_auc = float(ood_rows["auc"].max()) if not ood_rows.empty else np.nan
        gen_gap = (final_id_auc - final_ood_auc) if np.isfinite(final_id_auc) and np.isfinite(final_ood_auc) else np.nan

        rows.append(
            {
                "pooling": pooling,
                "layer": int(layer),
                "best_ood_auc": best_ood_auc,
                "final_id_auc": final_id_auc,
                "final_ood_auc": final_ood_auc,
                "gen_gap": gen_gap,
            }
        )

    return pd.DataFrame(rows)


def choose_target_layers(
    layer_metrics: pd.DataFrame,
    metric_mode: str,
    manual_layers: List[int],
    top_k_layers: int,
    poolings: Sequence[str],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}

    for pooling in poolings:
        if manual_layers:
            out[pooling] = sorted(set(manual_layers))
            continue

        m = layer_metrics[layer_metrics["pooling"] == pooling].copy()
        if m.empty:
            out[pooling] = []
            continue

        if metric_mode == "ood_only":
            score = m["best_ood_auc"].fillna(-1.0)
        else:
            # ood_plus_gap
            z_ood = (m["best_ood_auc"] - m["best_ood_auc"].mean()) / (m["best_ood_auc"].std(ddof=0) + 1e-8)
            z_gap = (m["gen_gap"].abs() - m["gen_gap"].abs().mean()) / (m["gen_gap"].abs().std(ddof=0) + 1e-8)
            score = z_ood + z_gap

        m = m.assign(_score=score)
        picks = m.sort_values("_score", ascending=False).head(max(1, top_k_layers))["layer"].astype(int).tolist()
        out[pooling] = sorted(set(picks))

    return out


def aggregate_sample_scores(inf: pd.DataFrame, use_abs: bool) -> pd.DataFrame:
    # collapse repeated entries (full matrix may contain multiple contributions)
    agg = inf.groupby(["pooling", "layer", "sample_id"], as_index=False)["score"].sum()
    agg["score_abs"] = agg["score"].abs() if use_abs else agg["score"]
    return agg


def build_overlap_matrix(df: pd.DataFrame, layers: Sequence[int], top_k: int, use_abs: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["layer_a", "layer_b", "jaccard", "intersection", "union"])

    top_sets: Dict[int, set] = {}
    score_col = "score_abs" if use_abs else "score"
    for layer in layers:
        sub = df[df["layer"] == layer].copy()
        sub = sub.sort_values(score_col, ascending=False).head(top_k)
        top_sets[layer] = set(sub["sample_id"].tolist())

    rows = []
    for la, lb in itertools.product(layers, layers):
        a = top_sets.get(la, set())
        b = top_sets.get(lb, set())
        inter = len(a & b)
        union = len(a | b)
        jac = float(inter / union) if union else 0.0
        rows.append({"layer_a": la, "layer_b": lb, "jaccard": jac, "intersection": inter, "union": union})

    return pd.DataFrame(rows)


def compute_uniqueness_scores(
    df: pd.DataFrame,
    target_layers: Sequence[int],
    method: str,
    top_k: int,
    use_abs: bool,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    score_col = "score_abs" if use_abs else "score"
    pivot = (
        df.pivot_table(index="sample_id", columns="layer", values=score_col, aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )

    all_layers = list(pivot.columns)
    rows = []

    # Precompute ranks per layer.
    ranks = {}
    for layer in all_layers:
        s = pivot[layer].sort_values(ascending=False)
        ranks[layer] = {sid: r + 1 for r, sid in enumerate(s.index.tolist())}

    topk_sets = {}
    for layer in all_layers:
        s = pivot[layer].sort_values(ascending=False).head(top_k)
        topk_sets[layer] = set(s.index.tolist())

    for sid, vals in pivot.iterrows():
        for target in target_layers:
            if target not in all_layers:
                continue
            target_val = float(vals[target])
            others = [l for l in all_layers if l != target]
            other_vals = [float(vals[l]) for l in others] if others else [0.0]
            other_mean = float(np.mean(other_vals))

            residualized = target_val - other_mean

            target_rank = ranks[target].get(sid, len(pivot) + 1)
            other_ranks = [ranks[l].get(sid, len(pivot) + 1) for l in others] if others else [target_rank]
            rank_diff = float(np.median(other_ranks) - target_rank)

            in_target_topk = sid in topk_sets[target]
            in_other_topk = any(sid in topk_sets[l] for l in others)
            topk_exclusive = 1 if (in_target_topk and not in_other_topk) else 0

            rows.append(
                {
                    "sample_id": sid,
                    "target_layer": int(target),
                    "target_score": target_val,
                    "other_mean_score": other_mean,
                    "residualized_score": residualized,
                    "rank_diff_score": rank_diff,
                    "topk_exclusive_score": topk_exclusive,
                }
            )

    out = pd.DataFrame(rows)
    return out


def label_membership(
    uniq: pd.DataFrame,
    primary_method: str,
    threshold_quantile: float,
) -> pd.DataFrame:
    if uniq.empty:
        return pd.DataFrame(columns=["sample_id", "group", "best_target_layer", "best_score"])

    if primary_method == "rank_diff":
        col = "rank_diff_score"
    elif primary_method == "topk_exclusive":
        col = "topk_exclusive_score"
    else:
        col = "residualized_score"

    # Sample-level table across target layers
    piv = uniq.pivot_table(index="sample_id", columns="target_layer", values=col, aggfunc="max").fillna(0.0)
    threshold = float(np.quantile(piv.to_numpy().flatten(), threshold_quantile))

    rows = []
    for sid, r in piv.iterrows():
        vals = r.to_dict()
        sorted_layers = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        best_layer, best_score = sorted_layers[0]
        num_high = sum(1 for _, v in vals.items() if v >= threshold)

        if num_high >= 2:
            group = "shared"
        elif best_score >= threshold:
            group = f"unique_target_layer_{best_layer}"
        else:
            group = "control"

        rows.append(
            {
                "sample_id": sid,
                "group": group,
                "best_target_layer": int(best_layer),
                "best_score": float(best_score),
                "threshold_used": threshold,
            }
        )

    return pd.DataFrame(rows)


def plot_overlap_heatmap(overlap_df: pd.DataFrame, layers: Sequence[int], out_path: str) -> None:
    if overlap_df.empty:
        return
    mat = overlap_df.pivot(index="layer_a", columns="layer_b", values="jaccard").reindex(index=layers, columns=layers)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Top-K Overlap (Jaccard) Across Layers")
    ax.set_xlabel("Layer B")
    ax.set_ylabel("Layer A")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    for i in range(len(layers)):
        for j in range(len(layers)):
            v = mat.values[i, j]
            txt = f"{v:.2f}" if np.isfinite(v) else "nan"
            ax.text(j, i, txt, ha="center", va="center", color="white" if v > 0.5 else "black", fontsize=9)

    fig.colorbar(im, ax=ax, label="Jaccard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_influence_profile_heatmap(df: pd.DataFrame, target_layers: Sequence[int], out_path: str, top_n: int = 60) -> None:
    if df.empty:
        return
    piv = df.pivot_table(index="sample_id", columns="layer", values="score_abs", aggfunc="sum").fillna(0.0)
    cols = [l for l in target_layers if l in piv.columns]
    if not cols:
        return

    sub = piv[cols].copy()
    sub["_max"] = sub.max(axis=1)
    sub = sub.sort_values("_max", ascending=False).head(top_n).drop(columns=["_max"])

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.16)))
    im = ax.imshow(sub.values, aspect="auto", cmap="magma")
    ax.set_title("Sample Influence Profiles (Top Samples)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sample")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="|score|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_uniqueness_scatter(uniq: pd.DataFrame, membership: pd.DataFrame, out_path: str) -> None:
    if uniq.empty:
        return

    layers = sorted(uniq["target_layer"].unique().tolist())
    if len(layers) < 2:
        return

    la, lb = layers[:2]
    p = uniq[uniq["target_layer"].isin([la, lb])].pivot_table(
        index="sample_id", columns="target_layer", values="residualized_score", aggfunc="max"
    ).fillna(0.0)
    if la not in p.columns or lb not in p.columns:
        return

    m = membership.set_index("sample_id") if not membership.empty else pd.DataFrame()
    grp = []
    for sid in p.index:
        if sid in m.index:
            grp.append(str(m.loc[sid, "group"]))
        else:
            grp.append("unlabeled")

    colors = {
        "shared": "#B04A8B",
        "control": "#808080",
        f"unique_target_layer_{la}": "#1FA77A",
        f"unique_target_layer_{lb}": "#4FA3C7",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for g in sorted(set(grp)):
        mask = np.array([x == g for x in grp])
        ax.scatter(
            p.iloc[mask][la],
            p.iloc[mask][lb],
            s=20,
            alpha=0.7,
            label=g,
            color=colors.get(g, "#222222"),
        )

    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.axvline(0.0, color="#444444", linewidth=1)
    ax.set_xlabel(f"Residualized score (layer {la})")
    ax.set_ylabel(f"Residualized score (layer {lb})")
    ax.set_title("Residualized Influence: Target Layer Contrast")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_json(path: str, data: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_for_pooling(
    root: str,
    pooling: str,
    args: argparse.Namespace,
    explicit_layers: List[int],
) -> dict:
    pdata = load_pooling_data(root, pooling, score_col=args.score_column, prefer_full=args.prefer_full)
    inf = aggregate_sample_scores(pdata.influence, use_abs=args.use_abs_scores)

    layer_metrics = compute_layer_metrics(pdata.checkpoint_metrics)
    target_map = choose_target_layers(
        layer_metrics=layer_metrics,
        metric_mode=args.target_layer_metric,
        manual_layers=explicit_layers,
        top_k_layers=args.top_k_layers,
        poolings=[pooling],
    )
    targets = target_map.get(pooling, [])
    if not targets:
        targets = sorted(inf["layer"].unique().tolist())[: max(1, args.top_k_layers)]

    inf = inf[inf["layer"].isin(sorted(inf["layer"].unique()))].copy()

    overlap = build_overlap_matrix(inf, layers=sorted(inf["layer"].unique().tolist()), top_k=args.top_k_overlap, use_abs=args.use_abs_scores)
    uniq = compute_uniqueness_scores(
        inf,
        target_layers=targets,
        method=args.uniqueness_method,
        top_k=args.top_k_overlap,
        use_abs=args.use_abs_scores,
    )
    membership = label_membership(uniq, primary_method=args.primary_method, threshold_quantile=args.uniqueness_quantile)

    out_dir = os.path.join(args.output_root, pooling)
    tables_dir = os.path.join(out_dir, "tables")
    figs_dir = os.path.join(out_dir, "figures")
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    inf.to_csv(os.path.join(tables_dir, "aggregated_sample_scores.csv"), index=False)
    layer_metrics.to_csv(os.path.join(tables_dir, "layer_metrics.csv"), index=False)
    overlap.to_csv(os.path.join(tables_dir, "cross_layer_overlap_matrix.csv"), index=False)
    uniq.to_csv(os.path.join(tables_dir, "sample_uniqueness_scores.csv"), index=False)
    membership.to_csv(os.path.join(tables_dir, "sample_membership_labels.csv"), index=False)

    plot_overlap_heatmap(overlap, layers=sorted(inf["layer"].unique().tolist()), out_path=os.path.join(figs_dir, "layer_overlap_heatmap.png"))
    plot_influence_profile_heatmap(inf, target_layers=targets, out_path=os.path.join(figs_dir, "sample_influence_profile_heatmap.png"), top_n=args.top_n_profile)
    plot_uniqueness_scatter(uniq, membership, out_path=os.path.join(figs_dir, "uniqueness_scatter.png"))

    meta = {
        "pooling": pooling,
        "target_layers": targets,
        "num_samples": int(inf["sample_id"].nunique()),
        "num_layers_available": int(inf["layer"].nunique()),
        "output_dir": out_dir,
    }
    save_json(os.path.join(out_dir, "run_metadata.json"), meta)
    return meta


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generalized post-hoc cross-layer influence analysis")
    p.add_argument("--input_root", type=str, required=True, help="Root containing pooling dirs or a direct pooling dir")
    p.add_argument("--output_root", type=str, required=True)

    p.add_argument("--pooling", type=str, default=None)
    p.add_argument("--poolings", nargs="+", default=None)
    p.add_argument("--layers", type=str, default="", help="Manual target layers, comma-separated")

    p.add_argument("--score_column", type=str, default="influence", choices=["influence", "grad_alignment"])
    p.add_argument("--prefer_full", action="store_true", help="Prefer sample_*_full_layer_*.csv when available")
    p.add_argument("--use_abs_scores", action="store_true", help="Rank samples by absolute score magnitude")
    p.add_argument("--use_signed_scores", action="store_true", help="Rank samples by signed score (overrides --use_abs_scores)")

    p.add_argument("--target_layer_metric", type=str, default="ood_plus_gap", choices=["manual", "ood_plus_gap", "ood_only"])
    p.add_argument("--top_k_layers", type=int, default=5)

    p.add_argument("--uniqueness_method", type=str, default="residualized", choices=["residualized", "rank_diff", "topk_exclusive"])
    p.add_argument("--primary_method", type=str, default="residualized", choices=["residualized", "rank_diff", "topk_exclusive"])
    p.add_argument("--top_k_overlap", type=int, default=100)
    p.add_argument("--uniqueness_quantile", type=float, default=0.9)

    p.add_argument("--top_n_profile", type=int, default=60)
    return p


def main() -> int:
    args = make_parser().parse_args()
    if not args.use_abs_scores and not args.use_signed_scores:
        args.use_abs_scores = True
    if args.use_signed_scores:
        args.use_abs_scores = False

    explicit_layers = parse_layers(args.layers)

    if args.poolings:
        poolings = args.poolings
    elif args.pooling:
        poolings = [args.pooling]
    else:
        poolings = discover_poolings(args.input_root)
        if not poolings:
            # Interpret input_root as a single pooling dir if files exist directly.
            poolings = [os.path.basename(args.input_root.rstrip("/"))]

    ensure_dir(args.output_root)

    all_meta = []
    for pooling in poolings:
        logger.info("Analyzing pooling=%s", pooling)
        meta = run_for_pooling(args.input_root, pooling, args, explicit_layers)
        all_meta.append(meta)

    save_json(os.path.join(args.output_root, "analysis_manifest.json"), {"runs": all_meta})
    logger.info("Done. Wrote outputs to %s", args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
