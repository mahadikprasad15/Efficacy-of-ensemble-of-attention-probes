#!/usr/bin/env python3
"""Analyze PCA consensus runs across L1 settings and optional retrain outputs."""

import argparse
import ast
import json
import os
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_indices(raw) -> List[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, (list, tuple)):
        return [int(x) for x in parsed]
    return []


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def filter_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if "threshold" not in df.columns:
        raise ValueError("Input consensus summary must include 'threshold'")
    mask = np.isclose(df["threshold"].astype(float), threshold, atol=1e-9)
    return df.loc[mask].copy()


def add_index_sets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["consensus_index_list"] = out["consensus_pc_indices"].apply(parse_indices)
    out["consensus_index_set"] = out["consensus_index_list"].apply(lambda xs: set(int(x) for x in xs))
    return out


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return np.nan
    return float(len(a & b) / len(union))


def load_run(run_dir: str, threshold: float, label: str) -> Dict[str, Optional[pd.DataFrame]]:
    consensus_path = os.path.join(run_dir, "consensus_summary.csv")
    retrain_path = os.path.join(run_dir, "consensus_retrain_summary.csv")

    consensus = safe_read_csv(consensus_path)
    if consensus is None:
        raise FileNotFoundError(f"Missing consensus summary: {consensus_path}")

    consensus_t = filter_threshold(consensus, threshold)
    consensus_t = add_index_sets(consensus_t)
    consensus_t["run_label"] = label

    retrain = safe_read_csv(retrain_path)
    if retrain is not None:
        retrain["run_label"] = label

    return {
        "consensus": consensus,
        "consensus_threshold": consensus_t,
        "retrain": retrain,
    }


def compute_pc_frequency(df_t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_t.iterrows():
        for pc in row["consensus_index_list"]:
            rows.append({
                "layer": int(row["layer"]),
                "K": int(row["K"]),
                "pc_index": int(pc),
            })
    if not rows:
        return pd.DataFrame(columns=["pc_index", "count", "freq"])

    expanded = pd.DataFrame(rows)
    out = (
        expanded.groupby("pc_index", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    out["freq"] = out["count"] / float(len(df_t))
    return out


def build_sweetspots(df_t: pd.DataFrame, run_label: str) -> pd.DataFrame:
    metric = "ood_test_auc" if df_t.get("ood_test_auc") is not None and df_t["ood_test_auc"].notna().any() else "id_val_auc"
    table = df_t.dropna(subset=[metric]).copy()
    if table.empty:
        return pd.DataFrame(columns=["run_label", "K", "best_layer", "metric", "best_value"])
    rows = []
    for k, sub in table.groupby("K"):
        best = sub.sort_values(metric, ascending=False).iloc[0]
        rows.append(
            {
                "run_label": run_label,
                "K": int(k),
                "best_layer": int(best["layer"]),
                "metric": metric,
                "best_value": float(best[metric]),
            }
        )
    return pd.DataFrame(rows)


def plot_jaccard_heatmap(jacc_df: pd.DataFrame, out_path: str) -> None:
    if jacc_df.empty:
        return
    pivot = jacc_df.pivot_table(index="K", columns="layer", values="jaccard")
    if pivot.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Jaccard")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Layer")
    plt.ylabel("K")
    plt.title("Consensus Mask Stability Across L1 Runs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_sparsity_vs_ood(df: pd.DataFrame, out_path: str) -> None:
    if df.empty or "ood_test_auc" not in df.columns:
        return
    table = df.dropna(subset=["ood_test_auc", "consensus_ratio"]).copy()
    if table.empty:
        return

    plt.figure(figsize=(8, 5))
    for label, sub in table.groupby("run_label"):
        plt.scatter(sub["consensus_ratio"], sub["ood_test_auc"], label=label, alpha=0.75)
    plt.xlabel("Consensus ratio")
    plt.ylabel("OOD AUC")
    plt.title("OOD Performance vs Consensus Sparsity")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_sweetspots(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    for label, sub in df.groupby("run_label"):
        sub_sorted = sub.sort_values("K")
        plt.plot(sub_sorted["K"], sub_sorted["best_layer"], marker="o", label=label)
    plt.xlabel("K")
    plt.ylabel("Best layer")
    plt.title("Layer Sweet Spots by K")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze extended PCA consensus outputs")
    parser.add_argument("--run_a_dir", type=str, required=True)
    parser.add_argument("--run_b_dir", type=str, required=True)
    parser.add_argument("--label_a", type=str, default="run_a")
    parser.add_argument("--label_b", type=str, default="run_b")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=25)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    run_a = load_run(args.run_a_dir, args.threshold, args.label_a)
    run_b = load_run(args.run_b_dir, args.threshold, args.label_b)

    a_t = run_a["consensus_threshold"]
    b_t = run_b["consensus_threshold"]

    a_view = a_t[["layer", "K", "num_consensus_pcs", "consensus_ratio", "id_val_auc", "ood_test_auc", "consensus_index_set"]].copy()
    b_view = b_t[["layer", "K", "num_consensus_pcs", "consensus_ratio", "id_val_auc", "ood_test_auc", "consensus_index_set"]].copy()
    a_view = a_view.rename(columns={
        "num_consensus_pcs": "num_consensus_pcs_a",
        "consensus_ratio": "consensus_ratio_a",
        "id_val_auc": "id_val_auc_a",
        "ood_test_auc": "ood_test_auc_a",
        "consensus_index_set": "consensus_index_set_a",
    })
    b_view = b_view.rename(columns={
        "num_consensus_pcs": "num_consensus_pcs_b",
        "consensus_ratio": "consensus_ratio_b",
        "id_val_auc": "id_val_auc_b",
        "ood_test_auc": "ood_test_auc_b",
        "consensus_index_set": "consensus_index_set_b",
    })

    merged = a_view.merge(b_view, on=["layer", "K"], how="inner")
    merged["jaccard"] = merged.apply(
        lambda r: jaccard(r["consensus_index_set_a"], r["consensus_index_set_b"]),
        axis=1,
    )
    merged["delta_ood_auc_b_minus_a"] = merged["ood_test_auc_b"] - merged["ood_test_auc_a"]
    merged["delta_id_auc_b_minus_a"] = merged["id_val_auc_b"] - merged["id_val_auc_a"]

    jaccard_out = merged[[
        "layer", "K", "num_consensus_pcs_a", "num_consensus_pcs_b",
        "consensus_ratio_a", "consensus_ratio_b",
        "id_val_auc_a", "id_val_auc_b", "ood_test_auc_a", "ood_test_auc_b",
        "jaccard", "delta_id_auc_b_minus_a", "delta_ood_auc_b_minus_a",
    ]].sort_values(["K", "layer"])
    jaccard_out.to_csv(os.path.join(args.output_dir, "jaccard_by_layer_k.csv"), index=False)

    sparsity_perf = pd.concat([
        a_t.assign(run_label=args.label_a),
        b_t.assign(run_label=args.label_b),
    ], ignore_index=True)
    sparsity_cols = [
        "run_label", "layer", "K", "num_consensus_pcs", "consensus_ratio", "id_val_auc", "ood_test_auc"
    ]
    sparsity_perf[sparsity_cols].sort_values(["run_label", "K", "layer"]).to_csv(
        os.path.join(args.output_dir, "sparsity_vs_performance.csv"),
        index=False,
    )

    freq_a = compute_pc_frequency(a_t)
    freq_b = compute_pc_frequency(b_t)
    freq_a.head(args.top_n).to_csv(os.path.join(args.output_dir, f"pc_frequency_top{args.top_n}_{args.label_a}.csv"), index=False)
    freq_b.head(args.top_n).to_csv(os.path.join(args.output_dir, f"pc_frequency_top{args.top_n}_{args.label_b}.csv"), index=False)

    sweet_a = build_sweetspots(a_t, args.label_a)
    sweet_b = build_sweetspots(b_t, args.label_b)
    sweetspots = pd.concat([sweet_a, sweet_b], ignore_index=True)
    sweetspots.to_csv(os.path.join(args.output_dir, "layer_sweetspots_by_k.csv"), index=False)

    retrain_tables = []
    for run, label in ((run_a, args.label_a), (run_b, args.label_b)):
        retrain_df = run.get("retrain")
        if retrain_df is None or retrain_df.empty:
            continue
        threshold_col = "source_threshold" if "source_threshold" in retrain_df.columns else None
        if threshold_col:
            retrain_df = retrain_df[np.isclose(retrain_df[threshold_col].astype(float), args.threshold, atol=1e-9)].copy()

        cons = run["consensus_threshold"][["layer", "K", "id_val_auc", "ood_test_auc"]].copy()
        cons = cons.rename(columns={"id_val_auc": "consensus_id_val_auc", "ood_test_auc": "consensus_ood_test_auc"})
        merged_retrain = retrain_df.merge(cons, on=["layer", "K"], how="left")
        merged_retrain["delta_id_auc_retrain_minus_consensus"] = (
            merged_retrain["id_val_auc"] - merged_retrain["consensus_id_val_auc"]
        )
        merged_retrain["delta_ood_auc_retrain_minus_consensus"] = (
            merged_retrain["ood_test_auc"] - merged_retrain["consensus_ood_test_auc"]
        )
        merged_retrain["run_label"] = label
        retrain_tables.append(merged_retrain)

    if retrain_tables:
        retrain_out = pd.concat(retrain_tables, ignore_index=True)
        retrain_out.sort_values(["run_label", "K", "layer"]).to_csv(
            os.path.join(args.output_dir, "retrain_benefit_by_layer_k.csv"),
            index=False,
        )

    plot_jaccard_heatmap(jaccard_out, os.path.join(args.output_dir, "jaccard_heatmap.png"))
    plot_sparsity_vs_ood(sparsity_perf, os.path.join(args.output_dir, "sparsity_vs_ood_scatter.png"))
    plot_sweetspots(sweetspots, os.path.join(args.output_dir, "layer_sweetspots.png"))

    summary = {
        "threshold": float(args.threshold),
        "run_a_dir": os.path.abspath(args.run_a_dir),
        "run_b_dir": os.path.abspath(args.run_b_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "rows": {
            "jaccard_by_layer_k": int(len(jaccard_out)),
            "sparsity_vs_performance": int(len(sparsity_perf)),
            "sweetspots": int(len(sweetspots)),
        },
        "top_n": int(args.top_n),
    }
    with open(os.path.join(args.output_dir, "analysis_manifest.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete.")
    print(f"Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
