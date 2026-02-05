#!/usr/bin/env python3
"""
Summarize attribution outputs into a single dashboard folder.

Inputs:
  results/probe_attribution/<model>/<dataset>/<pooling>/
  results/layer_agnostic_attribution/<model>/<dataset>/<pooling>/

Outputs:
  results/attribution_summary/
    pooled/<pooling>/
    layer_agnostic/
    combined_summary.csv
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_dynamics(rows: List[Dict[str, str]], out_path: str, title: str):
    epochs = [int(r["epoch"]) for r in rows]
    cos_vals = [float(r["cos_to_final"]) for r in rows]
    norms = [float(r["w_norm"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, cos_vals, color="#2E86AB", label="cos(w_t, w*)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cosine to final")
    ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, norms, color="#A23B72", label="||w_t||", alpha=0.7)
    ax2.set_ylabel("Weight norm")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_checkpoint_metrics(rows: List[Dict[str, str]], out_path: str, title: str):
    # split rows
    id_rows = [r for r in rows if r["split"] == "id"]
    ood_rows = [r for r in rows if r["split"] == "ood"]

    def series(rows, key):
        epochs = [int(r["epoch"]) for r in rows]
        vals = [float(r[key]) for r in rows]
        return epochs, vals

    fig, ax = plt.subplots(figsize=(8, 4))

    if id_rows:
        e, auc = series(id_rows, "auc")
        ax.plot(e, auc, label="ID AUC", color="#2E86AB")
    if ood_rows:
        e, auc = series(ood_rows, "auc")
        ax.plot(e, auc, label="OOD AUC", color="#E67E22")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layer_metric(rows: List[Dict[str, str]], out_path: str, title: str, value_key: str):
    layers = [int(r["layer"]) for r in rows]
    vals = [float(r[value_key]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(layers, vals, color="#2E86AB")
    ax.set_xlabel("Layer")
    ax.set_ylabel(value_key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_pooled_pooling(base_dir: str, out_dir: str) -> Dict[str, float]:
    ensure_dir(out_dir)

    # Summaries over layers
    summary = {"best_id_auc": 0.0, "best_ood_auc": 0.0, "best_id_epoch": None, "best_ood_epoch": None}

    # Combine checkpoint metrics across layers
    ckpt_rows = []
    for path in glob.glob(os.path.join(base_dir, "checkpoint_metrics_layer_*.csv")):
        ckpt_rows.extend(read_csv(path))
    if ckpt_rows:
        # Save merged checkpoint metrics
        merged_path = os.path.join(out_dir, "checkpoint_metrics.csv")
        with open(merged_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ckpt_rows[0].keys())
            writer.writeheader()
            writer.writerows(ckpt_rows)

        # Compute best metrics
        id_rows = [r for r in ckpt_rows if r["split"] == "id"]
        ood_rows = [r for r in ckpt_rows if r["split"] == "ood"]
        if id_rows:
            best = max(id_rows, key=lambda r: float(r["auc"]))
            summary["best_id_auc"] = float(best["auc"])
            summary["best_id_epoch"] = int(best["epoch"])
        if ood_rows:
            best = max(ood_rows, key=lambda r: float(r["auc"]))
            summary["best_ood_auc"] = float(best["auc"])
            summary["best_ood_epoch"] = int(best["epoch"])

        plot_checkpoint_metrics(ckpt_rows, os.path.join(out_dir, "checkpoint_auc.png"), "Checkpoint AUC (ID/OOD)")

    # Dynamics: average over layers (if present)
    dyn_rows = []
    for path in glob.glob(os.path.join(base_dir, "training_dynamics_layer_*.csv")):
        rows = read_csv(path)
        if rows:
            dyn_rows.extend(rows)
    if dyn_rows:
        # crude average by epoch
        by_epoch = {}
        for r in dyn_rows:
            e = int(r["epoch"])
            by_epoch.setdefault(e, []).append((float(r["cos_to_final"]), float(r["w_norm"])))
        avg_rows = []
        for e in sorted(by_epoch.keys()):
            cos_vals = [x[0] for x in by_epoch[e]]
            norm_vals = [x[1] for x in by_epoch[e]]
            avg_rows.append({"epoch": e, "cos_to_final": sum(cos_vals)/len(cos_vals), "w_norm": sum(norm_vals)/len(norm_vals)})
        plot_dynamics(avg_rows, os.path.join(out_dir, "dynamics.png"), "Training Dynamics (avg)")

    # Layer influence/progress: merge if present (use first layer file if one exists)
    layer_influence = glob.glob(os.path.join(base_dir, "layer_influence_layer_*.csv"))
    if layer_influence:
        rows = read_csv(layer_influence[0])
        plot_layer_metric(rows, os.path.join(out_dir, "layer_influence.png"), "Layer Influence", "mean_influence")

    layer_progress = glob.glob(os.path.join(base_dir, "layer_progress_layer_*.csv"))
    if layer_progress:
        rows = read_csv(layer_progress[0])
        plot_layer_metric(rows, os.path.join(out_dir, "layer_progress.png"), "Layer Progress", "mean_grad_alignment")

    # Sample summaries
    sample_files = glob.glob(os.path.join(base_dir, "sample_influence_top*.csv"))
    if sample_files:
        # Just copy first as representative
        rows = read_csv(sample_files[0])
        out_path = os.path.join(out_dir, "top_samples.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Save summary json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def summarize_layer_agnostic(base_dir: str, out_dir: str) -> Dict[str, float]:
    ensure_dir(out_dir)

    summary = {"best_id_auc": 0.0, "best_ood_auc": 0.0, "best_id_epoch": None, "best_ood_epoch": None}

    dyn_path = os.path.join(base_dir, "training_dynamics.csv")
    if os.path.exists(dyn_path):
        rows = read_csv(dyn_path)
        plot_dynamics(rows, os.path.join(out_dir, "dynamics.png"), "Training Dynamics")

    ckpt_path = os.path.join(base_dir, "checkpoint_metrics.csv")
    if os.path.exists(ckpt_path):
        rows = read_csv(ckpt_path)
        plot_checkpoint_metrics(rows, os.path.join(out_dir, "checkpoint_auc.png"), "Checkpoint AUC (ID/OOD)")

        id_rows = [r for r in rows if r["split"] == "id"]
        ood_rows = [r for r in rows if r["split"] == "ood"]
        if id_rows:
            best = max(id_rows, key=lambda r: float(r["auc"]))
            summary["best_id_auc"] = float(best["auc"])
            summary["best_id_epoch"] = int(best["epoch"])
        if ood_rows:
            best = max(ood_rows, key=lambda r: float(r["auc"]))
            summary["best_ood_auc"] = float(best["auc"])
            summary["best_ood_epoch"] = int(best["epoch"])

    layer_inf_path = os.path.join(base_dir, "layer_influence.csv")
    if os.path.exists(layer_inf_path):
        rows = read_csv(layer_inf_path)
        plot_layer_metric(rows, os.path.join(out_dir, "layer_influence.png"), "Layer Influence", "mean_influence")

    layer_prog_path = os.path.join(base_dir, "layer_progress.csv")
    if os.path.exists(layer_prog_path):
        rows = read_csv(layer_prog_path)
        plot_layer_metric(rows, os.path.join(out_dir, "layer_progress.png"), "Layer Progress", "mean_grad_alignment")

    sample_files = glob.glob(os.path.join(base_dir, "sample_influence_top*.csv"))
    if sample_files:
        rows = read_csv(sample_files[0])
        out_path = os.path.join(out_dir, "top_samples.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize attribution outputs")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--poolings", nargs="+", default=["mean", "max", "last", "attn"])
    parser.add_argument("--layer_agnostic_pooling", type=str, default="mean")
    parser.add_argument("--out_dir", type=str, default="results/attribution_summary")
    args = parser.parse_args()

    model_dir = args.model.replace("/", "_")
    pooled_base = os.path.join("results/probe_attribution", model_dir, args.dataset)
    la_base = os.path.join("results/layer_agnostic_attribution", model_dir, args.dataset, args.layer_agnostic_pooling)

    pooled_out = os.path.join(args.out_dir, "pooled")
    la_out = os.path.join(args.out_dir, "layer_agnostic")
    ensure_dir(pooled_out)
    ensure_dir(la_out)

    combined_rows = []

    for pooling in args.poolings:
        src = os.path.join(pooled_base, pooling)
        dst = os.path.join(pooled_out, pooling)
        if not os.path.exists(src):
            print(f"Skipping pooling {pooling}: {src} not found")
            continue
        summary = summarize_pooled_pooling(src, dst)
        combined_rows.append(["pooled", pooling, summary["best_id_auc"], summary["best_ood_auc"], summary["best_id_epoch"], summary["best_ood_epoch"]])

    if os.path.exists(la_base):
        la_summary = summarize_layer_agnostic(la_base, la_out)
        combined_rows.append(["layer_agnostic", args.layer_agnostic_pooling, la_summary["best_id_auc"], la_summary["best_ood_auc"], la_summary["best_id_epoch"], la_summary["best_ood_epoch"]])

    # Combined summary
    combined_path = os.path.join(args.out_dir, "combined_summary.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "pooling", "best_id_auc", "best_ood_auc", "best_id_epoch", "best_ood_epoch"])
        writer.writerows(combined_rows)

    print(f"✓ Wrote summary to {args.out_dir}")


if __name__ == "__main__":
    main()
