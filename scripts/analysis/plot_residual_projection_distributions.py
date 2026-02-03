#!/usr/bin/env python3
"""
Cross-Domain Residual Projection Distributions
==============================================

Plots KDE + histogram overlays for residual scores across domains/splits,
and reports AUC + KS statistics.
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from scipy.stats import gaussian_kde, ks_2samp
from tqdm import tqdm


def load_invariant_direction(invariant_dir: str, layer: int) -> np.ndarray:
    path = os.path.join(invariant_dir, f"invariant_layer_{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Invariant probe not found: {path}")
    state = torch.load(path, map_location="cpu")
    if "classifier.weight" not in state:
        raise ValueError(f"Missing classifier.weight in {path}")
    w = state["classifier.weight"].squeeze().cpu().numpy()
    return w


def load_norm_stats(probe_dir: str, layer: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    norm_path = os.path.join(probe_dir, f"norm_layer_{layer}.npz")
    if os.path.exists(norm_path):
        norm = np.load(norm_path)
        return norm["mean"], norm["std"]
    return None, None


def load_activations(
    activations_dir: str,
    layer: int,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = []
    with open(manifest_path, "r") as f:
        for line in f:
            manifest.append(json.loads(line))

    if max_samples is not None:
        manifest = manifest[:max_samples]

    shards = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))

    activations, labels = [], []
    for entry in manifest:
        eid = entry.get("id")
        if eid not in all_tensors:
            continue
        label = entry.get("label", -1)
        if label == -1:
            continue
        tensor = all_tensors[eid]
        if layer >= tensor.shape[0]:
            continue
        x_layer = tensor[layer].cpu().numpy()
        activations.append(x_layer)
        labels.append(int(label))
    return activations, labels


def normalize_tokens(x: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return x
    return (x - mean) / (std + 1e-8)


def pooled_scores(
    activations: List[np.ndarray],
    w: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> np.ndarray:
    scores = []
    for x in activations:
        x_norm = normalize_tokens(x, mean, std)
        pooled = x_norm.mean(axis=0)
        scores.append(float(np.dot(w, pooled)))
    return np.array(scores)


def select_top_layers(sweep_results: Dict, pooling: str, top_k: int) -> List[int]:
    results = sweep_results.get("results", {}).get(pooling, [])
    scored = []
    for r in results:
        if "error" in r:
            continue
        layer = r.get("layer")
        if layer is None:
            continue
        auc = None
        if "eval_on_insider" in r:
            auc = r["eval_on_insider"].get("invariant_core")
        elif "ood_auc" in r:
            if isinstance(r["ood_auc"], dict):
                auc = r["ood_auc"].get("invariant_core")
            else:
                try:
                    auc = float(r["ood_auc"])
                except Exception:
                    auc = None
        if auc is None:
            continue
        scored.append((auc, int(layer)))
    scored.sort(reverse=True)
    return [layer for _, layer in scored[:top_k]]


def plot_kde_hist(ax, scores_pos, scores_neg, bins="auto", title=""):
    if len(scores_pos) > 0:
        ax.hist(scores_pos, bins=bins, alpha=0.4, density=True, label="Deceptive")
    if len(scores_neg) > 0:
        ax.hist(scores_neg, bins=bins, alpha=0.4, density=True, label="Non-deceptive")

    if len(scores_pos) > 1:
        kde_pos = gaussian_kde(scores_pos)
        xmin = scores_pos.min()
        xmax = scores_pos.max()
        if len(scores_neg) > 0:
            xmin = min(xmin, scores_neg.min())
            xmax = max(xmax, scores_neg.max())
        xs = np.linspace(xmin, xmax, 200)
        ax.plot(xs, kde_pos(xs), color="tab:orange")
    if len(scores_neg) > 1:
        kde_neg = gaussian_kde(scores_neg)
        xmin = scores_neg.min()
        xmax = scores_neg.max()
        if len(scores_pos) > 0:
            xmin = min(xmin, scores_pos.min())
            xmax = max(xmax, scores_pos.max())
        xs = np.linspace(xmin, xmax, 200)
        ax.plot(xs, kde_neg(xs), color="tab:blue")

    ax.set_title(title)
    ax.set_xlabel("Residual Score")
    ax.set_ylabel("Density")
    ax.legend()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot residual projection distributions")
    parser.add_argument("--base_data_dir", required=True, help="Base data directory")
    parser.add_argument("--model", required=True, help="Model name for activations path")
    parser.add_argument("--sweep_results", required=True, help="Path to sweep_results.json")
    parser.add_argument("--invariant_probes_dir", required=True, help="Invariant probes dir")
    parser.add_argument("--combined_probes_dir", required=True, help="Combined probes dir (for norm stats)")
    parser.add_argument("--domain_a", default="Deception-Roleplaying")
    parser.add_argument("--domain_b", default="Deception-InsiderTrading")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bins", default="auto")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.sweep_results, "r") as f:
        sweep_results = json.load(f)

    layers = select_top_layers(sweep_results, args.pooling, args.top_k)
    if not layers:
        print("No layers found from sweep_results.")
        return 1

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    ks_rows = []

    for layer in layers:
        w_res = load_invariant_direction(args.invariant_probes_dir, layer)
        w_res = w_res / (np.linalg.norm(w_res) + 1e-8)

        mean_res, std_res = load_norm_stats(args.combined_probes_dir, layer)
        if mean_res is None:
            print("Warning: combined norm stats missing; using raw activations.")

        for split in splits:
            # Load both domains
            scores = {}
            labels = {}
            for domain in [args.domain_a, args.domain_b]:
                act_dir = os.path.join(args.base_data_dir, "activations", args.model, domain, split)
                acts, y = load_activations(act_dir, layer, max_samples=args.max_samples)
                scores[domain] = pooled_scores(acts, w_res, mean_res, std_res)
                labels[domain] = np.array(y)

            # Compute stats
            for domain in [args.domain_a, args.domain_b]:
                try:
                    auc = roc_auc_score(labels[domain], scores[domain])
                except Exception:
                    auc = 0.5
                summary_rows.append({
                    "layer": layer,
                    "split": split,
                    "domain": domain,
                    "auc": float(auc),
                })

            # KS between deceptive scores across domains
            a_pos = scores[args.domain_a][labels[args.domain_a] == 1]
            b_pos = scores[args.domain_b][labels[args.domain_b] == 1]
            if len(a_pos) > 1 and len(b_pos) > 1:
                ks_stat, ks_p = ks_2samp(a_pos, b_pos)
            else:
                ks_stat, ks_p = float("nan"), float("nan")
            ks_rows.append({
                "layer": layer,
                "split": split,
                "ks_deceptive": ks_stat,
                "ks_pvalue": ks_p,
            })

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            plot_kde_hist(
                axes[0],
                scores[args.domain_a][labels[args.domain_a] == 1],
                scores[args.domain_a][labels[args.domain_a] == 0],
                bins=args.bins,
                title=f"{args.domain_a} ({split})"
            )
            plot_kde_hist(
                axes[1],
                scores[args.domain_b][labels[args.domain_b] == 1],
                scores[args.domain_b][labels[args.domain_b] == 0],
                bins=args.bins,
                title=f"{args.domain_b} ({split})"
            )

            # Cross-domain deceptive overlay
            if len(a_pos) > 0:
                axes[2].hist(a_pos, bins=args.bins, alpha=0.4, density=True, label=args.domain_a)
            if len(b_pos) > 0:
                axes[2].hist(b_pos, bins=args.bins, alpha=0.4, density=True, label=args.domain_b)
            if len(a_pos) > 1:
                kde_a = gaussian_kde(a_pos)
                xmin = a_pos.min()
                xmax = a_pos.max()
                if len(b_pos) > 0:
                    xmin = min(xmin, b_pos.min())
                    xmax = max(xmax, b_pos.max())
                xs = np.linspace(xmin, xmax, 200)
                axes[2].plot(xs, kde_a(xs), color="tab:orange")
            if len(b_pos) > 1:
                kde_b = gaussian_kde(b_pos)
                xmin = b_pos.min()
                xmax = b_pos.max()
                if len(a_pos) > 0:
                    xmin = min(xmin, a_pos.min())
                    xmax = max(xmax, a_pos.max())
                xs = np.linspace(xmin, xmax, 200)
                axes[2].plot(xs, kde_b(xs), color="tab:blue")
            axes[2].set_title(f"Deceptive Only (KS={ks_stat:.3f})")
            axes[2].set_xlabel("Residual Score")
            axes[2].set_ylabel("Density")
            axes[2].legend()

            fig.suptitle(f"Residual Projection Distributions (Layer {layer}, {split})")
            plt.tight_layout()

            out_dir = os.path.join(args.output_dir, split)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"layer_{layer}_kde_hist.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: {out_path}")

            stats_path = os.path.join(out_dir, f"layer_{layer}_stats.json")
            stats = {
                "layer": layer,
                "split": split,
                "ks_deceptive": float(ks_stat) if ks_stat == ks_stat else None,
                "ks_pvalue": float(ks_p) if ks_p == ks_p else None,
                "auc": {
                    args.domain_a: [r["auc"] for r in summary_rows if r["layer"] == layer and r["split"] == split and r["domain"] == args.domain_a][0],
                    args.domain_b: [r["auc"] for r in summary_rows if r["layer"] == layer and r["split"] == split and r["domain"] == args.domain_b][0],
                },
            }
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"✓ Saved: {stats_path}")

    # Summary CSV
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    with open(summary_csv, "w") as f:
        f.write("layer,split,domain,auc\n")
        for row in summary_rows:
            f.write(f"{row['layer']},{row['split']},{row['domain']},{row['auc']}\n")
    print(f"✓ Saved: {summary_csv}")

    # Invariance vs Performance scatter
    for split in splits:
        # Collect per-layer avg AUC and KS
        layers_in_split = sorted({r["layer"] for r in summary_rows if r["split"] == split})
        if not layers_in_split:
            continue

        xs = []
        ys = []
        labels = []
        for layer in layers_in_split:
            a_auc = next((r["auc"] for r in summary_rows if r["layer"] == layer and r["split"] == split and r["domain"] == args.domain_a), None)
            b_auc = next((r["auc"] for r in summary_rows if r["layer"] == layer and r["split"] == split and r["domain"] == args.domain_b), None)
            ks = next((k["ks_deceptive"] for k in ks_rows if k["layer"] == layer and k["split"] == split), None)
            if a_auc is None or b_auc is None or ks is None or not (ks == ks):
                continue
            xs.append(ks)
            ys.append((a_auc + b_auc) / 2.0)
            labels.append(layer)

        if xs:
            plt.figure(figsize=(8, 6))
            plt.scatter(xs, ys, c="#4c78a8")
            for x, y, layer in zip(xs, ys, labels):
                plt.text(x, y, str(layer), fontsize=8, ha="right", va="bottom")
            plt.xlabel("KS (Deceptive: Roleplaying vs InsiderTrading)")
            plt.ylabel("Avg AUC (Roleplaying + InsiderTrading)")
            plt.title(f"Invariance vs Performance ({split})")
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(args.output_dir, f"invariance_vs_performance_{split}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
