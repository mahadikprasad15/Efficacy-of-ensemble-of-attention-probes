#!/usr/bin/env python3
"""
Plot Residual Patching Results
==============================

Aggregates layer_*.jsonl files produced by patch_residual_tokens.py and visualizes:
  - Top tokens by count and by sum(|delta|)
  - Top positions by count and by sum(|delta|)
  - Distribution of |delta_score|
  - Relative position histogram (pos / length)

Usage:
  python scripts/analysis/plot_residual_patching_summary.py \
    --input_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/invariant_core_patching \
    --output_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/invariant_core_patching/plots
"""

import os
import json
import glob
import argparse
import re
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


def parse_path(path: str):
    parts = path.split(os.sep)
    pooling = "unknown"
    split = "unknown"
    domain = "unknown"
    for i, part in enumerate(parts):
        if part in ["validation", "test", "train"]:
            split = part
            if i + 1 < len(parts):
                domain = parts[i + 1]
            if i - 1 >= 0:
                pooling = parts[i - 1]
            break
    layer = None
    m = re.search(r"layer_(\d+)\.jsonl$", os.path.basename(path))
    if m:
        layer = int(m.group(1))
    return pooling, split, domain, layer


def load_jsonl(input_dir: str):
    pattern = os.path.join(input_dir, "**", "layer_*.jsonl")
    paths = glob.glob(pattern, recursive=True)
    records = []
    for p in paths:
        pooling, split, domain, layer = parse_path(p)
        with open(p, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["_path"] = p
                rec["_pooling"] = pooling
                rec["_split"] = split
                rec["_domain"] = domain
                rec["_layer"] = layer
                records.append(rec)
    return records


def plot_bar(data, title, xlabel, output_path, top_k=20):
    items = data.most_common(top_k)
    if not items:
        return
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color="#4c78a8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_delta_hist(deltas, title, output_path):
    if len(deltas) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(deltas, bins="auto", color="#f58518", alpha=0.8)
    plt.title(title)
    plt.xlabel("|delta_score|")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_signed_delta_hist(deltas, title, output_path):
    if len(deltas) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(deltas, bins="auto", color="#e45756", alpha=0.8)
    plt.title(title)
    plt.xlabel("delta_score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_relative_position_hist(rel_pos_vals, title, output_path, bins=20):
    if len(rel_pos_vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(rel_pos_vals, bins=bins, color="#54a24b", alpha=0.8)
    plt.title(title)
    plt.xlabel("Position / Length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_contrib_delta(points, title, output_path):
    if len(points) == 0:
        return
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    plt.figure(figsize=(7, 6))
    if len(points) > 2000:
        plt.hexbin(x, y, gridsize=50, cmap="viridis", mincnt=1)
        plt.colorbar(label="Count")
    else:
        plt.scatter(x, y, s=8, alpha=0.6, color="#4c78a8")
    plt.axhline(0, color="#888888", linewidth=1)
    plt.axvline(0, color="#888888", linewidth=1)
    plt.title(title)
    plt.xlabel("Contribution")
    plt.ylabel("Delta Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def contrib_delta_stats(points):
    if len(points) == 0:
        return {"n": 0}
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    stats = {"n": int(len(points))}
    if len(points) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        stats["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    else:
        stats["pearson_r"] = None
    stats["quadrants"] = {
        "pos_pos": int(np.sum((x >= 0) & (y >= 0))),
        "pos_neg": int(np.sum((x >= 0) & (y < 0))),
        "neg_pos": int(np.sum((x < 0) & (y >= 0))),
        "neg_neg": int(np.sum((x < 0) & (y < 0))),
    }
    return stats


def plot_layer_hists(layer_relpos, output_dir, tag, bins=20):
    if not layer_relpos:
        return
    layers = sorted(layer_relpos.keys())
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    heatmap = []

    for layer in layers:
        vals = layer_relpos[layer]
        if not vals:
            counts = np.zeros(bins)
        else:
            counts, _ = np.histogram(vals, bins=bin_edges)
        heatmap.append(counts)

        plt.figure(figsize=(8, 5))
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(centers, counts, width=(1.0 / bins) * 0.9, color="#72b7b2")
        plt.title(f"Relative Position (Layer {layer})")
        plt.xlabel("Position / Length")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"relative_position_layer_{layer}_{tag}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    heatmap_arr = np.array(heatmap)
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap_arr, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Count")
    plt.yticks(range(len(layers)), [str(l) for l in layers])
    plt.xlabel("Relative Position Bin")
    plt.ylabel("Layer")
    plt.title("Relative Position Heatmap")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"relative_position_heatmap_{tag}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def collect_stats(records):
    token_count = Counter()
    token_weight = Counter()
    pos_count = Counter()
    pos_weight = Counter()
    delta_vals_abs = []
    delta_vals_signed = []
    rel_pos_vals = []
    contrib_delta = []
    layer_relpos = defaultdict(list)

    missing_token = 0
    total_token = 0

    for r in records:
        length = r.get("length", None)
        layer = r.get("_layer")
        for t in r.get("top_tokens", []):
            token = t.get("token")
            idx = t.get("index")
            delta = float(t.get("delta_score", 0.0))
            contrib = t.get("contribution", None)
            if contrib is not None:
                contrib_delta.append((float(contrib), delta))
            if token is not None:
                token_count[token] += 1
                token_weight[token] += abs(delta)
                total_token += 1
            else:
                missing_token += 1
            pos_count[str(idx)] += 1
            pos_weight[str(idx)] += abs(delta)
            delta_vals_abs.append(abs(delta))
            delta_vals_signed.append(delta)
            if length and length > 0:
                rel_pos = float(idx) / float(length)
                rel_pos_vals.append(rel_pos)
                if layer is not None:
                    layer_relpos[layer].append(rel_pos)

    return {
        "token_count": token_count,
        "token_weight": token_weight,
        "pos_count": pos_count,
        "pos_weight": pos_weight,
        "delta_vals_abs": delta_vals_abs,
        "delta_vals_signed": delta_vals_signed,
        "rel_pos_vals": rel_pos_vals,
        "contrib_delta": contrib_delta,
        "layer_relpos": layer_relpos,
        "missing_token": missing_token,
        "total_token": total_token,
    }


def process_group(records, output_dir, group_key, top_k=20, bins=20):
    safe_key = group_key.replace("/", "_")
    stratifiers = [
        ("all", lambda r: True),
        ("label_0", lambda r: r.get("label", None) == 0),
        ("label_1", lambda r: r.get("label", None) == 1),
        ("respos", lambda r: r.get("residual_score", 0.0) > 0),
        ("resneg", lambda r: r.get("residual_score", 0.0) < 0),
    ]

    for suffix, predicate in stratifiers:
        subset = [r for r in records if predicate(r)]
        if not subset:
            continue

        tag = safe_key if suffix == "all" else f"{safe_key}_{suffix}"
        stats = collect_stats(subset)

        if stats["missing_token"] > 0 and stats["total_token"] == 0:
            print(f"Warning: tokens missing for group {group_key} ({suffix}).")

        if stats["total_token"] > 0:
            plot_bar(
                stats["token_count"],
                f"Top Tokens by Count ({group_key}, {suffix})",
                "Count",
                os.path.join(output_dir, f"top_tokens_count_{tag}.png"),
                top_k=top_k,
            )
            plot_bar(
                stats["token_weight"],
                f"Top Tokens by |Delta| ({group_key}, {suffix})",
                "Sum |Delta|",
                os.path.join(output_dir, f"top_tokens_weight_{tag}.png"),
                top_k=top_k,
            )

        plot_bar(
            stats["pos_count"],
            f"Top Positions by Count ({group_key}, {suffix})",
            "Count",
            os.path.join(output_dir, f"top_positions_count_{tag}.png"),
            top_k=top_k,
        )
        plot_bar(
            stats["pos_weight"],
            f"Top Positions by |Delta| ({group_key}, {suffix})",
            "Sum |Delta|",
            os.path.join(output_dir, f"top_positions_weight_{tag}.png"),
            top_k=top_k,
        )
        plot_delta_hist(
            np.array(stats["delta_vals_abs"]),
            f"|Delta Score| Distribution ({group_key}, {suffix})",
            os.path.join(output_dir, f"delta_hist_{tag}.png"),
        )
        plot_signed_delta_hist(
            np.array(stats["delta_vals_signed"]),
            f"Delta Score Distribution ({group_key}, {suffix})",
            os.path.join(output_dir, f"delta_hist_signed_{tag}.png"),
        )
        plot_relative_position_hist(
            stats["rel_pos_vals"],
            f"Relative Position Distribution ({group_key}, {suffix})",
            os.path.join(output_dir, f"relative_position_{tag}.png"),
            bins=bins,
        )

        plot_contrib_delta(
            stats["contrib_delta"],
            f"Contribution vs Delta ({group_key}, {suffix})",
            os.path.join(output_dir, f"contrib_delta_{tag}.png"),
        )
        with open(os.path.join(output_dir, f"contrib_delta_stats_{tag}.json"), "w") as f:
            json.dump(contrib_delta_stats(stats["contrib_delta"]), f, indent=2)

        plot_layer_hists(stats["layer_relpos"], output_dir, tag, bins=bins)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot residual patching summaries")
    parser.add_argument("--input_dir", required=True, help="Base directory with summary JSONs")
    parser.add_argument("--output_dir", required=True, help="Output plots directory")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    records = load_jsonl(args.input_dir)
    if not records:
        print("No JSONL files found.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    grouped = defaultdict(list)
    for r in records:
        pooling = r.get("_pooling", "unknown")
        split = r.get("_split", "unknown")
        domain = r.get("_domain", "unknown")
        grouped[f"{pooling}/{split}/{domain}"].append(r)

    process_group(records, args.output_dir, "all", top_k=args.top_k)

    for key, group in grouped.items():
        process_group(group, args.output_dir, key, top_k=args.top_k)

    print(f"✓ Saved plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
