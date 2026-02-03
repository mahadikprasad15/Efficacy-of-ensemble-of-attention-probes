#!/usr/bin/env python3
"""
Plot Residual Patching Summaries
================================

Aggregates layer_*_summary.json files produced by patch_residual_tokens.py
and visualizes:
  - Top tokens by frequency
  - Top positions by frequency
  - Distribution of absolute delta scores

Usage:
  python scripts/analysis/plot_residual_patching_summary.py \
    --input_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/invariant_core_patching \
    --output_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/results/invariant_core_patching/plots
"""

import os
import json
import glob
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def load_summaries(input_dir: str):
    pattern = os.path.join(input_dir, "**", "layer_*_summary.json")
    paths = glob.glob(pattern, recursive=True)
    summaries = []
    for p in paths:
        with open(p, "r") as f:
            s = json.load(f)
        s["_path"] = p
        summaries.append(s)
    return summaries


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot residual patching summaries")
    parser.add_argument("--input_dir", required=True, help="Base directory with summary JSONs")
    parser.add_argument("--output_dir", required=True, help="Output plots directory")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    summaries = load_summaries(args.input_dir)
    if not summaries:
        print("No summary files found.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Global aggregation
    token_counter = Counter()
    pos_counter = Counter()
    delta_vals = []

    for s in summaries:
        for tok, cnt in s.get("top_tokens", []):
            token_counter[tok] += cnt
        for pos, cnt in s.get("top_positions", []):
            pos_counter[str(pos)] += cnt
        # delta_stats are summary, not per-sample; use mean_abs_delta as proxy
        if "delta_stats" in s and "mean_abs_delta" in s["delta_stats"]:
            delta_vals.append(s["delta_stats"]["mean_abs_delta"])

    plot_bar(
        token_counter,
        "Top Tokens (All Layers/Splits/Domains)",
        "Frequency",
        os.path.join(args.output_dir, "top_tokens_all.png"),
        top_k=args.top_k,
    )
    plot_bar(
        pos_counter,
        "Top Positions (All Layers/Splits/Domains)",
        "Frequency",
        os.path.join(args.output_dir, "top_positions_all.png"),
        top_k=args.top_k,
    )
    plot_delta_hist(
        np.array(delta_vals),
        "Mean |Delta Score| Across Summaries",
        os.path.join(args.output_dir, "delta_hist_all.png"),
    )

    # Split/domain specific plots
    by_group = {}
    for s in summaries:
        split = s.get("split", "unknown")
        domain = s.get("domain", "unknown")
        key = f"{split}/{domain}"
        by_group.setdefault(key, []).append(s)

    for key, group in by_group.items():
        token_counter = Counter()
        pos_counter = Counter()
        delta_vals = []
        for s in group:
            for tok, cnt in s.get("top_tokens", []):
                token_counter[tok] += cnt
            for pos, cnt in s.get("top_positions", []):
                pos_counter[str(pos)] += cnt
            if "delta_stats" in s and "mean_abs_delta" in s["delta_stats"]:
                delta_vals.append(s["delta_stats"]["mean_abs_delta"])

        safe_key = key.replace("/", "_")
        plot_bar(
            token_counter,
            f"Top Tokens ({key})",
            "Frequency",
            os.path.join(args.output_dir, f"top_tokens_{safe_key}.png"),
            top_k=args.top_k,
        )
        plot_bar(
            pos_counter,
            f"Top Positions ({key})",
            "Frequency",
            os.path.join(args.output_dir, f"top_positions_{safe_key}.png"),
            top_k=args.top_k,
        )
        plot_delta_hist(
            np.array(delta_vals),
            f"Mean |Delta Score| ({key})",
            os.path.join(args.output_dir, f"delta_hist_{safe_key}.png"),
        )

    print(f"✓ Saved plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
