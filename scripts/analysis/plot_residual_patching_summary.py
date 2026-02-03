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
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(input_dir: str):
    pattern = os.path.join(input_dir, "**", "layer_*.jsonl")
    paths = glob.glob(pattern, recursive=True)
    records = []
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["_path"] = p
                records.append(rec)
    return records


def parse_group(path: str):
    # Expect .../<split>/<domain>/layer_X.jsonl
    parts = path.split(os.sep)
    split = "unknown"
    domain = "unknown"
    for i, part in enumerate(parts):
        if part in ["validation", "test"]:
            split = part
            if i + 1 < len(parts):
                domain = parts[i + 1]
            break
    return split, domain


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

    records = load_jsonl(args.input_dir)
    if not records:
        print("No JSONL files found.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Global aggregation
    token_count = Counter()
    token_weight = Counter()
    pos_count = Counter()
    pos_weight = Counter()
    delta_vals = []
    rel_pos_vals = []
    missing_token = 0
    total_token = 0

    grouped = defaultdict(list)
    for r in records:
        split, domain = parse_group(r["_path"])
        grouped[f"{split}/{domain}"].append(r)

        length = r.get("length", None)
        for t in r.get("top_tokens", []):
            token = t.get("token")
            idx = t.get("index")
            delta = abs(t.get("delta_score", 0.0))
            if token is not None:
                token_count[token] += 1
                token_weight[token] += delta
                total_token += 1
            else:
                missing_token += 1
            pos_count[str(idx)] += 1
            pos_weight[str(idx)] += delta
            delta_vals.append(delta)
            if length and length > 0:
                rel_pos_vals.append(float(idx) / float(length))

    if missing_token > 0:
        print(f"Warning: {missing_token} token entries missing token text.")

    if total_token == 0:
        print("No token text found; skipping token plots (position-only mode).")
    else:
        plot_bar(
            token_count,
            "Top Tokens by Count (All Layers/Splits/Domains)",
            "Count",
            os.path.join(args.output_dir, "top_tokens_count_all.png"),
            top_k=args.top_k,
        )
        plot_bar(
            token_weight,
            "Top Tokens by |Delta| (All Layers/Splits/Domains)",
            "Sum |Delta|",
            os.path.join(args.output_dir, "top_tokens_weight_all.png"),
            top_k=args.top_k,
        )
    plot_bar(
        pos_count,
        "Top Positions by Count (All Layers/Splits/Domains)",
        "Count",
        os.path.join(args.output_dir, "top_positions_count_all.png"),
        top_k=args.top_k,
    )
    plot_bar(
        pos_weight,
        "Top Positions by |Delta| (All Layers/Splits/Domains)",
        "Sum |Delta|",
        os.path.join(args.output_dir, "top_positions_weight_all.png"),
        top_k=args.top_k,
    )
    plot_delta_hist(
        np.array(delta_vals),
        "|Delta Score| Distribution (All Layers/Splits/Domains)",
        os.path.join(args.output_dir, "delta_hist_all.png"),
    )

    if rel_pos_vals:
        plt.figure(figsize=(8, 5))
        plt.hist(rel_pos_vals, bins=20, color="#54a24b", alpha=0.8)
        plt.title("Relative Position Distribution (All)")
        plt.xlabel("Position / Length")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "relative_position_all.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Split/domain specific plots
    for key, group in grouped.items():
        token_count = Counter()
        token_weight = Counter()
        pos_count = Counter()
        pos_weight = Counter()
        delta_vals = []
        rel_pos_vals = []

        group_token_total = 0
        for r in group:
            length = r.get("length", None)
            for t in r.get("top_tokens", []):
                token = t.get("token")
                idx = t.get("index")
                delta = abs(t.get("delta_score", 0.0))
                if token is not None:
                    token_count[token] += 1
                    token_weight[token] += delta
                    group_token_total += 1
                pos_count[str(idx)] += 1
                pos_weight[str(idx)] += delta
                delta_vals.append(delta)
                if length and length > 0:
                    rel_pos_vals.append(float(idx) / float(length))

        safe_key = key.replace("/", "_")
        if group_token_total > 0:
            plot_bar(
                token_count,
                f"Top Tokens by Count ({key})",
                "Count",
                os.path.join(args.output_dir, f"top_tokens_count_{safe_key}.png"),
                top_k=args.top_k,
            )
            plot_bar(
                token_weight,
                f"Top Tokens by |Delta| ({key})",
                "Sum |Delta|",
                os.path.join(args.output_dir, f"top_tokens_weight_{safe_key}.png"),
                top_k=args.top_k,
            )
        plot_bar(
            pos_count,
            f"Top Positions by Count ({key})",
            "Count",
            os.path.join(args.output_dir, f"top_positions_count_{safe_key}.png"),
            top_k=args.top_k,
        )
        plot_bar(
            pos_weight,
            f"Top Positions by |Delta| ({key})",
            "Sum |Delta|",
            os.path.join(args.output_dir, f"top_positions_weight_{safe_key}.png"),
            top_k=args.top_k,
        )
        plot_delta_hist(
            np.array(delta_vals),
            f"|Delta Score| Distribution ({key})",
            os.path.join(args.output_dir, f"delta_hist_{safe_key}.png"),
        )
        if rel_pos_vals:
            plt.figure(figsize=(8, 5))
            plt.hist(rel_pos_vals, bins=20, color="#54a24b", alpha=0.8)
            plt.title(f"Relative Position Distribution ({key})")
            plt.xlabel("Position / Length")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"relative_position_{safe_key}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"✓ Saved plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
