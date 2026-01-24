#!/usr/bin/env python3
"""
Compare Per-Token vs Pooled Probes
==================================

Side-by-side comparison of:
1. Pooled probes (mean/max/last/attn pooling)
2. Per-token probes (Apollo approach)

Generates comparison plots and summary statistics.

Usage:
    python scripts/comparison/compare_per_token_vs_pooled.py \
        --pooled_results data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/mean/layer_results.json \
        --per_token_results data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/layer_results.json \
        --output_dir results/per_token_comparison
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compare per-token vs pooled probes")
    parser.add_argument('--pooled_results', type=str, required=True)
    parser.add_argument('--per_token_results', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/per_token_comparison')
    parser.add_argument('--pooled_label', type=str, default='Pooled (Mean)')
    parser.add_argument('--per_token_label', type=str, default='Per-Token (Apollo)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    with open(args.pooled_results, 'r') as f:
        pooled = json.load(f)
    
    with open(args.per_token_results, 'r') as f:
        per_token = json.load(f)
    
    # Extract layers and AUCs
    pooled_layers = [r['layer'] for r in pooled]
    pooled_aucs = [r.get('val_auc', r.get('auc', 0.5)) for r in pooled]
    
    per_token_layers = [r['layer'] for r in per_token]
    per_token_aucs = [r.get('sample_val_auc', r.get('val_auc', 0.5)) for r in per_token]
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(pooled_layers, pooled_aucs, 'b-o', label=args.pooled_label, linewidth=2, markersize=6)
    ax.plot(per_token_layers, per_token_aucs, 'r-s', label=args.per_token_label, linewidth=2, markersize=6)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Per-Token vs Pooled Probe Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparison.png'), dpi=150)
    
    # Summary stats
    best_pooled = max(pooled_aucs)
    best_per_token = max(per_token_aucs)
    best_pooled_layer = pooled_layers[np.argmax(pooled_aucs)]
    best_per_token_layer = per_token_layers[np.argmax(per_token_aucs)]
    
    mean_diff = np.mean(per_token_aucs) - np.mean(pooled_aucs)
    
    summary = {
        'pooled_best_auc': best_pooled,
        'pooled_best_layer': best_pooled_layer,
        'per_token_best_auc': best_per_token,
        'per_token_best_layer': best_per_token_layer,
        'per_token_improvement': best_per_token - best_pooled,
        'mean_diff': mean_diff
    }
    
    with open(os.path.join(args.output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{args.pooled_label}: Best AUC = {best_pooled:.4f} (Layer {best_pooled_layer})")
    print(f"{args.per_token_label}: Best AUC = {best_per_token:.4f} (Layer {best_per_token_layer})")
    print(f"Improvement: {best_per_token - best_pooled:+.4f}")
    print(f"Mean Diff: {mean_diff:+.4f}")
    print("=" * 60)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
