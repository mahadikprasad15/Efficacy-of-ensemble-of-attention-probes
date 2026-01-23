#!/usr/bin/env python3
"""
Compare Single-Domain vs Combined Training
===========================================
Loads results from both training approaches and generates comparison plots.

Usage:
    python scripts/domain_adaptation/compare_training_approaches.py \
        --single_a_results results/single_domain_roleplaying/results.json \
        --single_b_results results/single_domain_insidertrading/results.json \
        --combined_results results/combined_all_pooling/results.json \
        --output_dir results/training_comparison
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

POOLING_TYPES = ['mean', 'max', 'last', 'attn']
POOLING_COLORS = {
    'mean': '#2E86AB',
    'max': '#A23B72',
    'last': '#F18F01',
    'attn': '#06A77D'
}


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_combined_vs_single_bar(single_a, single_b, combined, output_path, label_a, label_b):
    """
    Bar chart comparing:
    - Single A → OOD B
    - Single B → OOD A  
    - Combined → A
    - Combined → B
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Helper to get best AUC for a pooling type
    def get_best_auc(results, pooling, key):
        if pooling not in results or not results[pooling]:
            return 0.5
        return max(r[key] for r in results[pooling])
    
    # Left plot: Performance on Domain A
    ax = axes[0]
    x = np.arange(len(POOLING_TYPES))
    width = 0.25
    
    single_a_id = [get_best_auc(single_a, p, 'auc_id') for p in POOLING_TYPES]
    single_b_ood_a = [get_best_auc(single_b, p, 'auc_ood') for p in POOLING_TYPES]  # B trained, tested on A
    combined_a = [get_best_auc(combined, p, 'auc_a') for p in POOLING_TYPES]
    
    ax.bar(x - width, single_a_id, width, label=f'Single ({label_a}) - ID', color='#27ae60', alpha=0.8)
    ax.bar(x, single_b_ood_a, width, label=f'Single ({label_b}) - OOD→{label_a}', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, combined_a, width, label='Combined', color='#3498db', alpha=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pooling Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance on {label_a}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POOLING_TYPES])
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Performance on Domain B
    ax = axes[1]
    
    single_b_id = [get_best_auc(single_b, p, 'auc_id') for p in POOLING_TYPES]
    single_a_ood_b = [get_best_auc(single_a, p, 'auc_ood') for p in POOLING_TYPES]  # A trained, tested on B
    combined_b = [get_best_auc(combined, p, 'auc_b') for p in POOLING_TYPES]
    
    ax.bar(x - width, single_b_id, width, label=f'Single ({label_b}) - ID', color='#27ae60', alpha=0.8)
    ax.bar(x, single_a_ood_b, width, label=f'Single ({label_a}) - OOD→{label_b}', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, combined_b, width, label='Combined', color='#3498db', alpha=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pooling Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance on {label_b}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POOLING_TYPES])
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Single-Domain vs Combined Training: All Pooling Strategies', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_ood_improvement(single_a, single_b, combined, output_path, label_a, label_b):
    """Show how much combined training improves OOD performance."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    def get_best_auc(results, pooling, key):
        if pooling not in results or not results[pooling]:
            return 0.5
        return max(r[key] for r in results[pooling])
    
    x = np.arange(len(POOLING_TYPES))
    width = 0.35
    
    # OOD improvement: Combined - Single OOD
    # For A→B: combined_b - single_a_ood_b
    # For B→A: combined_a - single_b_ood_a
    
    improvement_a_to_b = []
    improvement_b_to_a = []
    
    for p in POOLING_TYPES:
        single_a_ood = get_best_auc(single_a, p, 'auc_ood')
        single_b_ood = get_best_auc(single_b, p, 'auc_ood')
        comb_a = get_best_auc(combined, p, 'auc_a')
        comb_b = get_best_auc(combined, p, 'auc_b')
        
        improvement_a_to_b.append(comb_b - single_a_ood)  # How much better is combined on B vs single A→B
        improvement_b_to_a.append(comb_a - single_b_ood)  # How much better is combined on A vs single B→A
    
    bars1 = ax.bar(x - width/2, improvement_a_to_b, width, label=f'{label_a}→{label_b} OOD Improvement', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, improvement_b_to_a, width, label=f'{label_b}→{label_a} OOD Improvement', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:+.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -12), textcoords='offset points', 
                    ha='center', fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Pooling Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC Improvement (Combined - Single OOD)', fontsize=12, fontweight='bold')
    ax.set_title('OOD Performance Improvement from Combined Training', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POOLING_TYPES])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color positive/negative regions
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_summary_table(single_a, single_b, combined, label_a, label_b):
    """Generate a comprehensive summary table."""
    
    def get_best(results, pooling, key):
        if pooling not in results or not results[pooling]:
            return 0.5, -1
        best = max(results[pooling], key=lambda r: r[key])
        return best[key], best['layer']
    
    lines = []
    lines.append("=" * 100)
    lines.append("COMPREHENSIVE TRAINING APPROACH COMPARISON")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Pooling':<8} | {'Single A→A':<12} | {'Single A→B':<12} | {'Single B→B':<12} | {'Single B→A':<12} | {'Combined→A':<12} | {'Combined→B':<12}")
    lines.append("-" * 100)
    
    for pooling in POOLING_TYPES:
        sa_id, _ = get_best(single_a, pooling, 'auc_id')
        sa_ood, _ = get_best(single_a, pooling, 'auc_ood')
        sb_id, _ = get_best(single_b, pooling, 'auc_id')
        sb_ood, _ = get_best(single_b, pooling, 'auc_ood')
        ca, _ = get_best(combined, pooling, 'auc_a')
        cb, _ = get_best(combined, pooling, 'auc_b')
        
        lines.append(f"{pooling.upper():<8} | {sa_id:.4f}       | {sa_ood:.4f}       | {sb_id:.4f}       | {sb_ood:.4f}       | {ca:.4f}       | {cb:.4f}")
    
    lines.append("=" * 100)
    lines.append("")
    lines.append("KEY:")
    lines.append(f"  • Single A = Trained on {label_a} only")
    lines.append(f"  • Single B = Trained on {label_b} only")
    lines.append(f"  • Combined = Trained on {label_a} + {label_b}")
    lines.append(f"  • →A/→B = Evaluated on {label_a}/{label_b}")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_a_results', type=str, required=True)
    parser.add_argument('--single_b_results', type=str, required=True)
    parser.add_argument('--combined_results', type=str, required=True)
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--output_dir', type=str, default='results/training_comparison')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading results...")
    single_a = load_results(args.single_a_results)
    single_b = load_results(args.single_b_results)
    combined = load_results(args.combined_results)
    
    # Extract per-layer results
    single_a_results = single_a.get('results', single_a.get('per_layer_results', {}))
    single_b_results = single_b.get('results', single_b.get('per_layer_results', {}))
    combined_results = combined.get('per_layer_results', combined.get('results', {}))
    
    print("\nGenerating comparison plots...")
    
    plot_combined_vs_single_bar(
        single_a_results, single_b_results, combined_results,
        os.path.join(args.output_dir, 'combined_vs_single_bar.png'),
        args.label_a, args.label_b
    )
    
    plot_ood_improvement(
        single_a_results, single_b_results, combined_results,
        os.path.join(args.output_dir, 'ood_improvement.png'),
        args.label_a, args.label_b
    )
    
    # Summary table
    summary = generate_summary_table(single_a_results, single_b_results, combined_results, 
                                     args.label_a, args.label_b)
    print("\n" + summary)
    
    with open(os.path.join(args.output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\n✓ Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
