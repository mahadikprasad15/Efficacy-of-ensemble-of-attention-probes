"""
Compare ensemble results across ALL pooling strategies.

This script creates the ultimate comparison showing:
- All 4 pooling strategies (mean, max, last, attn)
- All 3 ensemble strategies (Mean, Weighted, Gated)
- Across different K% values
- On both validation and OOD datasets

Creates comprehensive heatmaps and comparison charts.

Usage:
    python scripts/compare_all_pooling_ensembles.py \
        --results_dir /content/drive/MyDrive/results/ensembles \
        --output_dir /content/drive/MyDrive/results/final_comparison

Expected structure:
    results_dir/
    ├── mean/
    │   ├── ensemble_k_sweep_validation.json
    │   └── ensemble_k_sweep_ood.json
    ├── max/
    ├── last/
    └── attn/

Output:
    - pooling_ensemble_heatmap.png       # Heatmap: Pooling × Ensemble → Best AUC
    - ensemble_strategy_comparison.png    # All strategies, all K% values
    - optimal_k_analysis.png              # Optimal K% for each combination
    - final_summary.txt                   # Complete comparison table
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Pooling order
POOLING_ORDER = ['mean', 'max', 'last', 'attn']
ENSEMBLE_STRATEGIES = ['mean', 'weighted', 'gated']

# Colors
POOLING_COLORS = {
    'mean': '#2E86AB',
    'max': '#A23B72',
    'last': '#F18F01',
    'attn': '#06A77D'
}


def load_results(results_dir: str, pooling: str, eval_type: str = 'validation') -> Dict:
    """Load ensemble results for a pooling strategy."""
    filename = f"ensemble_k_sweep_{eval_type}.json"
    path = os.path.join(results_dir, pooling, filename)

    if not os.path.exists(path):
        print(f"⚠️  Not found: {path}")
        return None

    with open(path, 'r') as f:
        return json.load(f)


def create_heatmap(
    all_results: Dict[str, Dict],
    eval_type: str,
    save_path: str
):
    """
    Create heatmap: Pooling (rows) × Ensemble (cols) → Best AUC.

    For each (pooling, ensemble) cell, shows the best AUC across all K% values.
    """
    # Prepare data matrix
    data = []
    row_labels = []

    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        row = []
        for ensemble_strategy in ENSEMBLE_STRATEGIES:
            # Find best AUC for this ensemble strategy across all K% values
            aucs = [r[ensemble_strategy]['auc'] for r in all_results[pooling]]
            best_auc = max(aucs)
            row.append(best_auc)

        data.append(row)
        row_labels.append(pooling.upper())

    if not data:
        print(f"No data for heatmap ({eval_type})")
        return

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        xticklabels=[s.capitalize() for s in ENSEMBLE_STRATEGIES],
        yticklabels=row_labels,
        cbar_kws={'label': 'Best AUC'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title(f'Best AUC: Pooling × Ensemble\n{eval_type.capitalize()} Set',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Ensemble Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pooling Strategy', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {save_path}")
    plt.close()


def create_ensemble_comparison_all(
    all_results: Dict[str, Dict],
    ensemble_strategy: str,
    eval_type: str,
    save_path: str
):
    """
    Compare one ensemble strategy across all pooling methods.

    Shows how Mean/Weighted/Gated performs for mean vs max vs last vs attn.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        results = all_results[pooling]
        k_values = [r['k_pct'] for r in results]
        aucs = [r[ensemble_strategy]['auc'] for r in results]
        accs = [r[ensemble_strategy]['acc'] for r in results]

        color = POOLING_COLORS.get(pooling, '#666666')

        # AUC
        ax1.plot(k_values, aucs, marker='o', linewidth=2.5, markersize=7,
                color=color, label=pooling.upper(), alpha=0.85)

        # Mark best
        best_idx = np.argmax(aucs)
        ax1.scatter([k_values[best_idx]], [aucs[best_idx]],
                   color=color, s=200, zorder=5, edgecolors='black',
                   linewidths=2.5, marker='*')

        # Accuracy
        ax2.plot(k_values, accs, marker='s', linewidth=2.5, markersize=7,
                color=color, label=pooling.upper(), alpha=0.85)

    # Style AUC
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax1.set_xlabel('Top-K% Layers', fontsize=13, fontweight='bold')
    ax1.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax1.set_title(f'{ensemble_strategy.capitalize()} Ensemble: All Pooling Strategies\n{eval_type.capitalize()} AUC vs K%',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.45, 1.0)

    # Style Accuracy
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax2.set_xlabel('Top-K% Layers', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'{ensemble_strategy.capitalize()} Ensemble: All Pooling Strategies\n{eval_type.capitalize()} Accuracy vs K%',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison: {save_path}")
    plt.close()


def create_optimal_k_analysis(
    all_results: Dict[str, Dict],
    eval_type: str,
    save_path: str
):
    """
    Analyze optimal K% for each (pooling, ensemble) combination.

    Bar chart showing which K% works best for each configuration.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    x_positions = []
    optimal_k_values = []
    colors_list = []
    labels_list = []

    pos = 0
    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        for ensemble_strategy in ENSEMBLE_STRATEGIES:
            results = all_results[pooling]

            # Find optimal K%
            best = max(results, key=lambda r: r[ensemble_strategy]['auc'])
            optimal_k = best['k_pct']

            x_positions.append(pos)
            optimal_k_values.append(optimal_k)
            colors_list.append(POOLING_COLORS.get(pooling, '#666666'))
            labels_list.append(f"{pooling.upper()}\n{ensemble_strategy.capitalize()}")

            pos += 1

        pos += 0.5  # Gap between pooling strategies

    bars = ax.bar(x_positions, optimal_k_values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Optimal K% Layers', fontsize=13, fontweight='bold')
    ax.set_title(f'Optimal Top-K% Selection\n{eval_type.capitalize()} Set',
                fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar, val in zip(bars, optimal_k_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{int(val)}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved optimal K analysis: {save_path}")
    plt.close()


def generate_final_summary(
    all_results: Dict[str, Dict],
    eval_type: str
) -> str:
    """Generate comprehensive summary table."""
    lines = []
    lines.append("=" * 110)
    lines.append(f"FINAL COMPARISON: ALL POOLING × ALL ENSEMBLE STRATEGIES - {eval_type.upper()}")
    lines.append("=" * 110)
    lines.append("")

    # Header
    lines.append(f"{'Pooling':<10} {'Ensemble':<12} {'Optimal K%':<12} {'# Layers':<12} {'Best AUC':<12} {'Best Acc':<12}")
    lines.append("-" * 110)

    overall_best_auc = 0
    overall_best = None

    for pooling in POOLING_ORDER:
        if pooling not in all_results or not all_results[pooling]:
            continue

        for ensemble_strategy in ENSEMBLE_STRATEGIES:
            results = all_results[pooling]

            # Find best for this combination
            best = max(results, key=lambda r: r[ensemble_strategy]['auc'])

            k_pct = best['k_pct']
            num_layers = best['num_layers']
            auc = best[ensemble_strategy]['auc']
            acc = best[ensemble_strategy]['acc']

            marker = ""
            if auc > overall_best_auc:
                overall_best_auc = auc
                overall_best = (pooling, ensemble_strategy, k_pct, auc)
                marker = " ⭐"

            lines.append(
                f"{pooling.upper():<10} {ensemble_strategy.capitalize():<12} "
                f"{k_pct:<12} {num_layers:<12} {auc:.4f}      {acc:.4f}      {marker}"
            )

    lines.append("=" * 110)
    if overall_best:
        pooling, ens, k, auc = overall_best
        lines.append(f"⭐ Best Overall: {pooling.upper()} + {ens.capitalize()} @ K={k}% | AUC={auc:.4f}")
    lines.append("=" * 110)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare all pooling × ensemble combinations")

    parser.add_argument("--results_dir", type=str, required=True,
                       help="Base directory containing results for all pooling strategies")
    parser.add_argument("--output_dir", type=str, default="results/final_comparison")
    parser.add_argument("--eval_type", type=str, default="validation",
                       choices=['validation', 'ood', 'both'])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 110)
    print("CROSS-POOLING ENSEMBLE COMPARISON")
    print("=" * 110)
    print()

    eval_types = ['validation', 'ood'] if args.eval_type == 'both' else [args.eval_type]

    for eval_type in eval_types:
        print(f"\nProcessing {eval_type} results...")

        # Load all results
        all_results = {}
        for pooling in POOLING_ORDER:
            results = load_results(args.results_dir, pooling, eval_type)
            if results:
                all_results[pooling] = results
                print(f"  ✓ Loaded {pooling}")

        if not all_results:
            print(f"  ❌ No results found for {eval_type}")
            continue

        print(f"\n  Loaded {len(all_results)} pooling strategies")

        # Create heatmap
        heatmap_path = os.path.join(args.output_dir, f"pooling_ensemble_heatmap_{eval_type}.png")
        create_heatmap(all_results, eval_type, heatmap_path)

        # Create comparison plots for each ensemble strategy
        for ensemble_strategy in ENSEMBLE_STRATEGIES:
            plot_path = os.path.join(args.output_dir, f"{ensemble_strategy}_comparison_{eval_type}.png")
            create_ensemble_comparison_all(all_results, ensemble_strategy, eval_type, plot_path)

        # Create optimal K analysis
        optimal_k_path = os.path.join(args.output_dir, f"optimal_k_analysis_{eval_type}.png")
        create_optimal_k_analysis(all_results, eval_type, optimal_k_path)

        # Generate summary
        summary = generate_final_summary(all_results, eval_type)
        print(f"\n{summary}")

        summary_path = os.path.join(args.output_dir, f"final_summary_{eval_type}.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"\n  ✓ Saved summary: {summary_path}")

    print("\n" + "=" * 110)
    print("✓ CROSS-POOLING COMPARISON COMPLETE")
    print("=" * 110)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 110)

    return 0


if __name__ == "__main__":
    exit(main())
