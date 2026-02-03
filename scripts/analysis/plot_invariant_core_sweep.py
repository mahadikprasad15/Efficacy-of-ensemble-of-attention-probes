#!/usr/bin/env python3
"""
Plot Invariant Core Sweep Results
==================================

Visualizes sweep results comparing:
- Invariant Core (domain-invariant residual)
- Roleplaying Probe (OOD: trained on Roleplaying, tested on InsiderTrading)
- InsiderTrading Probe (ID: trained & tested on InsiderTrading)
- Combined Probe

Usage:
    python scripts/analysis/plot_invariant_core_sweep.py \
        --results_path results/invariant_core_sweep/sweep_results.json \
        --output_dir results/invariant_core_sweep/plots

Output:
    - comparison_<pooling>.png: Per-pooling comparison plots
    - invariant_all_pooling.png: All pooling types for invariant core
    - best_layer_summary.png: Bar chart of best layers per method
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_results(results_path):
    """Load sweep results JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_comparison_per_pooling(results, output_dir):
    """
    For each pooling type, plot line chart comparing:
    - Invariant Core
    - Roleplaying Probe (OOD)
    - InsiderTrading Probe (ID) 
    - Combined Probe
    """
    print("\n📊 Generating per-pooling comparison plots...")
    
    for pooling, layer_results in results['results'].items():
        
        # Filter out errors
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            print(f"  ⚠ Skipping {pooling}: no valid results")
            continue
        
        layers = [r['layer'] for r in valid]
        
        # Extract AUCs (handle both old and new key names)
        def get_auc(r, key, fallback_key=None):
            auc_dict = r.get('eval_on_insider', r.get('ood_auc', {}))
            if key in auc_dict:
                return auc_dict[key]
            if fallback_key and fallback_key in auc_dict:
                return auc_dict[fallback_key]
            return 0.5
        
        auc_invariant = [get_auc(r, 'invariant_core') for r in valid]
        auc_roleplaying = [get_auc(r, 'roleplaying_OOD', 'roleplaying_raw') for r in valid]
        auc_insider = [get_auc(r, 'insider_ID', 'insider_raw') for r in valid]
        auc_combined = [get_auc(r, 'combined') for r in valid]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines
        ax.plot(layers, auc_invariant, 'o-', linewidth=2.5, markersize=6, 
                label='Invariant Core', color='#2ecc71')
        ax.plot(layers, auc_roleplaying, 's--', linewidth=1.5, markersize=5, 
                label='Roleplaying Probe (OOD)', color='#3498db', alpha=0.7)
        ax.plot(layers, auc_insider, '^--', linewidth=1.5, markersize=5, 
                label='InsiderTrading Probe (ID)', color='#e74c3c', alpha=0.7)
        ax.plot(layers, auc_combined, 'd-', linewidth=2, markersize=5, 
                label='Combined Probe', color='#9b59b6', alpha=0.8)
        
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
        
        # Mark best invariant layer
        best_idx = np.argmax(auc_invariant)
        best_layer = layers[best_idx]
        best_auc = auc_invariant[best_idx]
        ax.annotate(f'Best: L{best_layer}\nAUC={best_auc:.3f}', 
                    xy=(best_layer, best_auc), 
                    xytext=(best_layer + 2, min(best_auc + 0.05, 0.98)),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                    bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('AUC on InsiderTrading', fontsize=12)
        ax.set_title(f'InsiderTrading Eval: Invariant Core vs Probes ({pooling.upper()} pooling)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'comparison_{pooling}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_path}")


def plot_invariant_all_pooling(results, output_dir):
    """Plot invariant core across all pooling types."""
    print("\n📊 Generating invariant core comparison (all pooling)...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'mean': '#2ecc71', 'max': '#3498db', 'last': '#e74c3c', 'attn': '#9b59b6'}
    markers = {'mean': 'o', 'max': 's', 'last': '^', 'attn': 'd'}
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            continue
        
        layers = [r['layer'] for r in valid]
        auc_dict_key = 'eval_on_insider' if 'eval_on_insider' in valid[0] else 'ood_auc'
        auc_invariant = [r[auc_dict_key]['invariant_core'] for r in valid]
        
        ax.plot(layers, auc_invariant, f'{markers.get(pooling, "o")}-', 
                linewidth=2, markersize=6, 
                label=f'{pooling.upper()}', color=colors.get(pooling, 'gray'))
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUC on InsiderTrading', fontsize=12)
    ax.set_title('Invariant Core Performance Across Pooling Types', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'invariant_all_pooling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


def plot_best_layer_summary(results, output_dir):
    """Bar chart showing best layer and AUC for each method per pooling."""
    print("\n📊 Generating best layer summary...")
    
    summary_data = {}
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            continue
        
        auc_dict_key = 'eval_on_insider' if 'eval_on_insider' in valid[0] else 'ood_auc'
        
        # Find best for each method
        best_inv = max(valid, key=lambda r: r[auc_dict_key]['invariant_core'])
        best_rp = max(valid, key=lambda r: r[auc_dict_key].get('roleplaying_OOD', r[auc_dict_key].get('roleplaying_raw', 0)))
        best_id = max(valid, key=lambda r: r[auc_dict_key].get('insider_ID', r[auc_dict_key].get('insider_raw', 0)))
        best_comb = max(valid, key=lambda r: r[auc_dict_key]['combined'])
        
        summary_data[pooling] = {
            'Invariant Core': best_inv[auc_dict_key]['invariant_core'],
            'Roleplaying (OOD)': best_rp[auc_dict_key].get('roleplaying_OOD', best_rp[auc_dict_key].get('roleplaying_raw', 0)),
            'InsiderTrad (ID)': best_id[auc_dict_key].get('insider_ID', best_id[auc_dict_key].get('insider_raw', 0)),
            'Combined': best_comb[auc_dict_key]['combined']
        }
    
    # Create grouped bar chart
    poolings = list(summary_data.keys())
    methods = ['Invariant Core', 'Roleplaying (OOD)', 'InsiderTrad (ID)', 'Combined']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    x = np.arange(len(poolings))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [summary_data[p][method] for p in poolings]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=color, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Pooling Type', fontsize=12)
    ax.set_ylabel('Best AUC', fontsize=12)
    ax.set_title('Best AUC per Method (at optimal layer)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in poolings])
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.4, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'best_layer_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


def print_summary_table(results):
    """Print a text summary table."""
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS SUMMARY")
    print("=" * 70)
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            continue
        
        auc_dict_key = 'eval_on_insider' if 'eval_on_insider' in valid[0] else 'ood_auc'
        
        best_inv = max(valid, key=lambda r: r[auc_dict_key]['invariant_core'])
        best_rp = max(valid, key=lambda r: r[auc_dict_key].get('roleplaying_OOD', r[auc_dict_key].get('roleplaying_raw', 0)))
        best_id = max(valid, key=lambda r: r[auc_dict_key].get('insider_ID', r[auc_dict_key].get('insider_raw', 0)))
        best_comb = max(valid, key=lambda r: r[auc_dict_key]['combined'])
        
        print(f"\n{pooling.upper()}:")
        print(f"  Invariant Core:      Layer {best_inv['layer']:2d} → AUC {best_inv[auc_dict_key]['invariant_core']:.4f}")
        print(f"  Roleplaying (OOD):   Layer {best_rp['layer']:2d} → AUC {best_rp[auc_dict_key].get('roleplaying_OOD', best_rp[auc_dict_key].get('roleplaying_raw', 0)):.4f}")
        print(f"  InsiderTrad (ID):    Layer {best_id['layer']:2d} → AUC {best_id[auc_dict_key].get('insider_ID', best_id[auc_dict_key].get('insider_raw', 0)):.4f}")
        print(f"  Combined:            Layer {best_comb['layer']:2d} → AUC {best_comb[auc_dict_key]['combined']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot Invariant Core Sweep Results")
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to sweep_results.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as results)')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_path}")
    results = load_results(args.results_path)
    
    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.results_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    plot_comparison_per_pooling(results, output_dir)
    plot_invariant_all_pooling(results, output_dir)
    plot_best_layer_summary(results, output_dir)
    
    # Print summary
    print_summary_table(results)
    
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
