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


# ============================================================================
# NEW DECOMPOSITION ANALYSIS PLOTS
# ============================================================================

def plot_residual_norm_vs_layer(results, output_dir):
    """Plot 1: Residual norm across layers for each pooling type."""
    print("\n📊 Generating residual norm vs layer plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'mean': '#2ecc71', 'max': '#3498db', 'last': '#e74c3c', 'attn': '#9b59b6'}
    markers = {'mean': 'o', 'max': 's', 'last': '^', 'attn': 'd'}
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            continue
        
        layers = [r['layer'] for r in valid]
        residual_norms = [r['decomposition']['residual_norm'] for r in valid]
        
        ax.plot(layers, residual_norms, f'{markers.get(pooling, "o")}-', 
                linewidth=2, markersize=6, label=f'{pooling.upper()}', 
                color=colors.get(pooling, 'gray'))
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Residual Norm (Invariant Core Magnitude)', fontsize=12)
    ax.set_title('Invariant Core Magnitude Across Layers', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'residual_norm_vs_layer.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


def plot_residual_norm_vs_auc(results, output_dir):
    """Plot 2: Scatter of residual norm vs invariant core AUC."""
    print("\n📊 Generating residual norm vs AUC scatter plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'mean': '#2ecc71', 'max': '#3498db', 'last': '#e74c3c', 'attn': '#9b59b6'}
    
    all_norms, all_aucs = [], []
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            continue
        
        auc_dict_key = 'eval_on_insider' if 'eval_on_insider' in valid[0] else 'ood_auc'
        
        norms = [r['decomposition']['residual_norm'] for r in valid]
        aucs = [r[auc_dict_key]['invariant_core'] for r in valid]
        layers = [r['layer'] for r in valid]
        
        all_norms.extend(norms)
        all_aucs.extend(aucs)
        
        ax.scatter(norms, aucs, c=colors.get(pooling, 'gray'), label=pooling.upper(), 
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Annotate a few points with layer numbers
        for i in range(0, len(layers), 5):
            ax.annotate(f'L{layers[i]}', (norms[i], aucs[i]), fontsize=7, alpha=0.6)
    
    # Add correlation line
    if len(all_norms) > 2:
        z = np.polyfit(all_norms, all_aucs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_norms), max(all_norms), 100)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, linewidth=2)
        
        # Calculate correlation
        corr = np.corrcoef(all_norms, all_aucs)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Residual Norm', fontsize=12)
    ax.set_ylabel('Invariant Core AUC', fontsize=12)
    ax.set_title('Does Larger Invariant Signal → Better OOD Performance?', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'residual_norm_vs_auc.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


def plot_decomposition_coefficients(results, output_dir):
    """Plot 3: Decomposition coefficients (a, b) vs layer."""
    print("\n📊 Generating decomposition coefficients plot...")
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            continue
        
        layers = [r['layer'] for r in valid]
        a_vals = [r['decomposition']['a'] for r in valid]
        b_vals = [r['decomposition']['b'] for r in valid]
        residual_norms = [r['decomposition']['residual_norm'] for r in valid]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(layers, a_vals, 'o-', linewidth=2, markersize=6, 
                label='a (Roleplaying alignment)', color='#3498db')
        ax.plot(layers, b_vals, 's-', linewidth=2, markersize=6, 
                label='b (InsiderTrading orth. alignment)', color='#e74c3c')
        ax.plot(layers, residual_norms, '^--', linewidth=2, markersize=6, 
                label='Residual Norm', color='#2ecc71', alpha=0.7)
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontsize=12)
        ax.set_title(f'Decomposition: w_C = a·e1 + b·e2 + residual ({pooling.upper()})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'decomposition_coefficients_{pooling}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_path}")


def plot_ternary_composition(results, output_dir):
    """Plot 4: Stacked area showing relative contributions |a|², |b|², residual²."""
    print("\n📊 Generating ternary composition plots...")
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        if len(valid) == 0:
            continue
        
        layers = [r['layer'] for r in valid]
        a_sq = [r['decomposition']['a']**2 for r in valid]
        b_sq = [r['decomposition']['b']**2 for r in valid]
        r_sq = [r['decomposition']['residual_norm']**2 for r in valid]
        
        # Normalize to sum to 1
        totals = [a + b + r for a, b, r in zip(a_sq, b_sq, r_sq)]
        a_frac = [a/t for a, t in zip(a_sq, totals)]
        b_frac = [b/t for b, t in zip(b_sq, totals)]
        r_frac = [r/t for r, t in zip(r_sq, totals)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.stackplot(layers, a_frac, b_frac, r_frac,
                     labels=['|a|² (Roleplaying)', '|b|² (InsiderTrading orth.)', '|residual|² (Invariant)'],
                     colors=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Fraction of Combined Probe', fontsize=12)
        ax.set_title(f'Combined Probe Composition ({pooling.upper()})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'ternary_composition_{pooling}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_path}")


def plot_correlation_heatmap(results, output_dir):
    """Plot 5: Correlation heatmap between all metrics."""
    print("\n📊 Generating correlation heatmap...")
    
    # Collect all data across all pooling types
    all_data = []
    
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        for r in valid:
            auc_dict = r.get('eval_on_insider', r.get('ood_auc', {}))
            row = {
                'a': r['decomposition']['a'],
                'b': r['decomposition']['b'],
                'residual_norm': r['decomposition']['residual_norm'],
                'invariant_auc': auc_dict.get('invariant_core', 0),
                'roleplaying_auc': auc_dict.get('roleplaying_OOD', auc_dict.get('roleplaying_raw', 0)),
                'insider_auc': auc_dict.get('insider_ID', auc_dict.get('insider_raw', 0)),
                'combined_auc': auc_dict.get('combined', 0),
            }
            all_data.append(row)
    
    if len(all_data) < 3:
        print("  ⚠ Not enough data for correlation heatmap")
        return
    
    # Build correlation matrix
    keys = ['a', 'b', 'residual_norm', 'invariant_auc', 'roleplaying_auc', 'insider_auc', 'combined_auc']
    labels = ['a', 'b', 'Residual\nNorm', 'Invariant\nAUC', 'Roleplaying\nAUC', 'Insider\nAUC', 'Combined\nAUC']
    
    data_matrix = np.array([[d[k] for k in keys] for d in all_data])
    corr_matrix = np.corrcoef(data_matrix.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Correlation Between Decomposition & AUC Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


def plot_residual_norm_by_pooling(results, output_dir):
    """Plot 6: Compare residual norms across pooling types at each layer."""
    print("\n📊 Generating residual norm by pooling comparison...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    poolings = list(results['results'].keys())
    colors = {'mean': '#2ecc71', 'max': '#3498db', 'last': '#e74c3c', 'attn': '#9b59b6'}
    
    # Get common layers
    all_layers = set()
    for pooling, layer_results in results['results'].items():
        valid = [r for r in layer_results if 'error' not in r]
        all_layers.update([r['layer'] for r in valid])
    layers = sorted(all_layers)
    
    x = np.arange(len(layers))
    width = 0.2
    
    for i, pooling in enumerate(poolings):
        layer_results = results['results'][pooling]
        valid = [r for r in layer_results if 'error' not in r]
        layer_to_norm = {r['layer']: r['decomposition']['residual_norm'] for r in valid}
        
        norms = [layer_to_norm.get(l, 0) for l in layers]
        offset = (i - len(poolings)/2 + 0.5) * width
        ax.bar(x + offset, norms, width, label=pooling.upper(), 
               color=colors.get(pooling, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Residual Norm', fontsize=12)
    ax.set_title('Invariant Core Magnitude by Pooling Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'residual_norm_by_pooling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


# ============================================================================
# SUMMARY AND MAIN
# ============================================================================

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
    
    # Generate original plots
    plot_comparison_per_pooling(results, output_dir)
    plot_invariant_all_pooling(results, output_dir)
    plot_best_layer_summary(results, output_dir)
    
    # Generate new decomposition analysis plots
    plot_residual_norm_vs_layer(results, output_dir)
    plot_residual_norm_vs_auc(results, output_dir)
    plot_decomposition_coefficients(results, output_dir)
    plot_ternary_composition(results, output_dir)
    plot_correlation_heatmap(results, output_dir)
    plot_residual_norm_by_pooling(results, output_dir)
    
    # Print summary
    print_summary_table(results)
    
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
