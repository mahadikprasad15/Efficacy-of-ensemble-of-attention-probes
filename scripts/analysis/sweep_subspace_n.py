#!/usr/bin/env python3
"""
Subspace Analysis Sweep: Analyze Existing Bootstrap Probes with Different N
============================================================================
Loads previously trained probes and sweeps across different N values.
Compatible with analyze_bootstrap_subspaces_v2.py probes.

Fixes applied (matching v2):
- Sign-aligns weights before stacking
- No centering in SVD (avoids sign issues)

Usage:
    # First train probes with N=15 (the max):
    python scripts/analysis/analyze_bootstrap_subspaces_v2.py --n_probes 15 ...
    
    # Then sweep across N values:
    python scripts/analysis/sweep_subspace_n.py \
        --probes_dir /path/to/probes_subspaces_v2 \
        --layers 12,14,16,18,20 \
        --n_values 5,8,10,15 \
        --subspace_dim 5 \
        --output_dir results/subspace_sweep
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
import torch


# ============================================================================
# PROBE LOADING
# ============================================================================
def sign_align_weights(W, reference_idx=0):
    """
    Sign-align all weight vectors to the first one.
    Prevents SVD from being messed up by random sign flips.
    """
    W_aligned = W.copy()
    ref = W[:, reference_idx]
    
    for i in range(W.shape[1]):
        if np.dot(W[:, i], ref) < 0:
            W_aligned[:, i] = -W[:, i]
    
    return W_aligned


def load_bootstrap_probes(probes_dir, domain, layer, max_n=None):
    """
    Load bootstrap probes from disk and sign-align them.
    
    Returns:
        W: (input_dim, n_probes) matrix of weight vectors (sign-aligned)
        n_loaded: number of probes loaded
    """
    probe_dir = os.path.join(probes_dir, domain, f'layer_{layer}')
    
    if not os.path.exists(probe_dir):
        print(f"  Warning: Directory not found: {probe_dir}")
        return None, 0
    
    # Find all probe files
    probe_files = sorted([f for f in os.listdir(probe_dir) if f.startswith('probe_seed_') and f.endswith('.pt')])
    
    if max_n is not None:
        probe_files = probe_files[:max_n]
    
    weight_vectors = []
    for probe_file in probe_files:
        probe_path = os.path.join(probe_dir, probe_file)
        state_dict = torch.load(probe_path, map_location='cpu')
        
        # Extract weight vector
        w = state_dict['classifier.weight'].squeeze().numpy()
        w = w / (np.linalg.norm(w) + 1e-10)  # Unit normalize
        weight_vectors.append(w)
    
    if len(weight_vectors) == 0:
        return None, 0
    
    W = np.column_stack(weight_vectors)
    
    # Sign-align all weights to first one (matching v2)
    W = sign_align_weights(W, reference_idx=0)
    
    return W, len(weight_vectors)


# ============================================================================
# SUBSPACE ANALYSIS
# ============================================================================
def build_orthonormal_basis(W, k=None):
    """Build orthonormal basis using SVD (no centering - matching v2)."""
    # DON'T center - just use raw SVD to get principal directions
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    
    total_var = np.sum(s ** 2)
    explained_variance = (s ** 2) / total_var if total_var > 0 else s * 0
    
    if k is not None:
        k = min(k, U.shape[1])
        U = U[:, :k]
        explained_variance = explained_variance[:k]
    
    return U, explained_variance


def compute_principal_angles(U, V):
    """Compute principal angles between subspaces."""
    angles = subspace_angles(U, V)
    return angles


def compute_metrics(angles):
    """Compute summary metrics."""
    angles_deg = np.degrees(angles)
    return {
        'min_angle': float(angles_deg.min()) if len(angles) > 0 else 90,
        'max_angle': float(angles_deg.max()) if len(angles) > 0 else 90,
        'mean_angle': float(angles_deg.mean()) if len(angles) > 0 else 90,
        'n_aligned_30': int(np.sum(angles_deg < 30)),
        'n_aligned_15': int(np.sum(angles_deg < 15)),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_n_sweep(results, layers, n_values, output_path):
    """Plot min angle vs N for each layer."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(layers)))
    
    for i, layer in enumerate(layers):
        min_angles = [results[layer][n]['metrics']['min_angle'] for n in n_values]
        ax.plot(n_values, min_angles, marker='o', linewidth=2, markersize=10,
                color=colors[i], label=f'Layer {layer}')
    
    ax.axhline(y=15, color='green', linestyle='--', alpha=0.6, label='Strong (15°)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.6, label='Moderate (30°)')
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.6, label='Weak (60°)')
    
    ax.set_xlabel('Number of Probes (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Smallest Principal Angle (degrees)', fontsize=13, fontweight='bold')
    ax.set_title('Subspace Overlap vs Number of Bootstrap Probes\n(Lower = Stronger Shared Structure)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(n_values)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_layer_n_heatmap(results, layers, n_values, output_path):
    """Heatmap of min angles: layers × N."""
    data = np.array([[results[layer][n]['metrics']['min_angle'] 
                      for n in n_values] for layer in layers])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=90)
    
    ax.set_xticks(range(len(n_values)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([f'N={n}' for n in n_values], fontsize=11)
    ax.set_yticklabels([f'Layer {l}' for l in layers], fontsize=11)
    
    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(n_values)):
            val = data[i, j]
            color = 'white' if val > 45 else 'black'
            ax.text(j, i, f'{val:.1f}°', ha='center', va='center', 
                    color=color, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Number of Probes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title('Smallest Principal Angle (degrees)\n(Green = Aligned, Red = Orthogonal)', 
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Angle (degrees)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sweep N values for subspace analysis")
    parser.add_argument('--probes_dir', type=str, required=True, help='Directory with saved probes')
    parser.add_argument('--layers', type=str, default='12,14,16,18,20')
    parser.add_argument('--n_values', type=str, default='5,8,10,15', help='N values to sweep (comma-separated)')
    parser.add_argument('--subspace_dim', type=int, default=5, help='Subspace dimension k')
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--output_dir', type=str, default='results/subspace_sweep')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    layers = [int(l.strip()) for l in args.layers.split(',')]
    n_values = [int(n.strip()) for n in args.n_values.split(',')]
    
    print("=" * 70)
    print("SUBSPACE ANALYSIS N-SWEEP")
    print("Loading existing probes and analyzing with different N")
    print("=" * 70)
    print(f"Probes dir: {args.probes_dir}")
    print(f"Layers: {layers}")
    print(f"N values: {n_values}")
    print(f"Subspace dim: {args.subspace_dim}")
    print("=" * 70)
    
    results = {layer: {} for layer in layers}
    
    for layer in layers:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print("=" * 70)
        
        for n in n_values:
            print(f"\n  N = {n}:")
            
            # Load probes (limited to N)
            W_a, n_a = load_bootstrap_probes(args.probes_dir, args.label_a, layer, max_n=n)
            W_b, n_b = load_bootstrap_probes(args.probes_dir, args.label_b, layer, max_n=n)
            
            if W_a is None or W_b is None:
                print(f"    ⚠ Could not load probes for layer {layer}")
                continue
            
            print(f"    Loaded: {n_a} {args.label_a}, {n_b} {args.label_b} probes")
            
            # Build subspaces
            k = min(args.subspace_dim, n_a, n_b)
            U_a, var_a = build_orthonormal_basis(W_a, k)
            U_b, var_b = build_orthonormal_basis(W_b, k)
            
            # Compute principal angles
            angles = compute_principal_angles(U_a, U_b)
            metrics = compute_metrics(angles)
            
            results[layer][n] = {
                'angles_deg': [float(np.degrees(a)) for a in angles],
                'metrics': metrics,
                'n_probes_a': n_a,
                'n_probes_b': n_b,
                'variance_a': [float(v) for v in var_a],
                'variance_b': [float(v) for v in var_b]
            }
            
            print(f"    Min angle: {metrics['min_angle']:.1f}°, Mean: {metrics['mean_angle']:.1f}°")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Min Angle by Layer × N")
    print("=" * 70)
    
    print(f"\n{'Layer':<10}", end='')
    for n in n_values:
        print(f"{'N='+str(n):<10}", end='')
    print()
    print("-" * (10 + 10*len(n_values)))
    
    for layer in layers:
        print(f"{layer:<10}", end='')
        for n in n_values:
            if n in results[layer]:
                angle = results[layer][n]['metrics']['min_angle']
                print(f"{angle:.1f}°".ljust(10), end='')
            else:
                print("N/A".ljust(10), end='')
        print()
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\nGenerating visualizations...")
    
    plot_n_sweep(results, layers, n_values,
                 os.path.join(args.output_dir, 'n_sweep.png'))
    
    plot_layer_n_heatmap(results, layers, n_values,
                         os.path.join(args.output_dir, 'layer_n_heatmap.png'))
    
    # Save results
    with open(os.path.join(args.output_dir, 'sweep_results.json'), 'w') as f:
        json.dump({
            'config': {
                'layers': layers,
                'n_values': n_values,
                'subspace_dim': args.subspace_dim
            },
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
