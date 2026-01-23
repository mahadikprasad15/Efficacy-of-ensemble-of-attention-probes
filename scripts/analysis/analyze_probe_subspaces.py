#!/usr/bin/env python3
"""
Probe Subspace Analysis: Do Domains Share a Deception Subspace?
================================================================
Uses EXISTING probes across all layers to test the shared subspace hypothesis.

Key insight: Each layer's probe gives a different "view" of the deception direction.
Stacking probes from layers 0-27 builds a subspace. If domains share a subspace,
the principal angles between them will be small.

Method:
1. Load probe weight vectors from all layers (for given pooling type)
2. Stack: W_Roleplaying ∈ ℝ^(dim × n_layers), W_InsiderTrading ∈ ℝ^(dim × n_layers)
3. PCA to get orthonormal bases U, V of dimension k
4. Compute principal angles via SVD(U^T V)
5. Interpret: small angles = shared subspace

Usage:
    python scripts/analysis/analyze_probe_subspaces.py \
        --probes_a /path/to/probes \
        --probes_b /path/to/probes_flipped \
        --probes_combined /path/to/probes_combined \
        --dataset_a Deception-Roleplaying \
        --dataset_b Deception-InsiderTrading \
        --model meta-llama_Llama-3.2-3B-Instruct \
        --pooling mean \
        --subspace_dim 5 \
        --output_dir results/probe_subspace_analysis
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

# ============================================================================
# PROBE LOADING
# ============================================================================
def load_all_layer_weights(probes_dir, model, dataset, pooling, layers=range(28)):
    """
    Load probe weights from all specified layers.
    
    Returns:
        W: (input_dim, n_layers) matrix of stacked weight vectors
        loaded_layers: list of successfully loaded layer indices
    """
    weight_vectors = []
    loaded_layers = []
    
    for layer in layers:
        probe_path = os.path.join(probes_dir, model, dataset, pooling, f'probe_layer_{layer}.pt')
        
        if not os.path.exists(probe_path):
            continue
        
        try:
            state_dict = torch.load(probe_path, map_location='cpu')
            
            # Extract classifier weight (the direction vector)
            if 'classifier.weight' in state_dict:
                w = state_dict['classifier.weight'].squeeze().numpy()
            elif 'net.0.weight' in state_dict:
                # MLP: use first layer weights, take mean across hidden units
                W1 = state_dict['net.0.weight'].numpy()  # (hidden, input)
                w = W1.mean(axis=0)  # (input,)
            else:
                print(f"  Warning: Unknown architecture at layer {layer}")
                continue
            
            # Normalize to unit length
            w = w / (np.linalg.norm(w) + 1e-10)
            weight_vectors.append(w)
            loaded_layers.append(layer)
            
        except Exception as e:
            print(f"  Warning: Could not load layer {layer}: {e}")
    
    if len(weight_vectors) == 0:
        return None, []
    
    W = np.column_stack(weight_vectors)  # (input_dim, n_layers)
    return W, loaded_layers


# ============================================================================
# SUBSPACE ANALYSIS
# ============================================================================
def build_orthonormal_basis(W, k=None):
    """
    Build orthonormal basis from weight matrix using PCA.
    
    W: (input_dim, n_samples)
    k: number of dimensions (default: all)
    
    Returns:
        U: (input_dim, k) orthonormal basis
        explained_variance: variance explained by each component
    """
    # PCA: find principal directions
    W_centered = W - W.mean(axis=1, keepdims=True)
    
    # SVD of W (columns are samples)
    U, s, Vt = np.linalg.svd(W_centered, full_matrices=False)
    
    # Variance explained
    total_var = np.sum(s ** 2)
    explained_variance = (s ** 2) / total_var if total_var > 0 else s * 0
    
    if k is not None:
        k = min(k, U.shape[1])
        U = U[:, :k]
        explained_variance = explained_variance[:k]
    
    return U, explained_variance


def compute_principal_angles(U, V):
    """
    Compute principal angles between two subspaces.
    
    U, V: (input_dim, k) orthonormal bases
    
    Returns:
        angles: principal angles in radians (ascending order)
        cosines: cos(angles) = singular values of U^T V
    """
    # Use scipy for numerical stability
    angles = subspace_angles(U, V)  # Returns in ascending order
    cosines = np.cos(angles)
    return angles, cosines


def compute_subspace_overlap_metrics(angles):
    """
    Compute summary metrics for subspace overlap.
    """
    angles_deg = np.degrees(angles)
    
    return {
        'n_angles': len(angles),
        'min_angle': float(angles_deg.min()) if len(angles) > 0 else 90,
        'max_angle': float(angles_deg.max()) if len(angles) > 0 else 90,
        'mean_angle': float(angles_deg.mean()) if len(angles) > 0 else 90,
        'n_aligned': int(np.sum(angles_deg < 30)),  # Angles < 30° considered aligned
        'n_orthogonal': int(np.sum(angles_deg > 60)),  # Angles > 60° considered orthogonal
        'grassmann_distance': float(np.sqrt(np.sum(angles ** 2)))  # Grassmann distance
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_principal_angles_spectrum(results, output_path):
    """Plot principal angles as a spectrum."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    markers = ['o', 's', '^']
    
    for i, (name, data) in enumerate(results.items()):
        angles_deg = np.degrees(data['angles'])
        x = np.arange(len(angles_deg)) + 1
        
        ax.plot(x, angles_deg, marker=markers[i % len(markers)], 
                color=colors[i % len(colors)], linewidth=2, markersize=8,
                label=f"{name} (mean={data['metrics']['mean_angle']:.1f}°)")
    
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Aligned threshold (30°)')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Orthogonal threshold (60°)')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Completely orthogonal (90°)')
    
    ax.set_xlabel('Principal Angle Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Angle (degrees)', fontsize=13, fontweight='bold')
    ax.set_title('Principal Angles Between Probe Subspaces\n(Lower = More Overlap)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_explained_variance(variances, labels, output_path):
    """Plot explained variance for each domain's probe subspace."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (var, label) in enumerate(zip(variances, labels)):
        cumvar = np.cumsum(var)
        x = np.arange(1, len(var) + 1)
        ax.plot(x, cumvar * 100, marker='o', color=colors[i % len(colors)], 
                linewidth=2, markersize=4, label=label)
    
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% variance')
    
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax.set_title('Probe Weight Variance Across Layers\n(How "subspace-y" is the probe variation?)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(variances[0]) + 0.5)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_subspace_summary(results, output_path):
    """Create summary bar chart of subspace overlap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    mean_angles = [results[n]['metrics']['mean_angle'] for n in names]
    min_angles = [results[n]['metrics']['min_angle'] for n in names]
    n_aligned = [results[n]['metrics']['n_aligned'] for n in names]
    
    x = np.arange(len(names))
    width = 0.3
    
    bars1 = ax.bar(x - width, min_angles, width, label='Smallest Angle', color='#2ecc71')
    bars2 = ax.bar(x, mean_angles, width, label='Mean Angle', color='#3498db')
    
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Subspace Comparison', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('Subspace Overlap Summary\n(Lower = Shared Subspace)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 95)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{bar.get_height():.1f}°', ha='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{bar.get_height():.1f}°', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Probe Subspace Principal Angles Analysis")
    parser.add_argument('--probes_a', type=str, required=True)
    parser.add_argument('--probes_b', type=str, required=True)
    parser.add_argument('--probes_combined', type=str, default=None)
    parser.add_argument('--dataset_a', type=str, default='Deception-Roleplaying')
    parser.add_argument('--dataset_b', type=str, default='Deception-InsiderTrading')
    parser.add_argument('--model', type=str, default='meta-llama_Llama-3.2-3B-Instruct')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--subspace_dim', type=int, default=5, help='Dimension of subspace to compare')
    parser.add_argument('--output_dir', type=str, default='results/probe_subspace_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    label_a = args.dataset_a.replace('Deception-', '')
    label_b = args.dataset_b.replace('Deception-', '')
    
    print("=" * 70)
    print("PROBE SUBSPACE PRINCIPAL ANGLES ANALYSIS")
    print("Testing: Do domains share a deception subspace?")
    print("=" * 70)
    print(f"Pooling: {args.pooling}")
    print(f"Subspace dimension: {args.subspace_dim}")
    
    # ========================================================================
    # 1. LOAD PROBE WEIGHTS FROM ALL LAYERS
    # ========================================================================
    print("\n1. Loading probe weights from all layers...")
    
    W_a, layers_a = load_all_layer_weights(
        args.probes_a, args.model, args.dataset_a, args.pooling
    )
    print(f"   {label_a}: Loaded {len(layers_a)} probes, W shape: {W_a.shape if W_a is not None else 'None'}")
    
    W_b, layers_b = load_all_layer_weights(
        args.probes_b, args.model, args.dataset_b, args.pooling
    )
    print(f"   {label_b}: Loaded {len(layers_b)} probes, W shape: {W_b.shape if W_b is not None else 'None'}")
    
    W_comb = None
    if args.probes_combined:
        W_comb, layers_comb = load_all_layer_weights(
            args.probes_combined, args.model, 'Deception-Combined', args.pooling
        )
        print(f"   Combined: Loaded {len(layers_comb)} probes, W shape: {W_comb.shape if W_comb is not None else 'None'}")
    
    if W_a is None or W_b is None:
        print("ERROR: Could not load probes!")
        return 1
    
    # ========================================================================
    # 2. BUILD ORTHONORMAL BASES (SUBSPACES)
    # ========================================================================
    print("\n2. Building orthonormal bases for probe subspaces...")
    
    k = args.subspace_dim
    
    U_a, var_a = build_orthonormal_basis(W_a, k)
    print(f"   {label_a}: {k}-dim subspace explains {sum(var_a)*100:.1f}% variance")
    
    U_b, var_b = build_orthonormal_basis(W_b, k)
    print(f"   {label_b}: {k}-dim subspace explains {sum(var_b)*100:.1f}% variance")
    
    if W_comb is not None:
        U_comb, var_comb = build_orthonormal_basis(W_comb, k)
        print(f"   Combined: {k}-dim subspace explains {sum(var_comb)*100:.1f}% variance")
    
    # ========================================================================
    # 3. COMPUTE PRINCIPAL ANGLES
    # ========================================================================
    print("\n3. Computing principal angles between subspaces...")
    
    results = {}
    
    # A vs B
    angles_ab, cos_ab = compute_principal_angles(U_a, U_b)
    results[f'{label_a} ↔ {label_b}'] = {
        'angles': angles_ab,
        'cosines': cos_ab,
        'metrics': compute_subspace_overlap_metrics(angles_ab)
    }
    print(f"\n   {label_a} ↔ {label_b}:")
    print(f"      Angles: {[f'{np.degrees(a):.1f}°' for a in angles_ab]}")
    print(f"      Mean: {np.degrees(angles_ab).mean():.1f}°")
    
    if W_comb is not None:
        # A vs Combined
        angles_ac, cos_ac = compute_principal_angles(U_a, U_comb)
        results[f'{label_a} ↔ Combined'] = {
            'angles': angles_ac,
            'cosines': cos_ac,
            'metrics': compute_subspace_overlap_metrics(angles_ac)
        }
        print(f"\n   {label_a} ↔ Combined:")
        print(f"      Angles: {[f'{np.degrees(a):.1f}°' for a in angles_ac]}")
        print(f"      Mean: {np.degrees(angles_ac).mean():.1f}°")
        
        # B vs Combined
        angles_bc, cos_bc = compute_principal_angles(U_b, U_comb)
        results[f'{label_b} ↔ Combined'] = {
            'angles': angles_bc,
            'cosines': cos_bc,
            'metrics': compute_subspace_overlap_metrics(angles_bc)
        }
        print(f"\n   {label_b} ↔ Combined:")
        print(f"      Angles: {[f'{np.degrees(a):.1f}°' for a in angles_bc]}")
        print(f"      Mean: {np.degrees(angles_bc).mean():.1f}°")
    
    # ========================================================================
    # 4. INTERPRETATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    mean_angle_ab = results[f'{label_a} ↔ {label_b}']['metrics']['mean_angle']
    min_angle_ab = results[f'{label_a} ↔ {label_b}']['metrics']['min_angle']
    n_aligned_ab = results[f'{label_a} ↔ {label_b}']['metrics']['n_aligned']
    
    if min_angle_ab < 15:
        print(f"\n✅ STRONG EVIDENCE for shared subspace!")
        print(f"   Smallest angle = {min_angle_ab:.1f}° (< 15°)")
        print(f"   {n_aligned_ab}/{k} dimensions are aligned (< 30°)")
        hypothesis_supported = True
    elif min_angle_ab < 30:
        print(f"\n⚠️ MODERATE EVIDENCE for shared subspace")
        print(f"   Smallest angle = {min_angle_ab:.1f}° (15°-30°)")
        hypothesis_supported = True
    elif mean_angle_ab < 45:
        print(f"\n📊 WEAK EVIDENCE for shared subspace")
        print(f"   Mean angle = {mean_angle_ab:.1f}° (30°-45°)")
        print("   Subspaces partially overlap")
        hypothesis_supported = False
    else:
        print(f"\n❌ NO EVIDENCE for shared subspace")
        print(f"   Mean angle = {mean_angle_ab:.1f}° (> 45°)")
        print("   Domains use genuinely different directions")
        hypothesis_supported = False
    
    # ========================================================================
    # 5. GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n4. Generating visualizations...")
    
    plot_principal_angles_spectrum(results, 
                                    os.path.join(args.output_dir, 'principal_angles_spectrum.png'))
    
    variances = [var_a, var_b]
    var_labels = [label_a, label_b]
    if W_comb is not None:
        variances.append(var_comb)
        var_labels.append('Combined')
    
    plot_explained_variance(variances, var_labels,
                             os.path.join(args.output_dir, 'explained_variance.png'))
    
    plot_subspace_summary(results,
                          os.path.join(args.output_dir, 'subspace_summary.png'))
    
    # ========================================================================
    # 6. SAVE RESULTS
    # ========================================================================
    summary = {
        'config': {
            'pooling': args.pooling,
            'subspace_dim': k,
            'model': args.model,
            'n_layers_a': len(layers_a),
            'n_layers_b': len(layers_b)
        },
        'results': {
            name: {
                'angles_degrees': [float(np.degrees(a)) for a in data['angles']],
                'cosines': [float(c) for c in data['cosines']],
                'metrics': data['metrics']
            }
            for name, data in results.items()
        },
        'conclusion': {
            'hypothesis_supported': hypothesis_supported,
            'min_angle_ab': float(min_angle_ab),
            'mean_angle_ab': float(mean_angle_ab),
            'n_aligned_ab': n_aligned_ab
        }
    }
    
    with open(os.path.join(args.output_dir, 'subspace_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
