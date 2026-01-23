#!/usr/bin/env python3
"""
Effective Direction Subspace Analysis
======================================
Proper analysis of 2-layer ReLU MLP probes.

Instead of extracting a single "direction" via SVD(W1), we compute the 
**per-sample effective direction**:

    w_eff(x) = W1^T (m(x) ⊙ w2)

where:
- W1: first layer weights (hidden_dim × input_dim)
- w2: second layer weights (hidden_dim × 1)
- m(x): ReLU activation mask (which hidden units are active for sample x)

This gives us a cloud of directions per domain. We then:
1. Build decision subspaces via PCA on the effective directions
2. Compare subspaces via principal angles
3. Compute mean effective directions
4. Test if there's a shared subspace (invariant core hypothesis)

Usage:
    python scripts/analysis/analyze_effective_directions.py \
        --act_a /path/to/Deception-Roleplaying/validation \
        --act_b /path/to/Deception-InsiderTrading/validation \
        --probe_a /path/to/probe_a.pt \
        --probe_b /path/to/probe_b.pt \
        --probe_combined /path/to/probe_combined.pt \
        --layer 12 --pooling mean \
        --output_dir results/effective_direction_analysis
"""

import os
import sys
import json
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.linalg import subspace_angles

# ============================================================================
# PROBE ARCHITECTURES
# ============================================================================
class SequentialProbe(nn.Module):
    """2-layer MLP probe."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def get_effective_direction(self, x):
        """
        Compute w_eff(x) = W1^T (m(x) ⊙ w2)
        
        x: (batch_size, input_dim) or (input_dim,)
        Returns: (batch_size, input_dim) or (input_dim,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Get weights
        W1 = self.net[0].weight  # (hidden_dim, input_dim)
        b1 = self.net[0].bias    # (hidden_dim,)
        w2 = self.net[3].weight.squeeze()  # (hidden_dim,)
        
        # Compute pre-activations
        h = x @ W1.T + b1  # (batch, hidden_dim)
        
        # ReLU mask
        m = (h > 0).float()  # (batch, hidden_dim)
        
        # Effective direction: W1^T (m ⊙ w2)
        # For each sample: sum_i (m_i * w2_i * W1[i])
        weighted = m * w2.unsqueeze(0)  # (batch, hidden_dim)
        w_eff = weighted @ W1  # (batch, input_dim)
        
        if squeeze_output:
            return w_eff.squeeze(0)
        return w_eff


# ============================================================================
# DATA LOADING
# ============================================================================
def load_activations(act_dir, layer, pooling):
    """Load and pool activations."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations, labels = [], []
    for entry in manifest:
        eid = entry['id']
        if eid in all_tensors:
            tensor = all_tensors[eid]
            x_layer = tensor[layer, :, :]
            if pooling in ['mean', 'attn']:
                pooled = x_layer.mean(dim=0)
            elif pooling == 'max':
                pooled = x_layer.max(dim=0)[0]
            elif pooling == 'last':
                pooled = x_layer[-1, :]
            else:
                pooled = x_layer.mean(dim=0)
            activations.append(pooled)
            labels.append(entry['label'])
    
    return torch.stack(activations), np.array(labels)


def load_probe_as_sequential(probe_path, input_dim):
    """Load probe and convert to SequentialProbe for analysis."""
    device = torch.device('cpu')
    state_dict = torch.load(probe_path, map_location=device)
    
    # Handle different architectures
    if 'net.0.weight' in state_dict:
        hidden_dim = state_dict['net.0.weight'].shape[0]
        probe = SequentialProbe(input_dim, hidden_dim)
        probe.load_state_dict(state_dict)
        print(f"  ✓ Loaded SequentialProbe (hidden={hidden_dim})")
        return probe
    
    # Handle attention probes - extract classifier part
    if 'classifier.0.weight' in state_dict:
        # Attention probe with separate classifier MLP
        hidden_dim = state_dict['classifier.0.weight'].shape[0]
        
        # Create a SequentialProbe and map weights
        probe = SequentialProbe(input_dim, hidden_dim)
        new_state = {
            'net.0.weight': state_dict['classifier.0.weight'],
            'net.0.bias': state_dict['classifier.0.bias'],
            'net.3.weight': state_dict['classifier.3.weight'],
            'net.3.bias': state_dict['classifier.3.bias'],
        }
        probe.load_state_dict(new_state)
        print(f"  ✓ Extracted classifier from AttentionProbe (hidden={hidden_dim})")
        return probe
    
    if 'classifier.weight' in state_dict:
        # Simple classifier without hidden layer - can't do effective direction analysis
        print(f"  ⚠ Probe is linear (no hidden layer), using raw weights")
        return None
    
    print(f"  ⚠ Unknown probe architecture, keys: {list(state_dict.keys())[:5]}")
    return None


# ============================================================================
# EFFECTIVE DIRECTION ANALYSIS
# ============================================================================
def compute_effective_directions(probe, X, y=None):
    """
    Compute effective directions for all samples.
    
    Returns:
        G: (n_samples, input_dim) matrix of effective directions
        labels: corresponding labels if provided
    """
    probe.eval()
    with torch.no_grad():
        w_eff_all = probe.get_effective_direction(X)
    
    # Normalize each direction to unit length
    norms = w_eff_all.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w_eff_normalized = w_eff_all / norms
    
    return w_eff_normalized.numpy()


def compute_decision_subspace(G, n_components=10):
    """
    Compute decision subspace from effective directions matrix.
    
    G: (n_samples, input_dim)
    Returns: basis vectors (input_dim, n_components)
    """
    # PCA on the effective directions
    n_components = min(n_components, G.shape[0], G.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(G)
    
    # Return the principal components as columns
    return pca.components_.T, pca.explained_variance_ratio_


def compute_principal_angles(U1, U2):
    """
    Compute principal angles between two subspaces.
    
    U1, U2: (input_dim, k) orthonormal bases
    Returns: angles in radians
    """
    angles = subspace_angles(U1, U2)
    return angles


def compute_mean_effective_direction(probe, X):
    """
    Compute mean effective direction across samples.
    
    This is: E_x[w_eff(x)] = W1^T E_x[m(x) ⊙ w2]
    """
    probe.eval()
    with torch.no_grad():
        w_eff_all = probe.get_effective_direction(X)
    
    mean_dir = w_eff_all.mean(dim=0)
    mean_dir = mean_dir / (mean_dir.norm() + 1e-8)
    
    return mean_dir.numpy()


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_principal_angles(angles_dict, output_path):
    """Plot principal angles between subspace pairs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(list(angles_dict.values())[0]))
    width = 0.25
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (name, angles) in enumerate(angles_dict.items()):
        angles_deg = np.degrees(angles)
        ax.bar(x + i*width, angles_deg, width, label=name, color=colors[i % len(colors)])
    
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='Orthogonal')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Aligned')
    
    ax.set_xlabel('Principal Angle Index', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('Principal Angles Between Decision Subspaces\n(0° = aligned, 90° = orthogonal)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{i+1}' for i in x])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_effective_direction_clouds(G_a, G_b, G_comb, y_a, y_b, y_comb, 
                                      output_path, label_a, label_b):
    """Visualize effective direction clouds in 2D using PCA."""
    # Combine all directions for joint PCA
    G_all = np.vstack([G_a, G_b, G_comb])
    
    pca = PCA(n_components=2)
    pca.fit(G_all)
    
    G_a_2d = pca.transform(G_a)
    G_b_2d = pca.transform(G_b)
    G_comb_2d = pca.transform(G_comb)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each probe's effective directions
    for ax, G_2d, y, title, color in [
        (axes[0], G_a_2d, y_a, f'{label_a} Probe', '#3498db'),
        (axes[1], G_b_2d, y_b, f'{label_b} Probe', '#e74c3c'),
        (axes[2], G_comb_2d, y_comb, 'Combined Probe', '#2ecc71')
    ]:
        # Color by deception label
        truthful_mask = y == 0
        deceptive_mask = y == 1
        
        ax.scatter(G_2d[truthful_mask, 0], G_2d[truthful_mask, 1], 
                   c='lightblue', marker='o', s=50, alpha=0.6, 
                   label='Truthful', edgecolors=color)
        ax.scatter(G_2d[deceptive_mask, 0], G_2d[deceptive_mask, 1], 
                   c='darkblue', marker='s', s=50, alpha=0.6, 
                   label='Deceptive', edgecolors=color)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        ax.set_title(f'{title}\nEffective Direction Cloud', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Effective Directions w_eff(x) in Shared PCA Space', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_mean_direction_comparison(mean_dirs, output_path, label_a, label_b):
    """Compare mean effective directions."""
    mean_a, mean_b, mean_comb = mean_dirs[label_a], mean_dirs[label_b], mean_dirs['Combined']
    
    # Compute cosine similarities
    cos_ab = np.dot(mean_a, mean_b)
    cos_ac = np.dot(mean_a, mean_comb)
    cos_bc = np.dot(mean_b, mean_comb)
    
    # Create similarity matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    
    labels = [f'{label_a}\nMean', f'{label_b}\nMean', 'Combined\nMean']
    sim_matrix = np.array([
        [1.0, cos_ab, cos_ac],
        [cos_ab, 1.0, cos_bc],
        [cos_ac, cos_bc, 1.0]
    ])
    
    im = ax.imshow(sim_matrix, cmap='RdBu', vmin=-1, vmax=1)
    
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    for i in range(3):
        for j in range(3):
            color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.3f}', ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')
    
    ax.set_title('Cosine Similarity: Mean Effective Directions\n' +
                 'E_x[w_eff(x)] for each probe', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    print(f"\n  Mean Direction Cosine Similarities:")
    print(f"    {label_a} ↔ {label_b}: {cos_ab:.4f}")
    print(f"    {label_a} ↔ Combined: {cos_ac:.4f}")
    print(f"    {label_b} ↔ Combined: {cos_bc:.4f}")
    
    return {'a_b': cos_ab, 'a_comb': cos_ac, 'b_comb': cos_bc}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Effective Direction Subspace Analysis")
    parser.add_argument('--act_a', type=str, required=True)
    parser.add_argument('--act_b', type=str, required=True)
    parser.add_argument('--probe_a', type=str, required=True)
    parser.add_argument('--probe_b', type=str, required=True)
    parser.add_argument('--probe_combined', type=str, required=True)
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--n_components', type=int, default=10, help='Subspace dimension')
    parser.add_argument('--output_dir', type=str, default='results/effective_direction_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("EFFECTIVE DIRECTION SUBSPACE ANALYSIS")
    print("Computing w_eff(x) = W1^T (m(x) ⊙ w2) per sample")
    print("=" * 70)
    print(f"Layer: {args.layer}, Pooling: {args.pooling}")
    print(f"Subspace dimension: {args.n_components}")
    
    # ========================================================================
    # 1. LOAD ACTIVATIONS
    # ========================================================================
    print("\n1. Loading activations...")
    X_a, y_a = load_activations(args.act_a, args.layer, args.pooling)
    X_b, y_b = load_activations(args.act_b, args.layer, args.pooling)
    X_comb = torch.cat([X_a, X_b], dim=0)
    y_comb = np.concatenate([y_a, y_b])
    
    # Normalize
    mean = X_comb.mean(dim=0)
    std = X_comb.std(dim=0).clamp(min=1e-8)
    X_a = (X_a - mean) / std
    X_b = (X_b - mean) / std
    X_comb = (X_comb - mean) / std
    
    input_dim = X_a.shape[1]
    print(f"   {args.label_a}: {len(X_a)} samples, dim={input_dim}")
    print(f"   {args.label_b}: {len(X_b)} samples")
    
    # ========================================================================
    # 2. LOAD PROBES
    # ========================================================================
    print("\n2. Loading probes...")
    print(f"   {args.probe_a}")
    probe_a = load_probe_as_sequential(args.probe_a, input_dim)
    print(f"   {args.probe_b}")
    probe_b = load_probe_as_sequential(args.probe_b, input_dim)
    print(f"   {args.probe_combined}")
    probe_comb = load_probe_as_sequential(args.probe_combined, input_dim)
    
    if probe_a is None or probe_b is None or probe_comb is None:
        print("ERROR: Could not load all probes as SequentialProbe!")
        print("This analysis requires 2-layer MLPs with ReLU activation.")
        return 1
    
    # ========================================================================
    # 3. COMPUTE EFFECTIVE DIRECTIONS
    # ========================================================================
    print("\n3. Computing per-sample effective directions w_eff(x)...")
    
    # For each probe, compute effective directions on combined data
    G_a = compute_effective_directions(probe_a, X_comb)
    G_b = compute_effective_directions(probe_b, X_comb)
    G_comb = compute_effective_directions(probe_comb, X_comb)
    
    print(f"   G_a shape: {G_a.shape} (effective directions from {args.label_a} probe)")
    print(f"   G_b shape: {G_b.shape} (effective directions from {args.label_b} probe)")
    print(f"   G_comb shape: {G_comb.shape} (effective directions from Combined probe)")
    
    # ========================================================================
    # 4. COMPUTE DECISION SUBSPACES
    # ========================================================================
    print("\n4. Computing decision subspaces (PCA on effective directions)...")
    
    U_a, var_a = compute_decision_subspace(G_a, args.n_components)
    U_b, var_b = compute_decision_subspace(G_b, args.n_components)
    U_comb, var_comb = compute_decision_subspace(G_comb, args.n_components)
    
    print(f"   {args.label_a} subspace: {args.n_components} PCs explain {sum(var_a[:args.n_components]):.1%} variance")
    print(f"   {args.label_b} subspace: {args.n_components} PCs explain {sum(var_b[:args.n_components]):.1%} variance")
    print(f"   Combined subspace: {args.n_components} PCs explain {sum(var_comb[:args.n_components]):.1%} variance")
    
    # ========================================================================
    # 5. COMPUTE PRINCIPAL ANGLES BETWEEN SUBSPACES
    # ========================================================================
    print("\n5. Computing principal angles between subspaces...")
    
    angles_ab = compute_principal_angles(U_a, U_b)
    angles_ac = compute_principal_angles(U_a, U_comb)
    angles_bc = compute_principal_angles(U_b, U_comb)
    
    print(f"\n   Principal angles (degrees):")
    print(f"   {args.label_a} ↔ {args.label_b}:  {[f'{np.degrees(a):.1f}°' for a in angles_ab[:5]]}")
    print(f"   {args.label_a} ↔ Combined:       {[f'{np.degrees(a):.1f}°' for a in angles_ac[:5]]}")
    print(f"   {args.label_b} ↔ Combined:       {[f'{np.degrees(a):.1f}°' for a in angles_bc[:5]]}")
    
    # ========================================================================
    # 6. COMPUTE MEAN EFFECTIVE DIRECTIONS
    # ========================================================================
    print("\n6. Computing mean effective directions E_x[w_eff(x)]...")
    
    mean_a = compute_mean_effective_direction(probe_a, X_comb)
    mean_b = compute_mean_effective_direction(probe_b, X_comb)
    mean_comb = compute_mean_effective_direction(probe_comb, X_comb)
    
    mean_dirs = {
        args.label_a: mean_a,
        args.label_b: mean_b,
        'Combined': mean_comb
    }
    
    # ========================================================================
    # 7. GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n7. Generating visualizations...")
    
    # Principal angles plot
    angles_dict = {
        f'{args.label_a} ↔ {args.label_b}': angles_ab,
        f'{args.label_a} ↔ Combined': angles_ac,
        f'{args.label_b} ↔ Combined': angles_bc
    }
    plot_principal_angles(angles_dict, 
                          os.path.join(args.output_dir, 'principal_angles.png'))
    
    # Effective direction clouds
    plot_effective_direction_clouds(G_a, G_b, G_comb, y_comb, y_comb, y_comb,
                                     os.path.join(args.output_dir, 'effective_direction_clouds.png'),
                                     args.label_a, args.label_b)
    
    # Mean direction comparison
    cos_sims = plot_mean_direction_comparison(mean_dirs,
                                               os.path.join(args.output_dir, 'mean_direction_similarity.png'),
                                               args.label_a, args.label_b)
    
    # ========================================================================
    # 8. INTERPRETATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Check if subspaces are aligned (small principal angles) or orthogonal (90°)
    mean_angle_ab = np.mean(np.degrees(angles_ab[:5]))
    mean_angle_ac = np.mean(np.degrees(angles_ac[:5]))
    mean_angle_bc = np.mean(np.degrees(angles_bc[:5]))
    
    if mean_angle_ab < 30:
        print(f"✅ {args.label_a} and {args.label_b} decision subspaces are ALIGNED (avg {mean_angle_ab:.1f}°)")
        print("   → Single-domain probes use similar decision mechanisms")
    elif mean_angle_ab > 60:
        print(f"❌ {args.label_a} and {args.label_b} decision subspaces are ORTHOGONAL (avg {mean_angle_ab:.1f}°)")
        print("   → Single-domain probes use different decision mechanisms")
    else:
        print(f"⚠️ {args.label_a} and {args.label_b} decision subspaces are PARTIALLY aligned (avg {mean_angle_ab:.1f}°)")
    
    if mean_angle_ac < 30 and mean_angle_bc < 30:
        print(f"\n✅ Combined probe subspace aligns with BOTH single-domain probes")
        print("   → Combined training learned a shared mechanism!")
    elif mean_angle_ac < mean_angle_bc:
        print(f"\n⚠️ Combined probe is closer to {args.label_a} ({mean_angle_ac:.1f}° vs {mean_angle_bc:.1f}°)")
    else:
        print(f"\n⚠️ Combined probe is closer to {args.label_b} ({mean_angle_bc:.1f}° vs {mean_angle_ac:.1f}°)")
    
    # Check mean direction alignment
    if cos_sims['a_b'] > 0.5:
        print(f"\n✅ Mean effective directions are aligned (cos={cos_sims['a_b']:.3f})")
    elif cos_sims['a_b'] < -0.5:
        print(f"\n🔄 Mean effective directions are anti-aligned (cos={cos_sims['a_b']:.3f})")
    else:
        print(f"\n❌ Mean effective directions are nearly orthogonal (cos={cos_sims['a_b']:.3f})")
    
    # ========================================================================
    # 9. SAVE SUMMARY
    # ========================================================================
    summary = {
        'config': {
            'layer': args.layer,
            'pooling': args.pooling,
            'n_components': args.n_components
        },
        'principal_angles': {
            'a_b': [float(a) for a in angles_ab],
            'a_comb': [float(a) for a in angles_ac],
            'b_comb': [float(a) for a in angles_bc],
            'a_b_mean_deg': float(mean_angle_ab),
            'a_comb_mean_deg': float(mean_angle_ac),
            'b_comb_mean_deg': float(mean_angle_bc)
        },
        'mean_direction_cosines': {k: float(v) for k, v in cos_sims.items()},
        'variance_explained': {
            args.label_a: [float(v) for v in var_a],
            args.label_b: [float(v) for v in var_b],
            'Combined': [float(v) for v in var_comb]
        }
    }
    
    with open(os.path.join(args.output_dir, 'effective_direction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
