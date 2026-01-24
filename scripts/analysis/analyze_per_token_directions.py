#!/usr/bin/env python3
"""
Per-Token Probe Direction Analysis
===================================

Analyze and compare probe directions across different training approaches:
- Single-domain per-token probes (A, B)
- Combined per-token probes
- (Optional) Pooled probes

Computes:
1. Cosine similarity between probe weight vectors
2. Orthogonal residual directions
3. How much combined probe captures of single-domain directions
4. Visualization of probe directions in 2D

Usage:
    python scripts/analysis/analyze_per_token_directions.py \
        --probe_a data/probes_per_token/.../Deception-Roleplaying/probe_layer_20.pt \
        --probe_b data/probes_per_token_flipped/.../Deception-InsiderTrading/probe_layer_20.pt \
        --probe_combined data/probes_combined_per_token/.../Deception-Combined/probe_layer_20.pt \
        --output_dir results/per_token_direction_analysis
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from safetensors.torch import load_file

# ============================================================================
# MODEL
# ============================================================================
class PerTokenProbe(nn.Module):
    """Simple linear probe."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def get_direction(self):
        """Get normalized probe direction vector."""
        w = self.classifier.weight.squeeze().detach().cpu().numpy()
        return w / (np.linalg.norm(w) + 1e-8)


# ============================================================================
# LOADING
# ============================================================================
def load_per_token_probe(probe_path, device='cpu'):
    """Load a per-token probe."""
    if not os.path.exists(probe_path):
        return None
    
    state_dict = torch.load(probe_path, map_location=device)
    input_dim = state_dict['classifier.weight'].shape[1]
    
    model = PerTokenProbe(input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_activations(act_dir, layer, aggregation='mean'):
    """Load and aggregate activations."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        return None, None
    
    manifest = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            manifest[entry['id']] = entry
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations = []
    labels = []
    
    for eid, entry in manifest.items():
        if eid not in all_tensors:
            continue
        
        label = entry.get('label', -1)
        if label == -1:
            continue
        
        tensor = all_tensors[eid]
        if layer >= tensor.shape[0]:
            layer = tensor.shape[0] - 1
        
        x_layer = tensor[layer].numpy()
        
        if aggregation == 'mean':
            pooled = x_layer.mean(axis=0)
        elif aggregation == 'max':
            pooled = x_layer.max(axis=0)
        elif aggregation == 'last':
            pooled = x_layer[-1]
        else:
            pooled = x_layer.mean(axis=0)
        
        activations.append(pooled)
        labels.append(label)
    
    return np.array(activations), np.array(labels)


# ============================================================================
# ANALYSIS
# ============================================================================
def compute_orthogonal_residual(v_main, v_other):
    """
    Compute the component of v_other that is orthogonal to v_main.
    
    residual = v_other - proj(v_other onto v_main)
    """
    v_main = v_main / (np.linalg.norm(v_main) + 1e-8)
    v_other = v_other / (np.linalg.norm(v_other) + 1e-8)
    
    # Projection of v_other onto v_main
    proj = np.dot(v_other, v_main) * v_main
    
    # Orthogonal residual
    residual = v_other - proj
    residual = residual / (np.linalg.norm(residual) + 1e-8)
    
    return residual


def evaluate_direction(direction, X, y):
    """Evaluate a direction as a probe on given data."""
    # Compute scores
    scores = X @ direction
    
    try:
        auc = roc_auc_score(y, scores)
    except:
        auc = 0.5
    
    return auc


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_direction_similarity(directions, labels, output_path):
    """Plot cosine similarity matrix between directions."""
    n = len(directions)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = np.dot(directions[i], directions[j])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='RdBu', vmin=-1, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=11)
    
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.3f}', ha='center', va='center',
                    color=color, fontsize=12, fontweight='bold')
    
    ax.set_title('Probe Direction Cosine Similarity', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return sim_matrix


def plot_directions_2d(directions, labels, output_path):
    """Visualize directions in 2D using PCA."""
    # Stack directions
    D = np.vstack(directions)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    D_2d = pca.fit_transform(D)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.arrow(0, 0, D_2d[i, 0], D_2d[i, 1], 
                 head_width=0.05, head_length=0.03, fc=color, ec=color,
                 linewidth=2, label=label)
        ax.annotate(label, xy=(D_2d[i, 0], D_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=color)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title('Probe Directions in PCA Space', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_residual_analysis(results, output_path):
    """Plot residual direction analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart of residual AUCs
    ax = axes[0]
    x = np.arange(len(results['residual_aucs']))
    labels = list(results['residual_aucs'].keys())
    aucs = list(results['residual_aucs'].values())
    
    bars = ax.bar(x, aucs, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(labels)],
                  alpha=0.8, edgecolor='black')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Direction', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Residual Direction Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')
    
    # Cosine similarity summary
    ax = axes[1]
    sims = results['cosine_similarities']
    x = np.arange(len(sims))
    labels = list(sims.keys())
    values = list(sims.values())
    
    colors = ['#27ae60' if v > 0.5 else '#e74c3c' if v < -0.5 else '#f39c12' for v in values]
    ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Direction Pair', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Direction Alignment', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze per-token probe directions")
    parser.add_argument('--probe_a', type=str, required=True, help='Probe trained on domain A')
    parser.add_argument('--probe_b', type=str, required=True, help='Probe trained on domain B')
    parser.add_argument('--probe_combined', type=str, required=True, help='Probe trained on combined')
    parser.add_argument('--probe_pooled_a', type=str, default=None, help='Optional pooled probe A')
    parser.add_argument('--probe_pooled_b', type=str, default=None, help='Optional pooled probe B')
    parser.add_argument('--act_a', type=str, default=None, help='Domain A activations for eval')
    parser.add_argument('--act_b', type=str, default=None, help='Domain B activations for eval')
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='results/per_token_direction_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PER-TOKEN PROBE DIRECTION ANALYSIS")
    print("=" * 70)
    
    # Load probes
    print("\n1. Loading probes...")
    probe_a = load_per_token_probe(args.probe_a)
    probe_b = load_per_token_probe(args.probe_b)
    probe_comb = load_per_token_probe(args.probe_combined)
    
    if probe_a is None or probe_b is None or probe_comb is None:
        print("ERROR: Could not load all required probes!")
        return 1
    
    # Extract directions
    dir_a = probe_a.get_direction()
    dir_b = probe_b.get_direction()
    dir_comb = probe_comb.get_direction()
    
    print(f"   {args.label_a} probe: dim={len(dir_a)}")
    print(f"   {args.label_b} probe: dim={len(dir_b)}")
    print(f"   Combined probe: dim={len(dir_comb)}")
    
    # Compute cosine similarities
    print("\n2. Computing cosine similarities...")
    cos_ab = np.dot(dir_a, dir_b)
    cos_ac = np.dot(dir_a, dir_comb)
    cos_bc = np.dot(dir_b, dir_comb)
    
    print(f"   {args.label_a} ↔ {args.label_b}: {cos_ab:.4f}")
    print(f"   {args.label_a} ↔ Combined: {cos_ac:.4f}")
    print(f"   {args.label_b} ↔ Combined: {cos_bc:.4f}")
    
    # Compute orthogonal residuals
    print("\n3. Computing orthogonal residuals...")
    
    # Residual of B orthogonal to A: "What B captures that A doesn't"
    residual_b_orth_a = compute_orthogonal_residual(dir_a, dir_b)
    
    # Residual of A orthogonal to B
    residual_a_orth_b = compute_orthogonal_residual(dir_b, dir_a)
    
    # Residual of Combined orthogonal to A
    residual_comb_orth_a = compute_orthogonal_residual(dir_a, dir_comb)
    
    # Residual of Combined orthogonal to B
    residual_comb_orth_b = compute_orthogonal_residual(dir_b, dir_comb)
    
    print(f"   Residual B⊥A norm: {np.linalg.norm(residual_b_orth_a):.4f}")
    print(f"   Residual A⊥B norm: {np.linalg.norm(residual_a_orth_b):.4f}")
    
    # Collect all directions for visualization
    all_directions = [dir_a, dir_b, dir_comb, residual_b_orth_a]
    all_labels = [args.label_a, args.label_b, 'Combined', f'{args.label_b}⊥{args.label_a}']
    
    # Evaluate residuals if activations provided
    results = {
        'cosine_similarities': {
            f'{args.label_a}↔{args.label_b}': cos_ab,
            f'{args.label_a}↔Combined': cos_ac,
            f'{args.label_b}↔Combined': cos_bc
        },
        'residual_aucs': {}
    }
    
    if args.act_a and args.act_b:
        print("\n4. Evaluating directions on data...")
        
        X_a, y_a = load_activations(args.act_a, args.layer)
        X_b, y_b = load_activations(args.act_b, args.layer)
        
        if X_a is not None and X_b is not None:
            # Normalize
            X_all = np.vstack([X_a, X_b])
            mean = X_all.mean(axis=0)
            std = X_all.std(axis=0) + 1e-8
            X_a = (X_a - mean) / std
            X_b = (X_b - mean) / std
            X_comb = np.vstack([X_a, X_b])
            y_comb = np.concatenate([y_a, y_b])
            
            # Evaluate each direction
            directions_to_eval = {
                f'{args.label_a}_probe': dir_a,
                f'{args.label_b}_probe': dir_b,
                'Combined_probe': dir_comb,
                f'Residual_{args.label_b}⊥{args.label_a}': residual_b_orth_a,
                f'Residual_{args.label_a}⊥{args.label_b}': residual_a_orth_b
            }
            
            print("\n   Direction AUCs on combined data:")
            for name, direction in directions_to_eval.items():
                auc = evaluate_direction(direction, X_comb, y_comb)
                results['residual_aucs'][name] = auc
                print(f"     {name}: {auc:.4f}")
            
            # Also evaluate on each domain separately
            print(f"\n   AUCs on {args.label_a}:")
            for name, direction in directions_to_eval.items():
                auc = evaluate_direction(direction, X_a, y_a)
                print(f"     {name}: {auc:.4f}")
            
            print(f"\n   AUCs on {args.label_b}:")
            for name, direction in directions_to_eval.items():
                auc = evaluate_direction(direction, X_b, y_b)
                print(f"     {name}: {auc:.4f}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    sim_matrix = plot_direction_similarity(
        [dir_a, dir_b, dir_comb],
        [args.label_a, args.label_b, 'Combined'],
        os.path.join(args.output_dir, 'direction_similarity.png')
    )
    print("   ✓ direction_similarity.png")
    
    plot_directions_2d(
        all_directions, all_labels,
        os.path.join(args.output_dir, 'directions_2d.png')
    )
    print("   ✓ directions_2d.png")
    
    if results['residual_aucs']:
        plot_residual_analysis(results, os.path.join(args.output_dir, 'residual_analysis.png'))
        print("   ✓ residual_analysis.png")
    
    # Save results
    with open(os.path.join(args.output_dir, 'direction_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if cos_ab > 0.7:
        print(f"✅ Single-domain probes are ALIGNED (cos={cos_ab:.3f})")
        print("   → They likely detect similar deception features")
    elif cos_ab < 0.3:
        print(f"❌ Single-domain probes are ORTHOGONAL (cos={cos_ab:.3f})")
        print("   → They detect different deception features")
    else:
        print(f"⚠️ Single-domain probes are PARTIALLY aligned (cos={cos_ab:.3f})")
    
    if cos_ac > 0.7 and cos_bc > 0.7:
        print(f"\n✅ Combined probe aligns with BOTH domains")
        print("   → Successfully learned a shared representation!")
    elif cos_ac > cos_bc:
        print(f"\n⚠️ Combined probe is closer to {args.label_a} (cos={cos_ac:.3f} vs {cos_bc:.3f})")
    else:
        print(f"\n⚠️ Combined probe is closer to {args.label_b} (cos={cos_bc:.3f} vs {cos_ac:.3f})")
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
