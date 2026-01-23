#!/usr/bin/env python3
"""
Analyze Probe Directions: Why Do Some Probes Generalize?
=========================================================
LOADS EXISTING TRAINED PROBES - does NOT train new ones!

This script:
1. LOADS existing probes from disk (not training new ones)
2. Computes cosine similarity between probe weight vectors
3. Projects activations onto probe directions (not PCA)
4. Shows if deception is separable along the probe direction

Usage:
    python scripts/analysis/analyze_probe_directions.py \
        --act_a /path/to/Deception-Roleplaying/validation \
        --act_b /path/to/Deception-InsiderTrading/validation \
        --probe_a /path/to/probes/mean/probe_layer_20.pt \
        --probe_b /path/to/probes_flipped/mean/probe_layer_20.pt \
        --probe_combined /path/to/combined_probes/probe.pt \
        --layer 20 --pooling mean \
        --output_dir results/probe_direction_analysis
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

# Add path for LayerProbe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../actprobe/src'))

try:
    from actprobe.probes.models import LayerProbe
    HAS_LAYERPROBE = True
except ImportError:
    HAS_LAYERPROBE = False
    print("Warning: Could not import LayerProbe, will use simple probe structure")


# Simple probe for fallback
class SimpleProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


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
            if pooling == 'mean':
                pooled = x_layer.mean(dim=0)
            elif pooling == 'max':
                pooled = x_layer.max(dim=0)[0]
            elif pooling == 'last':
                pooled = x_layer[-1, :]
            activations.append(pooled.numpy())
            labels.append(entry['label'])
    
    return np.array(activations), np.array(labels)


def load_probe(probe_path, input_dim, pooling='mean'):
    """Load a probe from disk."""
    device = torch.device('cpu')
    
    if HAS_LAYERPROBE:
        try:
            probe = LayerProbe(input_dim=input_dim, pooling_type=pooling)
            probe.load_state_dict(torch.load(probe_path, map_location=device))
            print(f"  ✓ Loaded LayerProbe from {probe_path}")
            return probe
        except Exception as e:
            print(f"  Warning: Failed to load as LayerProbe: {e}")
    
    # Fallback to simple probe
    try:
        state_dict = torch.load(probe_path, map_location=device)
        # Try to infer structure from state dict
        if 'fc1.weight' in state_dict:
            hidden_dim = state_dict['fc1.weight'].shape[0]
            probe = SimpleProbe(input_dim, hidden_dim)
            probe.load_state_dict(state_dict)
        else:
            # Try loading directly
            probe = SimpleProbe(input_dim)
            probe.load_state_dict(state_dict)
        print(f"  ✓ Loaded SimpleProbe from {probe_path}")
        return probe
    except Exception as e:
        print(f"  ✗ Failed to load probe: {e}")
        return None


def get_probe_direction(probe):
    """
    Extract the primary direction the probe uses for classification.
    For a 2-layer MLP, we use SVD of the first layer weights.
    """
    state_dict = probe.state_dict()
    
    # Find the first linear layer weights
    for key in state_dict:
        if 'weight' in key and len(state_dict[key].shape) == 2:
            W = state_dict[key].cpu().numpy()
            # Use SVD to get the principal direction
            u, s, vt = np.linalg.svd(W, full_matrices=False)
            # Return the first right singular vector (most important input direction)
            return vt[0]
    
    raise ValueError("Could not find weight matrix in probe")


def project_onto_direction(X, direction):
    """Project X onto a direction vector."""
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return X @ direction


def plot_cosine_similarity_matrix(dir_a, dir_b, dir_comb, output_path, label_a, label_b):
    """Compute and visualize cosine similarity between probe directions."""
    # Normalize
    dir_a = dir_a / (np.linalg.norm(dir_a) + 1e-8)
    dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-8)
    dir_comb = dir_comb / (np.linalg.norm(dir_comb) + 1e-8)
    
    sim_ab = np.dot(dir_a, dir_b)
    sim_ac = np.dot(dir_a, dir_comb)
    sim_bc = np.dot(dir_b, dir_comb)
    
    labels = [f'{label_a}\nProbe', f'{label_b}\nProbe', 'Combined\nProbe']
    sim_matrix = np.array([
        [1.0, sim_ab, sim_ac],
        [sim_ab, 1.0, sim_bc],
        [sim_ac, sim_bc, 1.0]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 7))
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
    
    ax.set_title('Cosine Similarity Between Probe Directions\n' +
                 '(+1 = same, 0 = orthogonal, -1 = opposite)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    print(f"\n  Cosine Similarities:")
    print(f"    {label_a} ↔ {label_b}: {sim_ab:.4f}")
    print(f"    {label_a} ↔ Combined: {sim_ac:.4f}")
    print(f"    {label_b} ↔ Combined: {sim_bc:.4f}")
    
    return {'a_b': sim_ab, 'a_comb': sim_ac, 'b_comb': sim_bc}


def plot_projection_histograms(X_a, y_a, X_b, y_b, dir_a, dir_b, dir_comb, 
                                output_path, label_a, label_b):
    """Project activations onto each probe direction and show distributions."""
    # Normalize data
    X_all = np.vstack([X_a, X_b])
    mean_all = X_all.mean(0)
    std_all = X_all.std(0) + 1e-8
    X_a_norm = (X_a - mean_all) / std_all
    X_b_norm = (X_b - mean_all) / std_all
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    probes = [
        (dir_a, f'{label_a} Probe'),
        (dir_b, f'{label_b} Probe'),
        (dir_comb, 'Combined Probe')
    ]
    
    for col, (direction, title) in enumerate(probes):
        proj_a = project_onto_direction(X_a_norm, direction)
        proj_b = project_onto_direction(X_b_norm, direction)
        
        # Top: by domain
        ax = axes[0, col]
        ax.hist(proj_a, bins=30, alpha=0.6, label=label_a, color='blue', density=True)
        ax.hist(proj_b, bins=30, alpha=0.6, label=label_b, color='red', density=True)
        ax.set_xlabel('Projection', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title}\n(by Domain)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom: by label
        ax = axes[1, col]
        proj_all = np.concatenate([proj_a, proj_b])
        y_all = np.concatenate([y_a, y_b])
        
        ax.hist(proj_all[y_all == 0], bins=30, alpha=0.6, label='Truthful', color='green', density=True)
        ax.hist(proj_all[y_all == 1], bins=30, alpha=0.6, label='Deceptive', color='purple', density=True)
        ax.set_xlabel('Projection', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title}\n(by Deception Label)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Activation Projections onto Probe Directions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_pca_vs_probe(X_a, X_b, dir_a, dir_b, dir_comb, output_path, label_a, label_b):
    """Show probe directions relative to PCA directions."""
    X_all = np.vstack([X_a, X_b])
    mean_all = X_all.mean(0)
    std_all = X_all.std(0) + 1e-8
    X_all_norm = (X_all - mean_all) / std_all
    
    pca = PCA(n_components=2)
    pca.fit(X_all_norm)
    X_pca = pca.transform(X_all_norm)
    
    # Normalize probe directions
    dir_a = dir_a / np.linalg.norm(dir_a)
    dir_b = dir_b / np.linalg.norm(dir_b)
    dir_comb = dir_comb / np.linalg.norm(dir_comb)
    
    pc1, pc2 = pca.components_[0], pca.components_[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    domain_labels = np.array(['A'] * len(X_a) + ['B'] * len(X_b))
    for domain, color, marker, label in [('A', '#3498db', 'o', label_a), ('B', '#e74c3c', 's', label_b)]:
        mask = domain_labels == domain
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, marker=marker, alpha=0.3, s=30, label=label)
    
    # Project probe directions onto PCA plane and draw arrows
    scale = 15
    for direction, color, label in [(dir_a, 'blue', f'{label_a} Probe'), 
                                     (dir_b, 'red', f'{label_b} Probe'),
                                     (dir_comb, 'green', 'Combined Probe')]:
        proj1 = np.dot(direction, pc1)
        proj2 = np.dot(direction, pc2)
        ax.annotate('', xy=(proj1*scale, proj2*scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3))
        ax.text(proj1*scale*1.1, proj2*scale*1.1, label, fontsize=10, color=color, fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    ax.set_title('Probe Directions Projected onto PCA Space\n(arrows show where each probe "looks")', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze probe directions from existing trained probes")
    parser.add_argument('--act_a', type=str, required=True, help='Domain A activations dir')
    parser.add_argument('--act_b', type=str, required=True, help='Domain B activations dir')
    parser.add_argument('--probe_a', type=str, required=True, help='Path to probe trained on Domain A')
    parser.add_argument('--probe_b', type=str, required=True, help='Path to probe trained on Domain B')
    parser.add_argument('--probe_combined', type=str, required=True, help='Path to probe trained on Combined')
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--output_dir', type=str, default='results/probe_direction_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PROBE DIRECTION ANALYSIS (Loading Existing Probes)")
    print("=" * 70)
    print(f"Layer: {args.layer}, Pooling: {args.pooling}")
    
    # Load activations
    print("\n1. Loading activations...")
    X_a, y_a = load_activations(args.act_a, args.layer, args.pooling)
    X_b, y_b = load_activations(args.act_b, args.layer, args.pooling)
    print(f"   {args.label_a}: {len(X_a)} samples, dim={X_a.shape[1]}")
    print(f"   {args.label_b}: {len(X_b)} samples")
    
    input_dim = X_a.shape[1]
    
    # Load probes
    print("\n2. Loading probes...")
    probe_a = load_probe(args.probe_a, input_dim, args.pooling)
    probe_b = load_probe(args.probe_b, input_dim, args.pooling)
    probe_comb = load_probe(args.probe_combined, input_dim, args.pooling)
    
    if probe_a is None or probe_b is None or probe_comb is None:
        print("ERROR: Could not load all probes!")
        return 1
    
    # Get probe directions
    print("\n3. Extracting probe directions...")
    dir_a = get_probe_direction(probe_a)
    dir_b = get_probe_direction(probe_b)
    dir_comb = get_probe_direction(probe_comb)
    
    # Cosine similarity
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Cosine Similarity")
    print("=" * 70)
    cosine_sims = plot_cosine_similarity_matrix(
        dir_a, dir_b, dir_comb,
        os.path.join(args.output_dir, 'probe_cosine_similarity.png'),
        args.label_a, args.label_b
    )
    
    # Projection histograms
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Projection Histograms")
    print("=" * 70)
    plot_projection_histograms(
        X_a, y_a, X_b, y_b, dir_a, dir_b, dir_comb,
        os.path.join(args.output_dir, 'probe_projections.png'),
        args.label_a, args.label_b
    )
    
    # PCA comparison
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PCA vs Probe Directions")
    print("=" * 70)
    plot_pca_vs_probe(
        X_a, X_b, dir_a, dir_b, dir_comb,
        os.path.join(args.output_dir, 'pca_vs_probe.png'),
        args.label_a, args.label_b
    )
    
    # Save summary
    summary = {
        'probes': {
            'probe_a': args.probe_a,
            'probe_b': args.probe_b,
            'probe_combined': args.probe_combined
        },
        'cosine_similarities': cosine_sims,
        'layer': args.layer,
        'pooling': args.pooling
    }
    with open(os.path.join(args.output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
