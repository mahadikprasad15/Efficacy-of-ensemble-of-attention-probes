#!/usr/bin/env python3
"""
Analyze Probe Directions: Why Do Some Probes Generalize?
=========================================================
This script addresses the question:
- PCA shows domains are separated, but ATTN probe gets 0.81 OOD AUC
- How can this be if there's no shared space?

Answer: PCA captures VARIANCE, not CLASSIFICATION signal!
The probe finds a hyperplane that may be orthogonal to PCA directions.

This script:
1. Loads probes trained on Domain A, Domain B, and Combined
2. Computes cosine similarity between probe weight vectors
3. Projects activations onto the probe directions (not PCA)
4. Shows if deception is separable along the probe direction

Usage:
    python scripts/analysis/analyze_probe_directions.py \
        --act_a /path/to/Deception-Roleplaying/train \
        --act_b /path/to/Deception-InsiderTrading/train \
        --probe_a /path/to/probe_a.pt \
        --probe_b /path/to/probe_b.pt \
        --probe_combined /path/to/probe_combined.pt \
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

# Simple probe model to load weights
class SimpleProbe(nn.Module):
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
    
    def get_first_layer_weights(self):
        """Get the first layer weight matrix (hidden_dim x input_dim)"""
        return self.net[0].weight.data.cpu().numpy()


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


def train_simple_probe(X_train, y_train, device, epochs=30):
    """Train a simple probe and return it."""
    mean, std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train_n = (X_train - mean) / std
    
    model = SimpleProbe(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 32):
            batch_x = X_t[perm[i:i+32]].to(device)
            batch_y = y_t[perm[i:i+32]].to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    return model, mean, std


def get_probe_direction(model):
    """
    Get the effective "probe direction" from the model.
    For a 2-layer MLP, we use the first layer weights averaged across hidden units.
    This gives a rough sense of what input directions matter most.
    """
    W1 = model.net[0].weight.data.cpu().numpy()  # (hidden_dim, input_dim)
    # Use the principal direction of W1
    u, s, vt = np.linalg.svd(W1, full_matrices=False)
    # The first right singular vector is the most important input direction
    return vt[0]


def project_onto_direction(X, direction):
    """Project X onto a direction vector."""
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return X @ direction


def plot_probe_direction_comparison(X_a, y_a, X_b, y_b, probe_a, probe_b, probe_comb, 
                                    mean_a, std_a, mean_b, std_b, mean_comb, std_comb,
                                    output_path, label_a, label_b):
    """
    Project all activations onto each probe's direction and show distributions.
    """
    # Get probe directions
    dir_a = get_probe_direction(probe_a)
    dir_b = get_probe_direction(probe_b)
    dir_comb = get_probe_direction(probe_comb)
    
    # Normalize all data with combined stats for fair comparison
    X_all = np.vstack([X_a, X_b])
    mean_all = X_all.mean(0)
    std_all = X_all.std(0) + 1e-8
    
    X_a_norm = (X_a - mean_all) / std_all
    X_b_norm = (X_b - mean_all) / std_all
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    probes = [
        (dir_a, f'Probe trained on {label_a}'),
        (dir_b, f'Probe trained on {label_b}'),
        (dir_comb, f'Probe trained on Combined')
    ]
    
    for col, (direction, title) in enumerate(probes):
        # Project onto this direction
        proj_a = project_onto_direction(X_a_norm, direction)
        proj_b = project_onto_direction(X_b_norm, direction)
        
        # Top row: histogram by domain
        ax = axes[0, col]
        ax.hist(proj_a, bins=30, alpha=0.6, label=label_a, color='blue', density=True)
        ax.hist(proj_b, bins=30, alpha=0.6, label=label_b, color='red', density=True)
        ax.set_xlabel('Projection onto Probe Direction', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title}\n(Colored by Domain)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom row: histogram by deception label (both domains combined)
        ax = axes[1, col]
        proj_all = np.concatenate([proj_a, proj_b])
        y_all = np.concatenate([y_a, y_b])
        
        ax.hist(proj_all[y_all == 0], bins=30, alpha=0.6, label='Truthful', color='green', density=True)
        ax.hist(proj_all[y_all == 1], bins=30, alpha=0.6, label='Deceptive', color='purple', density=True)
        ax.set_xlabel('Projection onto Probe Direction', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title}\n(Colored by Deception Label)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Activation Projections onto Different Probe Directions\n' +
                 '(If deception is separable, Truthful and Deceptive should be separated)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_cosine_similarity_matrix(probe_a, probe_b, probe_comb, output_path, label_a, label_b):
    """
    Compute and visualize cosine similarity between probe directions.
    """
    dir_a = get_probe_direction(probe_a)
    dir_b = get_probe_direction(probe_b)
    dir_comb = get_probe_direction(probe_comb)
    
    # Normalize
    dir_a = dir_a / (np.linalg.norm(dir_a) + 1e-8)
    dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-8)
    dir_comb = dir_comb / (np.linalg.norm(dir_comb) + 1e-8)
    
    # Compute cosine similarities
    sim_ab = np.dot(dir_a, dir_b)
    sim_ac = np.dot(dir_a, dir_comb)
    sim_bc = np.dot(dir_b, dir_comb)
    
    # Create matrix
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
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = f'{sim_matrix[i, j]:.3f}'
            color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=14, fontweight='bold')
    
    ax.set_title('Cosine Similarity Between Probe Directions\n' +
                 '(+1 = same direction, 0 = orthogonal, -1 = opposite)',
                 fontsize=13, fontweight='bold')
    
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


def plot_pca_vs_probe_directions(X_a, X_b, probe_a, probe_b, probe_comb, output_path, label_a, label_b):
    """
    Show where probe directions lie relative to PCA directions.
    """
    X_all = np.vstack([X_a, X_b])
    mean_all = X_all.mean(0)
    std_all = X_all.std(0) + 1e-8
    X_all_norm = (X_all - mean_all) / std_all
    
    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(X_all_norm)
    
    # Get probe directions
    dir_a = get_probe_direction(probe_a)
    dir_b = get_probe_direction(probe_b)
    dir_comb = get_probe_direction(probe_comb)
    
    # Normalize
    dir_a = dir_a / np.linalg.norm(dir_a)
    dir_b = dir_b / np.linalg.norm(dir_b)
    dir_comb = dir_comb / np.linalg.norm(dir_comb)
    
    # Project probe directions onto PCA components
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    
    # Compute angles
    def angle_with_pca(direction, pc1, pc2):
        proj1 = np.dot(direction, pc1)
        proj2 = np.dot(direction, pc2)
        # Also compute component orthogonal to PCA plane
        in_pca_plane = proj1 * pc1 + proj2 * pc2
        orthogonal = np.linalg.norm(direction - in_pca_plane)
        return proj1, proj2, orthogonal
    
    proj_a = angle_with_pca(dir_a, pc1, pc2)
    proj_b = angle_with_pca(dir_b, pc1, pc2)
    proj_comb = angle_with_pca(dir_comb, pc1, pc2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: PCA scatter with probe direction arrows
    ax = axes[0]
    X_pca = pca.transform(X_all_norm)
    domain_labels = np.array(['A'] * len(X_a) + ['B'] * len(X_b))
    
    for domain, color, marker in [('A', '#3498db', 'o'), ('B', '#e74c3c', 's')]:
        mask = domain_labels == domain
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, marker=marker, 
                   alpha=0.3, s=30, label=f'{label_a if domain == "A" else label_b}')
    
    # Draw arrows for probe directions (projected onto PCA plane)
    arrow_scale = 20
    ax.annotate('', xy=(proj_a[0]*arrow_scale, proj_a[1]*arrow_scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.annotate('', xy=(proj_b[0]*arrow_scale, proj_b[1]*arrow_scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.annotate('', xy=(proj_comb[0]*arrow_scale, proj_comb[1]*arrow_scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    ax.set_title('Probe Directions in PCA Space', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Bar chart showing how much of probe is orthogonal to PCA
    ax = axes[1]
    probes_names = [f'{label_a}\nProbe', f'{label_b}\nProbe', 'Combined\nProbe']
    orthogonal_components = [proj_a[2], proj_b[2], proj_comb[2]]
    in_plane = [np.sqrt(proj_a[0]**2 + proj_a[1]**2),
                np.sqrt(proj_b[0]**2 + proj_b[1]**2),
                np.sqrt(proj_comb[0]**2 + proj_comb[1]**2)]
    
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, in_plane, width, label='In PCA plane', color='#3498db')
    ax.bar(x + width/2, orthogonal_components, width, label='Orthogonal to PCA', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(probes_names)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Probe Direction: In PCA Plane vs Orthogonal\n' +
                 '(High orthogonal = probe looks at different features than PCA)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    
    print(f"\n  Probe direction analysis:")
    print(f"    {label_a} probe: {in_plane[0]:.3f} in PCA plane, {orthogonal_components[0]:.3f} orthogonal")
    print(f"    {label_b} probe: {in_plane[1]:.3f} in PCA plane, {orthogonal_components[1]:.3f} orthogonal")
    print(f"    Combined probe: {in_plane[2]:.3f} in PCA plane, {orthogonal_components[2]:.3f} orthogonal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--act_a', type=str, required=True)
    parser.add_argument('--act_b', type=str, required=True)
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='results/probe_direction_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("PROBE DIRECTION ANALYSIS")
    print("Why can probes generalize if domains are separated in PCA?")
    print("=" * 70)
    print(f"Layer: {args.layer}, Pooling: {args.pooling}")
    print()
    
    # Load data
    print("Loading activations...")
    X_a, y_a = load_activations(args.act_a, args.layer, args.pooling)
    X_b, y_b = load_activations(args.act_b, args.layer, args.pooling)
    print(f"  {args.label_a}: {len(X_a)} samples")
    print(f"  {args.label_b}: {len(X_b)} samples")
    
    # Train probes
    print("\nTraining probes...")
    print(f"  Training on {args.label_a} only...")
    probe_a, mean_a, std_a = train_simple_probe(X_a, y_a, device, args.epochs)
    
    print(f"  Training on {args.label_b} only...")
    probe_b, mean_b, std_b = train_simple_probe(X_b, y_b, device, args.epochs)
    
    print(f"  Training on Combined...")
    X_comb = np.vstack([X_a, X_b])
    y_comb = np.concatenate([y_a, y_b])
    probe_comb, mean_comb, std_comb = train_simple_probe(X_comb, y_comb, device, args.epochs)
    
    # Analysis 1: Cosine similarity between probe directions
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Cosine Similarity Between Probe Directions")
    print("=" * 70)
    cosine_sims = plot_cosine_similarity_matrix(
        probe_a, probe_b, probe_comb,
        os.path.join(args.output_dir, 'probe_cosine_similarity.png'),
        args.label_a, args.label_b
    )
    
    # Analysis 2: Project activations onto probe directions
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Activation Projections onto Probe Directions")
    print("=" * 70)
    plot_probe_direction_comparison(
        X_a, y_a, X_b, y_b,
        probe_a, probe_b, probe_comb,
        mean_a, std_a, mean_b, std_b, mean_comb, std_comb,
        os.path.join(args.output_dir, 'probe_direction_projections.png'),
        args.label_a, args.label_b
    )
    
    # Analysis 3: PCA vs Probe directions
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Probe Directions vs PCA Directions")
    print("=" * 70)
    plot_pca_vs_probe_directions(
        X_a, X_b, probe_a, probe_b, probe_comb,
        os.path.join(args.output_dir, 'pca_vs_probe_directions.png'),
        args.label_a, args.label_b
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    if abs(cosine_sims['a_b']) < 0.3:
        print("• Single-domain probes point in DIFFERENT directions (low cosine similarity)")
        print("  → Each domain has its own 'deception' direction")
    else:
        print("• Single-domain probes point in SIMILAR directions")
        print("  → There may be a shared 'deception' direction")
    
    if abs(cosine_sims['a_comb']) > 0.5 or abs(cosine_sims['b_comb']) > 0.5:
        print("• Combined probe is similar to one of the single-domain probes")
    else:
        print("• Combined probe finds a COMPROMISE direction between the two domains")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
