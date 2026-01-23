#!/usr/bin/env python3
"""
Bootstrap Subspace Analysis: Proper Test of Shared Deception Subspace
======================================================================
The CORRECT approach to testing shared subspace hypothesis.

Key insight: Layer 12 and Layer 16 have DIFFERENT coordinate systems.
We must compare subspaces WITHIN the same layer.

Method:
1. Pick a fixed layer (e.g., 12, 16, 20)
2. Train N probes with different random seeds + bootstrapped samples
3. Stack those N weight vectors → build k-dimensional subspace
4. Compare Roleplaying vs InsiderTrading subspaces via principal angles

Usage:
    python scripts/analysis/analyze_bootstrap_subspaces.py \
        --act_a /path/to/Deception-Roleplaying/train \
        --act_b /path/to/Deception-InsiderTrading/train \
        --layers 12,14,16,18,20 \
        --pooling mean \
        --n_probes 10 \
        --subspace_dim 5 \
        --bootstrap_frac 0.8 \
        --output_dir results/bootstrap_subspace_analysis
"""

import os
import sys
import json
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from scipy.linalg import subspace_angles

# ============================================================================
# LINEAR PROBE (matching LayerProbe from actprobe)
# ============================================================================
class LinearProbe(nn.Module):
    """Simple linear probe: w^T x + b"""
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.classifier(x).squeeze(-1)
    
    def get_direction(self):
        """Return the weight vector (the probe direction)."""
        return self.classifier.weight.data.squeeze().clone()


# ============================================================================
# DATA LOADING
# ============================================================================
def load_activations(act_dir, layer, pooling):
    """Load and pool activations from a directory."""
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
            else:
                pooled = x_layer.mean(dim=0)
            
            activations.append(pooled)
            labels.append(entry['label'])
    
    X = torch.stack(activations)
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y


# ============================================================================
# PROBE TRAINING
# ============================================================================
def train_probe(X_train, y_train, seed, epochs=30, lr=1e-2, weight_decay=1e-4):
    """Train a single linear probe with given random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_train.shape[1]
    model = LinearProbe(input_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    # Normalize - convert to float32 (activations may be float16)
    X_train = X_train.float()
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-8)
    X_norm = (X_train - mean) / std
    
    # Training
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_norm))
        
        for i in range(0, len(X_norm), 32):
            batch_x = X_norm[perm[i:i+32]]
            batch_y = y_train[perm[i:i+32]]
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    
    # Get direction (in normalized space, so it's comparable)
    w = model.get_direction().numpy()
    w = w / (np.linalg.norm(w) + 1e-10)  # Unit normalize
    
    # Compute AUC
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_norm)).numpy()
    try:
        auc = roc_auc_score(y_train.numpy(), probs)
    except:
        auc = 0.5
    
    return model, w, auc, mean, std  # Return model and normalization stats


def train_bootstrap_probes(X, y, n_probes, bootstrap_frac=0.8, save_dir=None, domain_name=None, layer=None, **train_kwargs):
    """
    Train multiple probes with different seeds and bootstrapped samples.
    
    Returns:
        W: (input_dim, n_probes) matrix of weight vectors
        aucs: list of training AUCs
        models: list of trained models (if save_dir is provided, they're saved there)
    """
    weight_vectors = []
    aucs = []
    models = []
    
    n_samples = len(X)
    n_bootstrap = int(n_samples * bootstrap_frac)
    
    for i in range(n_probes):
        seed = i * 42 + 123  # Different seed for each probe
        
        # Bootstrap sample
        np.random.seed(seed)
        indices = np.random.choice(n_samples, n_bootstrap, replace=False)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Train probe
        model, w, auc, mean, std = train_probe(X_boot, y_boot, seed, **train_kwargs)
        weight_vectors.append(w)
        aucs.append(auc)
        models.append(model)
        
        # Save probe if save_dir is provided
        if save_dir is not None and domain_name is not None and layer is not None:
            probe_dir = os.path.join(save_dir, domain_name, f'layer_{layer}')
            os.makedirs(probe_dir, exist_ok=True)
            
            probe_path = os.path.join(probe_dir, f'probe_seed_{i}.pt')
            torch.save(model.state_dict(), probe_path)
            
            norm_path = os.path.join(probe_dir, f'norm_seed_{i}.npz')
            np.savez(norm_path, mean=mean.numpy(), std=std.numpy())
    
    W = np.column_stack(weight_vectors)  # (input_dim, n_probes)
    return W, aucs, models


# ============================================================================
# SUBSPACE ANALYSIS
# ============================================================================
def build_orthonormal_basis(W, k=None):
    """Build orthonormal basis from weight matrix using SVD."""
    # Center the weight vectors
    W_centered = W - W.mean(axis=1, keepdims=True)
    
    # SVD
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
    """Compute principal angles between subspaces."""
    angles = subspace_angles(U, V)
    cosines = np.cos(angles)
    return angles, cosines


def compute_metrics(angles):
    """Compute summary metrics."""
    angles_deg = np.degrees(angles)
    return {
        'n_angles': len(angles),
        'min_angle': float(angles_deg.min()) if len(angles) > 0 else 90,
        'max_angle': float(angles_deg.max()) if len(angles) > 0 else 90,
        'mean_angle': float(angles_deg.mean()) if len(angles) > 0 else 90,
        'median_angle': float(np.median(angles_deg)) if len(angles) > 0 else 90,
        'n_aligned_30': int(np.sum(angles_deg < 30)),
        'n_aligned_15': int(np.sum(angles_deg < 15)),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_layer_comparison(all_results, output_path):
    """Plot principal angles across layers."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    layers = sorted(all_results.keys())
    x = np.arange(len(layers))
    width = 0.35
    
    min_angles = [all_results[l]['metrics']['min_angle'] for l in layers]
    mean_angles = [all_results[l]['metrics']['mean_angle'] for l in layers]
    
    bars1 = ax.bar(x - width/2, min_angles, width, label='Smallest Angle', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, mean_angles, width, label='Mean Angle', color='#3498db', edgecolor='black')
    
    ax.axhline(y=15, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Strong alignment (15°)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='Moderate alignment (30°)')
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Weak alignment (60°)')
    
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Principal Angle (degrees)', fontsize=13, fontweight='bold')
    ax.set_title('Subspace Alignment Across Layers\n(Lower = Stronger Shared Structure)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend(loc='upper right', fontsize=10)
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


def plot_angle_spectrum(all_results, output_path):
    """Plot full angle spectrum for each layer."""
    n_layers = len(all_results)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 5), sharey=True)
    
    if n_layers == 1:
        axes = [axes]
    
    layers = sorted(all_results.keys())
    
    for ax, layer in zip(axes, layers):
        angles_deg = np.degrees(all_results[layer]['angles'])
        x = np.arange(1, len(angles_deg) + 1)
        
        ax.bar(x, angles_deg, color='#3498db', edgecolor='black')
        ax.axhline(y=15, color='green', linestyle='--', alpha=0.6)
        ax.axhline(y=30, color='orange', linestyle='--', alpha=0.6)
        ax.axhline(y=60, color='red', linestyle='--', alpha=0.6)
        
        ax.set_xlabel('Angle Index', fontsize=11)
        ax.set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 95)
        ax.grid(True, alpha=0.3, axis='y')
    
    axes[0].set_ylabel('Angle (degrees)', fontsize=11)
    fig.suptitle('Principal Angle Spectrum per Layer', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_probe_auc_distribution(aucs_a, aucs_b, layer, output_path):
    """Plot distribution of probe AUCs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(aucs_a, bins=15, alpha=0.6, label='Roleplaying', color='blue', edgecolor='black')
    ax.hist(aucs_b, bins=15, alpha=0.6, label='InsiderTrading', color='red', edgecolor='black')
    
    ax.axvline(x=np.mean(aucs_a), color='blue', linestyle='--', linewidth=2, 
               label=f'Roleplaying mean: {np.mean(aucs_a):.3f}')
    ax.axvline(x=np.mean(aucs_b), color='red', linestyle='--', linewidth=2,
               label=f'InsiderTrading mean: {np.mean(aucs_b):.3f}')
    
    ax.set_xlabel('Training AUC', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Layer {layer}: Bootstrap Probe AUC Distribution\n(N probes each)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Bootstrap Subspace Analysis")
    parser.add_argument('--act_a', type=str, required=True, help='Domain A training activations')
    parser.add_argument('--act_b', type=str, required=True, help='Domain B training activations')
    parser.add_argument('--layers', type=str, default='12,14,16,18,20', help='Layers to analyze (comma-separated)')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--n_probes', type=int, default=10, help='Number of bootstrap probes per domain')
    parser.add_argument('--subspace_dim', type=int, default=5, help='Subspace dimension k')
    parser.add_argument('--bootstrap_frac', type=float, default=0.8, help='Fraction of data for bootstrap')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--output_dir', type=str, default='results/bootstrap_subspace_analysis')
    parser.add_argument('--probes_dir', type=str, default=None, help='Directory to save probes (default: {output_dir}/probes)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set probes_dir default
    if args.probes_dir is None:
        args.probes_dir = os.path.join(args.output_dir, 'probes')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    layers = [int(l.strip()) for l in args.layers.split(',')]
    
    print("=" * 70)
    print("BOOTSTRAP SUBSPACE ANALYSIS")
    print("Proper test of shared deception subspace hypothesis")
    print("=" * 70)
    print(f"Layers: {layers}")
    print(f"Pooling: {args.pooling}")
    print(f"N probes per domain: {args.n_probes}")
    print(f"Subspace dimension: {args.subspace_dim}")
    print(f"Bootstrap fraction: {args.bootstrap_frac}")
    print("=" * 70)
    
    all_results = {}
    
    for layer in layers:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print("=" * 70)
        
        # 1. Load activations
        print(f"\n1. Loading activations (layer {layer}, pooling {args.pooling})...")
        X_a, y_a = load_activations(args.act_a, layer, args.pooling)
        X_b, y_b = load_activations(args.act_b, layer, args.pooling)
        print(f"   {args.label_a}: {len(X_a)} samples, dim={X_a.shape[1]}")
        print(f"   {args.label_b}: {len(X_b)} samples")
        
        # 2. Train bootstrap probes
        print(f"\n2. Training {args.n_probes} bootstrap probes per domain...")
        
        print(f"   Training {args.label_a} probes...")
        W_a, aucs_a, _ = train_bootstrap_probes(
            X_a, y_a, args.n_probes, 
            bootstrap_frac=args.bootstrap_frac, 
            epochs=args.epochs,
            save_dir=args.probes_dir,
            domain_name=args.label_a,
            layer=layer
        )
        print(f"      Mean AUC: {np.mean(aucs_a):.4f} ± {np.std(aucs_a):.4f}")
        print(f"      ✓ Saved to: {args.probes_dir}/{args.label_a}/layer_{layer}/")
        
        print(f"   Training {args.label_b} probes...")
        W_b, aucs_b, _ = train_bootstrap_probes(
            X_b, y_b, args.n_probes, 
            bootstrap_frac=args.bootstrap_frac, 
            epochs=args.epochs,
            save_dir=args.probes_dir,
            domain_name=args.label_b,
            layer=layer
        )
        print(f"      Mean AUC: {np.mean(aucs_b):.4f} ± {np.std(aucs_b):.4f}")
        print(f"      ✓ Saved to: {args.probes_dir}/{args.label_b}/layer_{layer}/")        
        # 3. Build subspaces
        print(f"\n3. Building {args.subspace_dim}-dim subspaces...")
        k = min(args.subspace_dim, args.n_probes)
        
        U_a, var_a = build_orthonormal_basis(W_a, k)
        print(f"   {args.label_a}: {k} PCs explain {sum(var_a)*100:.1f}% variance")
        
        U_b, var_b = build_orthonormal_basis(W_b, k)
        print(f"   {args.label_b}: {k} PCs explain {sum(var_b)*100:.1f}% variance")
        
        # 4. Compute principal angles
        print(f"\n4. Computing principal angles...")
        angles, cosines = compute_principal_angles(U_a, U_b)
        metrics = compute_metrics(angles)
        
        angles_deg = np.degrees(angles)
        print(f"   Angles: {[f'{a:.1f}°' for a in angles_deg]}")
        print(f"   Min: {metrics['min_angle']:.1f}°, Mean: {metrics['mean_angle']:.1f}°")
        print(f"   Aligned (<30°): {metrics['n_aligned_30']}/{k}")
        
        all_results[layer] = {
            'angles': angles,
            'cosines': cosines,
            'metrics': metrics,
            'variance_a': var_a.tolist(),
            'variance_b': var_b.tolist(),
            'aucs_a': aucs_a,
            'aucs_b': aucs_b
        }
        
        # Plot AUC distribution for this layer
        plot_probe_auc_distribution(
            aucs_a, aucs_b, layer,
            os.path.join(args.output_dir, f'auc_distribution_layer_{layer}.png')
        )
    
    # ========================================================================
    # OVERALL INTERPRETATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("OVERALL INTERPRETATION")
    print("=" * 70)
    
    for layer in layers:
        metrics = all_results[layer]['metrics']
        min_angle = metrics['min_angle']
        
        if min_angle < 15:
            verdict = "✅ STRONG shared subspace"
        elif min_angle < 30:
            verdict = "⚠️ MODERATE overlap"
        elif min_angle < 45:
            verdict = "📊 WEAK overlap"
        else:
            verdict = "❌ NO shared structure"
        
        print(f"   Layer {layer}: {verdict} (min={min_angle:.1f}°, mean={metrics['mean_angle']:.1f}°)")
    
    # Find best layer
    best_layer = min(layers, key=lambda l: all_results[l]['metrics']['min_angle'])
    best_min = all_results[best_layer]['metrics']['min_angle']
    
    print(f"\n🏆 Best alignment at Layer {best_layer} (min angle = {best_min:.1f}°)")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n5. Generating visualizations...")
    
    plot_layer_comparison(all_results, 
                          os.path.join(args.output_dir, 'layer_comparison.png'))
    
    plot_angle_spectrum(all_results,
                        os.path.join(args.output_dir, 'angle_spectrum.png'))
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    summary = {
        'config': {
            'layers': layers,
            'pooling': args.pooling,
            'n_probes': args.n_probes,
            'subspace_dim': args.subspace_dim,
            'bootstrap_frac': args.bootstrap_frac
        },
        'results': {
            str(layer): {
                'angles_degrees': [float(np.degrees(a)) for a in all_results[layer]['angles']],
                'metrics': all_results[layer]['metrics'],
                'variance_explained_a': all_results[layer]['variance_a'],
                'variance_explained_b': all_results[layer]['variance_b'],
                'mean_auc_a': float(np.mean(all_results[layer]['aucs_a'])),
                'mean_auc_b': float(np.mean(all_results[layer]['aucs_b']))
            }
            for layer in layers
        },
        'best_layer': best_layer,
        'best_min_angle': float(best_min)
    }
    
    with open(os.path.join(args.output_dir, 'bootstrap_subspace_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
