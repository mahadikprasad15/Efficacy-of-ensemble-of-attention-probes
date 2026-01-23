#!/usr/bin/env python3
"""
Bootstrap Subspace Analysis v2: Fixed Version
==============================================
Fixes from v1:
1. FIXED NORMALIZATION per domain/layer (all probes in same coordinate system)
2. True bootstrap (replace=True) not subsampling
3. Sign-align probes before stacking (prevents SVD issues)
4. Evaluate AUC on full dataset (not just training subset)
5. Added cos(angle) plots

Usage:
    python scripts/analysis/analyze_bootstrap_subspaces_v2.py \
        --act_a /path/to/Deception-Roleplaying/train \
        --act_b /path/to/Deception-InsiderTrading/train \
        --layers 12,14,16,18,20 \
        --pooling mean \
        --n_probes 15 \
        --subspace_dim 5 \
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
# LINEAR PROBE
# ============================================================================
class LinearProbe(nn.Module):
    """Simple linear probe: w^T x + b"""
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.classifier(x).squeeze(-1)
    
    def get_direction(self):
        return self.classifier.weight.data.squeeze().clone()


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
    
    X = torch.stack(activations).float()  # Convert to float32
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y


# ============================================================================
# PROBE TRAINING - with FIXED normalization
# ============================================================================
def train_probe(X_train, y_train, mean, std, seed, epochs=30, lr=1e-2, weight_decay=1e-4):
    """
    Train a single linear probe with FIXED normalization (passed in).
    
    Args:
        X_train: bootstrap sample (raw, unnormalized)
        y_train: labels for bootstrap sample
        mean, std: FIXED normalization from full dataset (same for all probes!)
        seed: random seed
    
    Returns:
        w: weight vector in NORMALIZED space (comparable across probes)
        auc: training AUC
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_train.shape[1]
    model = LinearProbe(input_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Normalize with FIXED mean/std (not recomputed!)
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
    
    # Get weight in normalized space
    w = model.get_direction().numpy()
    
    # Compute AUC
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_norm)).numpy()
    try:
        auc = roc_auc_score(y_train.numpy(), probs)
    except:
        auc = 0.5
    
    return model, w, auc


def evaluate_on_full(model, X_full, y_full, mean, std):
    """Evaluate probe on full dataset (sanity check)."""
    model.eval()
    X_norm = (X_full - mean) / std
    with torch.no_grad():
        probs = torch.sigmoid(model(X_norm)).numpy()
    try:
        auc = roc_auc_score(y_full.numpy(), probs)
    except:
        auc = 0.5
    return auc


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


def train_bootstrap_probes(X, y, mean, std, n_probes, save_dir=None, domain_name=None, layer=None, epochs=30):
    """
    Train multiple probes with TRUE bootstrap (replace=True).
    Uses FIXED normalization for all probes.
    Sign-aligns weights before returning.
    
    Returns:
        W: (input_dim, n_probes) matrix of weight vectors (normalized, sign-aligned)
        train_aucs: AUC on bootstrap sample
        full_aucs: AUC on full dataset (sanity check)
        models: trained models
    """
    weight_vectors = []
    train_aucs = []
    full_aucs = []
    models = []
    
    n_samples = len(X)
    
    for i in range(n_probes):
        seed = i * 42 + 123
        
        # TRUE bootstrap: sample WITH replacement
        np.random.seed(seed)
        indices = np.random.choice(n_samples, n_samples, replace=True)  # TRUE bootstrap!
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Train with FIXED normalization
        model, w, train_auc = train_probe(X_boot, y_boot, mean, std, seed, epochs=epochs)
        
        # Evaluate on FULL dataset (sanity check)
        full_auc = evaluate_on_full(model, X, y, mean, std)
        
        # Normalize w to unit length
        w = w / (np.linalg.norm(w) + 1e-10)
        
        weight_vectors.append(w)
        train_aucs.append(train_auc)
        full_aucs.append(full_auc)
        models.append(model)
        
        # Save probe if requested
        if save_dir is not None and domain_name is not None and layer is not None:
            probe_dir = os.path.join(save_dir, domain_name, f'layer_{layer}')
            os.makedirs(probe_dir, exist_ok=True)
            
            probe_path = os.path.join(probe_dir, f'probe_seed_{i}.pt')
            torch.save(model.state_dict(), probe_path)
    
    W = np.column_stack(weight_vectors)
    
    # Sign-align all weights to first one
    W = sign_align_weights(W, reference_idx=0)
    
    return W, train_aucs, full_aucs, models


# ============================================================================
# SUBSPACE ANALYSIS
# ============================================================================
def build_orthonormal_basis(W, k=None):
    """Build orthonormal basis using SVD (no centering to avoid sign issues)."""
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
    cosines = np.cos(angles)
    return {
        'min_angle': float(angles_deg.min()) if len(angles) > 0 else 90,
        'max_angle': float(angles_deg.max()) if len(angles) > 0 else 90,
        'mean_angle': float(angles_deg.mean()) if len(angles) > 0 else 90,
        'n_aligned_30': int(np.sum(angles_deg < 30)),
        'n_aligned_15': int(np.sum(angles_deg < 15)),
        'cosines': [float(c) for c in cosines]
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
    
    ax.axhline(y=15, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Strong (15°)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='Moderate (30°)')
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Weak (60°)')
    
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Principal Angle (degrees)', fontsize=13, fontweight='bold')
    ax.set_title('Subspace Alignment Across Layers (v2: Fixed Normalization)\n(Lower = Stronger Shared Structure)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 95)
    
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


def plot_cosine_spectrum(all_results, output_path):
    """Plot cos(angle) vs index - often easier to read than degrees."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(4*len(all_results), 5), sharey=True)
    
    if len(all_results) == 1:
        axes = [axes]
    
    layers = sorted(all_results.keys())
    
    for ax, layer in zip(axes, layers):
        cosines = all_results[layer]['metrics']['cosines']
        x = np.arange(1, len(cosines) + 1)
        
        ax.bar(x, cosines, color='#3498db', edgecolor='black')
        ax.axhline(y=0.97, color='green', linestyle='--', alpha=0.6, label='Strong (cos>0.97)')
        ax.axhline(y=0.87, color='orange', linestyle='--', alpha=0.6, label='Moderate')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, label='Weak')
        
        ax.set_xlabel('Principal Component', fontsize=11)
        ax.set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
    
    axes[0].set_ylabel('cos(angle) = Alignment', fontsize=11)
    fig.suptitle('Subspace Alignment: cos(principal angle) per Layer\n(1.0 = identical, 0 = orthogonal)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_auc_comparison(all_results, output_path):
    """Plot train vs full AUC to sanity check probe quality."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = sorted(all_results.keys())
    x = np.arange(len(layers))
    width = 0.2
    
    for i, domain in enumerate(['A', 'B']):
        key_train = f'mean_train_auc_{domain.lower()}'
        key_full = f'mean_full_auc_{domain.lower()}'
        
        train_aucs = [all_results[l][key_train] for l in layers]
        full_aucs = [all_results[l][key_full] for l in layers]
        
        color = '#3498db' if domain == 'A' else '#e74c3c'
        offset = -width if domain == 'A' else width
        
        ax.bar(x + offset - width/2, train_aucs, width, label=f'{domain} Train AUC', 
               color=color, alpha=0.6, edgecolor='black')
        ax.bar(x + offset + width/2, full_aucs, width, label=f'{domain} Full AUC', 
               color=color, alpha=1.0, edgecolor='black')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Probe Quality: Train vs Full Dataset AUC\n(Similar = stable probes, Full < Train = overfitting)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Bootstrap Subspace Analysis v2")
    parser.add_argument('--act_a', type=str, required=True)
    parser.add_argument('--act_b', type=str, required=True)
    parser.add_argument('--layers', type=str, default='12,14,16,18,20')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--n_probes', type=int, default=15)
    parser.add_argument('--subspace_dim', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--output_dir', type=str, default='results/bootstrap_subspace_v2')
    parser.add_argument('--probes_dir', type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.probes_dir is None:
        args.probes_dir = os.path.join(args.output_dir, 'probes')
    
    layers = [int(l.strip()) for l in args.layers.split(',')]
    
    print("=" * 70)
    print("BOOTSTRAP SUBSPACE ANALYSIS v2 (Fixed)")
    print("=" * 70)
    print("Fixes applied:")
    print("  ✓ Fixed normalization per domain/layer (all probes comparable)")
    print("  ✓ True bootstrap (replace=True)")
    print("  ✓ Sign-aligned weights")
    print("  ✓ AUC evaluated on full dataset")
    print("=" * 70)
    print(f"Layers: {layers}")
    print(f"Pooling: {args.pooling}")
    print(f"N probes: {args.n_probes}")
    print(f"Subspace dim: {args.subspace_dim}")
    print("=" * 70)
    
    all_results = {}
    
    for layer in layers:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print("=" * 70)
        
        # 1. Load activations
        print(f"\n1. Loading activations...")
        X_a, y_a = load_activations(args.act_a, layer, args.pooling)
        X_b, y_b = load_activations(args.act_b, layer, args.pooling)
        print(f"   {args.label_a}: {len(X_a)} samples")
        print(f"   {args.label_b}: {len(X_b)} samples")
        
        # 2. Compute FIXED normalization (once per domain!)
        print("\n2. Computing FIXED normalization...")
        mean_a = X_a.mean(dim=0)
        std_a = X_a.std(dim=0).clamp(min=1e-8)
        
        mean_b = X_b.mean(dim=0)
        std_b = X_b.std(dim=0).clamp(min=1e-8)
        
        # Save normalization stats
        norm_dir = os.path.join(args.probes_dir, 'normalization', f'layer_{layer}')
        os.makedirs(norm_dir, exist_ok=True)
        np.savez(os.path.join(norm_dir, f'{args.label_a}_norm.npz'), 
                 mean=mean_a.numpy(), std=std_a.numpy())
        np.savez(os.path.join(norm_dir, f'{args.label_b}_norm.npz'), 
                 mean=mean_b.numpy(), std=std_b.numpy())
        
        # 3. Train bootstrap probes with FIXED normalization
        print(f"\n3. Training {args.n_probes} bootstrap probes per domain...")
        
        print(f"   {args.label_a}...")
        W_a, train_aucs_a, full_aucs_a, _ = train_bootstrap_probes(
            X_a, y_a, mean_a, std_a, args.n_probes,
            save_dir=args.probes_dir, domain_name=args.label_a, layer=layer,
            epochs=args.epochs
        )
        print(f"      Train AUC: {np.mean(train_aucs_a):.4f} ± {np.std(train_aucs_a):.4f}")
        print(f"      Full AUC:  {np.mean(full_aucs_a):.4f} ± {np.std(full_aucs_a):.4f}")
        
        print(f"   {args.label_b}...")
        W_b, train_aucs_b, full_aucs_b, _ = train_bootstrap_probes(
            X_b, y_b, mean_b, std_b, args.n_probes,
            save_dir=args.probes_dir, domain_name=args.label_b, layer=layer,
            epochs=args.epochs
        )
        print(f"      Train AUC: {np.mean(train_aucs_b):.4f} ± {np.std(train_aucs_b):.4f}")
        print(f"      Full AUC:  {np.mean(full_aucs_b):.4f} ± {np.std(full_aucs_b):.4f}")
        
        # 4. Build subspaces
        print(f"\n4. Building {args.subspace_dim}-dim subspaces...")
        k = min(args.subspace_dim, args.n_probes)
        
        U_a, var_a = build_orthonormal_basis(W_a, k)
        U_b, var_b = build_orthonormal_basis(W_b, k)
        
        print(f"   {args.label_a}: top-{k} explain {sum(var_a)*100:.1f}% variance")
        print(f"   {args.label_b}: top-{k} explain {sum(var_b)*100:.1f}% variance")
        
        # 5. Compute principal angles
        print(f"\n5. Computing principal angles...")
        angles = compute_principal_angles(U_a, U_b)
        metrics = compute_metrics(angles)
        
        angles_deg = np.degrees(angles)
        print(f"   Angles: {[f'{a:.1f}°' for a in angles_deg]}")
        print(f"   Cosines: {[f'{c:.3f}' for c in metrics['cosines']]}")
        print(f"   Min: {metrics['min_angle']:.1f}°, Aligned (<30°): {metrics['n_aligned_30']}/{k}")
        
        all_results[layer] = {
            'angles': angles,
            'metrics': metrics,
            'variance_a': var_a.tolist(),
            'variance_b': var_b.tolist(),
            'mean_train_auc_a': float(np.mean(train_aucs_a)),
            'mean_train_auc_b': float(np.mean(train_aucs_b)),
            'mean_full_auc_a': float(np.mean(full_aucs_a)),
            'mean_full_auc_b': float(np.mean(full_aucs_b))
        }
    
    # ========================================================================
    # INTERPRETATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    for layer in layers:
        metrics = all_results[layer]['metrics']
        min_angle = metrics['min_angle']
        n_aligned = metrics['n_aligned_30']
        k = len(metrics['cosines'])
        
        if min_angle < 15:
            verdict = "✅ STRONG shared subspace"
        elif min_angle < 30:
            verdict = "⚠️ MODERATE overlap"
        elif min_angle < 45:
            verdict = "📊 WEAK overlap"
        else:
            verdict = "❌ NO shared structure"
        
        print(f"   Layer {layer}: {verdict}")
        print(f"            min={min_angle:.1f}°, aligned(<30°)={n_aligned}/{k}")
    
    best_layer = min(layers, key=lambda l: all_results[l]['metrics']['min_angle'])
    print(f"\n🏆 Best: Layer {best_layer} (min angle = {all_results[best_layer]['metrics']['min_angle']:.1f}°)")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n6. Generating visualizations...")
    
    plot_layer_comparison(all_results, 
                          os.path.join(args.output_dir, 'layer_comparison.png'))
    
    plot_cosine_spectrum(all_results,
                         os.path.join(args.output_dir, 'cosine_spectrum.png'))
    
    plot_auc_comparison(all_results,
                        os.path.join(args.output_dir, 'auc_comparison.png'))
    
    # Save results
    summary = {
        'config': {
            'layers': layers,
            'pooling': args.pooling,
            'n_probes': args.n_probes,
            'subspace_dim': args.subspace_dim,
            'fixes': ['fixed_normalization', 'true_bootstrap', 'sign_aligned', 'full_auc_eval']
        },
        'results': {
            str(layer): {
                'angles_degrees': [float(np.degrees(a)) for a in all_results[layer]['angles']],
                'cosines': all_results[layer]['metrics']['cosines'],
                'metrics': {k: v for k, v in all_results[layer]['metrics'].items() if k != 'cosines'},
                'variance_a': all_results[layer]['variance_a'],
                'variance_b': all_results[layer]['variance_b'],
                'auc': {
                    'train_a': all_results[layer]['mean_train_auc_a'],
                    'train_b': all_results[layer]['mean_train_auc_b'],
                    'full_a': all_results[layer]['mean_full_auc_a'],
                    'full_b': all_results[layer]['mean_full_auc_b']
                }
            }
            for layer in layers
        },
        'best_layer': best_layer,
        'best_min_angle': float(all_results[best_layer]['metrics']['min_angle'])
    }
    
    with open(os.path.join(args.output_dir, 'subspace_analysis_v2.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
