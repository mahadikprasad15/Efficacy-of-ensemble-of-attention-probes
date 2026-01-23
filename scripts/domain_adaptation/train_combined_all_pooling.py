#!/usr/bin/env python3
"""
Train All Pooling Types on Combined Dataset
============================================
Trains probes for all 4 pooling strategies (mean, max, last, attn) on combined
domain data (A+B), evaluating on each domain's validation set separately.

Produces:
1. Line plots: AUC vs Layers for each pooling, separate plots for each domain
2. Bar graphs: Best layer AUC per pooling for both domains side-by-side
3. Comprehensive results JSON

Usage:
    python scripts/domain_adaptation/train_combined_all_pooling.py \
        --train_a /path/to/domain_a/train \
        --val_a /path/to/domain_a/validation \
        --train_b /path/to/domain_b/train \
        --val_b /path/to/domain_b/validation \
        --output_dir results/combined_all_pooling
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
from sklearn.metrics import roc_auc_score, accuracy_score
from safetensors.torch import load_file
from tqdm import tqdm
from typing import Dict, List, Tuple

# ============================================================================
# POOLING STRATEGIES
# ============================================================================
POOLING_TYPES = ['mean', 'max', 'last', 'attn']
POOLING_COLORS = {
    'mean': '#2E86AB',  # Blue
    'max': '#A23B72',   # Purple
    'last': '#F18F01',  # Orange
    'attn': '#06A77D'   # Green
}

# ============================================================================
# MODELS
# ============================================================================
class SimpleProbe(nn.Module):
    """Simple 2-layer probe for a single pooling type."""
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


class AttentionPoolingProbe(nn.Module):
    """Probe with learned attention pooling over tokens."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: (B, T, D) -> attention -> (B, D)
        attn_weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)
        return self.classifier(pooled).squeeze(-1)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_layer_activations(act_dir: str, layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load activations for a specific layer (without pooling - keep sequence dim).
    Returns: (N, T, D) activations and labels
    """
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
            tensor = all_tensors[eid]  # (L, T, D)
            x_layer = tensor[layer, :, :]  # (T, D)
            activations.append(x_layer.numpy())
            labels.append(entry['label'])
    
    return np.array(activations), np.array(labels)


def pool_activations(x: np.ndarray, pooling: str) -> np.ndarray:
    """
    Apply pooling to activations.
    x: (N, T, D) -> (N, D)
    """
    if pooling == 'mean':
        return x.mean(axis=1)
    elif pooling == 'max':
        return x.max(axis=1)
    elif pooling == 'last':
        return x[:, -1, :]
    elif pooling == 'attn':
        return x  # Keep full sequence for attention pooling
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


# ============================================================================
# TRAINING
# ============================================================================
def train_probe(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val_a: np.ndarray, 
    y_val_a: np.ndarray,
    X_val_b: np.ndarray, 
    y_val_b: np.ndarray,
    pooling: str,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3
) -> Dict:
    """
    Train a probe and evaluate on both validation sets.
    
    Returns dict with:
        - auc_a: AUC on domain A validation
        - auc_b: AUC on domain B validation
        - acc_a: Accuracy on domain A validation
        - acc_b: Accuracy on domain B validation
    """
    # Handle attention pooling differently
    if pooling == 'attn':
        # X is (N, T, D)
        N, T, D = X_train.shape
        model = AttentionPoolingProbe(D).to(device)
        
        # Normalize along feature dimension
        mean = X_train.mean(axis=(0, 1))
        std = X_train.std(axis=(0, 1)) + 1e-8
        X_train_n = (X_train - mean) / std
        X_val_a_n = (X_val_a - mean) / std
        X_val_b_n = (X_val_b - mean) / std
    else:
        # X is (N, D)
        D = X_train.shape[1]
        model = SimpleProbe(D).to(device)
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_n = (X_train - mean) / std
        X_val_a_n = (X_val_a - mean) / std
        X_val_b_n = (X_val_b - mean) / std
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        
        for i in range(0, len(X_t), 32):
            batch_x = X_t[perm[i:i+32]].to(device)
            batch_y = y_t[perm[i:i+32]].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Domain A
        X_a_t = torch.tensor(X_val_a_n, dtype=torch.float32).to(device)
        probs_a = torch.sigmoid(model(X_a_t)).cpu().numpy()
        
        # Domain B
        X_b_t = torch.tensor(X_val_b_n, dtype=torch.float32).to(device)
        probs_b = torch.sigmoid(model(X_b_t)).cpu().numpy()
    
    try:
        auc_a = roc_auc_score(y_val_a, probs_a)
    except:
        auc_a = 0.5
    
    try:
        auc_b = roc_auc_score(y_val_b, probs_b)
    except:
        auc_b = 0.5
    
    acc_a = accuracy_score(y_val_a, (probs_a > 0.5).astype(int))
    acc_b = accuracy_score(y_val_b, (probs_b > 0.5).astype(int))
    
    return {
        'auc_a': auc_a,
        'auc_b': auc_b,
        'acc_a': acc_a,
        'acc_b': acc_b
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_layerwise_auc(
    results: Dict[str, List[Dict]], 
    domain_label: str,
    domain_key: str,
    output_path: str,
    label_a: str,
    label_b: str
):
    """Plot AUC vs Layer for all pooling types for one domain."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    best_overall = {'auc': 0, 'pooling': None, 'layer': None}
    
    for pooling in POOLING_TYPES:
        if pooling not in results:
            continue
        
        layers = [r['layer'] for r in results[pooling]]
        aucs = [r[domain_key] for r in results[pooling]]
        color = POOLING_COLORS.get(pooling, '#666666')
        
        ax.plot(layers, aucs, marker='o', linewidth=2.5, markersize=6,
                color=color, label=pooling.upper(), alpha=0.85)
        
        # Mark best layer
        best_idx = np.argmax(aucs)
        best_auc = aucs[best_idx]
        best_layer = layers[best_idx]
        
        ax.scatter([best_layer], [best_auc], color=color, s=200, zorder=5,
                   edgecolors='black', linewidths=2.5, marker='*')
        
        # Track overall best
        if best_auc > best_overall['auc']:
            best_overall = {'auc': best_auc, 'pooling': pooling, 'layer': best_layer}
    
    # Add annotation for overall best
    if best_overall['pooling']:
        ax.annotate(
            f"BEST: {best_overall['pooling'].upper()}\nLayer {best_overall['layer']}\nAUC: {best_overall['auc']:.3f}",
            xy=(best_overall['layer'], best_overall['auc']),
            xytext=(15, -15), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=POOLING_COLORS[best_overall['pooling']], alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
            fontsize=10, fontweight='bold'
        )
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5, label='Strong (0.7)')
    
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax.set_title(f'{domain_label} Validation AUC - Combined Training\n(All Pooling Strategies)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    return best_overall


def plot_bar_comparison(
    results: Dict[str, List[Dict]], 
    output_path: str,
    label_a: str,
    label_b: str
):
    """Create side-by-side bar plots comparing best AUC per pooling for both domains."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(POOLING_TYPES))
    width = 0.35
    
    aucs_a = []
    aucs_b = []
    best_layers_a = []
    best_layers_b = []
    
    for pooling in POOLING_TYPES:
        if pooling in results:
            # Find best layer for each domain
            best_a = max(results[pooling], key=lambda r: r['auc_a'])
            best_b = max(results[pooling], key=lambda r: r['auc_b'])
            
            aucs_a.append(best_a['auc_a'])
            aucs_b.append(best_b['auc_b'])
            best_layers_a.append(best_a['layer'])
            best_layers_b.append(best_b['layer'])
        else:
            aucs_a.append(0.5)
            aucs_b.append(0.5)
            best_layers_a.append(-1)
            best_layers_b.append(-1)
    
    # Create bars
    bars_a = ax.bar(x - width/2, aucs_a, width, label=f'{label_a}', 
                     color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars_b = ax.bar(x + width/2, aucs_b, width, label=f'{label_b}', 
                     color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar_a, bar_b) in enumerate(zip(bars_a, bars_b)):
        h_a = bar_a.get_height()
        h_b = bar_b.get_height()
        ax.annotate(f'{h_a:.3f}\n(L{best_layers_a[i]})', 
                    xy=(bar_a.get_x() + bar_a.get_width()/2, h_a),
                    xytext=(0, 3), textcoords='offset points', 
                    ha='center', fontsize=9, fontweight='bold')
        ax.annotate(f'{h_b:.3f}\n(L{best_layers_b[i]})', 
                    xy=(bar_b.get_x() + bar_b.get_width()/2, h_b),
                    xytext=(0, 3), textcoords='offset points', 
                    ha='center', fontsize=9, fontweight='bold')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Random')
    ax.set_xlabel('Pooling Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Layer AUC', fontsize=13, fontweight='bold')
    ax.set_title('Combined Training: Best Layer AUC per Pooling Strategy\n(Separate Evaluation on Each Domain)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POOLING_TYPES])
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_combined_layerwise(
    results: Dict[str, List[Dict]], 
    output_path: str,
    label_a: str,
    label_b: str
):
    """Create a 2-panel plot with both domains side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for ax, (domain_key, domain_label) in zip(axes, [('auc_a', label_a), ('auc_b', label_b)]):
        best_overall = {'auc': 0, 'pooling': None, 'layer': None}
        
        for pooling in POOLING_TYPES:
            if pooling not in results:
                continue
            
            layers = [r['layer'] for r in results[pooling]]
            aucs = [r[domain_key] for r in results[pooling]]
            color = POOLING_COLORS.get(pooling, '#666666')
            
            ax.plot(layers, aucs, marker='o', linewidth=2.5, markersize=6,
                    color=color, label=pooling.upper(), alpha=0.85)
            
            # Mark best layer
            best_idx = np.argmax(aucs)
            best_auc = aucs[best_idx]
            best_layer = layers[best_idx]
            
            ax.scatter([best_layer], [best_auc], color=color, s=200, zorder=5,
                       edgecolors='black', linewidths=2.5, marker='*')
            
            if best_auc > best_overall['auc']:
                best_overall = {'auc': best_auc, 'pooling': pooling, 'layer': best_layer}
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
        ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5, label='Strong (0.7)')
        
        ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
        ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
        ax.set_title(f'{domain_label} Validation', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0.4, 1.0)
    
    fig.suptitle('Combined Training: Layerwise AUC for All Pooling Strategies', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train all pooling types on combined data")
    parser.add_argument('--train_a', type=str, required=True, help='Domain A training activations dir')
    parser.add_argument('--val_a', type=str, required=True, help='Domain A validation activations dir')
    parser.add_argument('--train_b', type=str, required=True, help='Domain B training activations dir')
    parser.add_argument('--val_b', type=str, required=True, help='Domain B validation activations dir')
    parser.add_argument('--label_a', type=str, default='Roleplaying', help='Label for domain A')
    parser.add_argument('--label_b', type=str, default='InsiderTrading', help='Label for domain B')
    parser.add_argument('--layers', type=str, default='0-27', help='Layer range (e.g., 0-27 or 5,10,15)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per probe')
    parser.add_argument('--output_dir', type=str, default='results/combined_all_pooling')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse layers
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = list(map(int, args.layers.split(',')))
    
    print("=" * 70)
    print("COMBINED TRAINING: ALL POOLING STRATEGIES")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Domains: {args.label_a} + {args.label_b}")
    print(f"Layers: {layers}")
    print(f"Pooling types: {POOLING_TYPES}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)
    
    # Results storage
    all_results = {pooling: [] for pooling in POOLING_TYPES}
    
    # Train for each layer
    for layer in tqdm(layers, desc="Processing layers"):
        print(f"\n--- Layer {layer} ---")
        
        # Load activations for this layer
        print("  Loading activations...")
        X_train_a_raw, y_train_a = load_layer_activations(args.train_a, layer)
        X_val_a_raw, y_val_a = load_layer_activations(args.val_a, layer)
        X_train_b_raw, y_train_b = load_layer_activations(args.train_b, layer)
        X_val_b_raw, y_val_b = load_layer_activations(args.val_b, layer)
        
        print(f"  {args.label_a}: Train={len(y_train_a)}, Val={len(y_val_a)}")
        print(f"  {args.label_b}: Train={len(y_train_b)}, Val={len(y_val_b)}")
        
        # Combine training data
        X_train_combined_raw = np.vstack([X_train_a_raw, X_train_b_raw])
        y_train_combined = np.concatenate([y_train_a, y_train_b])
        
        # Train each pooling type
        for pooling in POOLING_TYPES:
            print(f"  Training {pooling.upper()} pooling...")
            
            # Apply pooling (except for attn which keeps sequence)
            if pooling == 'attn':
                X_train = X_train_combined_raw  # (N, T, D)
                X_val_a = X_val_a_raw
                X_val_b = X_val_b_raw
            else:
                X_train = pool_activations(X_train_combined_raw, pooling)  # (N, D)
                X_val_a = pool_activations(X_val_a_raw, pooling)
                X_val_b = pool_activations(X_val_b_raw, pooling)
            
            # Train and evaluate
            metrics = train_probe(
                X_train, y_train_combined,
                X_val_a, y_val_a,
                X_val_b, y_val_b,
                pooling, device, args.epochs
            )
            
            metrics['layer'] = layer
            all_results[pooling].append(metrics)
            
            print(f"    {args.label_a}: AUC={metrics['auc_a']:.4f}, Acc={metrics['acc_a']:.4f}")
            print(f"    {args.label_b}: AUC={metrics['auc_b']:.4f}, Acc={metrics['acc_b']:.4f}")
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Plot 1: Layerwise AUC for Domain A
    plot_layerwise_auc(
        all_results, args.label_a, 'auc_a',
        os.path.join(args.output_dir, f'layerwise_auc_{args.label_a}.png'),
        args.label_a, args.label_b
    )
    
    # Plot 2: Layerwise AUC for Domain B
    plot_layerwise_auc(
        all_results, args.label_b, 'auc_b',
        os.path.join(args.output_dir, f'layerwise_auc_{args.label_b}.png'),
        args.label_a, args.label_b
    )
    
    # Plot 3: Combined side-by-side layerwise
    plot_combined_layerwise(
        all_results,
        os.path.join(args.output_dir, 'layerwise_auc_combined.png'),
        args.label_a, args.label_b
    )
    
    # Plot 4: Bar comparison of best layers
    plot_bar_comparison(
        all_results,
        os.path.join(args.output_dir, 'bar_comparison_best_layer.png'),
        args.label_a, args.label_b
    )
    
    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Best Layer per Pooling Strategy")
    print("=" * 70)
    print(f"\n{'Pooling':<10} | {'Best Layer A':<15} | {'AUC A':<10} | {'Best Layer B':<15} | {'AUC B':<10}")
    print("-" * 70)
    
    summary = {}
    for pooling in POOLING_TYPES:
        if pooling not in all_results or not all_results[pooling]:
            continue
        
        best_a = max(all_results[pooling], key=lambda r: r['auc_a'])
        best_b = max(all_results[pooling], key=lambda r: r['auc_b'])
        
        print(f"{pooling.upper():<10} | Layer {best_a['layer']:<10} | {best_a['auc_a']:.4f}    | Layer {best_b['layer']:<10} | {best_b['auc_b']:.4f}")
        
        summary[pooling] = {
            'best_layer_a': best_a['layer'],
            'best_auc_a': best_a['auc_a'],
            'best_layer_b': best_b['layer'],
            'best_auc_b': best_b['auc_b']
        }
    
    print("=" * 70)
    
    # Find overall best
    all_aucs = [(p, max(r['auc_a'] for r in results), max(r['auc_b'] for r in results)) 
                for p, results in all_results.items() if results]
    
    best_for_a = max(all_aucs, key=lambda x: x[1])
    best_for_b = max(all_aucs, key=lambda x: x[2])
    
    print(f"\n⭐ Best for {args.label_a}: {best_for_a[0].upper()} (AUC: {best_for_a[1]:.4f})")
    print(f"⭐ Best for {args.label_b}: {best_for_b[0].upper()} (AUC: {best_for_b[2]:.4f})")
    
    # Save results
    output = {
        'config': {
            'label_a': args.label_a,
            'label_b': args.label_b,
            'layers': layers,
            'epochs': args.epochs
        },
        'per_layer_results': all_results,
        'summary': summary
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\n✓ Results saved to: {results_path}")
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
