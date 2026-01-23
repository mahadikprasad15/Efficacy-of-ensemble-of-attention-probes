#!/usr/bin/env python3
"""
Train All Pooling Types on Single Domain with OOD Evaluation
=============================================================
Trains probes for all 4 pooling strategies on ONE domain and evaluates on BOTH
domains (ID and OOD). This provides a baseline for comparison with combined training.

Produces:
1. Line plots: AUC vs Layers for each pooling, separate for ID and OOD
2. Bar graphs: ID vs OOD AUC per pooling
3. Comparison JSON

Usage:
    # Train on Roleplaying, test on both Roleplaying (ID) and InsiderTrading (OOD)
    python scripts/domain_adaptation/train_single_domain_all_pooling.py \
        --train_dir /path/to/Deception-Roleplaying/train \
        --val_id /path/to/Deception-Roleplaying/validation \
        --val_ood /path/to/Deception-InsiderTrading/validation \
        --label_id Roleplaying --label_ood InsiderTrading \
        --output_dir results/single_domain_roleplaying
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
# CONFIG
# ============================================================================
POOLING_TYPES = ['mean', 'max', 'last', 'attn']
POOLING_COLORS = {
    'mean': '#2E86AB',
    'max': '#A23B72',
    'last': '#F18F01',
    'attn': '#06A77D'
}

# ============================================================================
# MODELS
# ============================================================================
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


class AttentionPoolingProbe(nn.Module):
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
        attn_weights = torch.softmax(self.attn(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


# ============================================================================
# DATA
# ============================================================================
def load_layer_activations(act_dir: str, layer: int) -> Tuple[np.ndarray, np.ndarray]:
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
            activations.append(x_layer.numpy())
            labels.append(entry['label'])
    
    return np.array(activations), np.array(labels)


def pool_activations(x: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == 'mean':
        return x.mean(axis=1)
    elif pooling == 'max':
        return x.max(axis=1)
    elif pooling == 'last':
        return x[:, -1, :]
    elif pooling == 'attn':
        return x
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


# ============================================================================
# TRAINING
# ============================================================================
def train_and_evaluate(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val_id: np.ndarray, 
    y_val_id: np.ndarray,
    X_val_ood: np.ndarray, 
    y_val_ood: np.ndarray,
    pooling: str,
    device: torch.device,
    epochs: int = 20
) -> Dict:
    """Train on single domain, evaluate on both ID and OOD."""
    
    if pooling == 'attn':
        N, T, D = X_train.shape
        model = AttentionPoolingProbe(D).to(device)
        mean = X_train.mean(axis=(0, 1))
        std = X_train.std(axis=(0, 1)) + 1e-8
    else:
        D = X_train.shape[1]
        model = SimpleProbe(D).to(device)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
    
    X_train_n = (X_train - mean) / std
    X_val_id_n = (X_val_id - mean) / std
    X_val_ood_n = (X_val_ood - mean) / std
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    for epoch in range(epochs):
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
    with torch.no_grad():
        probs_id = torch.sigmoid(model(torch.tensor(X_val_id_n, dtype=torch.float32).to(device))).cpu().numpy()
        probs_ood = torch.sigmoid(model(torch.tensor(X_val_ood_n, dtype=torch.float32).to(device))).cpu().numpy()
    
    try:
        auc_id = roc_auc_score(y_val_id, probs_id)
    except:
        auc_id = 0.5
    
    try:
        auc_ood = roc_auc_score(y_val_ood, probs_ood)
    except:
        auc_ood = 0.5
    
    return {
        'auc_id': auc_id,
        'auc_ood': auc_ood,
        'acc_id': accuracy_score(y_val_id, (probs_id > 0.5).astype(int)),
        'acc_ood': accuracy_score(y_val_ood, (probs_ood > 0.5).astype(int))
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_layerwise_id_ood(results: Dict[str, List[Dict]], output_path: str, label_id: str, label_ood: str):
    """Plot layerwise AUC for ID and OOD side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (key, label) in zip(axes, [('auc_id', f'{label_id} (ID)'), ('auc_ood', f'{label_ood} (OOD)')]):
        for pooling in POOLING_TYPES:
            if pooling not in results:
                continue
            layers = [r['layer'] for r in results[pooling]]
            aucs = [r[key] for r in results[pooling]]
            color = POOLING_COLORS[pooling]
            
            ax.plot(layers, aucs, marker='o', linewidth=2.5, markersize=6,
                    color=color, label=pooling.upper(), alpha=0.85)
            
            best_idx = np.argmax(aucs)
            ax.scatter([layers[best_idx]], [aucs[best_idx]], color=color, s=200, 
                       zorder=5, edgecolors='black', linewidths=2.5, marker='*')
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5)
        ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
        ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 1.0)
    
    fig.suptitle(f'Single-Domain Training: Trained on {label_id}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_id_ood_gap(results: Dict[str, List[Dict]], output_path: str, label_id: str, label_ood: str):
    """Bar plot showing ID vs OOD performance gap for each pooling type."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(POOLING_TYPES))
    width = 0.35
    
    best_id = []
    best_ood = []
    best_layers = []
    
    for pooling in POOLING_TYPES:
        if pooling in results:
            # Use best ID layer for both metrics (fair comparison)
            best_by_id = max(results[pooling], key=lambda r: r['auc_id'])
            best_id.append(best_by_id['auc_id'])
            best_ood.append(best_by_id['auc_ood'])
            best_layers.append(best_by_id['layer'])
        else:
            best_id.append(0.5)
            best_ood.append(0.5)
            best_layers.append(-1)
    
    bars_id = ax.bar(x - width/2, best_id, width, label=f'{label_id} (ID)', 
                     color='#27ae60', alpha=0.8, edgecolor='black')
    bars_ood = ax.bar(x + width/2, best_ood, width, label=f'{label_ood} (OOD)', 
                      color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add gap annotations
    for i, (id_val, ood_val) in enumerate(zip(best_id, best_ood)):
        gap = id_val - ood_val
        mid_y = (id_val + ood_val) / 2
        ax.annotate(f'Gap: {gap:.3f}', xy=(i, mid_y), ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Value labels
    for bar, layer in zip(bars_id, best_layers):
        h = bar.get_height()
        ax.annotate(f'{h:.3f}\n(L{layer})', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    for bar in bars_ood:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pooling Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax.set_title(f'Single-Domain Training: ID vs OOD Gap\n(Trained on {label_id})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POOLING_TYPES])
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0.3, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='Training domain activations')
    parser.add_argument('--val_id', type=str, required=True, help='In-domain validation')
    parser.add_argument('--val_ood', type=str, required=True, help='Out-of-domain validation')
    parser.add_argument('--label_id', type=str, default='ID', help='Label for training domain')
    parser.add_argument('--label_ood', type=str, default='OOD', help='Label for OOD domain')
    parser.add_argument('--layers', type=str, default='0-27', help='Layer range')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='results/single_domain')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = list(map(int, args.layers.split(',')))
    
    print("=" * 70)
    print("SINGLE-DOMAIN TRAINING: ALL POOLING TYPES")
    print("=" * 70)
    print(f"Training on: {args.label_id}")
    print(f"OOD test: {args.label_ood}")
    print(f"Device: {device}")
    print(f"Layers: {layers}")
    print("=" * 70)
    
    all_results = {pooling: [] for pooling in POOLING_TYPES}
    
    for layer in tqdm(layers, desc="Processing layers"):
        print(f"\n--- Layer {layer} ---")
        
        X_train_raw, y_train = load_layer_activations(args.train_dir, layer)
        X_val_id_raw, y_val_id = load_layer_activations(args.val_id, layer)
        X_val_ood_raw, y_val_ood = load_layer_activations(args.val_ood, layer)
        
        print(f"  Train: {len(y_train)}, Val ID: {len(y_val_id)}, Val OOD: {len(y_val_ood)}")
        
        for pooling in POOLING_TYPES:
            print(f"  Training {pooling.upper()}...")
            
            if pooling == 'attn':
                X_train = X_train_raw
                X_val_id = X_val_id_raw
                X_val_ood = X_val_ood_raw
            else:
                X_train = pool_activations(X_train_raw, pooling)
                X_val_id = pool_activations(X_val_id_raw, pooling)
                X_val_ood = pool_activations(X_val_ood_raw, pooling)
            
            metrics = train_and_evaluate(X_train, y_train, X_val_id, y_val_id, 
                                         X_val_ood, y_val_ood, pooling, device, args.epochs)
            metrics['layer'] = layer
            all_results[pooling].append(metrics)
            
            print(f"    ID: {metrics['auc_id']:.4f}, OOD: {metrics['auc_ood']:.4f}, Gap: {metrics['auc_id']-metrics['auc_ood']:.4f}")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_layerwise_id_ood(all_results, 
                          os.path.join(args.output_dir, 'layerwise_id_ood.png'),
                          args.label_id, args.label_ood)
    
    plot_id_ood_gap(all_results,
                    os.path.join(args.output_dir, 'id_ood_gap_bar.png'),
                    args.label_id, args.label_ood)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Pooling':<10} | {'Best Layer':<12} | {'ID AUC':<10} | {'OOD AUC':<10} | {'Gap':<10}")
    print("-" * 60)
    
    for pooling in POOLING_TYPES:
        if not all_results[pooling]:
            continue
        best = max(all_results[pooling], key=lambda r: r['auc_id'])
        gap = best['auc_id'] - best['auc_ood']
        print(f"{pooling.upper():<10} | Layer {best['layer']:<6} | {best['auc_id']:.4f}    | {best['auc_ood']:.4f}    | {gap:+.4f}")
    
    # Save
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({
            'config': {'label_id': args.label_id, 'label_ood': args.label_ood, 'layers': layers},
            'results': all_results
        }, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
