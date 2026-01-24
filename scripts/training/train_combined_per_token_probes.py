#!/usr/bin/env python3
"""
Combined Per-Token Probe Training
=================================

Train probes on COMBINED datasets (A+B) using per-token approach.
Flattens tokens from both domains, replicates labels, trains linear probes.
Evaluates on each domain separately with multiple aggregation methods.

Usage:
    python scripts/training/train_combined_per_token_probes.py \
        --train_a data/activations/.../Deception-Roleplaying/train \
        --val_a data/activations/.../Deception-Roleplaying/validation \
        --train_b data/activations/.../Deception-InsiderTrading/train \
        --val_b data/activations/.../Deception-InsiderTrading/validation \
        --output_dir data/probes_combined_per_token \
        --model meta-llama_Llama-3.2-3B-Instruct
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AGGREGATION_METHODS = ['mean', 'max', 'last']
AGGREGATION_COLORS = {'mean': '#2E86AB', 'max': '#A23B72', 'last': '#F18F01'}


# ============================================================================
# MODEL
# ============================================================================
class PerTokenProbe(nn.Module):
    """Simple linear probe applied to each token independently."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_activations_with_tokens(activations_dir: str, layer: int):
    """Load activations keeping all tokens."""
    manifest_path = os.path.join(activations_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    manifest = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            manifest[entry['id']] = entry
    
    shards = sorted(glob.glob(os.path.join(activations_dir, 'shard_*.safetensors')))
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
        activations.append(x_layer)
        labels.append(label)
    
    return activations, labels


def flatten_tokens(activations, labels):
    """Flatten all tokens and replicate labels."""
    X_list = []
    y_list = []
    sample_indices = []
    
    for i, (x, y) in enumerate(zip(activations, labels)):
        T = x.shape[0]
        X_list.append(x)
        y_list.extend([y] * T)
        sample_indices.extend([i] * T)
    
    X_flat = np.vstack(X_list)
    y_flat = np.array(y_list)
    sample_indices = np.array(sample_indices)
    
    return X_flat, y_flat, sample_indices


# ============================================================================
# TRAINING
# ============================================================================
def train_probe(X_train, y_train, device, args):
    """Train a per-token probe."""
    input_dim = X_train.shape[1]
    model = PerTokenProbe(input_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    batch_size = min(args.batch_size, len(X_train))
    
    model.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idxs = perm[i:i+batch_size]
            batch_x = X_t[idxs].to(device)
            batch_y = y_t[idxs].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    
    return model, mean, std


def evaluate_sample_level(model, activations, labels, mean, std, device, aggregation='mean'):
    """Evaluate at sample level by aggregating token predictions."""
    model.eval()
    sample_probs = []
    
    with torch.no_grad():
        for x in activations:
            x_n = (x - mean) / std
            x_t = torch.tensor(x_n, dtype=torch.float32).to(device)
            logits = model(x_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            if aggregation == 'mean':
                sample_probs.append(probs.mean())
            elif aggregation == 'max':
                sample_probs.append(probs.max())
            elif aggregation == 'last':
                sample_probs.append(probs[-1])
            else:
                sample_probs.append(probs.mean())
    
    sample_probs = np.array(sample_probs)
    labels = np.array(labels)
    
    try:
        auc = roc_auc_score(labels, sample_probs)
    except:
        auc = 0.5
    
    acc = accuracy_score(labels, (sample_probs > 0.5).astype(int))
    return auc, acc


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_layerwise_results(results, output_dir, label_a, label_b):
    """Create layerwise plots for each aggregation method."""
    for agg in AGGREGATION_METHODS:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        layers = [r['layer'] for r in results[agg]]
        aucs_a = [r['auc_a'] for r in results[agg]]
        aucs_b = [r['auc_b'] for r in results[agg]]
        
        ax.plot(layers, aucs_a, 'b-o', label=label_a, linewidth=2, markersize=6)
        ax.plot(layers, aucs_b, 'r-s', label=label_b, linewidth=2, markersize=6)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(f'Combined Per-Token Training: {agg.upper()} Aggregation', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'layerwise_{agg}.png'), dpi=150)
        plt.close()


def plot_aggregation_comparison(results, output_dir, label_a, label_b):
    """Bar chart comparing all aggregations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (domain, label) in zip(axes, [('auc_a', label_a), ('auc_b', label_b)]):
        x = np.arange(len(AGGREGATION_METHODS))
        
        best_aucs = []
        best_layers = []
        colors = []
        
        for agg in AGGREGATION_METHODS:
            best = max(results[agg], key=lambda r: r[domain])
            best_aucs.append(best[domain])
            best_layers.append(best['layer'])
            colors.append(AGGREGATION_COLORS[agg])
        
        bars = ax.bar(x, best_aucs, color=colors, alpha=0.8, edgecolor='black')
        
        for i, (bar, layer) in enumerate(zip(bars, best_layers)):
            h = bar.get_height()
            ax.annotate(f'{h:.3f}\n(L{layer})', 
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Aggregation Method', fontsize=12)
        ax.set_ylabel('Best Layer AUC', fontsize=12)
        ax.set_title(f'{label} Validation', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in AGGREGATION_METHODS])
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Combined Per-Token Training: Aggregation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregation_comparison.png'), dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train combined per-token probes")
    parser.add_argument('--train_a', type=str, required=True)
    parser.add_argument('--val_a', type=str, required=True)
    parser.add_argument('--train_b', type=str, required=True)
    parser.add_argument('--val_b', type=str, required=True)
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--output_dir', type=str, default='data/probes_combined_per_token')
    parser.add_argument('--model', type=str, default='meta-llama_Llama-3.2-3B-Instruct')
    parser.add_argument('--layers', type=str, default='0-27')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse layers
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = list(map(int, args.layers.split(',')))
    
    # Output directory
    probe_dir = os.path.join(args.output_dir, args.model, 'Deception-Combined')
    os.makedirs(probe_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("COMBINED PER-TOKEN PROBE TRAINING")
    logger.info("=" * 70)
    logger.info(f"Train A: {args.train_a}")
    logger.info(f"Train B: {args.train_b}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Output: {probe_dir}")
    logger.info("=" * 70)
    
    # Results structure: {aggregation: [{layer, auc_a, auc_b}]}
    all_results = {agg: [] for agg in AGGREGATION_METHODS}
    
    for layer in tqdm(layers, desc="Training Layers"):
        logger.info(f"\n--- Layer {layer} ---")
        
        # Load data
        train_acts_a, train_labels_a = load_activations_with_tokens(args.train_a, layer)
        train_acts_b, train_labels_b = load_activations_with_tokens(args.train_b, layer)
        val_acts_a, val_labels_a = load_activations_with_tokens(args.val_a, layer)
        val_acts_b, val_labels_b = load_activations_with_tokens(args.val_b, layer)
        
        logger.info(f"  {args.label_a}: Train={len(train_acts_a)}, Val={len(val_acts_a)}")
        logger.info(f"  {args.label_b}: Train={len(train_acts_b)}, Val={len(val_acts_b)}")
        
        # Combine training data
        train_acts_combined = train_acts_a + train_acts_b
        train_labels_combined = train_labels_a + train_labels_b
        
        # Flatten tokens
        X_train_flat, y_train_flat, _ = flatten_tokens(train_acts_combined, train_labels_combined)
        logger.info(f"  Combined flattened tokens: {len(X_train_flat)}")
        
        # Train
        model, mean, std = train_probe(X_train_flat, y_train_flat, device, args)
        
        # Save probe
        probe_path = os.path.join(probe_dir, f'probe_layer_{layer}.pt')
        torch.save(model.state_dict(), probe_path)
        
        norm_path = os.path.join(probe_dir, f'norm_layer_{layer}.npz')
        np.savez(norm_path, mean=mean, std=std)
        
        # Evaluate with all aggregations
        for agg in AGGREGATION_METHODS:
            auc_a, acc_a = evaluate_sample_level(model, val_acts_a, val_labels_a, mean, std, device, agg)
            auc_b, acc_b = evaluate_sample_level(model, val_acts_b, val_labels_b, mean, std, device, agg)
            
            all_results[agg].append({
                'layer': layer,
                'auc_a': auc_a,
                'acc_a': acc_a,
                'auc_b': auc_b,
                'acc_b': acc_b
            })
        
        logger.info(f"  MEAN: {args.label_a}={all_results['mean'][-1]['auc_a']:.4f}, {args.label_b}={all_results['mean'][-1]['auc_b']:.4f}")
    
    # Save results
    results_path = os.path.join(probe_dir, 'layer_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    logger.info("\nGenerating visualizations...")
    plot_layerwise_results(all_results, probe_dir, args.label_a, args.label_b)
    plot_aggregation_comparison(all_results, probe_dir, args.label_a, args.label_b)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    for agg in AGGREGATION_METHODS:
        best_a = max(all_results[agg], key=lambda r: r['auc_a'])
        best_b = max(all_results[agg], key=lambda r: r['auc_b'])
        logger.info(f"{agg.upper()}: Best {args.label_a}={best_a['auc_a']:.4f} (L{best_a['layer']}), "
                   f"Best {args.label_b}={best_b['auc_b']:.4f} (L{best_b['layer']})")
    
    logger.info(f"\nSaved to: {probe_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
