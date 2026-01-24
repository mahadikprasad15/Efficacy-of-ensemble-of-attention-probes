#!/usr/bin/env python3
"""
Per-Token Probe Training (Apollo Research Approach)
====================================================

Instead of pooling tokens before training, this script flattens all token
activations and replicates sample labels to each token. This dramatically
increases effective training set size.

Approach:
    - Current (pooled): N samples → pool → (N, D) → 1 label each
    - Apollo (per-token): N samples × T tokens → (N*T, D) → label replicated T times

Usage:
    python scripts/training/train_per_token_probes.py \
        --train_activations data/activations/.../Deception-Roleplaying/train \
        --val_activations data/activations/.../Deception-Roleplaying/validation \
        --output_dir data/probes_per_token \
        --model meta-llama_Llama-3.2-3B-Instruct \
        --dataset Deception-Roleplaying

For flipped (OOD) training:
    python scripts/training/train_per_token_probes.py \
        --train_activations data/activations/.../Deception-InsiderTrading/train \
        --val_activations data/activations/.../Deception-InsiderTrading/validation \
        --output_dir data/probes_per_token_flipped \
        --model meta-llama_Llama-3.2-3B-Instruct \
        --dataset Deception-InsiderTrading
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
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL: Simple Linear Probe (no pooling!)
# ============================================================================
class PerTokenProbe(nn.Module):
    """Simple linear probe applied to each token independently."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) flattened token activations
        Returns:
            logits: (B, 1)
        """
        return self.classifier(x)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_activations_with_tokens(activations_dir: str, layer: int):
    """
    Load activations keeping all tokens (no pooling).
    
    Returns:
        activations: List of (T, D) arrays, one per sample
        labels: List of sample-level labels
        sample_ids: List of sample IDs
    """
    manifest_path = os.path.join(activations_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = {json.loads(line)['id']: json.loads(line) for line in open(manifest_path)}
    
    # Reload manifest properly
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
    sample_ids = []
    
    for eid, entry in manifest.items():
        if eid not in all_tensors:
            continue
        
        label = entry.get('label', -1)
        if label == -1:
            continue
        
        tensor = all_tensors[eid]  # (L, T, D)
        
        # Select layer
        if layer >= tensor.shape[0]:
            layer = tensor.shape[0] - 1
        
        x_layer = tensor[layer].numpy()  # (T, D)
        
        activations.append(x_layer)
        labels.append(label)
        sample_ids.append(eid)
    
    return activations, labels, sample_ids


def flatten_tokens(activations, labels):
    """
    Flatten all tokens and replicate labels.
    
    Args:
        activations: List of (T_i, D) arrays
        labels: List of sample-level labels
    
    Returns:
        X_flat: (sum(T_i), D) flattened token activations
        y_flat: (sum(T_i),) replicated labels
        sample_indices: Which sample each token belongs to
    """
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
def train_probe(X_train, y_train, X_val, y_val, device, args):
    """Train a per-token probe."""
    input_dim = X_train.shape[1]
    model = PerTokenProbe(input_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    batch_size = min(args.batch_size, len(X_train))
    
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0.0
        
        for i in range(0, len(X_t), batch_size):
            idxs = perm[i:i+batch_size]
            batch_x = X_t[idxs].to(device)
            batch_y = y_t[idxs].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
            val_logits = model(X_val_t).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        
        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except:
            val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_auc, mean, std


def evaluate_sample_level(model, activations, labels, mean, std, device):
    """
    Evaluate at sample level by aggregating token predictions.
    
    For each sample, apply probe to all tokens and aggregate via mean.
    """
    model.eval()
    sample_probs = []
    
    with torch.no_grad():
        for x in activations:
            x_n = (x - mean) / std
            x_t = torch.tensor(x_n, dtype=torch.float32).to(device)
            logits = model(x_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Aggregate: mean probability across tokens
            sample_probs.append(probs.mean())
    
    sample_probs = np.array(sample_probs)
    labels = np.array(labels)
    
    try:
        auc = roc_auc_score(labels, sample_probs)
    except:
        auc = 0.5
    
    acc = accuracy_score(labels, (sample_probs > 0.5).astype(int))
    
    return auc, acc, sample_probs


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train per-token probes (Apollo approach)")
    parser.add_argument('--train_activations', type=str, required=True)
    parser.add_argument('--val_activations', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/probes_per_token')
    parser.add_argument('--model', type=str, default='meta-llama_Llama-3.2-3B-Instruct')
    parser.add_argument('--dataset', type=str, default='Deception-Roleplaying')
    parser.add_argument('--layers', type=str, default='0-27', help='Layer range (e.g., 0-27)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Parse layers
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = list(map(int, args.layers.split(',')))
    
    # Output directory
    probe_dir = os.path.join(args.output_dir, args.model, args.dataset)
    os.makedirs(probe_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PER-TOKEN PROBE TRAINING (Apollo Approach)")
    logger.info("=" * 70)
    logger.info(f"Train: {args.train_activations}")
    logger.info(f"Val: {args.val_activations}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Output: {probe_dir}")
    logger.info("=" * 70)
    
    results = []
    
    for layer in tqdm(layers, desc="Training Layers"):
        logger.info(f"\n--- Layer {layer} ---")
        
        # Load data
        train_acts, train_labels, _ = load_activations_with_tokens(args.train_activations, layer)
        val_acts, val_labels, _ = load_activations_with_tokens(args.val_activations, layer)
        
        logger.info(f"  Train samples: {len(train_acts)}")
        logger.info(f"  Val samples: {len(val_acts)}")
        
        # Flatten tokens
        X_train_flat, y_train_flat, _ = flatten_tokens(train_acts, train_labels)
        X_val_flat, y_val_flat, val_sample_indices = flatten_tokens(val_acts, val_labels)
        
        logger.info(f"  Train tokens (flattened): {len(X_train_flat)}")
        logger.info(f"  Val tokens (flattened): {len(X_val_flat)}")
        
        # Train
        model, token_val_auc, mean, std = train_probe(
            X_train_flat, y_train_flat, X_val_flat, y_val_flat, device, args
        )
        
        # Evaluate at sample level
        sample_val_auc, sample_val_acc, _ = evaluate_sample_level(
            model, val_acts, val_labels, mean, std, device
        )
        
        logger.info(f"  Token-level Val AUC: {token_val_auc:.4f}")
        logger.info(f"  Sample-level Val AUC: {sample_val_auc:.4f} (Acc: {sample_val_acc:.4f})")
        
        # Save probe
        probe_path = os.path.join(probe_dir, f'probe_layer_{layer}.pt')
        torch.save(model.state_dict(), probe_path)
        
        # Save normalization stats
        norm_path = os.path.join(probe_dir, f'norm_layer_{layer}.npz')
        np.savez(norm_path, mean=mean, std=std)
        
        results.append({
            'layer': layer,
            'token_val_auc': float(token_val_auc),
            'sample_val_auc': float(sample_val_auc),
            'sample_val_acc': float(sample_val_acc),
            'n_train_tokens': len(X_train_flat),
            'n_val_tokens': len(X_val_flat)
        })
    
    # Save results
    results_path = os.path.join(probe_dir, 'layer_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    best = max(results, key=lambda x: x['sample_val_auc'])
    logger.info(f"Best layer: {best['layer']} (Sample AUC: {best['sample_val_auc']:.4f})")
    logger.info(f"Saved to: {probe_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
