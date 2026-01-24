#!/usr/bin/env python3
"""
Per-Token Probe Evaluation
==========================

Evaluate per-token probes on ID and OOD datasets.

For each sample:
1. Apply probe to all tokens → get (T,) logits
2. Aggregate token predictions to sample level via mean/max/vote
3. Compute AUC at sample level

Usage:
    # Evaluate on OOD dataset
    python scripts/evaluation/eval_per_token_probes.py \
        --probes_dir data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
        --ood_activations data/activations/.../Deception-InsiderTrading/validation \
        --output_dir results/per_token_ood

    # Evaluate on ID validation
    python scripts/evaluation/eval_per_token_probes.py \
        --probes_dir data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
        --id_activations data/activations/.../Deception-Roleplaying/validation \
        --output_dir results/per_token_id
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
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
    sample_ids = []
    
    for eid, entry in manifest.items():
        if eid not in all_tensors:
            continue
        
        label = entry.get('label', -1)
        if label == -1:
            continue
        
        tensor = all_tensors[eid]  # (L, T, D)
        
        if layer >= tensor.shape[0]:
            layer = tensor.shape[0] - 1
        
        x_layer = tensor[layer].numpy()  # (T, D)
        
        activations.append(x_layer)
        labels.append(label)
        sample_ids.append(eid)
    
    return activations, labels, sample_ids


def load_probe(probes_dir: str, layer: int, device):
    """Load trained probe and normalization stats."""
    probe_path = os.path.join(probes_dir, f'probe_layer_{layer}.pt')
    norm_path = os.path.join(probes_dir, f'norm_layer_{layer}.npz')
    
    if not os.path.exists(probe_path):
        return None, None, None
    
    # Load probe
    state_dict = torch.load(probe_path, map_location=device)
    input_dim = state_dict['classifier.weight'].shape[1]
    
    model = PerTokenProbe(input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load normalization stats
    if os.path.exists(norm_path):
        norm = np.load(norm_path)
        mean = norm['mean']
        std = norm['std']
    else:
        mean = None
        std = None
    
    return model, mean, std


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_sample_level(model, activations, labels, mean, std, device, aggregation='mean'):
    """
    Evaluate at sample level by aggregating token predictions.
    
    Args:
        aggregation: 'mean', 'max', or 'vote'
    """
    model.eval()
    sample_probs = []
    
    with torch.no_grad():
        for x in activations:
            if mean is not None and std is not None:
                x_n = (x - mean) / std
            else:
                x_n = x
            
            x_t = torch.tensor(x_n, dtype=torch.float32).to(device)
            logits = model(x_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Aggregate
            if aggregation == 'mean':
                sample_probs.append(probs.mean())
            elif aggregation == 'max':
                sample_probs.append(probs.max())
            elif aggregation == 'vote':
                sample_probs.append((probs > 0.5).mean())  # Fraction voting positive
            else:
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
    parser = argparse.ArgumentParser(description="Evaluate per-token probes")
    parser.add_argument('--probes_dir', type=str, required=True)
    parser.add_argument('--id_activations', type=str, default=None)
    parser.add_argument('--ood_activations', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/per_token_eval')
    parser.add_argument('--layers', type=str, default='0-27')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'max', 'vote'])
    args = parser.parse_args()
    
    if args.id_activations is None and args.ood_activations is None:
        logger.error("Must specify at least one of --id_activations or --ood_activations")
        return 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse layers
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = list(map(int, args.layers.split(',')))
    
    logger.info("=" * 70)
    logger.info("PER-TOKEN PROBE EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Probes: {args.probes_dir}")
    logger.info(f"ID: {args.id_activations}")
    logger.info(f"OOD: {args.ood_activations}")
    logger.info(f"Aggregation: {args.aggregation}")
    logger.info("=" * 70)
    
    results = {
        'id': [],
        'ood': []
    }
    
    for layer in tqdm(layers, desc="Evaluating Layers"):
        model, mean, std = load_probe(args.probes_dir, layer, device)
        
        if model is None:
            logger.warning(f"Probe not found for layer {layer}, skipping")
            continue
        
        layer_results = {'layer': layer}
        
        # ID evaluation
        if args.id_activations:
            acts, labels, _ = load_activations_with_tokens(args.id_activations, layer)
            auc, acc, _ = evaluate_sample_level(model, acts, labels, mean, std, device, args.aggregation)
            layer_results['id_auc'] = auc
            layer_results['id_acc'] = acc
            results['id'].append({'layer': layer, 'auc': auc, 'acc': acc})
        
        # OOD evaluation
        if args.ood_activations:
            acts, labels, _ = load_activations_with_tokens(args.ood_activations, layer)
            auc, acc, _ = evaluate_sample_level(model, acts, labels, mean, std, device, args.aggregation)
            layer_results['ood_auc'] = auc
            layer_results['ood_acc'] = acc
            results['ood'].append({'layer': layer, 'auc': auc, 'acc': acc})
        
        if args.id_activations and args.ood_activations:
            logger.info(f"  Layer {layer}: ID={layer_results.get('id_auc', 0):.4f}, OOD={layer_results.get('ood_auc', 0):.4f}")
        elif args.id_activations:
            logger.info(f"  Layer {layer}: ID={layer_results.get('id_auc', 0):.4f}")
        else:
            logger.info(f"  Layer {layer}: OOD={layer_results.get('ood_auc', 0):.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if results['id']:
        id_layers = [r['layer'] for r in results['id']]
        id_aucs = [r['auc'] for r in results['id']]
        ax.plot(id_layers, id_aucs, 'b-o', label='ID', linewidth=2, markersize=6)
        
        best_id = max(results['id'], key=lambda x: x['auc'])
        ax.scatter([best_id['layer']], [best_id['auc']], color='blue', s=200, zorder=5, marker='*')
    
    if results['ood']:
        ood_layers = [r['layer'] for r in results['ood']]
        ood_aucs = [r['auc'] for r in results['ood']]
        ax.plot(ood_layers, ood_aucs, 'r-s', label='OOD', linewidth=2, markersize=6)
        
        best_ood = max(results['ood'], key=lambda x: x['auc'])
        ax.scatter([best_ood['layer']], [best_ood['auc']], color='red', s=200, zorder=5, marker='*')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Per-Token Probe Evaluation\n(Aggregation: {args.aggregation})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'eval_layerwise.png'), dpi=150)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    
    if results['id']:
        best_id = max(results['id'], key=lambda x: x['auc'])
        logger.info(f"Best ID: Layer {best_id['layer']} (AUC: {best_id['auc']:.4f})")
    
    if results['ood']:
        best_ood = max(results['ood'], key=lambda x: x['auc'])
        logger.info(f"Best OOD: Layer {best_ood['layer']} (AUC: {best_ood['auc']:.4f})")
    
    logger.info(f"Saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
