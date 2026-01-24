#!/usr/bin/env python3
"""
Per-Token Probe Evaluation
==========================

Evaluate per-token probes on ID and OOD datasets with multiple aggregation methods.

For each sample:
1. Apply probe to all tokens → get (T,) logits
2. Aggregate token predictions to sample level via mean/max/last/vote
3. Compute AUC at sample level

By default, evaluates ALL aggregation methods and saves separate results for each.

Usage:
    # Evaluate on OOD dataset (all aggregations)
    python scripts/evaluation/eval_per_token_probes.py \
        --probes_dir data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
        --ood_activations data/activations/.../Deception-InsiderTrading/validation \
        --output_dir results/per_token_ood

    # Evaluate with specific aggregation only
    python scripts/evaluation/eval_per_token_probes.py \
        --probes_dir ... \
        --ood_activations ... \
        --aggregation mean
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

# All aggregation methods
AGGREGATION_METHODS = ['mean', 'max', 'last', 'vote']
AGGREGATION_COLORS = {
    'mean': '#2E86AB',
    'max': '#A23B72',
    'last': '#F18F01',
    'vote': '#06A77D'
}


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
    
    state_dict = torch.load(probe_path, map_location=device)
    input_dim = state_dict['classifier.weight'].shape[1]
    
    model = PerTokenProbe(input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
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
        aggregation: 'mean', 'max', 'last', or 'vote'
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
            elif aggregation == 'last':
                sample_probs.append(probs[-1])  # Last token
            elif aggregation == 'vote':
                sample_probs.append((probs > 0.5).mean())
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


def evaluate_all_aggregations(model, activations, labels, mean, std, device):
    """Evaluate with all aggregation methods."""
    results = {}
    for agg in AGGREGATION_METHODS:
        auc, acc, probs = evaluate_sample_level(model, activations, labels, mean, std, device, agg)
        results[agg] = {'auc': auc, 'acc': acc}
    return results


# ============================================================================
# PLOTTING
# ============================================================================
def plot_single_aggregation(results, eval_type, aggregation, output_dir):
    """Plot layerwise results for a single aggregation method."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = [r['layer'] for r in results]
    aucs = [r['auc'] for r in results]
    
    color = AGGREGATION_COLORS.get(aggregation, 'blue')
    ax.plot(layers, aucs, '-o', color=color, linewidth=2, markersize=6, label=eval_type.upper())
    
    best = max(results, key=lambda x: x['auc'])
    ax.scatter([best['layer']], [best['auc']], color=color, s=200, zorder=5, marker='*')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Per-Token Probe: {aggregation.upper()} Aggregation\n({eval_type.upper()} Evaluation)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'eval_{eval_type}_{aggregation}.png'), dpi=150)
    plt.close()


def plot_combined_aggregations(all_results, eval_type, output_dir):
    """Plot all aggregation methods on the same chart."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for agg in AGGREGATION_METHODS:
        if agg not in all_results:
            continue
        
        results = all_results[agg]
        layers = [r['layer'] for r in results]
        aucs = [r['auc'] for r in results]
        color = AGGREGATION_COLORS.get(agg, 'gray')
        
        ax.plot(layers, aucs, '-o', color=color, linewidth=2, markersize=5, 
                label=f'{agg.upper()}', alpha=0.85)
        
        # Mark best
        best = max(results, key=lambda x: x['auc'])
        ax.scatter([best['layer']], [best['auc']], color=color, s=150, zorder=5, 
                   marker='*', edgecolors='black', linewidths=1)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Per-Token Probe: All Aggregation Methods\n({eval_type.upper()} Evaluation)', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'eval_{eval_type}_all_aggregations.png'), dpi=150)
    plt.close()


def plot_aggregation_comparison_bar(all_results, eval_type, output_dir):
    """Bar chart comparing best AUC per aggregation method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    best_aucs = []
    best_layers = []
    colors = []
    
    for agg in AGGREGATION_METHODS:
        if agg not in all_results or not all_results[agg]:
            continue
        
        best = max(all_results[agg], key=lambda x: x['auc'])
        methods.append(agg.upper())
        best_aucs.append(best['auc'])
        best_layers.append(best['layer'])
        colors.append(AGGREGATION_COLORS.get(agg, 'gray'))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, best_aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, layer) in enumerate(zip(bars, best_layers)):
        h = bar.get_height()
        ax.annotate(f'{h:.3f}\n(L{layer})', 
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Aggregation Method', fontsize=12)
    ax.set_ylabel('Best Layer AUC', fontsize=12)
    ax.set_title(f'Per-Token Probe: Aggregation Method Comparison\n({eval_type.upper()} Evaluation)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'eval_{eval_type}_aggregation_bar.png'), dpi=150)
    plt.close()


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
    parser.add_argument('--aggregation', type=str, default='all', 
                        choices=['all', 'mean', 'max', 'last', 'vote'],
                        help="Aggregation method ('all' runs all methods)")
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
    
    # Determine which aggregations to run
    if args.aggregation == 'all':
        aggregations = AGGREGATION_METHODS
    else:
        aggregations = [args.aggregation]
    
    logger.info("=" * 70)
    logger.info("PER-TOKEN PROBE EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Probes: {args.probes_dir}")
    logger.info(f"ID: {args.id_activations}")
    logger.info(f"OOD: {args.ood_activations}")
    logger.info(f"Aggregations: {aggregations}")
    logger.info("=" * 70)
    
    # Results structure: {aggregation: {eval_type: [{layer, auc, acc}]}}
    all_results = {agg: {'id': [], 'ood': []} for agg in aggregations}
    
    for layer in tqdm(layers, desc="Evaluating Layers"):
        model, mean, std = load_probe(args.probes_dir, layer, device)
        
        if model is None:
            logger.warning(f"Probe not found for layer {layer}, skipping")
            continue
        
        # ID evaluation
        if args.id_activations:
            acts, labels, _ = load_activations_with_tokens(args.id_activations, layer)
            for agg in aggregations:
                auc, acc, _ = evaluate_sample_level(model, acts, labels, mean, std, device, agg)
                all_results[agg]['id'].append({'layer': layer, 'auc': auc, 'acc': acc})
        
        # OOD evaluation
        if args.ood_activations:
            acts, labels, _ = load_activations_with_tokens(args.ood_activations, layer)
            for agg in aggregations:
                auc, acc, _ = evaluate_sample_level(model, acts, labels, mean, std, device, agg)
                all_results[agg]['ood'].append({'layer': layer, 'auc': auc, 'acc': acc})
    
    # Save results for each aggregation
    for agg in aggregations:
        agg_dir = os.path.join(args.output_dir, agg)
        os.makedirs(agg_dir, exist_ok=True)
        
        results_path = os.path.join(agg_dir, 'eval_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results[agg], f, indent=2)
        
        # Plot per-aggregation results
        if all_results[agg]['id']:
            plot_single_aggregation(all_results[agg]['id'], 'id', agg, agg_dir)
        if all_results[agg]['ood']:
            plot_single_aggregation(all_results[agg]['ood'], 'ood', agg, agg_dir)
    
    # Combined plots (if running all aggregations)
    if len(aggregations) > 1:
        # ID combined
        if args.id_activations:
            id_combined = {agg: all_results[agg]['id'] for agg in aggregations}
            plot_combined_aggregations(id_combined, 'id', args.output_dir)
            plot_aggregation_comparison_bar(id_combined, 'id', args.output_dir)
        
        # OOD combined
        if args.ood_activations:
            ood_combined = {agg: all_results[agg]['ood'] for agg in aggregations}
            plot_combined_aggregations(ood_combined, 'ood', args.output_dir)
            plot_aggregation_comparison_bar(ood_combined, 'ood', args.output_dir)
    
    # ========================================================================
    # Save OOD Summary (for use by invariant_core analysis)
    # ========================================================================
    overall_best_auc = 0
    overall_best = {'aggregation': 'mean', 'layer': 20, 'auc': 0.5}
    
    if args.ood_activations:
        ood_summary = {
            'probes_dir': args.probes_dir,
            'ood_activations': args.ood_activations,
            'aggregations': {}
        }
        
        for agg in aggregations:
            if all_results[agg]['ood']:
                best = max(all_results[agg]['ood'], key=lambda x: x['auc'])
                ood_summary['aggregations'][agg] = {
                    'best_layer': best['layer'],
                    'best_auc': best['auc'],
                    'best_acc': best['acc']
                }
                if best['auc'] > overall_best_auc:
                    overall_best_auc = best['auc']
                    overall_best = {'aggregation': agg, 'layer': best['layer'], 'auc': best['auc']}
        
        ood_summary['overall_best'] = overall_best
        
        summary_path = os.path.join(args.output_dir, 'ood_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(ood_summary, f, indent=2)
        logger.info(f"Saved OOD summary: {summary_path}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    
    for agg in aggregations:
        logger.info(f"\n{agg.upper()} Aggregation:")
        if all_results[agg]['id']:
            best_id = max(all_results[agg]['id'], key=lambda x: x['auc'])
            logger.info(f"  Best ID: Layer {best_id['layer']} (AUC: {best_id['auc']:.4f})")
        if all_results[agg]['ood']:
            best_ood = max(all_results[agg]['ood'], key=lambda x: x['auc'])
            logger.info(f"  Best OOD: Layer {best_ood['layer']} (AUC: {best_ood['auc']:.4f})")
    
    if args.ood_activations:
        logger.info(f"\n*** Overall Best OOD: {overall_best['aggregation'].upper()} Layer {overall_best['layer']} (AUC: {overall_best_auc:.4f}) ***")
    
    logger.info(f"\nSaved to: {args.output_dir}")
    logger.info("=" * 70)



if __name__ == "__main__":
    main()
