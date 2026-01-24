#!/usr/bin/env python3
"""
Invariant Core Analysis for Per-Token Probes
=============================================

Analyze whether per-token probes learn a domain-invariant deception direction.

This script:
1. Loads per-token probes from:
   - Single domain A (probes_per_token)
   - Single domain B (probes_per_token_flipped)
   - Combined training (probes_combined_per_token)
2. Decomposes combined probe: w_C = a·ŵ_A + b·ŵ_B_orth + r (residual)
3. Tests each component for cross-domain generalization
4. If residual `r` generalizes best, it's the "invariant core"

Usage:
    python scripts/analysis/analyze_invariant_core_per_token_probes.py \
        --probes_a data/probes_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying \
        --probes_b data/probes_per_token_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading \
        --probes_combined data/probes_combined_per_token/meta-llama_Llama-3.2-3B-Instruct/Deception-Combined \
        --val_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
        --val_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
        --layer 20 \
        --output_dir results/invariant_core_per_token
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
from sklearn.metrics import roc_auc_score, accuracy_score

AGGREGATION_METHODS = ['mean', 'max', 'last']


# ============================================================================
# MODEL
# ============================================================================
class PerTokenProbe(nn.Module):
    """Simple linear probe."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def get_direction(self):
        """Get normalized probe direction vector."""
        w = self.classifier.weight.squeeze().detach().cpu().numpy()
        return w / (np.linalg.norm(w) + 1e-8)


# ============================================================================
# LOADING
# ============================================================================
def load_per_token_probe(probes_dir, layer, device='cpu'):
    """Load a per-token probe for a specific layer."""
    probe_path = os.path.join(probes_dir, f'probe_layer_{layer}.pt')
    if not os.path.exists(probe_path):
        return None
    
    state_dict = torch.load(probe_path, map_location=device)
    input_dim = state_dict['classifier.weight'].shape[1]
    
    model = PerTokenProbe(input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_norm_stats(probes_dir, layer):
    """Load normalization statistics."""
    norm_path = os.path.join(probes_dir, f'norm_layer_{layer}.npz')
    if os.path.exists(norm_path):
        norm = np.load(norm_path)
        return norm['mean'], norm['std']
    return None, None


def load_activations_with_tokens(activations_dir, layer):
    """Load activations keeping all tokens."""
    manifest_path = os.path.join(activations_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        return None, None
    
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


def load_pooled_activations(activations_dir, layer, pooling='mean'):
    """Load and pool activations."""
    acts, labels = load_activations_with_tokens(activations_dir, layer)
    if acts is None:
        return None, None
    
    pooled = []
    for x in acts:
        if pooling == 'mean':
            pooled.append(x.mean(axis=0))
        elif pooling == 'max':
            pooled.append(x.max(axis=0))
        elif pooling == 'last':
            pooled.append(x[-1])
        else:
            pooled.append(x.mean(axis=0))
    
    return np.array(pooled), np.array(labels)


# ============================================================================
# DECOMPOSITION
# ============================================================================
def decompose_combined_direction(w_C, w_A, w_B):
    """
    Decompose combined direction into:
    w_C = a * ŵ_A + b * ŵ_B_orth + r
    
    Where:
    - ŵ_A is unit vector in domain A direction
    - ŵ_B_orth is component of B orthogonal to ŵ_A (Gram-Schmidt)
    - r is residual orthogonal to both
    """
    w_A_unit = w_A / (np.linalg.norm(w_A) + 1e-10)
    w_B_unit = w_B / (np.linalg.norm(w_B) + 1e-10)
    
    # Gram-Schmidt
    e1 = w_A_unit
    proj_B_on_A = np.dot(w_B_unit, e1) * e1
    e2_unnorm = w_B_unit - proj_B_on_A
    e2_norm = np.linalg.norm(e2_unnorm)
    
    if e2_norm < 1e-8:
        print("  Warning: Single-domain directions are nearly parallel!")
        e2 = np.zeros_like(e1)
    else:
        e2 = e2_unnorm / e2_norm
    
    # Project w_C
    a = np.dot(w_C, e1)
    b = np.dot(w_C, e2)
    
    projection = a * e1 + b * e2
    residual = w_C - projection
    
    return {
        'a': a,
        'b': b,
        'residual': residual,
        'residual_norm': np.linalg.norm(residual),
        'e1': e1,
        'e2': e2,
        'projection': projection,
        'projection_norm': np.linalg.norm(projection)
    }


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_direction_on_pooled(direction, X, y):
    """Evaluate a direction as a linear classifier on pooled data."""
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    projections = X @ direction
    
    try:
        auc_pos = roc_auc_score(y, projections)
        auc_neg = roc_auc_score(y, -projections)
        auc = max(auc_pos, auc_neg)
    except:
        auc = 0.5
    
    preds = (projections > 0).astype(int)
    acc = accuracy_score(y, preds)
    
    return {'auc': auc, 'accuracy': acc}


def evaluate_direction_per_token(direction, activations, labels, aggregation='mean'):
    """Evaluate direction on per-token activations with aggregation."""
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    sample_probs = []
    
    for x in activations:
        token_scores = x @ direction
        probs = 1 / (1 + np.exp(-token_scores))  # Sigmoid
        
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
    
    return {'auc': auc, 'accuracy': acc}


def evaluate_all_directions(X_a, y_a, X_b, y_b, decomposition, aggregation='mean', per_token=False, acts_a=None, acts_b=None):
    """Evaluate all direction components on both domains."""
    results = {}
    
    directions = {
        'Domain A Direction (e1)': decomposition['e1'],
        'Domain B Orth (e2)': decomposition['e2'],
        'Residual (r)': decomposition['residual'],
        'Combined Projection': decomposition['projection']
    }
    
    for name, direction in directions.items():
        if np.linalg.norm(direction) < 1e-8:
            results[name] = {'domain_a': {'auc': 0.5}, 'domain_b': {'auc': 0.5}}
            continue
        
        if per_token and acts_a is not None and acts_b is not None:
            results[name] = {
                'domain_a': evaluate_direction_per_token(direction, acts_a, y_a, aggregation),
                'domain_b': evaluate_direction_per_token(direction, acts_b, y_b, aggregation)
            }
        else:
            results[name] = {
                'domain_a': evaluate_direction_on_pooled(direction, X_a, y_a),
                'domain_b': evaluate_direction_on_pooled(direction, X_b, y_b)
            }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_decomposition_analysis(decomposition, eval_results, output_path, label_a, label_b):
    """Create visualization of the decomposition and performance."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Component magnitudes
    ax = axes[0]
    components = [f'|a| ({label_a})', f'|b| ({label_b} orth)', '|r| (Residual)']
    magnitudes = [abs(decomposition['a']), abs(decomposition['b']), decomposition['residual_norm']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(components, magnitudes, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Combined Probe Decomposition\nw_C = a·e1 + b·e2 + r', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mag in zip(bars, magnitudes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mag:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # 2. Performance on Domain A
    ax = axes[1]
    directions = list(eval_results.keys())
    aucs_a = [eval_results[d]['domain_a']['auc'] for d in directions]
    
    bars = ax.barh(directions, aucs_a, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'Performance on {label_a}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, auc in zip(bars, aucs_a):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 3. Performance on Domain B
    ax = axes[2]
    aucs_b = [eval_results[d]['domain_b']['auc'] for d in directions]
    
    bars = ax.barh(directions, aucs_b, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'Performance on {label_b}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, auc in zip(bars, aucs_b):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_generalization_comparison(eval_results, output_path, label_a, label_b):
    """Compare generalization across domains for each component."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    directions = list(eval_results.keys())
    x = np.arange(len(directions))
    width = 0.35
    
    aucs_a = [eval_results[d]['domain_a']['auc'] for d in directions]
    aucs_b = [eval_results[d]['domain_b']['auc'] for d in directions]
    
    bars1 = ax.bar(x - width/2, aucs_a, width, label=label_a, color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, aucs_b, width, label=label_b, color='#e74c3c', edgecolor='black')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax.set_title('Which Direction Component Generalizes Best?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(directions, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, aucs_a):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, aucs_b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    # Highlight best
    avg_aucs = [(aucs_a[i] + aucs_b[i]) / 2 for i in range(len(directions))]
    best_idx = np.argmax(avg_aucs)
    ax.annotate(f'Best Average: {directions[best_idx]}', 
                xy=(best_idx, max(aucs_a[best_idx], aucs_b[best_idx]) + 0.08),
                fontsize=11, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_aggregation_comparison(all_results, output_path, label_a, label_b):
    """Compare results across aggregation methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, agg in zip(axes, AGGREGATION_METHODS):
        if agg not in all_results:
            continue
        
        eval_res = all_results[agg]
        directions = list(eval_res.keys())
        
        aucs_a = [eval_res[d]['domain_a']['auc'] for d in directions]
        aucs_b = [eval_res[d]['domain_b']['auc'] for d in directions]
        
        x = np.arange(len(directions))
        width = 0.35
        
        ax.bar(x - width/2, aucs_a, width, label=label_a, color='#3498db', alpha=0.8)
        ax.bar(x + width/2, aucs_b, width, label=label_b, color='#e74c3c', alpha=0.8)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('AUC', fontsize=11)
        ax.set_title(f'{agg.upper()} Aggregation', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(directions, rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Invariant Core Analysis: Per-Token Probes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Invariant Core Analysis for Per-Token Probes")
    parser.add_argument('--probes_a', type=str, required=True, help='Per-token probes dir for domain A')
    parser.add_argument('--probes_b', type=str, required=True, help='Per-token probes dir for domain B')
    parser.add_argument('--probes_combined', type=str, required=True, help='Combined per-token probes dir')
    parser.add_argument('--val_a', type=str, required=True, help='Domain A validation activations')
    parser.add_argument('--val_b', type=str, required=True, help='Domain B validation activations')
    parser.add_argument('--ood_results_a', type=str, default=None, 
                        help='Optional: OOD summary JSON for domain A probes (auto-detect best layer)')
    parser.add_argument('--ood_results_b', type=str, default=None,
                        help='Optional: OOD summary JSON for domain B probes (auto-detect best layer)')
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--layer', type=int, default=None, help='Layer to use (auto-detected if ood_results provided)')
    parser.add_argument('--output_dir', type=str, default='results/invariant_core_per_token')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("INVARIANT CORE ANALYSIS - PER-TOKEN PROBES")
    print("Decomposing combined probe to find domain-invariant direction")
    print("=" * 70)
    
    # ========================================================================
    # AUTO-DETECT BEST LAYER
    # ========================================================================
    # Priority: 1. Manual --layer, 2. Combined probe's best layer, 3. OOD results, 4. Default
    layer = args.layer
    
    # Try to load combined_summary.json for best combined layer
    combined_summary_path = os.path.join(args.probes_combined, 'combined_summary.json')
    if layer is None and os.path.exists(combined_summary_path):
        with open(combined_summary_path, 'r') as f:
            combined_data = json.load(f)
        best_comb = combined_data.get('best_combined', {})
        layer = best_comb.get('layer', None)
        print(f"Combined Summary: Best layer {layer} ({best_comb.get('aggregation', '?')}) "
              f"Avg AUC={best_comb.get('auc_avg', 0):.4f}")
    
    # Fallback: use OOD results if provided
    if layer is None:
        if args.ood_results_a and os.path.exists(args.ood_results_a):
            with open(args.ood_results_a, 'r') as f:
                ood_a = json.load(f)
            best_a = ood_a.get('overall_best', {})
            print(f"OOD Results A: Best layer {best_a.get('layer', '?')} ({best_a.get('aggregation', '?')}) AUC={best_a.get('auc', 0):.4f}")
            layer = best_a.get('layer', 20)
        
        if args.ood_results_b and os.path.exists(args.ood_results_b):
            with open(args.ood_results_b, 'r') as f:
                ood_b = json.load(f)
            best_b = ood_b.get('overall_best', {})
            print(f"OOD Results B: Best layer {best_b.get('layer', '?')} ({best_b.get('aggregation', '?')}) AUC={best_b.get('auc', 0):.4f}")
            # If we already have layer from A, average with B
            if args.ood_results_a and os.path.exists(args.ood_results_a):
                layer = (best_a.get('layer', 20) + best_b.get('layer', 20)) // 2
                print(f"Using average of OOD layers: {layer}")
            else:
                layer = best_b.get('layer', 20)
    
    if layer is None:
        layer = 20
        print(f"Using default layer: {layer}")
    
    print(f"\nLayer for analysis: {layer}")


    
    # ========================================================================
    # 1. LOAD PROBES
    # ========================================================================
    print("\n1. Loading per-token probes...")
    
    probe_a = load_per_token_probe(args.probes_a, layer)
    probe_b = load_per_token_probe(args.probes_b, layer)
    probe_comb = load_per_token_probe(args.probes_combined, layer)
    
    if probe_a is None or probe_b is None or probe_comb is None:
        print("ERROR: Could not load all required probes!")
        print(f"  probe_a ({args.probes_a}): {'Found' if probe_a else 'Missing'}")
        print(f"  probe_b ({args.probes_b}): {'Found' if probe_b else 'Missing'}")
        print(f"  probe_comb ({args.probes_combined}): {'Found' if probe_comb else 'Missing'}")
        return 1
    
    # Extract directions
    dir_a = probe_a.get_direction()
    dir_b = probe_b.get_direction()
    dir_comb = probe_comb.get_direction()
    
    print(f"   |w_A| = 1.0 (normalized), dim={len(dir_a)}")
    print(f"   |w_B| = 1.0 (normalized)")
    print(f"   |w_C| = 1.0 (normalized)")
    print(f"   cos(w_A, w_B) = {np.dot(dir_a, dir_b):.4f}")
    
    # ========================================================================
    # 2. LOAD ACTIVATIONS
    # ========================================================================
    print("\n2. Loading validation activations...")
    
    acts_a, y_a = load_activations_with_tokens(args.val_a, layer)
    acts_b, y_b = load_activations_with_tokens(args.val_b, layer)
    
    # Also load pooled for direction evaluation
    X_a, _ = load_pooled_activations(args.val_a, layer)
    X_b, _ = load_pooled_activations(args.val_b, layer)
    
    print(f"   {args.label_a}: {len(acts_a)} samples")
    print(f"   {args.label_b}: {len(acts_b)} samples")
    
    # Normalize (use combined norm stats if available)
    mean_comb, std_comb = load_norm_stats(args.probes_combined, layer)
    if mean_comb is not None:
        X_a = (X_a - mean_comb) / std_comb
        X_b = (X_b - mean_comb) / std_comb
        acts_a = [(x - mean_comb) / std_comb for x in acts_a]
        acts_b = [(x - mean_comb) / std_comb for x in acts_b]
    
    # ========================================================================
    # 3. DECOMPOSE COMBINED DIRECTION
    # ========================================================================
    print("\n3. Decomposing combined direction...")
    decomposition = decompose_combined_direction(dir_comb, dir_a, dir_b)
    
    print(f"   w_C = {decomposition['a']:.4f} * e1 + {decomposition['b']:.4f} * e2 + r")
    print(f"   |projection| = {decomposition['projection_norm']:.4f}")
    print(f"   |residual| = {decomposition['residual_norm']:.4f}")
    print(f"   Residual fraction: {decomposition['residual_norm']:.2%}")
    
    # ========================================================================
    # 4. EVALUATE ON ALL AGGREGATIONS
    # ========================================================================
    print("\n4. Evaluating direction components...")
    
    all_results = {}
    for agg in AGGREGATION_METHODS:
        print(f"\n   --- {agg.upper()} Aggregation ---")
        eval_results = evaluate_all_directions(
            X_a, y_a, X_b, y_b, decomposition,
            aggregation=agg, per_token=True, acts_a=acts_a, acts_b=acts_b
        )
        all_results[agg] = eval_results
        
        for name, results in eval_results.items():
            auc_a = results['domain_a']['auc']
            auc_b = results['domain_b']['auc']
            print(f"     {name}: {args.label_a}={auc_a:.4f}, {args.label_b}={auc_b:.4f}")
    
    # ========================================================================
    # 5. CONCLUSION
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Use mean aggregation for primary conclusion
    mean_results = all_results['mean']
    avg_aucs = {name: (r['domain_a']['auc'] + r['domain_b']['auc']) / 2 
                for name, r in mean_results.items()}
    best_component = max(avg_aucs, key=avg_aucs.get)
    
    residual_auc = avg_aucs.get('Residual (r)', 0)
    projection_auc = avg_aucs.get('Combined Projection', 0)
    
    if residual_auc > projection_auc and residual_auc > 0.6:
        print(f"✅ RESIDUAL GENERALIZES BEST! (Avg AUC: {residual_auc:.4f})")
        print("   The 'invariant core' hypothesis is SUPPORTED.")
        print("   There exists a domain-invariant deception direction.")
    elif best_component == 'Combined Projection':
        print(f"⚠️ Combined Projection works best (Avg AUC: {projection_auc:.4f})")
        print("   The combined probe uses a mix of domain-specific directions.")
    else:
        print(f"📊 Best component: {best_component} (Avg AUC: {avg_aucs[best_component]:.4f})")
    
    # ========================================================================
    # 6. VISUALIZATIONS
    # ========================================================================
    print("\n5. Generating visualizations...")
    
    plot_decomposition_analysis(
        decomposition, mean_results,
        os.path.join(args.output_dir, 'decomposition_analysis.png'),
        args.label_a, args.label_b
    )
    
    plot_generalization_comparison(
        mean_results,
        os.path.join(args.output_dir, 'generalization_comparison.png'),
        args.label_a, args.label_b
    )
    
    plot_aggregation_comparison(
        all_results,
        os.path.join(args.output_dir, 'aggregation_comparison.png'),
        args.label_a, args.label_b
    )
    
    # Save summary
    summary = {
        'config': {
            'layer': layer,
            'label_a': args.label_a,
            'label_b': args.label_b
        },
        'decomposition': {
            'a': float(decomposition['a']),
            'b': float(decomposition['b']),
            'residual_norm': float(decomposition['residual_norm']),
            'projection_norm': float(decomposition['projection_norm']),
            'cos_a_b': float(np.dot(dir_a, dir_b))
        },
        'evaluation': {
            agg: {name: {k: float(v['auc']) for k, v in results.items()}
                  for name, results in eval_res.items()}
            for agg, eval_res in all_results.items()
        },
        'conclusion': {
            'best_component': best_component,
            'best_avg_auc': float(avg_aucs[best_component]),
            'invariant_core_supported': bool(residual_auc > projection_auc and residual_auc > 0.6)
        }
    }
    
    with open(os.path.join(args.output_dir, 'invariant_core_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
