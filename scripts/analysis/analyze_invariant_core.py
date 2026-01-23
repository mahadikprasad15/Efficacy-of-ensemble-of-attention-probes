#!/usr/bin/env python3
"""
Invariant Core Analysis: Is There a Domain-Invariant Deception Direction?
==========================================================================
This script:
1. Loads best probes automatically from results JSON files
2. Decomposes combined probe: w_C = a·ŵ_R + b·ŵ_I + r (residual)
3. Tests each component (Roleplaying direction, InsiderTrading direction, residual) 
   for OOD generalization
4. If residual `r` generalizes best, it's the "invariant core"

The hypothesis: The combined probe learns something domain-invariant that the
single-domain probes don't capture. By removing the domain-specific components,
we can isolate this invariant representation.

Usage:
    python scripts/analysis/analyze_invariant_core.py \
        --results_a /path/to/results/ood_evaluation/ood_results_all_pooling.json \
        --results_b /path/to/results_flipped/ood_evaluation/ood_results_all_pooling.json \
        --results_combined /path/to/results/combined_all_pooling/results.json \
        --probes_a /path/to/data/probes \
        --probes_b /path/to/data/probes_flipped \
        --probes_combined /path/to/data/probes_combined \
        --val_a /path/to/Deception-Roleplaying/validation \
        --val_b /path/to/Deception-InsiderTrading/validation \
        --model meta-llama_Llama-3.2-3B-Instruct \
        --output_dir results/invariant_core_analysis
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

# ============================================================================
# PROBE ARCHITECTURES
# ============================================================================
class SequentialProbe(nn.Module):
    """Probe architecture used by train_combined_all_pooling.py"""
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
    """Probe with learned attention pooling (has pooling.query key)"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.pooling = nn.Linear(input_dim, 1)  # This produces "pooling.weight" and "pooling.bias"
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x: (B, T, D) -> attention -> (B, D)
        attn_weights = torch.softmax(self.pooling(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


# Add path for LayerProbe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../actprobe/src'))
try:
    from actprobe.probes.models import LayerProbe
    HAS_LAYERPROBE = True
except ImportError:
    HAS_LAYERPROBE = False


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
            # Apply pooling (attn uses mean for direction analysis)
            if pooling == 'mean' or pooling == 'attn':
                pooled = x_layer.mean(dim=0)
            elif pooling == 'max':
                pooled = x_layer.max(dim=0)[0]
            elif pooling == 'last':
                pooled = x_layer[-1, :]
            else:
                pooled = x_layer.mean(dim=0)  # Default to mean
            activations.append(pooled.numpy())
            labels.append(entry['label'])
    
    return np.array(activations), np.array(labels)


def load_probe(probe_path, input_dim):
    """Load a probe from disk, detecting architecture automatically."""
    device = torch.device('cpu')
    state_dict = torch.load(probe_path, map_location=device)
    
    # Check for attention pooling probe (has pooling.query or pooling.weight)
    if 'pooling.query' in state_dict or 'pooling.weight' in state_dict:
        # Get the direction from classifier weights instead of loading full model
        print(f"  ✓ Found attention probe, extracting direction from weights")
        return state_dict  # Return state_dict directly, we'll extract direction from it
    
    # Sequential probe (net.0.weight, net.3.weight)
    if 'net.0.weight' in state_dict:
        hidden_dim = state_dict['net.0.weight'].shape[0]
        probe = SequentialProbe(input_dim, hidden_dim)
        probe.load_state_dict(state_dict)
        print(f"  ✓ Loaded SequentialProbe from {probe_path}")
        return probe
    
    # LayerProbe from actprobe (classifier.weight)
    if 'classifier.weight' in state_dict:
        if HAS_LAYERPROBE:
            try:
                probe = LayerProbe(input_dim=input_dim, pooling_type='mean')
                probe.load_state_dict(state_dict)
                print(f"  ✓ Loaded LayerProbe from {probe_path}")
                return probe
            except:
                pass
        # Fallback: return state_dict for direction extraction
        print(f"  ✓ Found classifier weights, will extract direction")
        return state_dict
    
    print(f"  Warning: Unknown architecture for {probe_path}, keys: {list(state_dict.keys())[:5]}")
    return state_dict  # Return state_dict anyway


def get_probe_direction(probe_or_state_dict):
    """Extract the first layer weights as the probe direction."""
    # Handle state_dict directly
    if isinstance(probe_or_state_dict, dict):
        state_dict = probe_or_state_dict
    else:
        state_dict = probe_or_state_dict.state_dict()
    
    # Priority order for finding the weight matrix
    priority_keys = ['classifier.weight', 'net.0.weight', 'pooling.weight', 'pooling.query']
    
    for key in priority_keys:
        if key in state_dict:
            W = state_dict[key].cpu().numpy()
            if len(W.shape) == 2:
                u, s, vt = np.linalg.svd(W, full_matrices=False)
                return vt[0]
            elif len(W.shape) == 1:
                return W  # 1D case
    
    # Fallback: find any weight matrix
    for key in sorted(state_dict.keys()):
        if 'weight' in key and len(state_dict[key].shape) == 2:
            W = state_dict[key].cpu().numpy()
            u, s, vt = np.linalg.svd(W, full_matrices=False)
            return vt[0]
    
    raise ValueError(f"Could not extract direction, keys: {list(state_dict.keys())}")


# ============================================================================
# BEST PROBE FINDING
# ============================================================================
def find_best_probe_from_ood_json(json_path, target_domain='ood'):
    """
    Parse ood_results_all_pooling.json to find the best probe.
    Returns: dict with pooling, layer, auc
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    best = {'pooling': 'mean', 'layer': 20, 'auc': 0}  # Default values
    
    # Structure 1: Direct {pooling: {best_layer, best_auc, ...}}
    # This is the format from OOD evaluation scripts
    for pooling in ['mean', 'max', 'last', 'attn']:
        if pooling in data and isinstance(data[pooling], dict):
            pooling_data = data[pooling]
            # Check for direct best_layer/best_auc keys
            if 'best_layer' in pooling_data and 'best_auc' in pooling_data:
                auc = pooling_data['best_auc']
                layer = pooling_data['best_layer']
                if auc > best['auc']:
                    best = {'pooling': pooling, 'layer': layer, 'auc': auc}
    
    # If we found results, return
    if best['auc'] > 0:
        return best
    
    # Structure 2: 'results' key with nested structure
    if 'results' in data:
        for pooling, layers_data in data['results'].items():
            if isinstance(layers_data, dict):
                for layer_str, metrics in layers_data.items():
                    if isinstance(metrics, dict):
                        auc = metrics.get('ood_auc', metrics.get('auc', 0))
                        try:
                            layer = int(layer_str.replace('layer_', ''))
                        except:
                            layer = 20
                        if auc > best['auc']:
                            best = {'pooling': pooling, 'layer': layer, 'auc': auc}
    
    # Structure 3: 'per_layer_results' (combined format)
    if 'per_layer_results' in data:
        for pooling, layer_results in data['per_layer_results'].items():
            if isinstance(layer_results, list):
                for result in layer_results:
                    auc = result.get('auc_a', 0) + result.get('auc_b', 0)
                    if auc > best['auc']:
                        best = {'pooling': pooling, 'layer': result.get('layer', 20), 'auc': auc}
    
    # Structure 4: 'summary' key
    if 'summary' in data:
        for pooling, summary in data['summary'].items():
            auc = summary.get('best_id_auc', summary.get('best_auc', summary.get('auc', 0)))
            layer = summary.get('best_layer', summary.get('layer', 20))
            if auc > best['auc']:
                best = {'pooling': pooling, 'layer': layer, 'auc': auc}
    
    return best


def find_best_combined_probe(json_path):
    """Parse combined_all_pooling results.json to find best probe."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    best = {'pooling': None, 'layer': None, 'auc': 0}
    
    if 'per_layer_results' in data:
        for pooling, layer_results in data['per_layer_results'].items():
            for result in layer_results:
                # Use average of both domain AUCs
                auc = (result.get('auc_a', 0) + result.get('auc_b', 0)) / 2
                if auc > best['auc']:
                    best = {'pooling': pooling, 'layer': result['layer'], 'auc': auc}
    elif 'summary' in data:
        # Use summary data
        for pooling, summary in data['summary'].items():
            auc = (summary.get('best_auc_a', 0) + summary.get('best_auc_b', 0)) / 2
            layer = summary.get('best_layer_a', 20)  # Use layer A as representative
            if auc > best['auc']:
                best = {'pooling': pooling, 'layer': layer, 'auc': auc}
    
    return best


# ============================================================================
# DECOMPOSITION
# ============================================================================
def decompose_combined_direction(w_C, w_R, w_I):
    """
    Decompose combined direction into:
    w_C = a * ŵ_R + b * ŵ_I_orth + r
    
    Where:
    - ŵ_R is unit vector in Roleplaying direction
    - ŵ_I_orth is component of InsiderTrading orthogonal to ŵ_R (Gram-Schmidt)
    - r is residual orthogonal to both
    
    Returns: (a, b, r, w_R_unit, w_I_orth_unit)
    """
    # Normalize directions
    w_R_unit = w_R / (np.linalg.norm(w_R) + 1e-10)
    w_I_unit = w_I / (np.linalg.norm(w_I) + 1e-10)
    
    # Gram-Schmidt orthogonalization
    # e1 = ŵ_R
    e1 = w_R_unit
    # e2 = orthogonal component of ŵ_I
    proj_I_on_R = np.dot(w_I_unit, e1) * e1
    e2_unnorm = w_I_unit - proj_I_on_R
    e2_norm = np.linalg.norm(e2_unnorm)
    
    if e2_norm < 1e-8:
        # Directions are nearly parallel
        print("  Warning: Single-domain directions are nearly parallel!")
        e2 = np.zeros_like(e1)
    else:
        e2 = e2_unnorm / e2_norm
    
    # Project w_C onto the 2D subspace spanned by e1, e2
    a = np.dot(w_C, e1)  # Component along Roleplaying
    b = np.dot(w_C, e2)  # Component along orthogonal InsiderTrading
    
    # Residual = w_C - projection onto 2D plane
    projection = a * e1 + b * e2
    residual = w_C - projection
    
    return {
        'a': a,
        'b': b,
        'residual': residual,
        'residual_norm': np.linalg.norm(residual),
        'e1': e1,  # Roleplaying unit direction
        'e2': e2,  # Orthogonal InsiderTrading direction
        'projection': projection,
        'projection_norm': np.linalg.norm(projection)
    }


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_direction_as_classifier(X, y, direction, threshold=0):
    """
    Evaluate a direction vector as a linear classifier.
    Projects data onto direction and classifies based on threshold.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    projections = X @ direction
    
    # Try both signs and pick best
    preds_pos = (projections > threshold).astype(int)
    preds_neg = (projections < threshold).astype(int)
    
    try:
        auc_pos = roc_auc_score(y, projections)
        auc_neg = roc_auc_score(y, -projections)
        
        if auc_pos >= auc_neg:
            auc = auc_pos
            acc = accuracy_score(y, preds_pos)
        else:
            auc = auc_neg
            acc = accuracy_score(y, preds_neg)
    except:
        auc = 0.5
        acc = 0.5
    
    return {'auc': auc, 'accuracy': acc}


def evaluate_all_directions(X_a, y_a, X_b, y_b, decomposition):
    """Evaluate all direction components on both domains."""
    results = {}
    
    directions = {
        'Roleplaying Direction (e1)': decomposition['e1'],
        'InsiderTrading Orth (e2)': decomposition['e2'],
        'Residual (r)': decomposition['residual'],
        'Combined Projection': decomposition['projection']
    }
    
    for name, direction in directions.items():
        if np.linalg.norm(direction) < 1e-8:
            results[name] = {'domain_a': {'auc': 0.5, 'accuracy': 0.5},
                            'domain_b': {'auc': 0.5, 'accuracy': 0.5}}
            continue
            
        results[name] = {
            'domain_a': evaluate_direction_as_classifier(X_a, y_a, direction),
            'domain_b': evaluate_direction_as_classifier(X_b, y_b, direction)
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
    components = ['|a| (Roleplaying)', '|b| (InsiderTrading orth)', '|r| (Residual)']
    magnitudes = [abs(decomposition['a']), abs(decomposition['b']), decomposition['residual_norm']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(components, magnitudes, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Combined Probe Decomposition\nw_C = a·e1 + b·e2 + r', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mag in zip(bars, magnitudes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mag:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # 2. Performance comparison on Domain A
    ax = axes[1]
    directions = list(eval_results.keys())
    aucs_a = [eval_results[d]['domain_a']['auc'] for d in directions]
    
    bars = ax.barh(directions, aucs_a, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, label='Random')
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'Performance on {label_a}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, auc in zip(bars, aucs_a):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 3. Performance comparison on Domain B
    ax = axes[2]
    aucs_b = [eval_results[d]['domain_b']['auc'] for d in directions]
    
    bars = ax.barh(directions, aucs_b, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, label='Random')
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
    ax.set_title('Generalization Test: Which Component Works Best Across Domains?', 
                 fontsize=14, fontweight='bold')
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
    
    # Highlight best generalizing component
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


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Invariant Core Analysis")
    parser.add_argument('--results_a', type=str, required=True, help='OOD results JSON for domain A probes')
    parser.add_argument('--results_b', type=str, required=True, help='OOD results JSON for domain B probes')
    parser.add_argument('--results_combined', type=str, required=True, help='Combined training results JSON')
    parser.add_argument('--probes_a', type=str, required=True, help='Base dir for domain A probes')
    parser.add_argument('--probes_b', type=str, required=True, help='Base dir for domain B probes')
    parser.add_argument('--probes_combined', type=str, required=True, help='Base dir for combined probes')
    parser.add_argument('--val_a', type=str, required=True, help='Domain A validation activations')
    parser.add_argument('--val_b', type=str, required=True, help='Domain B validation activations')
    parser.add_argument('--model', type=str, default='meta-llama_Llama-3.2-3B-Instruct')
    parser.add_argument('--label_a', type=str, default='Roleplaying')
    parser.add_argument('--label_b', type=str, default='InsiderTrading')
    parser.add_argument('--output_dir', type=str, default='results/invariant_core_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("INVARIANT CORE ANALYSIS")
    print("Decomposing combined probe to find domain-invariant direction")
    print("=" * 70)
    
    # ========================================================================
    # 1. FIND BEST PROBES FROM RESULTS
    # ========================================================================
    print("\n1. Finding best probes from results JSONs...")
    
    best_a = find_best_probe_from_ood_json(args.results_a)
    print(f"   Best {args.label_a} probe: {best_a['pooling']} layer {best_a['layer']} (AUC: {best_a['auc']:.4f})")
    
    best_b = find_best_probe_from_ood_json(args.results_b)
    print(f"   Best {args.label_b} probe: {best_b['pooling']} layer {best_b['layer']} (AUC: {best_b['auc']:.4f})")
    
    best_comb = find_best_combined_probe(args.results_combined)
    print(f"   Best Combined probe: {best_comb['pooling']} layer {best_comb['layer']} (AUC: {best_comb['auc']:.4f})")
    
    # Use the same layer and pooling for fair comparison
    # Default to combined's best settings
    layer = best_comb['layer']
    pooling = best_comb['pooling']
    print(f"\n   Using layer={layer}, pooling={pooling} for analysis")
    
    # ========================================================================
    # 2. LOAD ACTIVATIONS
    # ========================================================================
    print("\n2. Loading validation activations...")
    X_a, y_a = load_activations(args.val_a, layer, pooling)
    X_b, y_b = load_activations(args.val_b, layer, pooling)
    print(f"   {args.label_a}: {len(X_a)} samples, dim={X_a.shape[1]}")
    print(f"   {args.label_b}: {len(X_b)} samples")
    
    # Normalize
    X_all = np.vstack([X_a, X_b])
    mean, std = X_all.mean(0), X_all.std(0) + 1e-8
    X_a_norm = (X_a - mean) / std
    X_b_norm = (X_b - mean) / std
    
    input_dim = X_a.shape[1]
    
    # ========================================================================
    # 3. LOAD PROBES
    # ========================================================================
    print("\n3. Loading probes...")
    
    # Construct probe paths
    probe_a_path = os.path.join(args.probes_a, args.model, f'Deception-{args.label_a}', 
                                 pooling, f'probe_layer_{layer}.pt')
    probe_b_path = os.path.join(args.probes_b, args.model, f'Deception-{args.label_b}', 
                                 pooling, f'probe_layer_{layer}.pt')
    probe_comb_path = os.path.join(args.probes_combined, args.model, 'Deception-Combined',
                                    pooling, f'probe_layer_{layer}.pt')
    
    print(f"   Loading: {probe_a_path}")
    probe_a = load_probe(probe_a_path, input_dim)
    
    print(f"   Loading: {probe_b_path}")
    probe_b = load_probe(probe_b_path, input_dim)
    
    print(f"   Loading: {probe_comb_path}")
    probe_comb = load_probe(probe_comb_path, input_dim)
    
    if probe_a is None or probe_b is None or probe_comb is None:
        print("ERROR: Could not load all probes!")
        return 1
    
    # ========================================================================
    # 4. EXTRACT DIRECTIONS
    # ========================================================================
    print("\n4. Extracting probe directions...")
    w_R = get_probe_direction(probe_a)  # Roleplaying direction
    w_I = get_probe_direction(probe_b)  # InsiderTrading direction
    w_C = get_probe_direction(probe_comb)  # Combined direction
    
    print(f"   |w_R| = {np.linalg.norm(w_R):.4f}")
    print(f"   |w_I| = {np.linalg.norm(w_I):.4f}")
    print(f"   |w_C| = {np.linalg.norm(w_C):.4f}")
    print(f"   cos(w_R, w_I) = {np.dot(w_R, w_I) / (np.linalg.norm(w_R) * np.linalg.norm(w_I)):.4f}")
    
    # ========================================================================
    # 5. DECOMPOSE COMBINED DIRECTION
    # ========================================================================
    print("\n5. Decomposing combined direction...")
    decomposition = decompose_combined_direction(w_C, w_R, w_I)
    
    print(f"   w_C = {decomposition['a']:.4f} * e1 + {decomposition['b']:.4f} * e2 + r")
    print(f"   |projection| = {decomposition['projection_norm']:.4f}")
    print(f"   |residual| = {decomposition['residual_norm']:.4f}")
    print(f"   Residual fraction: {decomposition['residual_norm'] / (np.linalg.norm(w_C) + 1e-10):.2%}")
    
    # ========================================================================
    # 6. EVALUATE EACH COMPONENT
    # ========================================================================
    print("\n6. Evaluating each direction component...")
    eval_results = evaluate_all_directions(X_a_norm, y_a, X_b_norm, y_b, decomposition)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Direction':<30} | {args.label_a + ' AUC':<15} | {args.label_b + ' AUC':<15} | {'Avg':<10}")
    print("-" * 70)
    
    for name, results in eval_results.items():
        auc_a = results['domain_a']['auc']
        auc_b = results['domain_b']['auc']
        avg = (auc_a + auc_b) / 2
        print(f"{name:<30} | {auc_a:<15.4f} | {auc_b:<15.4f} | {avg:<10.4f}")
    
    # ========================================================================
    # 7. CONCLUSION
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Find best generalizing component
    avg_aucs = {name: (r['domain_a']['auc'] + r['domain_b']['auc']) / 2 
                for name, r in eval_results.items()}
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
    # 8. GENERATE PLOTS
    # ========================================================================
    print("\n7. Generating visualizations...")
    
    plot_decomposition_analysis(
        decomposition, eval_results,
        os.path.join(args.output_dir, 'decomposition_analysis.png'),
        args.label_a, args.label_b
    )
    
    plot_generalization_comparison(
        eval_results,
        os.path.join(args.output_dir, 'generalization_comparison.png'),
        args.label_a, args.label_b
    )
    
    # Save summary
    summary = {
        'config': {
            'layer': layer,
            'pooling': pooling,
            'model': args.model
        },
        'best_probes': {
            'domain_a': best_a,
            'domain_b': best_b,
            'combined': best_comb
        },
        'decomposition': {
            'a': float(decomposition['a']),
            'b': float(decomposition['b']),
            'residual_norm': float(decomposition['residual_norm']),
            'projection_norm': float(decomposition['projection_norm'])
        },
        'evaluation': {name: {k: float(v['auc']) for k, v in results.items()} 
                       for name, results in eval_results.items()},
        'conclusion': {
            'best_component': best_component,
            'best_avg_auc': float(avg_aucs[best_component]),
            'invariant_core_supported': residual_auc > projection_auc and residual_auc > 0.6
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
