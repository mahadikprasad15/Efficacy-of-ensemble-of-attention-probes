#!/usr/bin/env python3
"""
Invariant Core Sweep: Run across all layers and pooling types
==============================================================

This script:
1. Sweeps across all layers and pooling types
2. Computes invariant core for each (layer, pooling) combination
3. Evaluates invariant core AND single-domain probes on OOD data
4. Generates comparison line plots
5. Saves all probes and results

Usage (Colab):
    python scripts/pipelines/sweep_invariant_core.py \
        --base_data_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data \
        --model meta-llama_Llama-3.2-3B-Instruct \
        --output_dir results/invariant_core_sweep

Output:
    results/invariant_core_sweep/
        ├── invariant_probes/           # All invariant core probes
        ├── sweep_results.json          # Full results for all layers/pooling
        ├── ood_comparison_<pooling>.png # Line plots per pooling type
        └── best_invariant_summary.json # Best layer for each pooling
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
from tqdm import tqdm

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../actprobe/src'))


# ============================================================================
# CORE FUNCTIONS (from run_invariant_core_pipeline.py)
# ============================================================================

def load_activations(act_dir, layer, pooling, num_samples=None):
    """Load and pool activations from a directory."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        return None, None
    
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    if num_samples is not None and num_samples < len(manifest):
        manifest = manifest[:num_samples]
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations, labels = [], []
    for entry in manifest:
        eid = entry['id']
        if eid in all_tensors:
            tensor = all_tensors[eid]
            if layer >= tensor.shape[0]:
                continue
            x_layer = tensor[layer, :, :]
            if pooling == 'mean' or pooling == 'attn':
                pooled = x_layer.mean(dim=0)
            elif pooling == 'max':
                pooled = x_layer.max(dim=0)[0]
            elif pooling == 'last':
                pooled = x_layer[-1, :]
            else:
                pooled = x_layer.mean(dim=0)
            activations.append(pooled.numpy())
            labels.append(entry['label'])
    
    if len(activations) == 0:
        return None, None
    return np.array(activations), np.array(labels)


def get_probe_direction(probe_path):
    """Load probe and extract direction."""
    if not os.path.exists(probe_path):
        return None
    
    state_dict = torch.load(probe_path, map_location='cpu')
    
    priority_keys = ['classifier.weight', 'net.0.weight', 'pooling.weight', 'pooling.query']
    
    for key in priority_keys:
        if key in state_dict:
            W = state_dict[key].cpu().numpy()
            if len(W.shape) == 2:
                u, s, vt = np.linalg.svd(W, full_matrices=False)
                return vt[0]
            elif len(W.shape) == 1:
                return W
    
    for key in sorted(state_dict.keys()):
        if 'weight' in key and len(state_dict[key].shape) == 2:
            W = state_dict[key].cpu().numpy()
            u, s, vt = np.linalg.svd(W, full_matrices=False)
            return vt[0]
    
    return None


def decompose_combined_direction(w_C, w_R, w_I):
    """Gram-Schmidt decomposition to find invariant core."""
    w_R_unit = w_R / (np.linalg.norm(w_R) + 1e-10)
    w_I_unit = w_I / (np.linalg.norm(w_I) + 1e-10)
    
    e1 = w_R_unit
    proj_I_on_R = np.dot(w_I_unit, e1) * e1
    e2_unnorm = w_I_unit - proj_I_on_R
    e2_norm = np.linalg.norm(e2_unnorm)
    
    if e2_norm < 1e-8:
        e2 = np.zeros_like(e1)
    else:
        e2 = e2_unnorm / e2_norm
    
    a = np.dot(w_C, e1)
    b = np.dot(w_C, e2)
    projection = a * e1 + b * e2
    residual = w_C - projection
    
    return {
        'a': a, 'b': b,
        'residual': residual,
        'residual_norm': np.linalg.norm(residual),
        'e1': e1, 'e2': e2,
        'projection': projection
    }


def evaluate_direction(X, y, direction):
    """Evaluate a direction as classifier, return AUC."""
    if direction is None or np.linalg.norm(direction) < 1e-10:
        return 0.5
    
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    projections = X @ direction
    
    try:
        auc_pos = roc_auc_score(y, projections)
        auc_neg = roc_auc_score(y, -projections)
        return max(auc_pos, auc_neg)
    except:
        return 0.5


# ============================================================================
# SWEEP LOGIC
# ============================================================================

def run_sweep(args):
    """Run sweep across all layers and pooling types."""
    
    base_dir = args.base_data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    pooling_types = ['mean', 'max', 'last', 'attn']
    layers = list(range(args.min_layer, args.max_layer + 1))
    
    # OOD activation directory
    ood_act_dir = os.path.join(base_dir, 'activations', args.model, 
                               args.ood_domain, args.ood_split)
    
    print("=" * 70)
    print("INVARIANT CORE SWEEP")
    print("=" * 70)
    print(f"Layers: {args.min_layer} to {args.max_layer}")
    print(f"Pooling types: {pooling_types}")
    print(f"OOD data: {ood_act_dir}")
    print(f"Output: {output_dir}")
    
    # Check OOD data exists
    if not os.path.exists(os.path.join(ood_act_dir, 'manifest.jsonl')):
        print(f"ERROR: OOD manifest not found at {ood_act_dir}")
        return 1
    
    # Results storage
    all_results = {
        'config': {
            'model': args.model,
            'layers': layers,
            'pooling_types': pooling_types,
            'ood_domain': args.ood_domain,
            'ood_split': args.ood_split,
            'num_ood_samples': args.num_ood_samples
        },
        'results': {}
    }
    
    # Probe output directory
    probe_output_dir = os.path.join(output_dir, 'invariant_probes')
    os.makedirs(probe_output_dir, exist_ok=True)
    
    # Run sweep
    for pooling in pooling_types:
        print(f"\n{'='*70}")
        print(f"POOLING: {pooling.upper()}")
        print(f"{'='*70}")
        
        all_results['results'][pooling] = []
        
        # Probe directories
        probes_a_base = os.path.join(base_dir, 'probes', args.model, 
                                      args.domain_a, pooling)
        probes_b_base = os.path.join(base_dir, 'probes_flipped', args.model, 
                                      args.domain_b, pooling)
        probes_comb_base = os.path.join(base_dir, 'probes_combined', args.model, 
                                         'Deception-Combined', pooling)
        
        for layer in tqdm(layers, desc=f"Layers ({pooling})"):
            
            # Load probe directions
            probe_a_path = os.path.join(probes_a_base, f'probe_layer_{layer}.pt')
            probe_b_path = os.path.join(probes_b_base, f'probe_layer_{layer}.pt')
            probe_comb_path = os.path.join(probes_comb_base, f'probe_layer_{layer}.pt')
            
            w_R = get_probe_direction(probe_a_path)
            w_I = get_probe_direction(probe_b_path)
            w_C = get_probe_direction(probe_comb_path)
            
            if w_R is None or w_I is None or w_C is None:
                # Missing probes for this layer
                all_results['results'][pooling].append({
                    'layer': layer,
                    'error': 'missing_probes'
                })
                continue
            
            # Compute invariant core
            decomp = decompose_combined_direction(w_C, w_R, w_I)
            residual = decomp['residual']
            
            # Load OOD activations
            X_ood, y_ood = load_activations(ood_act_dir, layer, pooling, 
                                            args.num_ood_samples)
            
            if X_ood is None:
                all_results['results'][pooling].append({
                    'layer': layer,
                    'error': 'no_activations'
                })
                continue
            
            # Normalize
            mean, std = X_ood.mean(0), X_ood.std(0) + 1e-8
            X_ood_norm = (X_ood - mean) / std
            
            # Evaluate all directions on OOD
            auc_invariant = evaluate_direction(X_ood_norm, y_ood, residual)
            auc_roleplaying = evaluate_direction(X_ood_norm, y_ood, decomp['e1'])
            auc_insider = evaluate_direction(X_ood_norm, y_ood, decomp['e2'])
            auc_combined = evaluate_direction(X_ood_norm, y_ood, w_C)
            auc_projection = evaluate_direction(X_ood_norm, y_ood, decomp['projection'])
            
            # Also evaluate single-domain probes directly (not orthogonalized)
            auc_roleplaying_raw = evaluate_direction(X_ood_norm, y_ood, w_R)
            auc_insider_raw = evaluate_direction(X_ood_norm, y_ood, w_I)
            
            layer_result = {
                'layer': layer,
                'ood_auc': {
                    'invariant_core': float(auc_invariant),
                    'roleplaying_e1': float(auc_roleplaying),
                    'insider_orth_e2': float(auc_insider),
                    'combined': float(auc_combined),
                    'projection': float(auc_projection),
                    'roleplaying_raw': float(auc_roleplaying_raw),
                    'insider_raw': float(auc_insider_raw)
                },
                'decomposition': {
                    'a': float(decomp['a']),
                    'b': float(decomp['b']),
                    'residual_norm': float(decomp['residual_norm'])
                }
            }
            
            all_results['results'][pooling].append(layer_result)
            
            # Save invariant probe
            residual_normalized = residual / (np.linalg.norm(residual) + 1e-10)
            probe_state = {
                'classifier.weight': torch.from_numpy(residual_normalized.reshape(1, -1)).float(),
                'classifier.bias': torch.zeros(1)
            }
            
            layer_probe_dir = os.path.join(probe_output_dir, pooling)
            os.makedirs(layer_probe_dir, exist_ok=True)
            torch.save(probe_state, os.path.join(layer_probe_dir, f'invariant_layer_{layer}.pt'))
    
    # Save full results
    results_path = os.path.join(output_dir, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved: {results_path}")
    
    # Generate plots
    generate_comparison_plots(all_results, output_dir)
    
    # Find and save best configurations
    save_best_summary(all_results, output_dir)
    
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    
    return 0


def generate_comparison_plots(all_results, output_dir):
    """Generate line plots comparing invariant core vs ID probes on OOD."""
    
    print("\nGenerating comparison plots...")
    
    for pooling, layer_results in all_results['results'].items():
        
        # Filter out errors
        valid_results = [r for r in layer_results if 'error' not in r]
        if len(valid_results) == 0:
            continue
        
        layers = [r['layer'] for r in valid_results]
        
        # Extract AUCs
        auc_invariant = [r['ood_auc']['invariant_core'] for r in valid_results]
        auc_roleplaying = [r['ood_auc']['roleplaying_raw'] for r in valid_results]
        auc_insider = [r['ood_auc']['insider_raw'] for r in valid_results]
        auc_combined = [r['ood_auc']['combined'] for r in valid_results]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(layers, auc_invariant, 'o-', linewidth=2.5, markersize=6, 
                label='Invariant Core', color='#2ecc71')
        ax.plot(layers, auc_roleplaying, 's--', linewidth=1.5, markersize=5, 
                label='Roleplaying (ID→OOD)', color='#3498db', alpha=0.7)
        ax.plot(layers, auc_insider, '^--', linewidth=1.5, markersize=5, 
                label='InsiderTrading (ID→OOD)', color='#e74c3c', alpha=0.7)
        ax.plot(layers, auc_combined, 'd-', linewidth=2, markersize=5, 
                label='Combined Probe', color='#9b59b6', alpha=0.8)
        
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
        
        # Mark best layer for invariant
        best_idx = np.argmax(auc_invariant)
        best_layer = layers[best_idx]
        best_auc = auc_invariant[best_idx]
        ax.annotate(f'Best: L{best_layer}\nAUC={best_auc:.3f}', 
                    xy=(best_layer, best_auc), xytext=(best_layer + 2, best_auc + 0.05),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                    bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('OOD AUC', fontsize=12)
        ax.set_title(f'OOD Generalization: Invariant Core vs ID Probes ({pooling.upper()} pooling)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'ood_comparison_{pooling}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_path}")
    
    # Also create combined plot with all pooling types for invariant core
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'mean': '#2ecc71', 'max': '#3498db', 'last': '#e74c3c', 'attn': '#9b59b6'}
    
    for pooling, layer_results in all_results['results'].items():
        valid_results = [r for r in layer_results if 'error' not in r]
        if len(valid_results) == 0:
            continue
        
        layers = [r['layer'] for r in valid_results]
        auc_invariant = [r['ood_auc']['invariant_core'] for r in valid_results]
        
        ax.plot(layers, auc_invariant, 'o-', linewidth=2, markersize=5, 
                label=f'{pooling.upper()}', color=colors.get(pooling, 'gray'))
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('OOD AUC', fontsize=12)
    ax.set_title('Invariant Core OOD Performance Across Pooling Types', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'invariant_core_all_pooling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


def save_best_summary(all_results, output_dir):
    """Find and save best configurations."""
    
    summary = {}
    
    for pooling, layer_results in all_results['results'].items():
        valid_results = [r for r in layer_results if 'error' not in r]
        if len(valid_results) == 0:
            summary[pooling] = {'error': 'no_valid_results'}
            continue
        
        # Best invariant core
        best_invariant = max(valid_results, key=lambda r: r['ood_auc']['invariant_core'])
        
        # Best ID probes on OOD
        best_roleplaying = max(valid_results, key=lambda r: r['ood_auc']['roleplaying_raw'])
        best_insider = max(valid_results, key=lambda r: r['ood_auc']['insider_raw'])
        best_combined = max(valid_results, key=lambda r: r['ood_auc']['combined'])
        
        summary[pooling] = {
            'invariant_core': {
                'best_layer': best_invariant['layer'],
                'best_auc': best_invariant['ood_auc']['invariant_core']
            },
            'roleplaying_id_probe': {
                'best_layer': best_roleplaying['layer'],
                'best_auc': best_roleplaying['ood_auc']['roleplaying_raw']
            },
            'insider_id_probe': {
                'best_layer': best_insider['layer'],
                'best_auc': best_insider['ood_auc']['insider_raw']
            },
            'combined_probe': {
                'best_layer': best_combined['layer'],
                'best_auc': best_combined['ood_auc']['combined']
            }
        }
    
    summary_path = os.path.join(output_dir, 'best_invariant_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {summary_path}")
    
    # Print summary
    print("\n" + "-" * 50)
    print("BEST CONFIGURATIONS (OOD AUC)")
    print("-" * 50)
    for pooling, data in summary.items():
        if 'error' in data:
            continue
        print(f"\n{pooling.upper()}:")
        print(f"  Invariant Core:   Layer {data['invariant_core']['best_layer']:2d} → AUC {data['invariant_core']['best_auc']:.4f}")
        print(f"  Roleplaying ID:   Layer {data['roleplaying_id_probe']['best_layer']:2d} → AUC {data['roleplaying_id_probe']['best_auc']:.4f}")
        print(f"  InsiderTrad ID:   Layer {data['insider_id_probe']['best_layer']:2d} → AUC {data['insider_id_probe']['best_auc']:.4f}")
        print(f"  Combined:         Layer {data['combined_probe']['best_layer']:2d} → AUC {data['combined_probe']['best_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Invariant Core Sweep")
    parser.add_argument('--base_data_dir', type=str, required=True,
                        help='Base data directory')
    parser.add_argument('--model', type=str, default='meta-llama_Llama-3.2-3B-Instruct',
                        help='Model name')
    parser.add_argument('--output_dir', type=str, default='results/invariant_core_sweep',
                        help='Output directory (NOT in data/)')
    parser.add_argument('--min_layer', type=int, default=1, help='Minimum layer')
    parser.add_argument('--max_layer', type=int, default=28, help='Maximum layer')
    parser.add_argument('--num_ood_samples', type=int, default=None,
                        help='Number of OOD samples (None = all)')
    parser.add_argument('--domain_a', type=str, default='Deception-Roleplaying',
                        help='Domain A (ID probe source)')
    parser.add_argument('--domain_b', type=str, default='Deception-InsiderTrading',
                        help='Domain B (ID probe source / OOD target)')
    parser.add_argument('--ood_domain', type=str, default='Deception-InsiderTrading',
                        help='OOD evaluation domain')
    parser.add_argument('--ood_split', type=str, default='test',
                        help='OOD split (test or validation)')
    args = parser.parse_args()
    
    return run_sweep(args)


if __name__ == "__main__":
    sys.exit(main())
