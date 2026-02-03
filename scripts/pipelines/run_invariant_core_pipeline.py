#!/usr/bin/env python3
"""
Invariant Core Pipeline: Compute and Save Domain-Invariant Deception Probe
===========================================================================

This script:
1. Loads single-domain probes (Roleplaying, InsiderTrading) and combined probe
2. Performs Gram-Schmidt decomposition to find the invariant core
3. Evaluates all direction components on OOD data
4. Saves the invariant core probe for later use

Usage (Colab):
    python scripts/pipelines/run_invariant_core_pipeline.py \
        --base_data_dir /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data \
        --model meta-llama_Llama-3.2-3B-Instruct \
        --layer 16 \
        --pooling mean \
        --num_ood_samples 100

Output:
    data/invariant_probes/<model>/layer_<N>_<pooling>/
        ├── invariant_core_probe.pt     # Usable probe file
        ├── residual_direction.npy      # Raw numpy direction
        ├── e1_roleplaying.npy          # Roleplaying basis vector
        └── e2_insider_orth.npy         # Orthogonal InsiderTrading basis

    data/results/invariant_probe_analysis/
        ├── decomposition_summary.json  # Coefficients, norms, metrics
        └── ood_evaluation.json         # OOD AUROC for each component
"""

import os
import sys
import json
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../actprobe/src'))

try:
    from actprobe.probes.models import LayerProbe
    HAS_LAYERPROBE = True
except ImportError:
    HAS_LAYERPROBE = False


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
    """Probe with learned attention pooling"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.pooling = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        attn_weights = torch.softmax(self.pooling(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


class InvariantCoreProbe(nn.Module):
    """
    Probe that uses the invariant core direction.
    Simple linear classifier with the residual direction as weights.
    """
    def __init__(self, direction: np.ndarray):
        super().__init__()
        input_dim = len(direction)
        self.classifier = nn.Linear(input_dim, 1, bias=False)
        # Set weights to the direction
        with torch.no_grad():
            self.classifier.weight.copy_(torch.from_numpy(direction.reshape(1, -1)).float())
    
    def forward(self, x):
        return self.classifier(x).squeeze(-1)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_activations(act_dir, layer, pooling, num_samples=None):
    """Load and pool activations from a directory."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    # Limit samples if specified
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
            x_layer = tensor[layer, :, :]
            # Apply pooling
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
    
    return np.array(activations), np.array(labels)


def load_probe(probe_path, input_dim):
    """Load a probe from disk, detecting architecture automatically."""
    device = torch.device('cpu')
    state_dict = torch.load(probe_path, map_location=device)
    
    # Check for attention pooling probe
    if 'pooling.query' in state_dict or 'pooling.weight' in state_dict:
        print(f"  ✓ Found attention probe, extracting direction from weights")
        return state_dict
    
    # Sequential probe
    if 'net.0.weight' in state_dict:
        hidden_dim = state_dict['net.0.weight'].shape[0]
        probe = SequentialProbe(input_dim, hidden_dim)
        probe.load_state_dict(state_dict)
        print(f"  ✓ Loaded SequentialProbe from {os.path.basename(probe_path)}")
        return probe
    
    # LayerProbe (classifier.weight)
    if 'classifier.weight' in state_dict:
        if HAS_LAYERPROBE:
            try:
                probe = LayerProbe(input_dim=input_dim, pooling_type='mean')
                probe.load_state_dict(state_dict)
                print(f"  ✓ Loaded LayerProbe from {os.path.basename(probe_path)}")
                return probe
            except:
                pass
        print(f"  ✓ Found classifier weights, will extract direction")
        return state_dict
    
    print(f"  Warning: Unknown architecture, keys: {list(state_dict.keys())[:5]}")
    return state_dict


def get_probe_direction(probe_or_state_dict):
    """Extract the first layer weights as the probe direction."""
    if isinstance(probe_or_state_dict, dict):
        state_dict = probe_or_state_dict
    else:
        state_dict = probe_or_state_dict.state_dict()
    
    priority_keys = ['classifier.weight', 'net.0.weight', 'pooling.weight', 'pooling.query']
    
    for key in priority_keys:
        if key in state_dict:
            W = state_dict[key].cpu().numpy()
            if len(W.shape) == 2:
                u, s, vt = np.linalg.svd(W, full_matrices=False)
                return vt[0]
            elif len(W.shape) == 1:
                return W
    
    # Fallback
    for key in sorted(state_dict.keys()):
        if 'weight' in key and len(state_dict[key].shape) == 2:
            W = state_dict[key].cpu().numpy()
            u, s, vt = np.linalg.svd(W, full_matrices=False)
            return vt[0]
    
    raise ValueError(f"Could not extract direction, keys: {list(state_dict.keys())}")


# ============================================================================
# GRAM-SCHMIDT DECOMPOSITION
# ============================================================================
def decompose_combined_direction(w_C, w_R, w_I):
    """
    Decompose combined direction into:
    w_C = a * ŵ_R + b * ŵ_I_orth + r
    
    Where:
    - ŵ_R is unit vector in Roleplaying direction (e1)
    - ŵ_I_orth is component of InsiderTrading orthogonal to ŵ_R (e2)
    - r is residual orthogonal to both (INVARIANT CORE)
    
    Returns: dict with all components
    """
    # Step 1: Normalize
    w_R_unit = w_R / (np.linalg.norm(w_R) + 1e-10)
    w_I_unit = w_I / (np.linalg.norm(w_I) + 1e-10)
    
    # Step 2: Gram-Schmidt orthogonalization
    e1 = w_R_unit
    proj_I_on_R = np.dot(w_I_unit, e1) * e1
    e2_unnorm = w_I_unit - proj_I_on_R
    e2_norm = np.linalg.norm(e2_unnorm)
    
    if e2_norm < 1e-8:
        print("  Warning: Single-domain directions are nearly parallel!")
        e2 = np.zeros_like(e1)
    else:
        e2 = e2_unnorm / e2_norm
    
    # Step 3: Project w_C onto 2D subspace
    a = np.dot(w_C, e1)
    b = np.dot(w_C, e2)
    
    # Step 4: Compute residual (INVARIANT CORE)
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
def evaluate_direction_as_classifier(X, y, direction, threshold=0):
    """Evaluate a direction vector as a linear classifier."""
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    projections = X @ direction
    
    try:
        auc_pos = roc_auc_score(y, projections)
        auc_neg = roc_auc_score(y, -projections)
        
        if auc_pos >= auc_neg:
            auc = auc_pos
            preds = (projections > threshold).astype(int)
        else:
            auc = auc_neg
            preds = (projections < threshold).astype(int)
        acc = accuracy_score(y, preds)
    except:
        auc = 0.5
        acc = 0.5
    
    return {'auc': auc, 'accuracy': acc}


def evaluate_all_directions(X, y, decomposition, domain_name="OOD"):
    """Evaluate all direction components on given data."""
    results = {}
    
    directions = {
        'roleplaying_e1': decomposition['e1'],
        'insider_orth_e2': decomposition['e2'],
        'residual': decomposition['residual'],
        'combined_projection': decomposition['projection']
    }
    
    for name, direction in directions.items():
        if np.linalg.norm(direction) < 1e-8:
            results[name] = {'auc': 0.5, 'accuracy': 0.5}
            continue
        results[name] = evaluate_direction_as_classifier(X, y, direction)
    
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Invariant Core Pipeline")
    parser.add_argument('--base_data_dir', type=str, required=True,
                        help='Base data directory (e.g., /content/drive/MyDrive/.../data)')
    parser.add_argument('--model', type=str, default='meta-llama_Llama-3.2-3B-Instruct',
                        help='Model name')
    parser.add_argument('--layer', type=int, default=16, help='Layer to analyze')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'last', 'attn'], help='Pooling type')
    parser.add_argument('--num_ood_samples', type=int, default=100,
                        help='Number of OOD samples for evaluation')
    parser.add_argument('--domain_a', type=str, default='Deception-Roleplaying',
                        help='Domain A name')
    parser.add_argument('--domain_b', type=str, default='Deception-InsiderTrading',
                        help='Domain B name')
    args = parser.parse_args()
    
    print("=" * 70)
    print("INVARIANT CORE PIPELINE")
    print("Computing and saving domain-invariant deception probe")
    print("=" * 70)
    
    # ========================================================================
    # 1. SETUP PATHS
    # ========================================================================
    print("\n1. Setting up paths...")
    
    base_dir = args.base_data_dir
    
    # Probe paths
    probes_a_dir = os.path.join(base_dir, 'probes', args.model, args.domain_a, args.pooling)
    probes_b_dir = os.path.join(base_dir, 'probes_flipped', args.model, args.domain_b, args.pooling)
    probes_comb_dir = os.path.join(base_dir, 'probes_combined', args.model, 'Deception-Combined', args.pooling)
    
    # Activation paths (for OOD evaluation)
    ood_act_dir = os.path.join(base_dir, 'activations', args.model, args.domain_b, 'test')
    
    # Output paths
    invariant_probe_dir = os.path.join(base_dir, 'invariant_probes', args.model, 
                                        f'layer_{args.layer}_{args.pooling}')
    results_dir = os.path.join(base_dir, 'results', 'invariant_probe_analysis')
    
    os.makedirs(invariant_probe_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"   Probes A: {probes_a_dir}")
    print(f"   Probes B: {probes_b_dir}")
    print(f"   Probes Combined: {probes_comb_dir}")
    print(f"   OOD Activations: {ood_act_dir}")
    print(f"   Output: {invariant_probe_dir}")
    
    # ========================================================================
    # 2. LOAD PROBES
    # ========================================================================
    print("\n2. Loading probes...")
    
    probe_a_path = os.path.join(probes_a_dir, f'probe_layer_{args.layer}.pt')
    probe_b_path = os.path.join(probes_b_dir, f'probe_layer_{args.layer}.pt')
    probe_comb_path = os.path.join(probes_comb_dir, f'probe_layer_{args.layer}.pt')
    
    # Check existence
    for path, name in [(probe_a_path, 'Domain A'), (probe_b_path, 'Domain B'), 
                        (probe_comb_path, 'Combined')]:
        if not os.path.exists(path):
            print(f"   ERROR: {name} probe not found: {path}")
            return 1
    
    # Determine input dimension from first probe
    state = torch.load(probe_a_path, map_location='cpu')
    for key in ['classifier.weight', 'net.0.weight', 'pooling.weight']:
        if key in state:
            input_dim = state[key].shape[-1]
            break
    else:
        input_dim = 3072  # Default for Llama-3.2-3B
    
    print(f"   Input dimension: {input_dim}")
    
    probe_a = load_probe(probe_a_path, input_dim)
    probe_b = load_probe(probe_b_path, input_dim)
    probe_comb = load_probe(probe_comb_path, input_dim)
    
    # ========================================================================
    # 3. EXTRACT DIRECTIONS
    # ========================================================================
    print("\n3. Extracting probe directions...")
    
    w_R = get_probe_direction(probe_a)  # Roleplaying
    w_I = get_probe_direction(probe_b)  # InsiderTrading
    w_C = get_probe_direction(probe_comb)  # Combined
    
    print(f"   |w_R| = {np.linalg.norm(w_R):.4f}")
    print(f"   |w_I| = {np.linalg.norm(w_I):.4f}")
    print(f"   |w_C| = {np.linalg.norm(w_C):.4f}")
    cos_RI = np.dot(w_R, w_I) / (np.linalg.norm(w_R) * np.linalg.norm(w_I) + 1e-10)
    print(f"   cos(w_R, w_I) = {cos_RI:.4f}")
    
    # ========================================================================
    # 4. GRAM-SCHMIDT DECOMPOSITION
    # ========================================================================
    print("\n4. Performing Gram-Schmidt decomposition...")
    
    decomposition = decompose_combined_direction(w_C, w_R, w_I)
    
    print(f"   w_C = {decomposition['a']:.4f} * e1 + {decomposition['b']:.4f} * e2 + r")
    print(f"   |projection| = {decomposition['projection_norm']:.4f}")
    print(f"   |residual| = {decomposition['residual_norm']:.4f}")
    residual_fraction = decomposition['residual_norm'] / (np.linalg.norm(w_C) + 1e-10)
    print(f"   Residual fraction: {residual_fraction:.2%}")
    
    # Verify orthogonality
    cos_r_e1 = np.dot(decomposition['residual'], decomposition['e1'])
    cos_r_e2 = np.dot(decomposition['residual'], decomposition['e2'])
    print(f"   Orthogonality check: cos(r, e1)={cos_r_e1:.6f}, cos(r, e2)={cos_r_e2:.6f}")
    
    # ========================================================================
    # 5. EVALUATE ON OOD DATA
    # ========================================================================
    print("\n5. Evaluating on OOD data...")
    
    ood_results = {}
    if os.path.exists(ood_act_dir):
        try:
            X_ood, y_ood = load_activations(ood_act_dir, args.layer, args.pooling, 
                                            args.num_ood_samples)
            print(f"   Loaded {len(X_ood)} OOD samples")
            
            # Normalize
            mean, std = X_ood.mean(0), X_ood.std(0) + 1e-8
            X_ood_norm = (X_ood - mean) / std
            
            ood_results = evaluate_all_directions(X_ood_norm, y_ood, decomposition, "OOD")
            
            print("\n   OOD Results:")
            for name, res in ood_results.items():
                print(f"     {name}: AUC={res['auc']:.4f}, Acc={res['accuracy']:.4f}")
        except Exception as e:
            print(f"   Warning: Could not load OOD data: {e}")
    else:
        print(f"   Warning: OOD activation directory not found: {ood_act_dir}")
    
    # ========================================================================
    # 6. SAVE INVARIANT CORE PROBE
    # ========================================================================
    print("\n6. Saving invariant core probe...")
    
    residual = decomposition['residual']
    
    # Normalize residual for the probe
    residual_normalized = residual / (np.linalg.norm(residual) + 1e-10)
    
    # Create probe state dict (compatible with LayerProbe format)
    probe_state = {
        'classifier.weight': torch.from_numpy(residual_normalized.reshape(1, -1)).float(),
        'classifier.bias': torch.zeros(1)
    }
    
    # Save probe
    probe_path = os.path.join(invariant_probe_dir, 'invariant_core_probe.pt')
    torch.save(probe_state, probe_path)
    print(f"   ✓ Saved: {probe_path}")
    
    # Save raw directions
    np.save(os.path.join(invariant_probe_dir, 'residual_direction.npy'), residual)
    np.save(os.path.join(invariant_probe_dir, 'e1_roleplaying.npy'), decomposition['e1'])
    np.save(os.path.join(invariant_probe_dir, 'e2_insider_orth.npy'), decomposition['e2'])
    print(f"   ✓ Saved direction vectors (.npy)")
    
    # ========================================================================
    # 7. SAVE ANALYSIS RESULTS
    # ========================================================================
    print("\n7. Saving analysis results...")
    
    summary = {
        'config': {
            'model': args.model,
            'layer': args.layer,
            'pooling': args.pooling,
            'domain_a': args.domain_a,
            'domain_b': args.domain_b,
            'num_ood_samples': args.num_ood_samples
        },
        'direction_norms': {
            'w_R': float(np.linalg.norm(w_R)),
            'w_I': float(np.linalg.norm(w_I)),
            'w_C': float(np.linalg.norm(w_C))
        },
        'cosine_similarity': {
            'w_R_w_I': float(cos_RI)
        },
        'decomposition': {
            'a': float(decomposition['a']),
            'b': float(decomposition['b']),
            'residual_norm': float(decomposition['residual_norm']),
            'projection_norm': float(decomposition['projection_norm']),
            'residual_fraction': float(residual_fraction)
        },
        'orthogonality_check': {
            'cos_residual_e1': float(cos_r_e1),
            'cos_residual_e2': float(cos_r_e2)
        },
        'ood_evaluation': {
            name: {'auc': float(res['auc']), 'accuracy': float(res['accuracy'])}
            for name, res in ood_results.items()
        } if ood_results else {},
        'output_paths': {
            'invariant_probe': probe_path,
            'residual_direction': os.path.join(invariant_probe_dir, 'residual_direction.npy'),
            'e1_roleplaying': os.path.join(invariant_probe_dir, 'e1_roleplaying.npy'),
            'e2_insider_orth': os.path.join(invariant_probe_dir, 'e2_insider_orth.npy')
        }
    }
    
    summary_path = os.path.join(results_dir, 'decomposition_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Saved: {summary_path}")
    
    # ========================================================================
    # 8. SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    print(f"\nInvariant Core Probe saved to:")
    print(f"  {probe_path}")
    
    if ood_results:
        residual_auc = ood_results.get('residual', {}).get('auc', 0)
        print(f"\nResidual OOD AUC: {residual_auc:.4f}")
        
        if residual_auc > 0.7:
            print("✅ The invariant core shows strong OOD generalization!")
        elif residual_auc > 0.6:
            print("📊 The invariant core shows moderate OOD generalization.")
        else:
            print("⚠️  The invariant core may not capture domain-invariant features.")
    
    print("\nNext steps:")
    print("  1. Run semantic analysis:")
    print(f"     python scripts/analysis/analyze_invariant_core_semantics.py \\")
    print(f"         --base_data_dir {args.base_data_dir} \\")
    print(f"         --invariant_probe_dir {invariant_probe_dir} \\")
    print(f"         --output_dir {results_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
