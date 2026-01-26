#!/usr/bin/env python3
"""
Find Max-Activating Examples for Activation Oracle Analysis
============================================================

This script finds examples that maximally activate different probe directions:
1. Domain A direction (w_R - Roleplaying)
2. Domain B direction (w_I - InsiderTrading)
3. Residual direction (r - orthogonal to both, from invariant core analysis)

The key insight: if residual r generalizes best OOD while being orthogonal to both
domain-specific directions, what does it actually capture?

Use these max-activating examples with pretrained Activation Oracles to understand
what each direction is detecting.

Output:
- max_activating_examples.json: Text + metadata for top examples per direction
- activations_for_ao/: Saved activations in format for AO input

Usage:
    python scripts/analysis/find_max_activating_for_ao.py \
        --probes_a data/probes/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/attn \
        --probes_b data/probes_flipped/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/attn \
        --probes_combined data/probes_combined/meta-llama_Llama-3.2-3B-Instruct/Deception-Combined/attn \
        --activations_a data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation \
        --activations_b data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-InsiderTrading/validation \
        --layer 20 \
        --top_k 10 \
        --output_dir results/ao_analysis
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
from safetensors.torch import load_file, save_file
from collections import defaultdict


# ============================================================================
# LOADING UTILITIES
# ============================================================================
def load_probe_direction(probe_path):
    """Load probe and extract its direction vector."""
    state_dict = torch.load(probe_path, map_location='cpu')

    # Priority order for finding weights
    for key in ['classifier.weight', 'net.0.weight', 'pooling.weight']:
        if key in state_dict:
            W = state_dict[key].cpu().numpy()
            if len(W.shape) == 2:
                # SVD to get principal direction
                u, s, vt = np.linalg.svd(W, full_matrices=False)
                return vt[0]
            else:
                return W

    raise ValueError(f"Could not extract direction from {probe_path}")


def load_activations_with_text(act_dir, layer, pooling='mean'):
    """Load activations with their associated text and metadata."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]

    # Load all shards
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))

    samples = []
    for entry in manifest:
        eid = entry['id']
        if eid not in all_tensors:
            continue

        tensor = all_tensors[eid]  # (L, T, D)
        x_layer = tensor[layer, :, :]  # (T, D)

        # Pool activations
        if pooling == 'mean':
            pooled = x_layer.mean(dim=0).numpy()
        elif pooling == 'max':
            pooled = x_layer.max(dim=0)[0].numpy()
        elif pooling == 'last':
            pooled = x_layer[-1, :].numpy()
        else:
            pooled = x_layer.mean(dim=0).numpy()

        samples.append({
            'id': eid,
            'activation': pooled,
            'raw_activation': x_layer.numpy(),  # Keep full sequence for AO
            'label': entry.get('label', -1),
            'text': entry.get('text', entry.get('completion', '')),
            'prompt': entry.get('prompt', ''),
            'scenario': entry.get('scenario', ''),
            'metadata': {k: v for k, v in entry.items()
                        if k not in ['id', 'label', 'text', 'completion', 'prompt', 'scenario']}
        })

    return samples


def decompose_combined_direction(w_C, w_R, w_I):
    """
    Decompose combined direction into:
    w_C = a * e1 + b * e2 + r

    Where e1 is Roleplaying direction, e2 is orthogonalized InsiderTrading,
    and r is the residual (invariant core).
    """
    # Normalize
    w_R_unit = w_R / (np.linalg.norm(w_R) + 1e-10)
    w_I_unit = w_I / (np.linalg.norm(w_I) + 1e-10)

    # Gram-Schmidt: e1 = w_R, e2 = orthogonal component of w_I
    e1 = w_R_unit
    proj_I_on_R = np.dot(w_I_unit, e1) * e1
    e2_unnorm = w_I_unit - proj_I_on_R
    e2_norm = np.linalg.norm(e2_unnorm)
    e2 = e2_unnorm / (e2_norm + 1e-10) if e2_norm > 1e-8 else np.zeros_like(e1)

    # Project w_C onto 2D plane
    a = np.dot(w_C, e1)
    b = np.dot(w_C, e2)
    projection = a * e1 + b * e2
    residual = w_C - projection

    return {
        'e1': e1,  # Roleplaying direction
        'e2': e2,  # Orthogonalized InsiderTrading
        'residual': residual / (np.linalg.norm(residual) + 1e-10),  # Normalized residual
        'residual_raw': residual,
        'a': a,
        'b': b,
        'cos_R_I': np.dot(w_R_unit, w_I_unit)
    }


# ============================================================================
# MAX ACTIVATION FINDING
# ============================================================================
def find_max_activating_examples(samples, direction, top_k=10, label_filter=None):
    """
    Find examples that maximally project onto the given direction.

    Returns top_k examples with highest |projection|, split by sign.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    results = []
    for sample in samples:
        if label_filter is not None and sample['label'] != label_filter:
            continue

        proj = np.dot(sample['activation'], direction)
        results.append({
            'id': sample['id'],
            'projection': float(proj),
            'abs_projection': float(abs(proj)),
            'label': sample['label'],
            'text': sample['text'],
            'prompt': sample['prompt'],
            'scenario': sample['scenario'],
            'raw_activation': sample['raw_activation']
        })

    # Sort by absolute projection
    results.sort(key=lambda x: x['abs_projection'], reverse=True)

    # Get top positive and negative
    positive = [r for r in results if r['projection'] > 0][:top_k]
    negative = [r for r in results if r['projection'] < 0][:top_k]

    return {
        'top_positive': positive,
        'top_negative': negative,
        'all_sorted': results[:top_k * 2]
    }


def compute_direction_stats(samples, direction):
    """Compute statistics for projections onto a direction."""
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    projections = []
    labels = []
    for sample in samples:
        proj = np.dot(sample['activation'], direction)
        projections.append(proj)
        labels.append(sample['label'])

    projections = np.array(projections)
    labels = np.array(labels)

    # Stats by label
    honest_projs = projections[labels == 0] if 0 in labels else []
    deceptive_projs = projections[labels == 1] if 1 in labels else []

    return {
        'mean': float(np.mean(projections)),
        'std': float(np.std(projections)),
        'min': float(np.min(projections)),
        'max': float(np.max(projections)),
        'honest_mean': float(np.mean(honest_projs)) if len(honest_projs) > 0 else None,
        'deceptive_mean': float(np.mean(deceptive_projs)) if len(deceptive_projs) > 0 else None,
        'separation': float(np.mean(deceptive_projs) - np.mean(honest_projs)) if len(honest_projs) > 0 and len(deceptive_projs) > 0 else None
    }


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================
def format_example_for_ao(example, direction_name):
    """Format an example for Activation Oracle analysis."""
    return {
        'direction': direction_name,
        'id': example['id'],
        'projection': example['projection'],
        'label': 'deceptive' if example['label'] == 1 else 'honest',
        'prompt': example['prompt'][:500] if example['prompt'] else '',
        'response': example['text'][:1000] if example['text'] else '',
        'scenario': example['scenario'][:300] if example['scenario'] else '',
        # Suggested AO queries
        'ao_queries': [
            "Is this response deceptive or honest?",
            "What is the model trying to accomplish?",
            "Is the model being evasive or straightforward?",
            "What emotional tone does this response have?",
            "Is the model hiding something?"
        ]
    }


def save_activations_for_ao(examples, output_dir, direction_name):
    """Save activations in format suitable for AO input."""
    ao_dir = os.path.join(output_dir, 'activations_for_ao', direction_name)
    os.makedirs(ao_dir, exist_ok=True)

    for i, ex in enumerate(examples):
        # Save raw activations (full sequence)
        if 'raw_activation' in ex and ex['raw_activation'] is not None:
            act_path = os.path.join(ao_dir, f'example_{i}_{ex["id"]}.safetensors')
            save_file({'activations': torch.tensor(ex['raw_activation'])}, act_path)

    print(f"  Saved {len(examples)} activations to {ao_dir}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Find max-activating examples for AO analysis")
    parser.add_argument('--probes_a', type=str, required=True, help='Domain A probes dir')
    parser.add_argument('--probes_b', type=str, required=True, help='Domain B probes dir')
    parser.add_argument('--probes_combined', type=str, required=True, help='Combined probes dir')
    parser.add_argument('--activations_a', type=str, required=True, help='Domain A activations')
    parser.add_argument('--activations_b', type=str, required=True, help='Domain B activations')
    parser.add_argument('--layer', type=int, required=True, help='Layer to analyze')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'last'])
    parser.add_argument('--top_k', type=int, default=10, help='Number of top examples per direction')
    parser.add_argument('--output_dir', type=str, default='results/ao_analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("FINDING MAX-ACTIVATING EXAMPLES FOR ACTIVATION ORACLE ANALYSIS")
    print("=" * 70)

    # ========================================================================
    # 1. LOAD PROBE DIRECTIONS
    # ========================================================================
    print("\n1. Loading probe directions...")

    probe_a_path = os.path.join(args.probes_a, f'probe_layer_{args.layer}.pt')
    probe_b_path = os.path.join(args.probes_b, f'probe_layer_{args.layer}.pt')
    probe_c_path = os.path.join(args.probes_combined, f'probe_layer_{args.layer}.pt')

    w_R = load_probe_direction(probe_a_path)
    w_I = load_probe_direction(probe_b_path)
    w_C = load_probe_direction(probe_c_path)

    print(f"   w_R (Roleplaying): dim={len(w_R)}")
    print(f"   w_I (InsiderTrading): dim={len(w_I)}")
    print(f"   w_C (Combined): dim={len(w_C)}")

    # ========================================================================
    # 2. DECOMPOSE TO GET RESIDUAL
    # ========================================================================
    print("\n2. Computing residual direction (invariant core)...")

    decomp = decompose_combined_direction(w_C, w_R, w_I)

    print(f"   cos(w_R, w_I) = {decomp['cos_R_I']:.4f}")
    print(f"   Decomposition: w_C = {decomp['a']:.4f}*e1 + {decomp['b']:.4f}*e2 + r")
    print(f"   |residual| = {np.linalg.norm(decomp['residual_raw']):.4f}")

    directions = {
        'roleplaying_e1': decomp['e1'],
        'insidertrading_e2': decomp['e2'],
        'residual_r': decomp['residual'],
        'combined_wC': w_C / (np.linalg.norm(w_C) + 1e-10)
    }

    # ========================================================================
    # 3. LOAD ACTIVATIONS
    # ========================================================================
    print("\n3. Loading activations...")

    samples_a = load_activations_with_text(args.activations_a, args.layer, args.pooling)
    samples_b = load_activations_with_text(args.activations_b, args.layer, args.pooling)
    all_samples = samples_a + samples_b

    print(f"   Domain A: {len(samples_a)} samples")
    print(f"   Domain B: {len(samples_b)} samples")
    print(f"   Total: {len(all_samples)} samples")

    # Tag samples with domain
    for s in samples_a:
        s['domain'] = 'Roleplaying'
    for s in samples_b:
        s['domain'] = 'InsiderTrading'

    # ========================================================================
    # 4. FIND MAX-ACTIVATING EXAMPLES
    # ========================================================================
    print("\n4. Finding max-activating examples...")

    results = {}

    for dir_name, direction in directions.items():
        print(f"\n   --- {dir_name} ---")

        # Stats
        stats_a = compute_direction_stats(samples_a, direction)
        stats_b = compute_direction_stats(samples_b, direction)
        stats_all = compute_direction_stats(all_samples, direction)

        print(f"   Domain A separation: {stats_a['separation']:.4f}" if stats_a['separation'] else "   Domain A: insufficient labels")
        print(f"   Domain B separation: {stats_b['separation']:.4f}" if stats_b['separation'] else "   Domain B: insufficient labels")

        # Find max-activating
        max_act = find_max_activating_examples(all_samples, direction, args.top_k)

        # Format for output
        results[dir_name] = {
            'stats': {
                'domain_a': stats_a,
                'domain_b': stats_b,
                'all': stats_all
            },
            'top_positive': [format_example_for_ao(ex, dir_name) for ex in max_act['top_positive']],
            'top_negative': [format_example_for_ao(ex, dir_name) for ex in max_act['top_negative']]
        }

        # Show top examples
        print(f"   Top positive projections:")
        for ex in max_act['top_positive'][:3]:
            domain = 'A' if any(s['id'] == ex['id'] for s in samples_a) else 'B'
            label = 'D' if ex['label'] == 1 else 'H'
            print(f"     [{domain}/{label}] proj={ex['projection']:.3f}: {ex['text'][:60]}...")

        # Save activations for AO
        save_activations_for_ao(max_act['top_positive'] + max_act['top_negative'],
                               args.output_dir, dir_name)

    # ========================================================================
    # 5. SAVE RESULTS
    # ========================================================================
    print("\n5. Saving results...")

    # Remove raw activations from JSON output (too large)
    for dir_name in results:
        for ex in results[dir_name]['top_positive'] + results[dir_name]['top_negative']:
            if 'raw_activation' in ex:
                del ex['raw_activation']

    output_path = os.path.join(args.output_dir, 'max_activating_examples.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: {output_path}")

    # Save directions
    directions_path = os.path.join(args.output_dir, 'directions.npz')
    np.savez(directions_path, **directions)
    print(f"   Saved: {directions_path}")

    # ========================================================================
    # 6. GENERATE AO QUERY SUGGESTIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUGGESTED ACTIVATION ORACLE QUERIES")
    print("=" * 70)

    print("""
For the RESIDUAL direction (invariant core), ask the AO:
1. "Is this response deceptive?"
   - Compare answers for residual vs domain-specific directions

2. "What is unusual about this response?"
   - Residual should capture domain-invariant deception patterns

3. "Is the model being evasive?"
   - The residual might capture evasiveness rather than domain-specific lies

4. "What is the model's intent?"
   - Compare intent descriptions across directions

HYPOTHESIS TO TEST:
- If residual r captures "general deceptiveness" while e1/e2 capture domain-specific patterns
- AO should describe r-activated examples with more abstract deception concepts
- While e1/e2 examples might mention domain-specific cues (roleplaying, financial)
""")

    # Save AO usage guide
    guide_path = os.path.join(args.output_dir, 'AO_USAGE_GUIDE.md')
    with open(guide_path, 'w') as f:
        f.write("""# Activation Oracle Analysis Guide

## Setup

1. Clone the AO repo:
```bash
git clone https://github.com/adamkarvonen/activation_oracles
cd activation_oracles
```

2. Download pretrained AO for Llama-3:
```python
from huggingface_hub import snapshot_download
snapshot_download("activation-oracles/llama-3-8b-ao", local_dir="./ao_model")
```

## Using AO on Your Max-Activating Examples

The activations are saved in `activations_for_ao/{direction_name}/`.

For each example:
1. Load the activation tensor
2. Inject into AO at layer 1 residual stream
3. Query with natural language

### Example Queries

For residual direction (invariant core):
- "Is this response deceptive?"
- "What is the model hiding?"
- "Is the model being evasive or direct?"

For domain-specific directions:
- "Is this a roleplay response?"
- "Is this about financial decisions?"

### What to Look For

1. **If residual captures general deception:**
   - AO should describe r-examples with abstract deception words
   - e1/e2 examples should have domain-specific descriptions

2. **If residual is noise:**
   - AO descriptions for r-examples will be incoherent
   - Low confidence in deception classification

3. **If residual is complementary signal:**
   - AO might describe different aspects of deception
   - e.g., e1 = "lying about identity", e2 = "lying about actions", r = "evasive language"

## Key Experiment

Compare AO responses for:
- Top residual-activated DECEPTIVE examples
- Top e1-activated DECEPTIVE examples
- Top e2-activated DECEPTIVE examples

Do they describe different types of deception?
""")

    print(f"\n   Saved: {guide_path}")
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
