#!/usr/bin/env python3
"""
Find Examples Uniquely Captured by Residual Direction
======================================================

The residual r from invariant core analysis is orthogonal to both domain-specific
directions but still generalizes well. This script finds examples where:

- High projection onto residual r
- Low projection onto e1 (Roleplaying direction)
- Low projection onto e2 (InsiderTrading direction)

These are the "pure invariant core" examples - deceptive in a way that neither
domain-specific probe captures, but the residual does.

Use these with Activation Oracles to understand WHAT the invariant core detects.

Usage:
    python scripts/analysis/find_uniquely_residual_examples.py \
        --probes_a data/probes/.../attn \
        --probes_b data/probes_flipped/.../attn \
        --probes_combined data/probes_combined/.../attn \
        --activations_a data/activations/.../validation \
        --activations_b data/activations/.../validation \
        --layer 20 \
        --output_dir results/uniquely_residual
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
from safetensors.torch import load_file, save_file


def load_probe_direction(probe_path):
    """Load probe and extract its direction vector."""
    state_dict = torch.load(probe_path, map_location='cpu')
    for key in ['classifier.weight', 'net.0.weight', 'pooling.weight']:
        if key in state_dict:
            W = state_dict[key].cpu().numpy()
            if len(W.shape) == 2:
                u, s, vt = np.linalg.svd(W, full_matrices=False)
                return vt[0]
            return W
    raise ValueError(f"Could not extract direction from {probe_path}")


def load_activations_with_text(act_dir, layer, pooling='mean'):
    """Load activations with text."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]

    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))

    samples = []
    for entry in manifest:
        eid = entry['id']
        if eid not in all_tensors:
            continue

        tensor = all_tensors[eid]
        x_layer = tensor[layer, :, :]

        if pooling == 'mean':
            pooled = x_layer.mean(dim=0).numpy()
        elif pooling == 'last':
            pooled = x_layer[-1, :].numpy()
        else:
            pooled = x_layer.mean(dim=0).numpy()

        samples.append({
            'id': eid,
            'activation': pooled,
            'raw_activation': x_layer.numpy(),
            'label': entry.get('label', -1),
            'text': entry.get('text', entry.get('completion', '')),
            'prompt': entry.get('prompt', ''),
            'scenario': entry.get('scenario', '')
        })

    return samples


def decompose_combined_direction(w_C, w_R, w_I):
    """Decompose combined into e1, e2, residual."""
    w_R_unit = w_R / (np.linalg.norm(w_R) + 1e-10)
    w_I_unit = w_I / (np.linalg.norm(w_I) + 1e-10)

    e1 = w_R_unit
    proj_I_on_R = np.dot(w_I_unit, e1) * e1
    e2_unnorm = w_I_unit - proj_I_on_R
    e2 = e2_unnorm / (np.linalg.norm(e2_unnorm) + 1e-10)

    a = np.dot(w_C, e1)
    b = np.dot(w_C, e2)
    projection = a * e1 + b * e2
    residual = w_C - projection

    return {
        'e1': e1,
        'e2': e2,
        'residual': residual / (np.linalg.norm(residual) + 1e-10)
    }


def compute_uniqueness_score(sample, e1, e2, residual):
    """
    Compute how "uniquely residual" an example is.

    High score = high |r projection|, low |e1| and |e2| projections
    """
    act = sample['activation']
    proj_e1 = abs(np.dot(act, e1))
    proj_e2 = abs(np.dot(act, e2))
    proj_r = abs(np.dot(act, residual))

    # Uniqueness = how much residual captures relative to domain-specific
    # Higher is more uniquely captured by residual
    domain_max = max(proj_e1, proj_e2) + 1e-10
    uniqueness = proj_r / domain_max

    return {
        'proj_e1': float(proj_e1),
        'proj_e2': float(proj_e2),
        'proj_r': float(proj_r),
        'uniqueness': float(uniqueness),
        'signed_proj_r': float(np.dot(act, residual))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probes_a', type=str, required=True)
    parser.add_argument('--probes_b', type=str, required=True)
    parser.add_argument('--probes_combined', type=str, required=True)
    parser.add_argument('--activations_a', type=str, required=True)
    parser.add_argument('--activations_b', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='results/uniquely_residual')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("FINDING EXAMPLES UNIQUELY CAPTURED BY RESIDUAL")
    print("=" * 70)

    # Load directions
    print("\n1. Loading probe directions...")
    w_R = load_probe_direction(os.path.join(args.probes_a, f'probe_layer_{args.layer}.pt'))
    w_I = load_probe_direction(os.path.join(args.probes_b, f'probe_layer_{args.layer}.pt'))
    w_C = load_probe_direction(os.path.join(args.probes_combined, f'probe_layer_{args.layer}.pt'))

    decomp = decompose_combined_direction(w_C, w_R, w_I)
    e1, e2, residual = decomp['e1'], decomp['e2'], decomp['residual']

    # Load activations
    print("\n2. Loading activations...")
    samples_a = load_activations_with_text(args.activations_a, args.layer, args.pooling)
    samples_b = load_activations_with_text(args.activations_b, args.layer, args.pooling)

    for s in samples_a:
        s['domain'] = 'Roleplaying'
    for s in samples_b:
        s['domain'] = 'InsiderTrading'

    all_samples = samples_a + samples_b
    print(f"   Total samples: {len(all_samples)}")

    # Compute uniqueness scores
    print("\n3. Computing uniqueness scores...")
    for sample in all_samples:
        scores = compute_uniqueness_score(sample, e1, e2, residual)
        sample.update(scores)

    # Find uniquely residual examples (deceptive)
    deceptive = [s for s in all_samples if s['label'] == 1]
    honest = [s for s in all_samples if s['label'] == 0]

    print(f"   Deceptive: {len(deceptive)}, Honest: {len(honest)}")

    # Sort by uniqueness
    deceptive_unique = sorted(deceptive, key=lambda x: x['uniqueness'], reverse=True)
    honest_unique = sorted(honest, key=lambda x: x['uniqueness'], reverse=True)

    # Also find examples where residual is HIGH but domain-specific is LOW
    # These are truly "orthogonal" examples
    def orthogonal_score(s):
        # High r, low e1 AND e2
        return s['proj_r'] - 0.5 * (s['proj_e1'] + s['proj_e2'])

    deceptive_orthogonal = sorted(deceptive, key=orthogonal_score, reverse=True)

    print("\n" + "=" * 70)
    print("UNIQUELY RESIDUAL DECEPTIVE EXAMPLES")
    print("(High residual projection, low domain-specific projections)")
    print("=" * 70)

    results = {'uniquely_residual_deceptive': [], 'uniquely_residual_honest': []}

    print("\nTop uniquely-residual DECEPTIVE examples:")
    print("-" * 70)
    for i, s in enumerate(deceptive_unique[:args.top_k]):
        print(f"\n[{i+1}] Domain: {s['domain']}, Uniqueness: {s['uniqueness']:.2f}")
        print(f"    proj_r={s['proj_r']:.3f}, proj_e1={s['proj_e1']:.3f}, proj_e2={s['proj_e2']:.3f}")
        print(f"    Text: {s['text'][:200]}...")

        results['uniquely_residual_deceptive'].append({
            'rank': i + 1,
            'id': s['id'],
            'domain': s['domain'],
            'uniqueness': s['uniqueness'],
            'proj_r': s['proj_r'],
            'proj_e1': s['proj_e1'],
            'proj_e2': s['proj_e2'],
            'text': s['text'][:500],
            'prompt': s['prompt'][:300] if s['prompt'] else ''
        })

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Analyze what's common in uniquely-residual examples
    domain_counts = {'Roleplaying': 0, 'InsiderTrading': 0}
    for s in deceptive_unique[:args.top_k]:
        domain_counts[s['domain']] += 1

    print(f"\nDomain distribution in top-{args.top_k} uniquely-residual deceptive:")
    print(f"  Roleplaying: {domain_counts['Roleplaying']}")
    print(f"  InsiderTrading: {domain_counts['InsiderTrading']}")

    if domain_counts['Roleplaying'] > 0.7 * args.top_k:
        print("\n  -> Residual might capture Roleplaying-specific deception missed by e1")
    elif domain_counts['InsiderTrading'] > 0.7 * args.top_k:
        print("\n  -> Residual might capture InsiderTrading-specific deception missed by e2")
    else:
        print("\n  -> Residual captures cross-domain deception patterns!")
        print("     This supports the INVARIANT CORE hypothesis.")

    # Compute average projections for uniquely-residual vs typical
    print("\n" + "=" * 70)
    print("PROJECTION STATISTICS")
    print("=" * 70)

    unique_top = deceptive_unique[:args.top_k]
    typical = deceptive_unique[args.top_k:]

    if len(typical) > 0:
        print(f"\nUniquely-residual (top {args.top_k}):")
        print(f"  Mean proj_r:  {np.mean([s['proj_r'] for s in unique_top]):.4f}")
        print(f"  Mean proj_e1: {np.mean([s['proj_e1'] for s in unique_top]):.4f}")
        print(f"  Mean proj_e2: {np.mean([s['proj_e2'] for s in unique_top]):.4f}")

        print(f"\nTypical deceptive (rest):")
        print(f"  Mean proj_r:  {np.mean([s['proj_r'] for s in typical]):.4f}")
        print(f"  Mean proj_e1: {np.mean([s['proj_e1'] for s in typical]):.4f}")
        print(f"  Mean proj_e2: {np.mean([s['proj_e2'] for s in typical]):.4f}")

    # Save results
    print("\n4. Saving results...")

    output_path = os.path.join(args.output_dir, 'uniquely_residual_examples.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: {output_path}")

    # Save activations for AO
    ao_dir = os.path.join(args.output_dir, 'activations_for_ao')
    os.makedirs(ao_dir, exist_ok=True)

    for i, s in enumerate(deceptive_unique[:args.top_k]):
        act_path = os.path.join(ao_dir, f'unique_{i}_{s["domain"]}_{s["id"]}.safetensors')
        save_file({'activations': torch.tensor(s['raw_activation'])}, act_path)

    print(f"   Saved activations to {ao_dir}")

    # AO query suggestions
    print("\n" + "=" * 70)
    print("ACTIVATION ORACLE QUERIES FOR THESE EXAMPLES")
    print("=" * 70)
    print("""
These examples are UNIQUELY captured by the residual direction - domain-specific
probes miss them. Ask the AO:

1. "Is this response deceptive? Why or why not?"
   - If AO says yes, the residual captures real deception
   - If AO says no, the residual might capture a false positive pattern

2. "What makes this response unusual?"
   - Look for commonalities across uniquely-residual examples

3. "Is the model being evasive, misleading, or dishonest?"
   - These might be different aspects of deception

4. "What is the model's goal in this response?"
   - Uniquely-residual examples might have a specific deceptive intent

KEY INSIGHT:
If these examples share a common pattern that AO identifies, that pattern
is what the "invariant core" is detecting - the domain-invariant deception signal.
""")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
