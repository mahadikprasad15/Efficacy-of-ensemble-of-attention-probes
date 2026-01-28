#!/usr/bin/env python3
"""
Sanity checks for prompted-probing activations.

Implements the three required sanity checks:
1. Suffix changes activation: Compute cosine similarity between base vs suffix final-token vectors
2. Label balance preserved: Compare label distributions before/after
3. Baseline reproducibility: Verify activations are valid

Usage:
    python scripts/analysis/sanity_check_prompted_probing.py \
        --base_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
        --prompted_dir data/prompted_activations/meta-llama_Llama-3.2-3B-Instruct/suffix_deception_yesno/Deception-Roleplaying/train \
        --sample_size 20 \
        --layer 20

Expected output:
    - Cosine similarity < 0.99 (suffix changes activation) 
    - Label balance ratio within 0.8-1.2 of original
    - Activation shapes and value ranges are valid
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
import torch
from safetensors.torch import load_file
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))


def load_activations_with_labels(act_dir: str, layer: int = 20, pooling: str = 'last'):
    """Load activations from a directory and return (activations, labels, ids)."""
    manifest_path = os.path.join(act_dir, 'manifest.jsonl')
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]
    
    shards = sorted(glob.glob(os.path.join(act_dir, 'shard_*.safetensors')))
    all_tensors = {}
    for shard_path in shards:
        all_tensors.update(load_file(shard_path))
    
    activations = []
    labels = []
    ids = []
    
    for entry in manifest:
        eid = entry['id']
        if eid not in all_tensors:
            continue
        
        tensor = all_tensors[eid]
        
        # Handle different tensor shapes
        if len(tensor.shape) == 2:
            # Prompted-probing: (L, D)
            x_layer = tensor[layer, :]  # (D,)
        elif len(tensor.shape) == 3:
            # Base: (L, T, D)
            x_layer_seq = tensor[layer, :, :]  # (T, D)
            if pooling == 'last':
                x_layer = x_layer_seq[-1, :]
            elif pooling == 'mean':
                x_layer = x_layer_seq.mean(dim=0)
            else:
                x_layer = x_layer_seq[-1, :]
        else:
            print(f"Unexpected shape for {eid}: {tensor.shape}")
            continue
        
        activations.append(x_layer.numpy())
        labels.append(entry['label'])
        ids.append(eid)
    
    return np.array(activations), np.array(labels), ids


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def check_suffix_changes_activation(base_acts, prompted_acts, base_ids, prompted_ids, sample_size=20):
    """
    Check 1: Verify suffix changes the final-token activation.
    
    For matched samples, compute cosine similarity between base (last token)
    and prompted (final token after suffix) activations.
    
    If cosine ~1.0 for everything, the suffix is not being included correctly.
    """
    print("\n" + "=" * 70)
    print("CHECK 1: SUFFIX CHANGES ACTIVATION")
    print("=" * 70)
    
    # Find common IDs
    base_id_to_idx = {id_: i for i, id_ in enumerate(base_ids)}
    prompted_id_to_idx = {id_: i for i, id_ in enumerate(prompted_ids)}
    
    common_ids = set(base_ids) & set(prompted_ids)
    print(f"Common samples: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("  ⚠️  No common IDs found between base and prompted activations!")
        print("  This may be expected if using different data splits.")
        return False
    
    # Sample subset
    sample_ids = list(common_ids)[:sample_size]
    
    similarities = []
    for id_ in sample_ids:
        base_vec = base_acts[base_id_to_idx[id_]]
        prompted_vec = prompted_acts[prompted_id_to_idx[id_]]
        
        sim = cosine_similarity(base_vec, prompted_vec)
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    print(f"\nCosine similarity (base vs prompted) on {len(sample_ids)} samples:")
    print(f"  Mean: {avg_sim:.4f}")
    print(f"  Min:  {min_sim:.4f}")
    print(f"  Max:  {max_sim:.4f}")
    
    if avg_sim > 0.99:
        print("\n  ⚠️  FAIL: Average cosine similarity > 0.99")
        print("  The suffix may not be changing activations as expected!")
        return False
    else:
        print(f"\n  ✓ PASS: Suffix is changing activations (avg cosine = {avg_sim:.4f})")
        return True


def check_label_balance(base_labels, prompted_labels):
    """
    Check 2: Verify label balance is preserved.
    """
    print("\n" + "=" * 70)
    print("CHECK 2: LABEL BALANCE PRESERVED")
    print("=" * 70)
    
    base_counts = Counter(base_labels)
    prompted_counts = Counter(prompted_labels)
    
    print(f"\nBase activations:")
    print(f"  Honest (0): {base_counts[0]}")
    print(f"  Deceptive (1): {base_counts[1]}")
    
    print(f"\nPrompted activations:")
    print(f"  Honest (0): {prompted_counts[0]}")
    print(f"  Deceptive (1): {prompted_counts[1]}")
    
    # Check ratios
    base_ratio = base_counts[0] / (base_counts[1] + 1e-8)
    prompted_ratio = prompted_counts[0] / (prompted_counts[1] + 1e-8)
    
    print(f"\nLabel ratio (honest/deceptive):")
    print(f"  Base: {base_ratio:.2f}")
    print(f"  Prompted: {prompted_ratio:.2f}")
    
    ratio_change = prompted_ratio / (base_ratio + 1e-8)
    
    if 0.8 <= ratio_change <= 1.2:
        print(f"\n  ✓ PASS: Label balance preserved (ratio change = {ratio_change:.2f})")
        return True
    else:
        print(f"\n  ⚠️  WARN: Label balance shifted significantly (ratio change = {ratio_change:.2f})")
        return False


def check_activation_validity(activations, name="activations"):
    """
    Check 3: Verify activations are valid (no NaN, reasonable range).
    """
    print("\n" + "=" * 70)
    print(f"CHECK 3: ACTIVATION VALIDITY ({name})")
    print("=" * 70)
    
    has_nan = np.isnan(activations).any()
    has_inf = np.isinf(activations).any()
    
    mean_val = np.mean(activations)
    std_val = np.std(activations)
    min_val = np.min(activations)
    max_val = np.max(activations)
    
    print(f"\nShape: {activations.shape}")
    print(f"Mean:  {mean_val:.4f}")
    print(f"Std:   {std_val:.4f}")
    print(f"Min:   {min_val:.4f}")
    print(f"Max:   {max_val:.4f}")
    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print(f"\n  ⚠️  FAIL: Invalid values detected!")
        return False
    
    if std_val < 0.001:
        print(f"\n  ⚠️  WARN: Very low variance - activations may be degenerate")
        return False
    
    print(f"\n  ✓ PASS: Activations are valid")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sanity checks for prompted-probing activations"
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Path to base activations directory (optional for comparison)"
    )
    parser.add_argument(
        "--prompted_dir",
        type=str,
        required=True,
        help="Path to prompted-probing activations directory"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer to analyze"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        choices=["mean", "max", "last"],
        help="Pooling for base activations (only used if base has T dimension)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of samples for cosine similarity check"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PROMPTED-PROBING SANITY CHECKS")
    print("=" * 70)
    print(f"Prompted dir: {args.prompted_dir}")
    if args.base_dir:
        print(f"Base dir: {args.base_dir}")
    print(f"Layer: {args.layer}")
    
    # Load prompted activations
    print("\nLoading prompted activations...")
    prompted_acts, prompted_labels, prompted_ids = load_activations_with_labels(
        args.prompted_dir, args.layer, args.pooling
    )
    print(f"  Loaded {len(prompted_ids)} samples")
    
    # Check validity of prompted activations
    valid_prompted = check_activation_validity(prompted_acts, "prompted")
    
    # Check label balance
    base_acts, base_labels, base_ids = None, None, None
    if args.base_dir and os.path.exists(args.base_dir):
        print("\nLoading base activations...")
        base_acts, base_labels, base_ids = load_activations_with_labels(
            args.base_dir, args.layer, args.pooling
        )
        print(f"  Loaded {len(base_ids)} samples")
        
        valid_base = check_activation_validity(base_acts, "base")
        balance_ok = check_label_balance(base_labels, prompted_labels)
        suffix_ok = check_suffix_changes_activation(
            base_acts, prompted_acts, base_ids, prompted_ids, args.sample_size
        )
    else:
        print("\n  ℹ️  No base activations provided, skipping comparison checks")
        balance_ok = True
        suffix_ok = True
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = valid_prompted and balance_ok and suffix_ok
    
    if all_passed:
        print("  ✓ All checks passed!")
    else:
        print("  ⚠️  Some checks failed - review output above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
