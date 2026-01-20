"""
Validate cached deception detection data quality.

Checks:
    1. Data balance (honest vs deceptive ratio)
    2. Activation tensor shapes
    3. Manifest completeness
    4. Sample examples for manual inspection
    5. Basic keyword analysis

Usage:
    python scripts/validate_deception_data.py \
        --activations_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train
"""

import argparse
import json
import os
from collections import Counter
from safetensors.torch import load_file
import torch

def load_manifest(manifest_path):
    """Load manifest.jsonl into list of dicts"""
    examples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def validate_data_quality(activations_dir):
    """Run all validation checks"""

    print("="*70)
    print("DECEPTION DETECTION DATA VALIDATION")
    print("="*70)
    print(f"Directory: {activations_dir}\n")

    # ========================================================================
    # 1. Load manifest
    # ========================================================================

    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")
        return False

    examples = load_manifest(manifest_path)
    print(f"✓ Loaded manifest: {len(examples)} examples\n")

    # ========================================================================
    # 2. Data balance check
    # ========================================================================

    print("-"*70)
    print("DATA BALANCE")
    print("-"*70)

    labels = [ex['label'] for ex in examples]
    label_counts = Counter(labels)

    honest_count = label_counts.get(0, 0)
    deceptive_count = label_counts.get(1, 0)
    unknown_count = label_counts.get(-1, 0)

    total = len(examples)

    print(f"Total examples: {total}")
    print(f"  • Honest (0):     {honest_count:4d} ({100*honest_count/total:5.1f}%)")
    print(f"  • Deceptive (1):  {deceptive_count:4d} ({100*deceptive_count/total:5.1f}%)")
    print(f"  • Unknown (-1):   {unknown_count:4d} ({100*unknown_count/total:5.1f}%)")

    # Check balance
    balance_ratio = honest_count / max(deceptive_count, 1)
    print(f"\nHonest/Deceptive ratio: {balance_ratio:.2f}")

    if 0.4 <= balance_ratio <= 2.5:
        print("✓ Dataset is reasonably balanced")
    else:
        print(f"⚠️  Dataset is imbalanced! Ratio should be between 0.4-2.5")

    if unknown_count > 0.05 * total:
        print(f"⚠️  High proportion of unknown labels: {100*unknown_count/total:.1f}%")
    elif unknown_count > 0:
        print(f"⚠️  Some unknown labels present: {unknown_count}")
    else:
        print("✓ No unknown labels")

    # ========================================================================
    # 3. Activation tensor validation
    # ========================================================================

    print("\n" + "-"*70)
    print("ACTIVATION TENSORS")
    print("-"*70)

    # Find all shard files
    shard_files = sorted([
        f for f in os.listdir(activations_dir)
        if f.startswith('shard_') and f.endswith('.safetensors')
    ])

    if not shard_files:
        print(f"❌ No shard files found in {activations_dir}")
        return False

    print(f"Found {len(shard_files)} shard file(s)")

    # Load first shard to check shapes
    first_shard_path = os.path.join(activations_dir, shard_files[0])
    tensors = load_file(first_shard_path)

    print(f"\nChecking {shard_files[0]}:")
    print(f"  • Number of tensors: {len(tensors)}")

    # Get first tensor shape
    first_tensor = next(iter(tensors.values()))
    expected_shape = first_tensor.shape
    print(f"  • Tensor shape: {tuple(expected_shape)}")
    print(f"  • Dtype: {first_tensor.dtype}")

    # Validate shape is (L_prime, T_prime, D)
    if len(expected_shape) != 3:
        print(f"❌ Expected 3D tensors (L, T, D), got {len(expected_shape)}D")
        return False

    L, T, D = expected_shape
    print(f"  • Layers (L): {L}")
    print(f"  • Tokens (T): {T}")
    print(f"  • Dimensions (D): {D}")

    # Common model dimensions
    model_dims = {
        2048: "1B model",
        3072: "3B model",
        4096: "7B model"
    }
    if D in model_dims:
        print(f"  ✓ Recognized as {model_dims[D]}")

    # Check all tensors have same shape
    print(f"\nValidating all tensors...")
    shape_issues = 0
    for tensor_id, tensor in tensors.items():
        if tensor.shape != expected_shape:
            print(f"⚠️  Shape mismatch in {tensor_id}: {tensor.shape}")
            shape_issues += 1

    if shape_issues == 0:
        print(f"✓ All {len(tensors)} tensors have consistent shape")
    else:
        print(f"❌ Found {shape_issues} tensors with wrong shape")

    # ========================================================================
    # 4. Generation length statistics
    # ========================================================================

    print("\n" + "-"*70)
    print("GENERATION STATISTICS")
    print("-"*70)

    gen_lengths = [ex['generation_length'] for ex in examples]

    print(f"Generation lengths:")
    print(f"  • Min: {min(gen_lengths)}")
    print(f"  • Max: {max(gen_lengths)}")
    print(f"  • Mean: {sum(gen_lengths)/len(gen_lengths):.1f}")
    print(f"  • Median: {sorted(gen_lengths)[len(gen_lengths)//2]}")

    # Check for very short generations
    very_short = sum(1 for x in gen_lengths if x < 10)
    if very_short > 0:
        print(f"⚠️  {very_short} examples have < 10 tokens generated")

    # ========================================================================
    # 5. Keyword analysis (simple deception markers)
    # ========================================================================

    print("\n" + "-"*70)
    print("KEYWORD ANALYSIS")
    print("-"*70)

    # Common deception markers
    deceptive_keywords = [
        'emergency', 'sick', 'hospital', 'family', 'accident',
        'forgot', 'didn\'t', 'wasn\'t', 'couldn\'t'
    ]

    # Common honesty markers
    honest_keywords = [
        'honestly', 'truth', 'actually', 'overslept', 'procrastinated',
        'my fault', 'apologize', 'sorry'
    ]

    # Count keyword occurrences by label
    deceptive_examples = [ex for ex in examples if ex['label'] == 1]
    honest_examples = [ex for ex in examples if ex['label'] == 0]

    if deceptive_examples:
        deceptive_texts = [ex['generated_text'].lower() for ex in deceptive_examples]
        dec_with_markers = sum(
            1 for text in deceptive_texts
            if any(kw in text for kw in deceptive_keywords)
        )
        print(f"Deceptive examples with deceptive keywords: "
              f"{dec_with_markers}/{len(deceptive_examples)} "
              f"({100*dec_with_markers/len(deceptive_examples):.1f}%)")

    if honest_examples:
        honest_texts = [ex['generated_text'].lower() for ex in honest_examples]
        hon_with_markers = sum(
            1 for text in honest_texts
            if any(kw in text for kw in honest_keywords)
        )
        print(f"Honest examples with honesty keywords: "
              f"{hon_with_markers}/{len(honest_examples)} "
              f"({100*hon_with_markers/len(honest_examples):.1f}%)")

    # ========================================================================
    # 6. Sample examples for manual inspection
    # ========================================================================

    print("\n" + "-"*70)
    print("SAMPLE EXAMPLES (Manual Inspection)")
    print("-"*70)

    print("\n[HONEST EXAMPLES]")
    honest_samples = [ex for ex in examples if ex['label'] == 0][:3]
    for i, ex in enumerate(honest_samples, 1):
        print(f"\n{i}. ID: {ex['id']}")
        print(f"   Scenario: {ex['scenario'][:100]}...")
        print(f"   Generated: {ex['generated_text'][:200]}")

    print("\n[DECEPTIVE EXAMPLES]")
    deceptive_samples = [ex for ex in examples if ex['label'] == 1][:3]
    for i, ex in enumerate(deceptive_samples, 1):
        print(f"\n{i}. ID: {ex['id']}")
        print(f"   Scenario: {ex['scenario'][:100]}...")
        print(f"   Generated: {ex['generated_text'][:200]}")

    # ========================================================================
    # 7. Final summary
    # ========================================================================

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    issues = []

    if balance_ratio < 0.4 or balance_ratio > 2.5:
        issues.append("Imbalanced dataset")

    if unknown_count > 0.05 * total:
        issues.append("High proportion of unknown labels")

    if shape_issues > 0:
        issues.append("Inconsistent tensor shapes")

    if very_short > 0.1 * total:
        issues.append("Many very short generations")

    if not issues:
        print("✓ ALL CHECKS PASSED")
        print("\nData quality looks good!")
        print("Ready for probe training.")
    else:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
        print("\nReview the output above and consider:")
        print("  1. Re-running with different prompts")
        print("  2. Manual labeling for unclear examples")
        print("  3. Filtering out unknown labels")

    print("="*70)

    return len(issues) == 0

def main():
    parser = argparse.ArgumentParser(
        description="Validate cached deception detection data"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Directory containing cached activations (e.g., data/activations/meta-llama_.../Deception-Roleplaying/train)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.activations_dir):
        print(f"❌ Directory not found: {args.activations_dir}")
        return 1

    success = validate_data_quality(args.activations_dir)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
