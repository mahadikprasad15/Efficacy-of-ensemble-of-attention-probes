#!/usr/bin/env python3
"""
Diagnostic script to check activation file structure and contents.

Usage:
    python scripts/diagnose_activations.py \
        --activations_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/validation
"""

import argparse
import os
import glob
import json
from safetensors.torch import load_file
from pathlib import Path


def diagnose_activations(activations_dir: str):
    """Diagnose activation directory contents."""

    print("=" * 90)
    print("ACTIVATION DIRECTORY DIAGNOSTIC")
    print("=" * 90)
    print(f"Directory: {activations_dir}")
    print()

    # Check if directory exists
    if not os.path.exists(activations_dir):
        print(f"❌ Directory does not exist!")
        print(f"   Check your path: {activations_dir}")
        return

    print("✓ Directory exists")
    print()

    # List all files in directory
    all_files = list(Path(activations_dir).rglob("*"))
    print(f"📁 Found {len(all_files)} total files/directories:")
    for f in sorted(all_files)[:20]:  # Show first 20
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"   📁 {f.name}/")
    if len(all_files) > 20:
        print(f"   ... and {len(all_files) - 20} more")
    print()

    # Find safetensors shards
    shard_pattern = os.path.join(activations_dir, "shard_*.safetensors")
    shards = sorted(glob.glob(shard_pattern))

    if not shards:
        print("❌ No shard_*.safetensors files found!")
        print()
        print("Expected files:")
        print("   - shard_0.safetensors")
        print("   - shard_1.safetensors")
        print("   - ...")
        print()
        print("Possible issues:")
        print("   1. Wrong directory (check if train/validation/test)")
        print("   2. Activations not cached yet (run cache_deception_activations.py)")
        print("   3. Files in subdirectory (check one level deeper)")
        print()

        # Check subdirectories
        subdirs = [d for d in Path(activations_dir).iterdir() if d.is_dir()]
        if subdirs:
            print("Found subdirectories:")
            for subdir in subdirs:
                sub_shards = list(subdir.glob("shard_*.safetensors"))
                print(f"   📁 {subdir.name}/ - {len(sub_shards)} shards")
        return

    print(f"✓ Found {len(shards)} shard files:")
    for shard_path in shards:
        size_mb = os.path.getsize(shard_path) / 1024 / 1024
        print(f"   {os.path.basename(shard_path)} ({size_mb:.2f} MB)")
    print()

    # Check first shard contents
    print(f"Checking first shard: {os.path.basename(shards[0])}")
    try:
        shard_data = load_file(shards[0])
        print(f"✓ Successfully loaded shard")
        print()

        # List all keys
        print(f"Keys in shard: {len(shard_data)}")
        sample_keys = list(shard_data.keys())[:10]
        for key in sample_keys:
            tensor_shape = shard_data[key].shape
            tensor_dtype = shard_data[key].dtype
            print(f"   {key}: shape={tensor_shape}, dtype={tensor_dtype}")
        if len(shard_data) > 10:
            print(f"   ... and {len(shard_data) - 10} more keys")
        print()

        # Count samples (just IDs, not _activations/_label format)
        all_keys = list(shard_data.keys())

        print(f"Samples in shard:")
        print(f"   Total tensor keys: {len(all_keys)}")
        print()

        print("Key format analysis:")
        print(f"   Sample keys: {all_keys[:5]}")
        if len(all_keys) > 5:
            print(f"   ... and {len(all_keys) - 5} more")
        print()

        # Check if using old format (_activations, _label) or new format (just ID)
        activation_keys = [k for k in all_keys if k.endswith("_activations")]
        label_keys = [k for k in all_keys if k.endswith("_label")]

        if len(activation_keys) > 0:
            print("✓ Using OLD format (keys with _activations and _label suffixes)")
            print(f"   Activation keys: {len(activation_keys)}")
            print(f"   Label keys: {len(label_keys)}")
        else:
            print("✓ Using NEW format (keys are sample IDs, labels in manifest.jsonl)")
            print(f"   This is the correct format for deception datasets!")
            print(f"   Tensor keys: {len(all_keys)}")
        print()

        # Check activation shapes
        first_key = all_keys[0]
        first_tensor = shard_data[first_key]
        print(f"Sample activation shape: {first_tensor.shape}")
        print(f"   Expected: (num_layers, num_tokens, hidden_dim)")
        print(f"   Got: {first_tensor.shape}")
        print()

        # Count total samples across all shards
        print("Counting total samples across all shards...")
        total_samples = 0
        for shard_path in shards:
            shard_data = load_file(shard_path)
            total_samples += len(shard_data.keys())

        print(f"✓ Total tensors in shards: {total_samples}")
        print()

        # Check manifest for labels
        manifest_path = os.path.join(activations_dir, "manifest.jsonl")
        if os.path.exists(manifest_path):
            print("✓ Found manifest.jsonl")
            manifest_entries = 0
            label_counts = {"honest": 0, "deceptive": 0, "unknown": 0}

            with open(manifest_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    manifest_entries += 1
                    label = entry.get('label', -1)
                    if label == 0:
                        label_counts["honest"] += 1
                    elif label == 1:
                        label_counts["deceptive"] += 1
                    else:
                        label_counts["unknown"] += 1

            print(f"   Total entries: {manifest_entries}")
            print(f"   Honest (0): {label_counts['honest']}")
            print(f"   Deceptive (1): {label_counts['deceptive']}")
            print(f"   Unknown (-1): {label_counts['unknown']}")
            print()

            valid_samples = label_counts["honest"] + label_counts["deceptive"]
            if valid_samples == 0:
                print("❌ No valid samples with labels 0 or 1!")
                print("   All samples have unknown label (-1)")
            else:
                print(f"✅ Activations directory is valid with {valid_samples} labeled samples!")
        else:
            print("❌ No manifest.jsonl found!")
            print("   Labels are required for evaluation")
            print("   This might be an old format activation cache")

    except Exception as e:
        print(f"❌ Error loading shard: {e}")
        print()
        import traceback
        traceback.print_exc()

    print()
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Diagnose activation directory")
    parser.add_argument("--activations_dir", type=str, required=True,
                       help="Path to activations directory")
    args = parser.parse_args()

    diagnose_activations(args.activations_dir)


if __name__ == "__main__":
    main()
