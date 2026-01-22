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

        # Count samples (keys ending in _activations)
        activation_keys = [k for k in shard_data.keys() if k.endswith("_activations")]
        label_keys = [k for k in shard_data.keys() if k.endswith("_label")]

        print(f"Samples in shard:")
        print(f"   Activation keys: {len(activation_keys)}")
        print(f"   Label keys: {len(label_keys)}")
        print()

        if len(activation_keys) == 0:
            print("❌ No activation keys found!")
            print("   Expected keys ending in '_activations'")
            print()
            print("Actual key format:")
            if shard_data.keys():
                first_key = list(shard_data.keys())[0]
                print(f"   Example: {first_key}")
            return

        if len(label_keys) == 0:
            print("❌ No label keys found!")
            print("   Expected keys ending in '_label'")
            return

        # Check if all activations have corresponding labels
        missing_labels = []
        for act_key in activation_keys:
            label_key = act_key.replace("_activations", "_label")
            if label_key not in shard_data:
                missing_labels.append(act_key)

        if missing_labels:
            print(f"⚠️  {len(missing_labels)} activation keys missing labels:")
            for key in missing_labels[:5]:
                print(f"   {key}")
            if len(missing_labels) > 5:
                print(f"   ... and {len(missing_labels) - 5} more")
            print()
        else:
            print("✓ All activations have corresponding labels")
            print()

        # Check activation shapes
        first_act_key = activation_keys[0]
        first_act = shard_data[first_act_key]
        print(f"Sample activation shape: {first_act.shape}")
        print(f"   Expected: (num_layers, num_tokens, hidden_dim)")
        print(f"   Got: {first_act.shape}")
        print()

        # Count total samples across all shards
        print("Counting total samples across all shards...")
        total_samples = 0
        for shard_path in shards:
            shard_data = load_file(shard_path)
            shard_activations = [k for k in shard_data.keys() if k.endswith("_activations")]
            shard_labels = [k for k in shard_data.keys() if k.endswith("_label")]

            # Count valid samples (both activation and label)
            valid_samples = 0
            for act_key in shard_activations:
                label_key = act_key.replace("_activations", "_label")
                if label_key in shard_data:
                    valid_samples += 1

            total_samples += valid_samples

        print(f"✓ Total valid samples: {total_samples}")
        print()

        if total_samples == 0:
            print("❌ No valid samples found!")
            print("   Each sample needs both:")
            print("   - <id>_activations tensor")
            print("   - <id>_label tensor")
        else:
            print("✅ Activations directory is valid and ready for evaluation!")

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
