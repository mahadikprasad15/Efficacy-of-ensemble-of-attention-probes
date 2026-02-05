#!/usr/bin/env python3
"""
Merge two activation directories (A + B) into a combined dataset.

Expected input structure:
  src_a/{train,validation,test}/
  src_b/{train,validation,test}/

Output:
  out_dir/{train,validation,test}/

This script copies shards and merges manifests, remapping shard indices
to match the new combined shard order.
"""

import argparse
import glob
import json
import os
import re
import shutil


def list_shards(split_dir: str):
    shard_paths = glob.glob(os.path.join(split_dir, "shard_*.safetensors"))
    shards = []
    for path in shard_paths:
        m = re.search(r"shard_(\d+)\.safetensors$", os.path.basename(path))
        if not m:
            continue
        idx = int(m.group(1))
        shards.append((idx, path))
    shards.sort(key=lambda x: x[0])
    return shards


def load_manifest(split_dir: str):
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        return []
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def merge_split(src_a, src_b, out_split, force=False):
    os.makedirs(out_split, exist_ok=True)

    existing_shards = glob.glob(os.path.join(out_split, "shard_*.safetensors"))
    manifest_path = os.path.join(out_split, "manifest.jsonl")
    if not force and existing_shards and os.path.exists(manifest_path):
        print(f"⚠️  Output already exists at {out_split}, skipping (use --force to overwrite).")
        return

    # Clear output dir if force
    if force:
        for f in existing_shards:
            os.remove(f)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    # Gather shards
    shards_a = list_shards(src_a) if src_a and os.path.exists(src_a) else []
    shards_b = list_shards(src_b) if src_b and os.path.exists(src_b) else []

    shard_map_a = {}
    shard_map_b = {}
    new_idx = 0

    # Copy A shards
    for old_idx, path in shards_a:
        new_name = f"shard_{new_idx:03d}.safetensors"
        shutil.copy2(path, os.path.join(out_split, new_name))
        shard_map_a[old_idx] = new_idx
        new_idx += 1

    # Copy B shards
    for old_idx, path in shards_b:
        new_name = f"shard_{new_idx:03d}.safetensors"
        shutil.copy2(path, os.path.join(out_split, new_name))
        shard_map_b[old_idx] = new_idx
        new_idx += 1

    # Merge manifests with updated shard indices
    entries_a = load_manifest(src_a) if src_a and os.path.exists(src_a) else []
    entries_b = load_manifest(src_b) if src_b and os.path.exists(src_b) else []

    with open(manifest_path, "w") as out_f:
        for entry in entries_a:
            old = entry.get("shard", 0)
            if old in shard_map_a:
                entry["shard"] = shard_map_a[old]
            out_f.write(json.dumps(entry) + "\n")

        for entry in entries_b:
            old = entry.get("shard", 0)
            if old in shard_map_b:
                entry["shard"] = shard_map_b[old]
            out_f.write(json.dumps(entry) + "\n")

    print(f"✓ Merged split: {out_split}")


def main():
    parser = argparse.ArgumentParser(description="Merge activation dirs A + B into combined dataset")
    parser.add_argument("--src_a", type=str, required=True, help="Path to dataset A activations (base dir)")
    parser.add_argument("--src_b", type=str, required=True, help="Path to dataset B activations (base dir)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output base dir for combined activations")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    splits = ["train", "validation", "test"]
    for split in splits:
        src_a_split = os.path.join(args.src_a, split)
        src_b_split = os.path.join(args.src_b, split)
        out_split = os.path.join(args.out_dir, split)

        # Skip if neither split exists
        if not os.path.exists(src_a_split) and not os.path.exists(src_b_split):
            print(f"Skipping {split}: no input data found")
            continue

        merge_split(src_a_split, src_b_split, out_split, force=args.force)


if __name__ == "__main__":
    main()
