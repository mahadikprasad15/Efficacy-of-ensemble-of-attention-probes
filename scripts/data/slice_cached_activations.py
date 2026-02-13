#!/usr/bin/env python3
"""
Slice cached RAW activations using boundary indices and (optionally) resample.

This is intended for datasets cached with full prompt+completion activations,
where the manifest contains boundary indices:
  - system_end_idx
  - user_end_idx
  - completion_end_idx

We produce a new resampled activation directory compatible with existing
evaluation scripts (e.g., evaluate_ood_all_pooling.py).

Examples:
  # Slice completion-only and resample to (L'=28, T'=64)
  python scripts/data/slice_cached_activations.py \
    --raw_activations_dir data/activations_raw/meta-llama_Llama-3.2-3B-Instruct/Deception-AILiar/train \
    --output_dir data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-AILiar-completion/train \
    --slice_type completion
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file

# Add src to path
import sys
sys.path.append(os.path.join(os.getcwd(), "actprobe", "src"))

from actprobe.features.resample import resample_activations


def load_manifest(manifest_path: str) -> List[dict]:
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def slice_indices(entry: dict, slice_type: str, tensor_len: int) -> Optional[tuple]:
    system_end = entry.get("system_end_idx")
    user_end = entry.get("user_end_idx")
    completion_end = entry.get("completion_end_idx")

    if slice_type == "full":
        start, end = 0, completion_end or tensor_len
    elif slice_type == "completion":
        if user_end is None or completion_end is None:
            return None
        start, end = user_end, completion_end
    elif slice_type == "prompt":
        if user_end is None:
            return None
        start, end = 0, user_end
    elif slice_type == "system":
        if system_end is None:
            return None
        start, end = 0, system_end
    elif slice_type == "user":
        if system_end is None or user_end is None:
            return None
        start, end = system_end, user_end
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    # Clamp to tensor length
    start = max(0, min(start, tensor_len))
    end = max(0, min(end, tensor_len))
    if end <= start:
        return None
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description="Slice cached RAW activations and resample.")
    parser.add_argument("--raw_activations_dir", type=str, required=True,
                        help="Directory with raw activations + manifest.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for resampled sliced activations")
    parser.add_argument("--slice_type", type=str, default="completion",
                        choices=["full", "completion", "prompt", "system", "user"],
                        help="Which token span to keep")
    parser.add_argument("--L_prime", type=int, default=28,
                        help="Target number of layers for resampling")
    parser.add_argument("--T_prime", type=int, default=64,
                        help="Target number of tokens for resampling")
    parser.add_argument("--save_raw", action="store_true",
                        help="Also save sliced raw activations")
    parser.add_argument("--raw_output_dir", type=str, default=None,
                        help="Output directory for sliced raw activations (if --save_raw)")
    parser.add_argument("--shard_size", type=int, default=100,
                        help="Number of samples per output shard")
    args = parser.parse_args()

    manifest_path = os.path.join(args.raw_activations_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    entries = load_manifest(manifest_path)
    if not entries:
        raise ValueError(f"No entries found in {manifest_path}")

    # Group entries by source shard
    by_shard: Dict[int, List[dict]] = {}
    for entry in entries:
        shard = entry.get("shard")
        if shard is None:
            continue
        by_shard.setdefault(int(shard), []).append(entry)

    if not by_shard:
        raise ValueError("No shard info found in manifest entries")

    os.makedirs(args.output_dir, exist_ok=True)
    out_manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    out_manifest = open(out_manifest_path, "w")

    raw_out_dir = None
    raw_manifest = None
    if args.save_raw:
        raw_out_dir = args.raw_output_dir or (args.output_dir + "_raw")
        os.makedirs(raw_out_dir, exist_ok=True)
        raw_manifest = open(os.path.join(raw_out_dir, "manifest.jsonl"), "w")

    buffer_resampled = {}
    buffer_raw = {}
    out_shard_idx = 0
    count = 0

    for shard_idx in sorted(by_shard.keys()):
        shard_path = os.path.join(args.raw_activations_dir, f"shard_{shard_idx}.safetensors")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        shard_data = load_file(shard_path)

        for entry in by_shard[shard_idx]:
            eid = entry.get("id")
            if eid not in shard_data:
                continue
            tensor = shard_data[eid]
            # tensor shape: (L, T, D)
            T = tensor.shape[1]
            sl = slice_indices(entry, args.slice_type, T)
            if sl is None:
                continue
            start, end = sl
            sliced = tensor[:, start:end, :]

            # Resample to fixed shape
            resampled = resample_activations(sliced, target_L=args.L_prime, target_T=args.T_prime)
            if resampled is None:
                continue

            buffer_resampled[eid] = resampled
            if args.save_raw:
                buffer_raw[eid] = sliced

            # Write manifest entry
            meta = {
                "id": eid,
                "label": entry.get("label"),
                "generation_length": sliced.shape[1],
                "shard": out_shard_idx,
                "slice_type": args.slice_type,
                "slice_start": start,
                "slice_end": end,
                "source_shard": shard_idx,
            }
            out_manifest.write(json.dumps(meta) + "\n")

            if raw_manifest is not None:
                raw_manifest.write(json.dumps(meta) + "\n")

            count += 1
            if count % args.shard_size == 0:
                out_path = os.path.join(args.output_dir, f"shard_{out_shard_idx}.safetensors")
                save_file(buffer_resampled, out_path)
                buffer_resampled = {}
                if args.save_raw:
                    raw_path = os.path.join(raw_out_dir, f"shard_{out_shard_idx}.safetensors")
                    save_file(buffer_raw, raw_path)
                    buffer_raw = {}
                out_shard_idx += 1

    # Flush leftovers
    if buffer_resampled:
        out_path = os.path.join(args.output_dir, f"shard_{out_shard_idx}.safetensors")
        save_file(buffer_resampled, out_path)
        if args.save_raw:
            raw_path = os.path.join(raw_out_dir, f"shard_{out_shard_idx}.safetensors")
            save_file(buffer_raw, raw_path)

    out_manifest.close()
    if raw_manifest is not None:
        raw_manifest.close()

    print(f"Wrote {count} sliced examples to {args.output_dir}")
    if args.save_raw:
        print(f"Wrote sliced raw activations to {raw_out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
