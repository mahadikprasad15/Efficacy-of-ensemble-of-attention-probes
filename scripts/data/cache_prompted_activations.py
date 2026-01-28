#!/usr/bin/env python3
"""
Cache activations for prompted-probing style evaluation.

This script reads prepared JSONL files (from prepare_prompted_datasets.py) and
caches the final-token activations for each sample.

Pipeline:
    1. Load prepared dataset (JSONL with model_input)
    2. Forward pass through model
    3. Extract final-token activation at each layer → shape (L, D)
    4. Save as safetensors shards + manifest.jsonl

Usage:
    python scripts/data/cache_prompted_activations.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --prepared_dataset data/prepared_datasets/suffix_deception_yesno/Deception-Roleplaying/train.jsonl \
        --output_dir data/prompted_activations \
        --batch_size 4

Output structure:
    data/prompted_activations/
        meta-llama_Llama-3.2-3B-Instruct/
            suffix_deception_yesno/
                Deception-Roleplaying/
                    train/
                        manifest.jsonl
                        shard_000.safetensors  # Contains (L, D) tensors per sample
"""

import argparse
import os
import sys
import json
import torch
import logging
from tqdm import tqdm
from safetensors.torch import save_file

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.llm.activations import ActivationRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prepared_dataset(jsonl_path: str):
    """Load prepared dataset from JSONL file."""
    items = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Cache final-token activations for prompted-probing"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model for activation extraction (e.g., meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--prepared_dataset",
        type=str,
        required=True,
        help="Path to prepared JSONL file (from prepare_prompted_datasets.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/prompted_activations",
        help="Base output directory for activations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for forward passes"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if activations already exist"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("CACHE PROMPTED-PROBING ACTIVATIONS")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Prepared dataset: {args.prepared_dataset}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 70)
    
    # ========================================================================
    # 1. Load prepared dataset
    # ========================================================================
    
    logger.info("\n[1/4] Loading prepared dataset...")
    
    if not os.path.exists(args.prepared_dataset):
        logger.error(f"Prepared dataset not found: {args.prepared_dataset}")
        return 1
    
    items = load_prepared_dataset(args.prepared_dataset)
    logger.info(f"  Loaded {len(items)} items")
    
    if args.limit:
        items = items[:args.limit]
        logger.info(f"  Limited to {len(items)} items")
    
    # Parse path to extract suffix_type, dataset_name, split
    # Expected: .../suffix_deception_yesno/Deception-Roleplaying/train.jsonl
    path_parts = args.prepared_dataset.split(os.sep)
    split = os.path.splitext(path_parts[-1])[0]  # train, validation, test
    dataset_name = path_parts[-2]  # Deception-Roleplaying
    suffix_type = path_parts[-3]  # suffix_deception_yesno
    
    logger.info(f"  Suffix: {suffix_type}")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Split: {split}")
    
    # ========================================================================
    # 2. Setup output directory
    # ========================================================================
    
    model_name_safe = args.model.replace("/", "_")
    save_dir = os.path.join(
        args.output_dir,
        model_name_safe,
        suffix_type,
        dataset_name,
        split
    )
    os.makedirs(save_dir, exist_ok=True)
    
    manifest_path = os.path.join(save_dir, "manifest.jsonl")
    
    # Check if already exists
    if not args.force and os.path.exists(manifest_path):
        import glob
        existing_shards = glob.glob(os.path.join(save_dir, "shard_*.safetensors"))
        if existing_shards:
            logger.info(f"\n⚠️  Activations already exist in {save_dir}")
            logger.info(f"   Found {len(existing_shards)} shard(s)")
            logger.info(f"   Use --force to regenerate")
            return 0
    
    logger.info(f"\n[2/4] Output directory: {save_dir}")
    
    # ========================================================================
    # 3. Load model and extract activations
    # ========================================================================
    
    logger.info(f"\n[3/4] Loading model: {args.model}...")
    runner = ActivationRunner(args.model, dtype=torch.float16, hf_token=args.hf_token)
    logger.info("  ✓ Model loaded")
    
    # Process in batches
    logger.info(f"\n[4/4] Extracting final-token activations...")
    
    SHARD_SIZE = 100
    buffer_tensors = {}
    shard_idx = 0
    manifest_entries = []
    
    pbar = tqdm(total=len(items), desc="Processing", unit="sample", ncols=100)
    
    for batch_start in range(0, len(items), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(items))
        batch_items = items[batch_start:batch_end]
        
        # Get model inputs
        model_inputs = [item['model_input'] for item in batch_items]
        
        try:
            # Extract final-token activations
            activations = runner.get_final_token_activations(model_inputs)
            
            for i, (item, tensor) in enumerate(zip(batch_items, activations)):
                item_id = item['id']
                
                # Store tensor
                buffer_tensors[item_id] = tensor  # Shape: (L, D)
                
                # Create manifest entry
                manifest_entries.append({
                    'id': item_id,
                    'label': item['label'],
                    'shard': shard_idx,
                    'activation_shape': list(tensor.shape),
                    'char_count': item.get('char_count', len(item['model_input'])),
                    'source_completion': item.get('source_completion', '')[:100],
                })
            
            pbar.update(len(batch_items))
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(len(batch_items))
            continue
        
        # Save shard when buffer is full
        if len(buffer_tensors) >= SHARD_SIZE or batch_end == len(items):
            if buffer_tensors:
                shard_name = f"shard_{shard_idx:03d}.safetensors"
                shard_path = os.path.join(save_dir, shard_name)
                
                try:
                    save_file(buffer_tensors, shard_path)
                    logger.info(f"  ✓ Saved {shard_name} with {len(buffer_tensors)} tensors")
                except Exception as e:
                    logger.error(f"Failed to save shard: {e}")
                
                buffer_tensors = {}
                shard_idx += 1
    
    pbar.close()
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    label_counts = {"honest": 0, "deceptive": 0}
    for entry in manifest_entries:
        if entry['label'] == 0:
            label_counts["honest"] += 1
        elif entry['label'] == 1:
            label_counts["deceptive"] += 1
    
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Processed: {len(manifest_entries)} samples")
    logger.info(f"  Honest: {label_counts['honest']}")
    logger.info(f"  Deceptive: {label_counts['deceptive']}")
    logger.info(f"Shards: {shard_idx}")
    logger.info(f"Output: {save_dir}")
    
    # Print sample activation shape
    if manifest_entries:
        sample_shape = manifest_entries[0]['activation_shape']
        logger.info(f"Activation shape per sample: {sample_shape} (L, D)")
    
    return 0


if __name__ == "__main__":
    exit(main())
