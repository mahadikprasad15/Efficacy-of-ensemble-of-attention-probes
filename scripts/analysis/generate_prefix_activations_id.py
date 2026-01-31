#!/usr/bin/env python3
"""
Generate With-Prefix Activations for ID (Roleplaying) Dataset.

This script loads the Roleplaying dataset and caches activations
using a trained soft prefix. Designed to complete the data needed
for probe geometry analysis.

Usage:
    python scripts/analysis/generate_prefix_activations_id.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --soft_prefix_ckpt /path/to/soft_prefix/layer_agnostic_last_layer15 \
        --data_yaml data/apollo_raw/roleplaying/dataset.yaml \
        --cached_manifest data/activations/.../train/manifest.jsonl \
        --output_dir /path/to/soft_prefix_activations/Deception-Roleplaying/train \
        --split train
"""

import argparse
import os
import sys
import json
import yaml
import torch
import logging
from tqdm import tqdm
from safetensors.torch import save_file, load_file

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoftPrefixActivationExtractor:
    """Extract activations with soft prefix prepended at embedding level."""
    
    def __init__(self, model, tokenizer, soft_prefix: torch.Tensor):
        self.model = model
        self.tokenizer = tokenizer
        self.soft_prefix = soft_prefix  # [P, D]
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # Move prefix to device/dtype
        self.soft_prefix = self.soft_prefix.to(device=self.device, dtype=self.dtype)
        
        logger.info(f"SoftPrefixActivationExtractor initialized: prefix shape {soft_prefix.shape}")
    
    @torch.no_grad()
    def get_final_token_activations(self, texts: list) -> list:
        """
        Extract final token activations across all layers with soft prefix.
        
        Returns:
            List of tensors shape (L, D) for each input
        """
        # Tokenize
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(self.device)
        
        input_ids = inputs.input_ids  # [B, T]
        attention_mask = inputs.attention_mask  # [B, T]
        B, T = input_ids.shape
        P = self.soft_prefix.shape[0]
        
        # Get token embeddings
        token_embeds = self.model.model.embed_tokens(input_ids)  # [B, T, D]
        
        # Expand and prepend prefix
        prefix_embeds = self.soft_prefix.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]
        combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # [B, P+T, D]
        
        # Attention mask for prefix
        prefix_mask = torch.ones(B, P, device=self.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, P+T]
        
        # Forward pass
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True
        )
        
        # Extract hidden states (exclude embeddings)
        hs_tuple = outputs.hidden_states[1:]  # L layers
        stacked = torch.stack(hs_tuple, dim=1)  # [B, L, P+T, D]
        
        # Get final token for each sample
        results = []
        for i in range(B):
            real_len = combined_mask[i].sum().item()
            final_idx = int(real_len) - 1
            results.append(stacked[i, :, final_idx, :].cpu())  # [L, D]
        
        return results


def load_roleplaying_data_with_manifest(yaml_path: str, manifest_path: str, split: str = "train") -> list:
    """
    Load Roleplaying data by matching YAML with cached manifest.
    
    Returns list of dicts with: id, text, label
    """
    import random
    
    # Load YAML
    with open(yaml_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    
    logger.info(f"Loaded {len(raw_data)} scenarios from YAML")
    
    # Load manifest for labels and completions
    cached = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                cached[entry['id']] = entry
        logger.info(f"Loaded {len(cached)} entries from manifest")
    
    # Build items
    items = []
    for idx, item in enumerate(raw_data):
        example_id = f"roleplaying_{idx}"
        
        scenario = item.get('scenario', '')
        question = item.get('question', '')
        answer_prefix = item.get('answer_prefix', '')
        
        # Get completion and label from cache
        if example_id in cached:
            entry = cached[example_id]
            completion = entry.get('generated_text', '')
            label = entry.get('label', -1)
            
            if label == -1 or not completion:
                continue
            
            # Build text (same format as original training)
            text = f"{scenario}\n\n{question}\n{answer_prefix} {completion}"
            
            items.append({
                'id': example_id,
                'text': text,
                'label': label,
                'metadata': {
                    'scenario': scenario,
                    'question': question,
                    'completion': completion
                }
            })
    
    logger.info(f"Built {len(items)} items from matched data")
    
    # Split data (same as original)
    random.seed(42)
    random.shuffle(items)
    
    n = len(items)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    if split == "train":
        return items[:train_end]
    elif split == "validation":
        return items[train_end:val_end]
    elif split == "test":
        return items[val_end:]
    else:
        return items


def main():
    parser = argparse.ArgumentParser(description="Generate with-prefix activations for ID dataset")
    
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model name")
    parser.add_argument("--soft_prefix_ckpt", type=str, required=True,
                       help="Path to soft prefix checkpoint directory")
    parser.add_argument("--data_yaml", type=str, required=True,
                       help="Path to roleplaying dataset YAML")
    parser.add_argument("--cached_manifest", type=str, required=True,
                       help="Path to cached manifest.jsonl with completions/labels")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for activations")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace token")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("GENERATE WITH-PREFIX ACTIVATIONS (ID)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Soft prefix: {args.soft_prefix_ckpt}")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n[1/4] Loading dataset...")
    items = load_roleplaying_data_with_manifest(
        args.data_yaml, args.cached_manifest, args.split
    )
    
    if args.limit:
        items = items[:args.limit]
    
    logger.info(f"Loaded {len(items)} samples")
    
    # Count labels
    n_honest = sum(1 for x in items if x['label'] == 0)
    n_deceptive = sum(1 for x in items if x['label'] == 1)
    logger.info(f"  Honest: {n_honest}, Deceptive: {n_deceptive}")
    
    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"\n[2/4] Output: {args.output_dir}")
    
    # Load model
    logger.info(f"\n[3/4] Loading model: {args.model}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.hf_token, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.hf_token, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    
    # Load soft prefix
    prefix_path = os.path.join(args.soft_prefix_ckpt, 'prefix.safetensors')
    if not os.path.exists(prefix_path):
        raise FileNotFoundError(f"Soft prefix not found: {prefix_path}")
    
    tensors = load_file(prefix_path)
    soft_prefix = tensors['soft_prefix']  # [P, D]
    logger.info(f"Loaded soft prefix with shape {soft_prefix.shape}")
    
    # Create extractor
    extractor = SoftPrefixActivationExtractor(model, tokenizer, soft_prefix)
    
    # Extract activations
    logger.info(f"\n[4/4] Extracting activations...")
    
    SHARD_SIZE = 100
    buffer_tensors = {}
    shard_idx = 0
    manifest_entries = []
    
    pbar = tqdm(total=len(items), desc="Processing", unit="sample")
    
    for batch_start in range(0, len(items), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(items))
        batch_items = items[batch_start:batch_end]
        
        texts = [item['text'] for item in batch_items]
        
        try:
            activations = extractor.get_final_token_activations(texts)
            
            for item, tensor in zip(batch_items, activations):
                item_id = item['id']
                label = item['label']
                
                buffer_tensors[item_id] = tensor
                
                manifest_entries.append({
                    'id': item_id,
                    'label': label,
                    'shard': shard_idx,
                    'activation_shape': list(tensor.shape),
                })
            
            pbar.update(len(batch_items))
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(len(batch_items))
            continue
        
        # Save shard when buffer full
        if len(buffer_tensors) >= SHARD_SIZE or batch_end == len(items):
            if buffer_tensors:
                shard_name = f"shard_{shard_idx:03d}.safetensors"
                shard_path = os.path.join(args.output_dir, shard_name)
                save_file(buffer_tensors, shard_path)
                logger.info(f"  ✓ Saved {shard_name} with {len(buffer_tensors)} tensors")
                buffer_tensors = {}
                shard_idx += 1
    
    pbar.close()
    
    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Processed: {len(manifest_entries)} samples")
    logger.info(f"  Honest: {n_honest}")
    logger.info(f"  Deceptive: {n_deceptive}")
    logger.info(f"Shards: {shard_idx}")
    logger.info(f"Output: {args.output_dir}")
    
    if manifest_entries:
        sample_shape = manifest_entries[0]['activation_shape']
        logger.info(f"Activation shape per sample: {sample_shape} (L, D)")
    
    return 0


if __name__ == "__main__":
    exit(main())
