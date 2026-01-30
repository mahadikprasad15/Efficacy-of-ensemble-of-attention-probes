#!/usr/bin/env python3
"""
Cache activations with soft prefix for OOD evaluation.

This script loads InsiderTrading (or other raw datasets) and caches activations
using a trained soft prefix. Output format matches existing activation caching
for compatibility with evaluation scripts.

Usage:
    python scripts/data/cache_soft_prefix_activations.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset Deception-InsiderTrading \
        --soft_prefix_ckpt checkpoints/soft_prefix/.../
        --output_dir data/soft_prefix_activations
"""

import argparse
import os
import sys
import json
import torch
import logging
from tqdm import tqdm
from safetensors.torch import save_file, load_file

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.datasets.deception_loaders import DeceptionInsiderTradingDataset, DeceptionRoleplayingDataset
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


def main():
    parser = argparse.ArgumentParser(description="Cache activations with soft prefix")
    
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Deception-InsiderTrading", "Deception-Roleplaying"],
                        help="Dataset to cache")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--soft_prefix_ckpt", type=str, required=True,
                        help="Path to soft prefix checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="data/soft_prefix_activations",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for forward passes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--data_dir", type=str, default="data/apollo_raw",
                        help="Base directory for Apollo data")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("CACHE SOFT PREFIX ACTIVATIONS")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Soft prefix: {args.soft_prefix_ckpt}")
    logger.info("=" * 70)
    
    # ========================================================================
    # Load Dataset
    # ========================================================================
    
    logger.info("\n[1/4] Loading dataset...")
    
    if args.dataset == "Deception-InsiderTrading":
        data_file = os.path.join(args.data_dir, "insider_trading/llama-70b-3.3-generations.json")
        dataset = DeceptionInsiderTradingDataset(
            split=args.split,
            limit=args.limit,
            data_file=data_file
        )
    elif args.dataset == "Deception-Roleplaying":
        data_file = os.path.join(args.data_dir, "roleplaying/dataset.yaml")
        dataset = DeceptionRoleplayingDataset(
            split=args.split,
            limit=args.limit,
            data_file=data_file
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset.load_data()
    logger.info(f"Loaded {len(dataset)} examples")
    
    # ========================================================================
    # Setup Output Directory
    # ========================================================================
    
    model_name = args.model.replace("/", "_")
    save_dir = os.path.join(
        args.output_dir,
        model_name,
        args.dataset,
        args.split
    )
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"\n[2/4] Output directory: {save_dir}")
    
    # ========================================================================
    # Load Model and Soft Prefix
    # ========================================================================
    
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
    
    # ========================================================================
    # Extract Activations
    # ========================================================================
    
    logger.info(f"\n[4/4] Extracting activations...")
    
    SHARD_SIZE = 100
    buffer_tensors = {}
    shard_idx = 0
    manifest_entries = []
    
    # Collect all items
    all_items = [dataset[i] for i in range(len(dataset))]
    
    pbar = tqdm(total=len(all_items), desc="Processing", unit="sample")
    
    for batch_start in range(0, len(all_items), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(all_items))
        batch_items = all_items[batch_start:batch_end]
        
        # Get text to process - use full_context for InsiderTrading, prompt for Roleplaying
        texts = []
        for item in batch_items:
            if "full_context" in item and item["full_context"]:
                # InsiderTrading: use full transcript
                texts.append(item["full_context"])
            elif "completion" in item and item["completion"]:
                # Fallback to completion if full_context empty
                texts.append(item["prompt"] + " " + item["completion"])
            else:
                # Roleplaying or other: use prompt
                texts.append(item["prompt"])
        
        try:
            activations = extractor.get_final_token_activations(texts)
            
            for i, (item, tensor) in enumerate(zip(batch_items, activations)):
                item_id = item["metadata"]["id"]
                label = item.get("gold_label", -1)
                
                # Store tensor
                buffer_tensors[item_id] = tensor  # [L, D]
                
                # Manifest entry
                manifest_entries.append({
                    'id': item_id,
                    'label': label,
                    'shard': shard_idx,
                    'activation_shape': list(tensor.shape),
                    'dataset': item["metadata"].get("dataset", args.dataset),
                    'made_trade': item["metadata"].get("made_trade", "")
                })
            
            pbar.update(len(batch_items))
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(len(batch_items))
            continue
        
        # Save shard when buffer full
        if len(buffer_tensors) >= SHARD_SIZE or batch_end == len(all_items):
            if buffer_tensors:
                shard_name = f"shard_{shard_idx:03d}.safetensors"
                shard_path = os.path.join(save_dir, shard_name)
                save_file(buffer_tensors, shard_path)
                logger.info(f"  ✓ Saved {shard_name} with {len(buffer_tensors)} tensors")
                buffer_tensors = {}
                shard_idx += 1
    
    pbar.close()
    
    # Save manifest
    manifest_path = os.path.join(save_dir, "manifest.jsonl")
    with open(manifest_path, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Summary
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
    
    if manifest_entries:
        sample_shape = manifest_entries[0]['activation_shape']
        logger.info(f"Activation shape per sample: {sample_shape} (L, D)")


if __name__ == "__main__":
    main()
