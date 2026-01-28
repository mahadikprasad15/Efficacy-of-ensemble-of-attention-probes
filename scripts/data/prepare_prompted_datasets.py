#!/usr/bin/env python3
"""
Prepare datasets for prompted-probing by constructing MODEL_INPUT texts.

This script creates JSONL files with PASSAGE + DELIM + SUFFIX for each sample.
The output is model-agnostic and can be reused across different models.

Pipeline:
    1. Load existing cached data (manifest with completions) OR raw dataset files
    2. Build PASSAGE for each dataset type
    3. Construct MODEL_INPUT = PASSAGE + DELIM + SUFFIX
    4. Save as JSONL files in prepared_datasets/

Usage:
    # From existing cached activations (has completions in manifest):
    python scripts/data/prepare_prompted_datasets.py \
        --source_type cached \
        --source_path data/activations/meta-llama_Llama-3.2-3B-Instruct/Deception-Roleplaying/train \
        --dataset_type roleplaying \
        --suffix_type deception_yesno \
        --output_dir data/prepared_datasets

    # From raw Apollo JSON (InsiderTrading has pre-generated responses):
    python scripts/data/prepare_prompted_datasets.py \
        --source_type raw \
        --source_path data/apollo_raw/insider_trading/llama-70b-3.3-generations.json \
        --dataset_type insider \
        --suffix_type deception_yesno \
        --split train \
        --output_dir data/prepared_datasets

Output JSONL format:
    {
        "id": "roleplaying_001",
        "passage": "{scenario}\n\n{question}\n{answer_prefix} {completion}",
        "model_input": "{passage}\n</passage>\nPlease evaluate whether...",
        "label": 1,
        "source_completion": "I had a family emergency...",
        "token_count": 247,
        "metadata": {...}
    }
"""

import argparse
import os
import sys
import json
import glob
import logging
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.datasets.templates import (
    PROMPTED_PROBING_DELIM,
    PROMPTED_PROBING_SUFFIX_DECEPTION,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Suffix configurations
# ============================================================================

SUFFIX_CONFIG = {
    "deception_yesno": PROMPTED_PROBING_SUFFIX_DECEPTION,
    # Add more suffixes here as needed:
    # "fabricated_excuse": PROMPTED_PROBING_SUFFIX_FABRICATED,
}

# ============================================================================
# PASSAGE construction functions
# ============================================================================

def build_passage_roleplaying(scenario: str, question: str, answer_prefix: str, completion: str) -> str:
    """
    Build PASSAGE for Roleplaying dataset.
    
    Format matches the original generation prompt + completion:
        {scenario}
        
        {question}
        {answer_prefix} {completion}
    """
    return f"{scenario}\n\n{question}\n{answer_prefix} {completion}"


def build_passage_insider(scenario: str, trade_message: str) -> str:
    """
    Build PASSAGE for InsiderTrading dataset.
    
    Format:
        {scenario}
        Report:
        {trade_message}
    """
    return f"{scenario}\nReport:\n{trade_message}"


def build_model_input(passage: str, suffix: str) -> str:
    """Construct full MODEL_INPUT = PASSAGE + DELIM + SUFFIX"""
    return passage + PROMPTED_PROBING_DELIM + suffix


# ============================================================================
# Source loaders
# ============================================================================

def load_from_cached_manifest(manifest_dir: str, dataset_type: str) -> List[Dict[str, Any]]:
    """
    Load data from existing cached activations manifest.
    
    Expects manifest.jsonl with 'generated_text' (completion) and metadata.
    """
    manifest_path = os.path.join(manifest_dir, "manifest.jsonl")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    logger.info(f"Loading from cached manifest: {manifest_path}")
    
    items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            items.append(entry)
    
    logger.info(f"  Loaded {len(items)} entries from manifest")
    
    # Convert to standardized format
    prepared = []
    for entry in items:
        try:
            completion = entry.get('generated_text', '')
            label = entry.get('label', -1)
            
            if label == -1:
                logger.warning(f"Skipping {entry.get('id')}: no valid label")
                continue
            
            if not completion:
                logger.warning(f"Skipping {entry.get('id')}: empty completion")
                continue
            
            # Build metadata based on dataset type
            if dataset_type == "roleplaying":
                # For roleplaying, we need scenario from metadata
                # The manifest should have 'scenario' truncated; we may need to reconstruct
                scenario = entry.get('scenario', '')
                
                # We don't have question/answer_prefix in the manifest
                # Fall back to using completion directly with scenario
                passage = build_passage_roleplaying(
                    scenario=scenario,
                    question="",  # Not available in manifest
                    answer_prefix="",
                    completion=completion
                )
            elif dataset_type == "insider":
                scenario = entry.get('scenario', '')
                passage = build_passage_insider(scenario, completion)
            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")
            
            prepared.append({
                'id': entry.get('id'),
                'completion': completion,
                'label': label,
                'passage': passage,
                'metadata': {
                    'scenario': scenario[:500],
                    'source': 'cached_manifest',
                    'original_generation_length': entry.get('generation_length'),
                }
            })
            
        except Exception as e:
            logger.warning(f"Error processing entry: {e}")
            continue
    
    logger.info(f"  Prepared {len(prepared)} valid entries")
    return prepared


def load_from_raw_insider_trading(json_path: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load directly from Apollo's raw InsiderTrading JSON.
    
    This is the preferred method for InsiderTrading since it has full data.
    """
    import random
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Raw data not found: {json_path}")
    
    logger.info(f"Loading from raw InsiderTrading JSON: {json_path}")
    
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    logger.info(f"  Loaded {len(raw_data)} rollouts")
    
    # Label mapping
    LABEL_MAP = {"yes": 1, "no": 0, "invalid": -1}
    
    all_items = []
    skipped = {"no_label": 0, "invalid": 0, "no_trade_message": 0}
    
    for idx, rollout in enumerate(raw_data):
        metadata = rollout.get("metadata", {})
        made_trade = metadata.get("made_trade")
        
        if made_trade is None:
            skipped["no_label"] += 1
            continue
        
        label = LABEL_MAP.get(made_trade.lower().strip(), -1)
        if label == -1:
            skipped["invalid"] += 1
            continue
        
        # Get scenario from transcript
        transcript = rollout.get("transcript", [])
        scenario = ""
        for msg in transcript:
            if msg.get("role") == "system":
                scenario = msg.get("content", "")
                break
        
        # Get trade_message - this is the key content
        trade_message = metadata.get("trade_message", "")
        if not trade_message:
            skipped["no_trade_message"] += 1
            continue
        
        passage = build_passage_insider(scenario, trade_message)
        
        all_items.append({
            'id': f"insider_{idx}",
            'completion': trade_message,
            'label': label,
            'passage': passage,
            'metadata': {
                'scenario': scenario[:500],
                'made_trade': made_trade,
                'source': 'raw_apollo',
            }
        })
    
    logger.info(f"  Valid items: {len(all_items)}")
    logger.info(f"  Skipped: {skipped}")
    
    # Split data
    random.seed(42)
    random.shuffle(all_items)
    
    n = len(all_items)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    if split == "train":
        return all_items[:train_end]
    elif split == "validation":
        return all_items[train_end:val_end]
    elif split == "test":
        return all_items[val_end:]
    else:
        return all_items


def load_from_raw_roleplaying(yaml_path: str, cached_manifest_path: Optional[str], split: str = "train") -> List[Dict[str, Any]]:
    """
    Load from raw Roleplaying YAML + optionally match with cached completions.
    
    If cached_manifest_path is provided, we use the generated completions from there.
    Otherwise, we'd need to generate completions (not implemented - requires model).
    """
    import yaml
    import random
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    
    logger.info(f"Loading from raw Roleplaying YAML: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    
    logger.info(f"  Loaded {len(raw_data)} scenarios")
    
    # Load cached completions if available
    cached_completions = {}
    if cached_manifest_path and os.path.exists(cached_manifest_path):
        logger.info(f"  Loading cached completions from: {cached_manifest_path}")
        with open(cached_manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                cached_completions[entry.get('id')] = entry
        logger.info(f"    Found {len(cached_completions)} cached completions")
    
    all_items = []
    for idx, item in enumerate(raw_data):
        example_id = f"roleplaying_{idx}"
        
        scenario = item.get('scenario', '')
        question = item.get('question', '')
        answer_prefix = item.get('answer_prefix', '')
        
        # Try to get completion from cache
        if example_id in cached_completions:
            cached = cached_completions[example_id]
            completion = cached.get('generated_text', '')
            label = cached.get('label', -1)
        else:
            # No cached completion - skip (would need model to generate)
            logger.debug(f"No cached completion for {example_id}, skipping")
            continue
        
        if label == -1 or not completion:
            continue
        
        passage = build_passage_roleplaying(scenario, question, answer_prefix, completion)
        
        all_items.append({
            'id': example_id,
            'completion': completion,
            'label': label,
            'passage': passage,
            'metadata': {
                'scenario': scenario[:500],
                'question': question,
                'answer_prefix': answer_prefix,
                'source': 'raw_yaml_with_cached_completion',
            }
        })
    
    # Split
    random.seed(42)
    random.shuffle(all_items)
    
    n = len(all_items)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    if split == "train":
        return all_items[:train_end]
    elif split == "validation":
        return all_items[train_end:val_end]
    elif split == "test":
        return all_items[val_end:]
    else:
        return all_items


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for prompted-probing by constructing MODEL_INPUT texts"
    )
    
    parser.add_argument(
        "--source_type",
        type=str,
        required=True,
        choices=["cached", "raw"],
        help="Source type: 'cached' for manifest from cached activations, 'raw' for original data files"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to source data (manifest directory for 'cached', or JSON/YAML file for 'raw')"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["roleplaying", "insider"],
        help="Dataset type: 'roleplaying' or 'insider'"
    )
    parser.add_argument(
        "--suffix_type",
        type=str,
        default="deception_yesno",
        choices=list(SUFFIX_CONFIG.keys()),
        help="Suffix type to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test", "all"],
        help="Data split (only used for 'raw' source_type)"
    )
    parser.add_argument(
        "--cached_manifest",
        type=str,
        default=None,
        help="Path to cached manifest with completions (for roleplaying + raw source)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/prepared_datasets",
        help="Output directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PREPARE PROMPTED DATASETS")
    logger.info("=" * 70)
    logger.info(f"Source type: {args.source_type}")
    logger.info(f"Source path: {args.source_path}")
    logger.info(f"Dataset type: {args.dataset_type}")
    logger.info(f"Suffix type: {args.suffix_type}")
    logger.info("=" * 70)
    
    # Get suffix
    suffix = SUFFIX_CONFIG[args.suffix_type]
    logger.info(f"\nSuffix preview:\n  {suffix[:80]}...")
    
    # Load data based on source type
    if args.source_type == "cached":
        items = load_from_cached_manifest(args.source_path, args.dataset_type)
    else:  # raw
        if args.dataset_type == "insider":
            items = load_from_raw_insider_trading(args.source_path, args.split)
        elif args.dataset_type == "roleplaying":
            items = load_from_raw_roleplaying(
                args.source_path,
                args.cached_manifest,
                args.split
            )
        else:
            raise ValueError(f"Unknown dataset_type: {args.dataset_type}")
    
    # Apply limit
    if args.limit:
        items = items[:args.limit]
        logger.info(f"\nLimited to {len(items)} examples")
    
    # Build model_input for each item
    logger.info(f"\nBuilding MODEL_INPUT for {len(items)} items...")
    
    prepared_items = []
    for item in items:
        model_input = build_model_input(item['passage'], suffix)
        
        prepared_items.append({
            'id': item['id'],
            'passage': item['passage'],
            'model_input': model_input,
            'label': item['label'],
            'source_completion': item['completion'],
            'char_count': len(model_input),
            'metadata': item['metadata']
        })
    
    # Create output directory
    dataset_name = "Deception-Roleplaying" if args.dataset_type == "roleplaying" else "Deception-InsiderTrading"
    output_path = os.path.join(
        args.output_dir,
        f"suffix_{args.suffix_type}",
        dataset_name,
        f"{args.split}.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    logger.info(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        for item in prepared_items:
            f.write(json.dumps(item) + '\n')
    
    # Summary
    label_counts = {"honest": 0, "deceptive": 0}
    for item in prepared_items:
        if item['label'] == 0:
            label_counts["honest"] += 1
        elif item['label'] == 1:
            label_counts["deceptive"] += 1
    
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total items: {len(prepared_items)}")
    logger.info(f"  Honest: {label_counts['honest']}")
    logger.info(f"  Deceptive: {label_counts['deceptive']}")
    logger.info(f"Output: {output_path}")
    
    # Print sample
    if prepared_items:
        logger.info(f"\nSample MODEL_INPUT (first 500 chars):")
        logger.info("-" * 50)
        logger.info(prepared_items[0]['model_input'][:500])
        logger.info("-" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())
