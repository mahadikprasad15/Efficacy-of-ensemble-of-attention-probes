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
    PROMPTED_PROBING_SUFFIX_FABRICATED,
    PROMPTED_PROBING_SUFFIX_INCONSISTENCY,
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
    "suffix_deception_yesno": PROMPTED_PROBING_SUFFIX_DECEPTION,
    "suffix_deception_fabricated": PROMPTED_PROBING_SUFFIX_FABRICATED,
    "suffix_inconsistency": PROMPTED_PROBING_SUFFIX_INCONSISTENCY,
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
    """
    Construct full MODEL_INPUT with explicit markup.
    
    Format:
        <passage>
        {passage}
        </passage>
        {suffix}
        Answer:
    
    This is explicit and doesn't rely on magic delimiter constants.
    """
    # Ensure suffix doesn't already have Answer: (avoid double-insertion)
    suffix = suffix.rstrip()
    if suffix.endswith("Answer:"):
        suffix = suffix[:-7].rstrip()  # Remove trailing Answer:
    
    return f"<passage>\n{passage}\n</passage>\n{suffix}\nAnswer:"


# ============================================================================
# Source loaders
# ============================================================================

def load_from_cached_manifest_with_yaml(
    manifest_dir: str, 
    raw_yaml_path: str, 
    dataset_type: str
) -> List[Dict[str, Any]]:
    """
    Load data by matching cached manifest (for completions/labels) with raw YAML (for full prompts).
    
    This is the **correct** approach for roleplaying data:
    - Raw YAML has: scenario, question, answer_prefix, honest_completion, deceptive_completion
    - Manifest has: generated_text (completion), label, id
    
    The passage is built faithfully from the original generation format:
        {scenario}\n\n{question}\n{answer_prefix} {completion}
    
    CRITICAL: We must NOT truncate or modify the scenario in any way.
    """
    import yaml
    
    manifest_path = os.path.join(manifest_dir, "manifest.jsonl")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    if not os.path.exists(raw_yaml_path):
        raise FileNotFoundError(
            f"Raw YAML not found: {raw_yaml_path}\n"
            "For roleplaying data, you MUST provide --raw_yaml pointing to the original dataset.yaml"
        )
    
    logger.info(f"Loading from cached manifest: {manifest_path}")
    logger.info(f"Matching with raw YAML: {raw_yaml_path}")
    
    # Load raw YAML to get full scenario/question/answer_prefix
    with open(raw_yaml_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    logger.info(f"  Loaded {len(raw_data)} scenarios from raw YAML")
    
    # Build lookup by index (manifest uses roleplaying_{idx} format)
    yaml_lookup = {}
    for idx, item in enumerate(raw_data):
        yaml_lookup[f"roleplaying_{idx}"] = {
            'scenario': item.get('scenario', ''),
            'question': item.get('question', ''),
            'answer_prefix': item.get('answer_prefix', ''),
            'honest_completion': item.get('honest_completion', ''),
            'deceptive_completion': item.get('deceptive_completion', ''),
        }
    
    # Load manifest for completions and labels
    manifest_items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            manifest_items.append(entry)
    
    logger.info(f"  Loaded {len(manifest_items)} entries from manifest")
    
    # Match and build passages
    prepared = []
    matched = 0
    skipped = {'no_match': 0, 'no_label': 0, 'no_completion': 0}
    
    for entry in manifest_items:
        try:
            example_id = entry.get('id')
            completion = entry.get('generated_text', '')
            label = entry.get('label', -1)
            
            if label == -1:
                skipped['no_label'] += 1
                continue
            
            if not completion:
                skipped['no_completion'] += 1
                continue
            
            # Look up full prompt info from YAML
            if example_id not in yaml_lookup:
                skipped['no_match'] += 1
                logger.warning(f"No YAML match for {example_id}")
                continue
            
            yaml_info = yaml_lookup[example_id]
            
            # Build passage with FULL scenario, question, answer_prefix
            # This is the faithful reconstruction of the original generation prompt
            passage = build_passage_roleplaying(
                scenario=yaml_info['scenario'],  # NO TRUNCATION!
                question=yaml_info['question'],
                answer_prefix=yaml_info['answer_prefix'],
                completion=completion
            )
            
            # Sanity check: ensure no truncation artifacts
            if '...' in yaml_info['scenario']:
                logger.warning(f"{example_id}: Scenario contains '...' - possible truncation in source")
            
            prepared.append({
                'id': example_id,
                'completion': completion,
                'label': label,
                'passage': passage,
                'metadata': {
                    'scenario': yaml_info['scenario'],  # Full scenario
                    'question': yaml_info['question'],
                    'answer_prefix': yaml_info['answer_prefix'],
                    'source': 'raw_yaml_with_cached_completion',
                    'original_generation_length': entry.get('generation_length'),
                }
            })
            matched += 1
            
        except Exception as e:
            logger.warning(f"Error processing {entry.get('id')}: {e}")
            continue
    
    logger.info(f"  Successfully matched: {matched}")
    logger.info(f"  Skipped: {skipped}")
    
    return prepared


def load_from_cached_manifest(manifest_dir: str, dataset_type: str) -> List[Dict[str, Any]]:
    """
    DEPRECATED for roleplaying data - use load_from_cached_manifest_with_yaml instead.
    
    This function is kept for backward compatibility with InsiderTrading data only,
    where the manifest contains sufficient information.
    """
    if dataset_type == "roleplaying":
        raise ValueError(
            "For roleplaying data, you MUST provide --raw_yaml pointing to the original YAML file.\n"
            "The cached manifest lacks the full scenario, question, and answer_prefix fields.\n"
            "Example: --source_type cached --source_path <manifest_dir> --raw_yaml /path/to/dataset.yaml"
        )
    
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
    
    # Convert to standardized format (InsiderTrading only)
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
            
            scenario = entry.get('scenario', '')
            passage = build_passage_insider(scenario, completion)
            
            prepared.append({
                'id': entry.get('id'),
                'completion': completion,
                'label': label,
                'passage': passage,
                'metadata': {
                    'scenario': scenario,
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
        
        # CONSISTENCY ASSERTIONS - catch data corruption early
        # Assert made_trade is exactly "yes" or "no" (not "invalid" or other)
        made_trade_normalized = made_trade.lower().strip()
        if made_trade_normalized not in {"yes", "no"}:
            skipped["invalid"] += 1
            logger.debug(f"insider_{idx}: skipping made_trade='{made_trade}'")
            continue
        
        # Optional: check trade_message is consistent with label
        has_trade_action = "execute_trade" in trade_message.lower() or "buy" in trade_message.lower() or "sell" in trade_message.lower()
        if label == 1 and not has_trade_action:
            logger.warning(f"insider_{idx}: label=deceptive but no trade action in message")
        if label == 0 and has_trade_action:
            logger.warning(f"insider_{idx}: label=honest but trade action found in message")
        
        passage = build_passage_insider(scenario, trade_message)
        
        all_items.append({
            'id': f"insider_{idx}",
            'completion': trade_message,
            'label': label,
            'passage': passage,
            'metadata': {
                'scenario': scenario,  # FULL scenario, no truncation
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
                'scenario': scenario,  # Full scenario, NO truncation
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
        "--raw_yaml",
        type=str,
        default=None,
        help="Path to raw YAML dataset file (REQUIRED for roleplaying to get full scenario/question/answer_prefix)"
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
        if args.dataset_type == "roleplaying":
            # For roleplaying, we MUST have raw_yaml to get full scenario/question/answer_prefix
            if not args.raw_yaml:
                raise ValueError(
                    "For roleplaying data, you MUST provide --raw_yaml pointing to the original YAML file.\n"
                    "The cached manifest lacks the full scenario, question, and answer_prefix fields.\n"
                    "Example: --source_type cached --source_path <manifest_dir> --raw_yaml /path/to/dataset.yaml"
                )
            items = load_from_cached_manifest_with_yaml(
                args.source_path, 
                args.raw_yaml, 
                args.dataset_type
            )
        else:
            items = load_from_cached_manifest(args.source_path, args.dataset_type)
    else:  # raw source
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
    truncation_warnings = 0
    ending_issues = 0
    tag_issues = 0
    question_missing = 0
    
    for item in items:
        model_input = build_model_input(item['passage'], suffix)
        
        # SANITY CHECK 1: Detect truncation artifacts in ACTUAL passage (not metadata)
        if '...' in item['passage']:
            truncation_warnings += 1
            if truncation_warnings <= 3:
                logger.warning(f"TRUNCATION DETECTED in {item['id']}: '...' found in passage")
        
        # SANITY CHECK 2: Validate tag structure
        if model_input.count('<passage>') != 1:
            tag_issues += 1
            if tag_issues <= 3:
                logger.error(f"{item['id']}: Wrong number of <passage> tags: {model_input.count('<passage>')}")
        if model_input.count('</passage>') != 1:
            tag_issues += 1
            if tag_issues <= 3:
                logger.error(f"{item['id']}: Wrong number of </passage> tags: {model_input.count('</passage>')}")
        
        # SANITY CHECK 3: Ensure model_input ends with Answer:
        if not model_input.strip().endswith("Answer:"):
            ending_issues += 1
            if ending_issues <= 3:
                logger.error(f"{item['id']}: model_input doesn't end with 'Answer:'")
        
        # SANITY CHECK 4: For roleplaying, ensure question AND answer_prefix are in passage
        if args.dataset_type == "roleplaying":
            question = item.get('metadata', {}).get('question', '')
            answer_prefix = item.get('metadata', {}).get('answer_prefix', '')
            if question and question not in item['passage']:
                question_missing += 1
                if question_missing <= 3:
                    logger.warning(f"{item['id']}: question line missing from passage!")
            if answer_prefix and answer_prefix not in item['passage']:
                logger.warning(f"{item['id']}: answer_prefix missing from passage!")
        
        prepared_items.append({
            'id': item['id'],
            'passage': item['passage'],
            'model_input': model_input,
            'label': item['label'],
            'source_completion': item['completion'],
            'char_count': len(model_input),
            'metadata': item['metadata']
        })
    
    # Summary of issues
    if truncation_warnings > 0:
        logger.warning(f"\n⚠️  {truncation_warnings} items have '...' in passage (possible truncation)")
    if tag_issues > 0:
        logger.error(f"\n❌ {tag_issues} items have tag structure issues")
    if ending_issues > 0:
        logger.error(f"\n❌ {ending_issues} items don't end with 'Answer:'")
    
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
