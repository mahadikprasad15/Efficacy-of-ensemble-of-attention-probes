"""
CLI script to cache activations (resampled).

Usage:
    python scripts/cache_activations.py --model meta-llama/Llama-3.2-1B-Instruct --dataset Movies --extractor regex
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

from actprobe.datasets.loaders import (
    HotpotQADataset, HotpotQAWCDataset, TriviaQADataset, MoviesDataset, IMDBDataset
)
from actprobe.llm.activations import ActivationRunner
from actprobe.features.resample import resample_activations
from actprobe.evaluation.answer_extraction import RegexExtractor, LLMExtractor
from actprobe.evaluation.scoring import ExactMatchEvaluator, ClassificationEvaluator
from actprobe.llm.generate import CerebrasGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_MAP = {
    "HotpotQA": HotpotQADataset,
    "HotpotQA-WC": HotpotQAWCDataset,
    "TriviaQA": TriviaQADataset,
    "Movies": MoviesDataset,
    "IMDB": IMDBDataset
}

LABELER_MAP = {
    "IMDB": ClassificationEvaluator,
    "default": ExactMatchEvaluator
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_MAP.keys()))
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--L_prime", type=int, default=16, help="Target layers (16 for 1B)")
    parser.add_argument("--T_prime", type=int, default=64, help="Target token bins")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="data/activations")
    
    # Labeling/Evaluator Config
    parser.add_argument("--extractor", type=str, default="regex", choices=["regex", "llm"], help="Method to extract answer from generation")
    parser.add_argument("--eval_model", type=str, default="llama3.1-70b", help="Model for LLM extraction (Cerebras)")
    parser.add_argument("--cerebras_api_key", type=str, default=None)
    
    args = parser.parse_args()

    # 1. Load Dataset
    logger.info(f"Loading {args.dataset} ({args.split})...")
    DSClass = DATASET_MAP[args.dataset]
    ds = DSClass(split=args.split, limit=args.limit)
    ds.load_data()
    logger.info(f"Loaded {len(ds)} examples.")

    # 2. Setup Model
    logger.info(f"Loading Model {args.model}...")
    runner = ActivationRunner(args.model, dtype=torch.float16)

    # 3. Prepare Pipeline Components
    if args.extractor == "llm":
        logger.info(f"Using LLM Extractor with {args.eval_model} (Cerebras)")
        try:
            gen = CerebrasGenerator(model_name=args.eval_model, api_key=args.cerebras_api_key)
            extractor = LLMExtractor(generator=gen)
        except Exception as e:
            logger.error(f"Failed to init Cerebras: {e}. Falling back to Regex.")
            extractor = RegexExtractor()
    else:
        extractor = RegexExtractor()
        
    EvaluatorClass = LABELER_MAP.get(args.dataset, LABELER_MAP["default"])
    evaluator = EvaluatorClass()
    logger.info(f"Using Evaluator: {EvaluatorClass.__name__}")

    # 4. Prepare Output
    save_dir = os.path.join(args.output_dir, args.model.replace("/", "_"), args.dataset, args.split)
    os.makedirs(save_dir, exist_ok=True)
    
    manifest_path = os.path.join(save_dir, "manifest.jsonl")
    
    # 5. Processing Loop
    logger.info("Starting processing...")
    
    manifest_file = open(manifest_path, "w")
    
    batch_prompts = []
    batch_ids = []
    batch_golds = []
    
    shard_idx = 0
    buffer_tensors = {} # id -> tensor
    
    # Simple sharding: save every N examples
    SHARD_SIZE = 100
    
    for i, item in enumerate(tqdm(ds)):
        batch_prompts.append(item['prompt'])
        batch_ids.append(item['metadata']['id'])
        batch_golds.append(item['gold_answers'])
        
        if len(batch_prompts) == args.batch_size or i == len(ds) - 1:
            # Run Batch
            try:
                # Generate and Get Activations
                completions, raw_list = runner.generate_and_get_activations(batch_prompts)
                
                # Batch Extraction if LLM? 
                # Our LLMExtractor has batch_extract
                if isinstance(extractor, LLMExtractor):
                    predicted_answers = extractor.batch_extract(completions)
                else:
                    predicted_answers = [extractor.extract(c) for c in completions]
                
                for j, raw_tensor in enumerate(raw_list):
                    # 1. Evaluation
                    completion = completions[j]
                    predicted_answer = predicted_answers[j]
                    golds = batch_golds[j]
                    is_correct = evaluator.evaluate(predicted_answer, golds)
                    label = 1 if is_correct else 0
                    
                    # 2. Resample Activations
                    # raw_tensor is (L, T, D)
                    resampled = resample_activations(raw_tensor, target_L=args.L_prime, target_T=args.T_prime)
                    # (L', T', D)
                    
                    eid = batch_ids[j]
                    buffer_tensors[eid] = resampled
                    
                    # 3. Write Manifest
                    meta = {
                        "id": eid,
                        "gold_answers": golds,
                        "generated_text": completion,
                        "predicted_answer": predicted_answer,
                        "label": label,
                        "shard": shard_idx
                    }
                    manifest_file.write(json.dumps(meta) + "\n")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_ids}: {e}")
                import traceback
                traceback.print_exc()
            
            # Flush Check
            if len(buffer_tensors) >= SHARD_SIZE or i == len(ds) - 1:
                # Save Shard
                shard_name = f"shard_{shard_idx:03d}.safetensors"
                shard_path = os.path.join(save_dir, shard_name)
                
                try:
                    save_file(buffer_tensors, shard_path)
                    logger.info(f"Saved {shard_name} with {len(buffer_tensors)} tensors.")
                except Exception as e:
                    logger.error(f"Failed to save shard: {e}")
                
                buffer_tensors = {}
                shard_idx += 1
            
            # Reset Batch
            batch_prompts = []
            batch_ids = []
            batch_golds = []

    manifest_file.close()
    logger.info("Done.")

if __name__ == "__main__":
    main()
