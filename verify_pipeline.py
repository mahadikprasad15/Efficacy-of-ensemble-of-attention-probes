"""
Verification script for ACT-ViT Dataset Pipeline.
Runs a small sanity check on dataset loading and scoring.
"""

import logging
import sys
import os

# Add actprobe to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.datasets.loaders import MoviesDataset, IMDBDataset
from actprobe.llm.generate import CerebrasGenerator
from actprobe.evaluation.answer_extraction import RegexExtractor
from actprobe.evaluation.scoring import ExactMatchEvaluator, ClassificationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Pipeline Verification...")

    # 1. Verify Movies Dataset (Synthetic)
    logger.info("1. Testing MoviesDataset (Synthetic)...")
    movies_ds = MoviesDataset(split="validation", limit=2)
    movies_ds.load_data()
    
    for i, item in enumerate(movies_ds):
        print(f"[{i}] Prompt: {item['prompt'][:50]}... | Gold: {item['gold_answers']}")
        
    # 2. Verify IMDB Dataset (1-shot)
    # logger.info("2. Testing IMDBDataset...")
    # try:
    #     imdb_ds = IMDBDataset(split="test", limit=2)
    #     imdb_ds.load_data()
    #     for i, item in enumerate(imdb_ds):
    #         print(f"[{i}] Prompt: {item['prompt'][:50]}... | Gold: {item['gold_answers']}")
    # except Exception as e:
    #     logger.warning(f"Skipping IMDB test (dataset might need download): {e}")

    # 3. Verify Scoring Logic (Mock)
    logger.info("3. Testing Scoring Logic...")
    scorer = ExactMatchEvaluator()
    
    # Correct case
    pred = "Keanu Reeves"
    golds = ["Keanu Reeves"]
    assert scorer.evaluate(pred, golds), "Failed Exact Match (Correct)"
    
    # Incorrect case
    pred = "Brad Pitt"
    assert not scorer.evaluate(pred, golds), "Failed Exact Match (Incorrect)"
    
    # Normalization case
    pred = "keanu reeves."
    assert scorer.evaluate(pred, golds), "Failed Exact Match (Normalization)"
    
    logger.info("Scoring logic verified.")
    
    logger.info("Verification Complete!")

if __name__ == "__main__":
    main()
