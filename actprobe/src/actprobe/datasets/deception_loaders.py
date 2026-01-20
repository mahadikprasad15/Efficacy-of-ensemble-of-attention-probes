"""
Deception dataset loaders for Apollo Research datasets.

These datasets are for detecting strategic deception (lying) rather than
factual hallucination. Labels are assigned via LLM judge after generation.
"""

import os
import yaml
import random
from typing import List, Dict, Any, Optional
from .base import BaseDataset
from .templates import DECEPTION_ROLEPLAYING

class DeceptionRoleplayingDataset(BaseDataset):
    """
    Apollo Research Roleplaying Dataset

    Dataset Structure (from YAML):
        - scenario: Context describing a deceptive situation
        - question: User's question to the role-player
        - answer_prefix: Speaker label (e.g., "Student:", "Driver:")
        - honest_completion: Reference truthful response (NOT USED for generation)
        - deceptive_completion: Reference deceptive response (NOT USED for generation)

    Pipeline:
        1. Create prompts from scenarios (neutral, no deception trigger)
        2. Generate completions using target model (e.g., Llama-3.2-3B)
        3. Label completions using LLM judge (Cerebras Llama-8B)
        4. Extract activations from generated tokens
        5. Train probes to detect deception

    Output Format:
        {
            "prompt": str,           # Formatted scenario + question + answer_prefix
            "gold_label": None,      # Will be assigned after generation + labeling
            "metadata": {
                "dataset": "Deception-Roleplaying",
                "id": str,
                "split": str,
                "scenario": str,     # Full scenario text
                "question": str,
                "honest_reference": str,    # For validation only
                "deceptive_reference": str  # For validation only
            }
        }

    Note: Unlike QA datasets, we don't have gold_answers. Instead, we have
    gold_label which is assigned after generation via LLM judge.
    """

    def __init__(self,
                 split: str = "train",
                 limit: Optional[int] = None,
                 data_file: str = "data/apollo_raw/roleplaying/dataset.yaml",
                 random_seed: int = 42):
        """
        Args:
            split: "train", "validation", or "test"
            limit: Maximum number of examples to load
            data_file: Path to downloaded YAML file
            random_seed: Seed for reproducible train/val/test splits
        """
        super().__init__(split=split, limit=limit)
        self.data_file = data_file
        self.random_seed = random_seed

        # Check if data file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please run: python scripts/download_apollo_data.py"
            )

    def load_data(self):
        """
        Load YAML and create prompts.

        Creates train/val/test splits:
        - 60% train
        - 20% validation
        - 20% test

        Splits are deterministic (based on random_seed).
        """
        # Read YAML
        with open(self.data_file, 'r') as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected list in YAML, got {type(raw_data)}")

        # Process each scenario
        all_examples = []
        for idx, item in enumerate(raw_data):
            # Validate required fields
            required_fields = ['scenario', 'question', 'answer_prefix',
                             'honest_completion', 'deceptive_completion']
            missing = [f for f in required_fields if f not in item]
            if missing:
                raise ValueError(f"Missing fields in item {idx}: {missing}")

            # Create neutral prompt (model generates without deception trigger)
            prompt = DECEPTION_ROLEPLAYING.format(
                scenario=item['scenario'],
                question=item['question'],
                answer_prefix=item['answer_prefix']
            )

            # Create example (no label yet - will be assigned after generation)
            example = {
                "prompt": prompt,
                "gold_label": None,  # Placeholder - assigned during caching
                "metadata": {
                    "dataset": "Deception-Roleplaying",
                    "id": f"roleplaying_{idx}",
                    "split": self.split,
                    "scenario": item['scenario'],
                    "question": item['question'],
                    "answer_prefix": item['answer_prefix'],
                    "honest_reference": item['honest_completion'],
                    "deceptive_reference": item['deceptive_completion']
                }
            }

            all_examples.append(example)

        # Create deterministic train/val/test splits
        random.seed(self.random_seed)
        random.shuffle(all_examples)

        n = len(all_examples)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)

        if self.split == "train":
            self.data = all_examples[:train_end]
        elif self.split == "validation":
            self.data = all_examples[train_end:val_end]
        elif self.split == "test":
            self.data = all_examples[val_end:]
        else:
            # If split is "all" or unknown, return everything
            self.data = all_examples

        # Apply limit if specified
        self.apply_limit()

        # Log summary
        print(f"Loaded {len(self.data)} examples for split '{self.split}'")
        print(f"  Total scenarios in YAML: {n}")
        print(f"  Train split: {train_end} ({100*train_end/n:.1f}%)")
        print(f"  Val split: {val_end - train_end} ({100*(val_end-train_end)/n:.1f}%)")
        print(f"  Test split: {n - val_end} ({100*(n-val_end)/n:.1f}%)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single example"""
        return self.data[idx]
