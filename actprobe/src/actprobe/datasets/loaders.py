"""
Concrete dataset implementations for ACT-ViT pipeline.
"""

import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from .base import BaseDataset
from .templates import (
    HOTPOT_NO_CONTEXT,
    HOTPOT_WITH_CONTEXT,
    TRIVIA_QA,
    MOVIES_TEMPLATE,
    IMDB_1_SHOT
)

class HotpotQADataset(BaseDataset):
    """HotpotQA (No Context) Dataset."""
    def load_data(self):
        # Using 'distractor' configuration, validation split
        ds = load_dataset("hotpot_qa", "distractor", split=self.split)
        
        self.data = []
        for item in ds:
            self.data.append({
                "prompt": HOTPOT_NO_CONTEXT.format(question=item['question']),
                "gold_answers": [item['answer']],
                "metadata": {
                    "dataset": "HotpotQA",
                    "id": item['id'],
                    "split": self.split
                }
            })
        self.apply_limit()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class HotpotQAWCDataset(BaseDataset):
    """HotpotQA (With Context) Dataset."""
    def load_data(self):
        ds = load_dataset("hotpot_qa", "distractor", split=self.split)
        
        self.data = []
        for item in ds:
            # Format context blocks
            # context is list of [title, sentences]
            formatted_context = []
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                block = f"Document: {title}\n" + " ".join(sentences)
                formatted_context.append(block)
            
            context_str = "\n\n".join(formatted_context)
            
            self.data.append({
                "prompt": HOTPOT_WITH_CONTEXT.format(
                    question=item['question'],
                    context_blocks=context_str
                ),
                "gold_answers": [item['answer']],
                "metadata": {
                    "dataset": "HotpotQA-WC",
                    "id": item['id'],
                    "split": self.split
                }
            })
        self.apply_limit()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class TriviaQADataset(BaseDataset):
    """TriviaQA (No Context) Dataset."""
    def load_data(self):
        # 'rc' configuration is standard, though we ignore context
        # using 'validation' split usually
        # Note: trivia_qa usually has 'search_results', 'entity_pages' etc in 'rc'
        # We need the 'answer' field which has 'aliases'
        ds = load_dataset("trivia_qa", "rc", split=self.split)
        
        self.data = []
        for item in ds:
            # Normalize list of answers
            # item['answer'] dict has 'value' (main answer), 'aliases' (list)
            golds = [item['answer']['value']] + item['answer']['aliases']
            # Dedup and filter
            golds = list(set([g for g in golds if g]))

            self.data.append({
                "prompt": TRIVIA_QA.format(question=item['question']),
                "gold_answers": golds,
                "metadata": {
                    "dataset": "TriviaQA",
                    "id": item['question_id'],
                    "split": self.split
                }
            })
        self.apply_limit()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class IMDBDataset(BaseDataset):
    """IMDB Sentiment Dataset (1-shot)."""
    def load_data(self):
        # Only has train/test/unsupervised. We typically use 'test' or part of 'train'.
        # Assuming 'test' for validation/OOD unless specified.
        # If split='validation' is passed (default), handle mapping to 'test' or split train.
        hf_split = 'test' if self.split == 'validation' else self.split
        ds = load_dataset("imdb", split=hf_split)
        
        # We need a fixed example for 1-shot. 
        # Ideally, we pick one balanced example. 
        # For simplicity/reproducibility, we'll hardcode one or pick first.
        # Let's hardcode a distinct positive example as the shot.
        shot_review = "This movie was fantastic! The acting was great and the plot kept me engaged."
        # shot_label = "Positive" (implicit in template)

        label_map = {0: "Negative", 1: "Positive"}

        self.data = []
        for item in ds:
            self.data.append({
                "prompt": IMDB_1_SHOT.format(
                    example_review=shot_review,
                    review=item['text']
                ),
                "gold_answers": [label_map[item['label']]],
                "metadata": {
                    "dataset": "IMDB",
                    "id": f"imdb_{random.randint(0, 1000000)}", # no ID in imdb
                    "split": hf_split
                }
            })
        self.apply_limit()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class MoviesDataset(BaseDataset):
    """Movies Dataset (Templated). Synthetic or from file."""
    def __init__(self, split="validation", limit=None, source_file=None):
         super().__init__(split, limit)
         self.source_file = source_file

    def load_data(self):
        # For now, generate synthetic if no file.
        # Format: "Who acted as {person} in the movie {movie}?"
        if not self.source_file:
             self._generate_synthetic()
        else:
             # TODO: specific loading logic if columns known
             pass
        self.apply_limit()

    def _generate_synthetic(self):
        # A few manually curated facts for demonstration
        facts = [
             ("Keanu Reeves", "The Matrix"),
             ("Elijah Wood", "The Lord of the Rings"),
             ("Leonardo DiCaprio", "Inception"),
             ("Robert Downey Jr.", "Avengers: Endgame"),
             ("Christian Bale", "The Dark Knight")
        ]
        
        # Expand slightly by repeating or assume user will provide file for real experiments
        # For the protocol, just these 5 repeated or limited is fine for checking pipeline
        
        # Correct synthetic facts (Character, Movie, Actor)
        full_facts = [
            ("Neo", "The Matrix", "Keanu Reeves"),
            ("Frodo Baggins", "The Lord of the Rings: The Fellowship of the Ring", "Elijah Wood"),
            ("Dom Cobb", "Inception", "Leonardo DiCaprio"),
            ("Tony Stark", "Avengers: Endgame", "Robert Downey Jr."),
            ("Bruce Wayne", "The Dark Knight", "Christian Bale")
        ]

        self.data = []
        for char, movie, actor in full_facts:
             self.data.append({
                 "prompt": MOVIES_TEMPLATE.format(person=char, movie=movie),
                 "gold_answers": [actor],
                 "metadata": {
                     "dataset": "Movies",
                     "id": f"movies_syn_{char}_{movie}",
                     "split": "synthetic"
                 }
             })

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
