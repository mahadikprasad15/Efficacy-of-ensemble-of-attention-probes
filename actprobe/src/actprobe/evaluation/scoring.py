"""
Scoring and Evaluation logic.
"""

from typing import List, Union
from actprobe.utils.normalization import normalize_answer

class Evaluator:
    def evaluate(self, prediction: str, golds: List[str]) -> bool:
        raise NotImplementedError

class ExactMatchEvaluator(Evaluator):
    """
    Checks if normalized prediction matches any normalized gold answer.
    Used for QA (Hotpot, Trivia, Movies).
    """
    def evaluate(self, prediction: str, golds: List[str]) -> bool:
        norm_pred = normalize_answer(prediction)
        norm_golds = [normalize_answer(g) for g in golds]
        
        # Exact match logic
        return norm_pred in norm_golds

class ClassificationEvaluator(Evaluator):
    """
    Checks binary classification (Positive/Negative).
    """
    def evaluate(self, prediction: str, golds: List[str]) -> bool:
        # Normalize prediction to just "positive" or "negative"
        pred = normalize_answer(prediction)
        
        # Simple mapping
        if "positive" in pred:
            clean_pred = "positive"
        elif "negative" in pred:
            clean_pred = "negative"
        else:
            # If unclear, distinct from gold likely, or we can treat as wrong
            clean_pred = "unclear"

        # Gold is expected to be "Positive" or "Negative"
        clean_golds = [normalize_answer(g) for g in golds]
        
        return clean_pred in clean_golds
