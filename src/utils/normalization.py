"""
String normalization utilities for answer comparison.
Adapted from official SQuAD and HotpotQA evaluation scripts.
"""

import re
import string

def normalize_answer(s: str) -> str:
    """Normalize answer string by removing articles, punctuation, and whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
