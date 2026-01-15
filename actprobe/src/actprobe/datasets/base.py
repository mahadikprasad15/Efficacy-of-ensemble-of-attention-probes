"""
Abstract Base Class for ACT-ViT datasets.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    """
    
    def __init__(self, split: str = "validation", limit: Optional[int] = None):
        self.split = split
        self.limit = limit
        self.data = []

    @abstractmethod
    def load_data(self):
        """
        Load data from source (HuggingFace or local).
        Populates self.data with raw examples.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a formatted example.
        Must return a dict with keys:
            - prompt: str (ready for chat template or direct usage)
            - gold_answers: List[str]
            - metadata: Dict (dataset_name, id, split)
        """
        pass

    def __len__(self) -> int:
        return len(self.data)

    def apply_limit(self):
        """Helper to enforce the limit on loaded data."""
        if self.limit is not None and len(self.data) > self.limit:
            self.data = self.data[:self.limit]
