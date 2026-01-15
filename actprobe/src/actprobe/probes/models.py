"""
Full Probe Model.
Combines Pooling -> Classifier.
"""

import torch
import torch.nn as nn
from .pooling import MeanPooling, MaxPooling, LastTokenPooling, LearnedAttentionPooling

POOLING_MAP = {
    "mean": MeanPooling,
    "max": MaxPooling,
    "last": LastTokenPooling,
    "attn": LearnedAttentionPooling
}

class LayerProbe(nn.Module):
    """
    Single Layer Probe: Pooling -> Linear -> Sigmoid/Logits
    """
    def __init__(self, input_dim: int, pooling_type: str = "mean"):
        super().__init__()
        if pooling_type not in POOLING_MAP:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
            
        PoolClass = POOLING_MAP[pooling_type]
        if pooling_type == "attn":
            self.pooling = PoolClass(input_dim)
        else:
            self.pooling = PoolClass()
            
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        pooled = self.pooling(x) # (B, D)
        logits = self.classifier(pooled) # (B, 1)
        return logits
