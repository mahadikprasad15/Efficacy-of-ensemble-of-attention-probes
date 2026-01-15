"""
Token pooling strategies for reducing (T, D) -> (D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasePooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, D)
        """
        raise NotImplementedError

class MeanPooling(BasePooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> mean(dim=1) -> (B, D)
        return x.mean(dim=1)

class MaxPooling(BasePooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> max(dim=1) -> (B, D)
        return x.max(dim=1)[0]

class LastTokenPooling(BasePooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> last index -> (B, D)
        return x[:, -1, :]

class LearnedAttentionPooling(BasePooling):
    """
    Learns a query vector q to compute attention scores over tokens.
    a_t = softmax(x_t * q)
    z = sum(a_t * x_t)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # scores = x @ q -> (B, T, 1)
        scores = torch.matmul(x, self.query)
        weights = F.softmax(scores, dim=1) # (B, T, 1)
        
        # Weighted sum: (B, T, D) * (B, T, 1) -> sum(dim=1) -> (B, D)
        pooled = (x * weights).sum(dim=1)
        return pooled
