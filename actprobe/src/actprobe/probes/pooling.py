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

    Optionally returns attention weights for entropy analysis (Spec section 9A).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim, 1))
        self.last_attention_weights = None  # Store for analysis

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (B, T, D)
            return_attention: If True, returns (pooled, attention_weights)

        Returns:
            pooled: (B, D) or (pooled, weights) if return_attention=True
        """
        # x: (B, T, D)
        # scores = x @ q -> (B, T, 1)
        scores = torch.matmul(x, self.query)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)

        # Store for analysis (spec: log entropy of attention weights)
        self.last_attention_weights = weights.detach()

        # Weighted sum: (B, T, D) * (B, T, 1) -> sum(dim=1) -> (B, D)
        pooled = (x * weights).sum(dim=1)

        if return_attention:
            return pooled, weights.squeeze(-1)  # (B, D), (B, T)
        return pooled

    def compute_attention_entropy(self):
        """
        Compute Shannon entropy of last attention weights.
        Returns mean entropy across batch.

        H(weights) = -sum(w * log(w))
        Lower entropy = more concentrated attention (spec requirement).
        """
        if self.last_attention_weights is None:
            return None

        weights = self.last_attention_weights.squeeze(-1)  # (B, T)
        # Avoid log(0)
        weights = torch.clamp(weights, min=1e-10)
        entropy = -(weights * torch.log(weights)).sum(dim=1)  # (B,)
        return entropy.mean().item()
