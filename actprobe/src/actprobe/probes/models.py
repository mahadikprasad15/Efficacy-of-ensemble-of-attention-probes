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


class TwoLevelAttentionProbe(nn.Module):
    """
    Variant C: Two-Level Attention Probe

    Architecture:
    1. Token-level attention: Pool tokens within each layer (per-layer learned attention)
    2. Layer-level attention: Pool across layers (learned layer attention)
    3. Final classifier

    This is the "two attention knobs" approach from the spec.

    Input: (B, L, T, D)
    Output: (B, 1) logits
    """
    def __init__(self, input_dim: int, num_layers: int, shared_token_attn: bool = True):
        """
        Args:
            input_dim: Hidden dimension (D)
            num_layers: Number of layers (L)
            shared_token_attn: If True, use same attention query for all layers.
                              If False, learn separate attention for each layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.shared_token_attn = shared_token_attn

        # Token-level attention (within each layer)
        if shared_token_attn:
            self.token_attention = LearnedAttentionPooling(input_dim)
        else:
            # Per-layer token attention
            self.token_attention = nn.ModuleList([
                LearnedAttentionPooling(input_dim) for _ in range(num_layers)
            ])

        # Layer-level attention (across layers)
        # Takes (B, L, D) and outputs (B, D)
        self.layer_attention = LearnedAttentionPooling(input_dim)

        # Final classifier
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, T, D)

        Returns:
            logits: (B, 1)
        """
        B, L, T, D = x.shape

        # Step 1: Token pooling for each layer
        # Output: (B, L, D)
        layer_features = []

        for l in range(L):
            x_layer = x[:, l, :, :]  # (B, T, D)

            if self.shared_token_attn:
                pooled = self.token_attention(x_layer)  # (B, D)
            else:
                pooled = self.token_attention[l](x_layer)  # (B, D)

            layer_features.append(pooled)

        # Stack: (B, L, D)
        layer_features = torch.stack(layer_features, dim=1)

        # Step 2: Layer pooling across all layers
        # Input: (B, L, D) -> treat as (B, T=L, D)
        pooled = self.layer_attention(layer_features)  # (B, D)

        # Step 3: Classification
        logits = self.classifier(pooled)  # (B, 1)

        return logits
