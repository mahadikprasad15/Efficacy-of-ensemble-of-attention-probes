"""
Full Probe Model.
Combines Pooling -> Classifier.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from .pooling import (
    MeanPooling,
    MaxPooling,
    LastTokenPooling,
    LearnedAttentionPooling,
    GMHAAttentionPooling,
    DirectQueryHeadPooling,
    MultiMaxPooling,
    RollingAttentionPooling,
    RMSNorm1D,
)


class IdentityPooling(nn.Module):
    """No-op pooling for final-token activations that are already (B, D)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) or (B, T, D) where T=1
        if x.dim() == 3:
            return x.squeeze(1)  # (B, 1, D) -> (B, D)
        return x  # (B, D) -> (B, D)


POOLING_MAP = {
    "mean": MeanPooling,
    "max": MaxPooling,
    "last": LastTokenPooling,
    "attn": LearnedAttentionPooling,
    "none": IdentityPooling,
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


class MultiAttentionProbe(nn.Module):
    """
    Multi-head attention probe for a single layer input.

    Input: (B, T, D)
    Output:
      - (B, 1) for binary tasks (num_classes=1)
      - (B, C) for multiclass tasks
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        num_classes: int = 1,
        variant: str = "gmha",
        topk_tokens: int = 4,
        rolling_window: int = 64,
        rolling_stride: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.variant = variant

        if variant == "gmha":
            self.pooling = GMHAAttentionPooling(input_dim=input_dim, num_heads=num_heads)
            classifier_input_dim = input_dim
        elif variant == "direct_q":
            self.pooling = DirectQueryHeadPooling(input_dim=input_dim, num_heads=num_heads)
            classifier_input_dim = input_dim * num_heads
        elif variant == "multimax":
            self.pooling = MultiMaxPooling(
                input_dim=input_dim,
                num_heads=num_heads,
                topk_tokens=topk_tokens,
            )
            classifier_input_dim = input_dim
        elif variant == "rolling":
            self.pooling = RollingAttentionPooling(
                input_dim=input_dim,
                num_heads=num_heads,
                window_size=rolling_window,
                stride=rolling_stride,
            )
            classifier_input_dim = input_dim
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        out_dim = 1 if num_classes <= 1 else num_classes
        self.classifier = nn.Linear(classifier_input_dim, out_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = self.pooling(x, attention_mask=attention_mask)
        return self.classifier(pooled)


class MultiLayerMultiAttentionProbe(nn.Module):
    """
    Multi-layer attention probe with per-layer RMSNorm and L*T flattening.

    Input: (B, L, T, D)
    Process:
      1) select input layers
      2) per-layer RMSNorm
      3) reshape B,L,T,D -> B,(L*T),D
      4) multi-head attention pooling + classifier
    """

    def __init__(
        self,
        input_dim: int,
        input_layers: Sequence[int],
        num_heads: int = 8,
        num_classes: int = 1,
        variant: str = "gmha",
        topk_tokens: int = 4,
        rolling_window: int = 64,
        rolling_stride: int = 32,
    ):
        super().__init__()
        if not input_layers:
            raise ValueError("input_layers must be non-empty for multi-layer probe")

        self.input_dim = input_dim
        self.input_layers = [int(x) for x in input_layers]
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.variant = variant
        self.norms = nn.ModuleList([RMSNorm1D(input_dim) for _ in self.input_layers])

        if variant == "gmha":
            self.pooling = GMHAAttentionPooling(input_dim=input_dim, num_heads=num_heads)
            classifier_input_dim = input_dim
        elif variant == "direct_q":
            self.pooling = DirectQueryHeadPooling(input_dim=input_dim, num_heads=num_heads)
            classifier_input_dim = input_dim * num_heads
        elif variant == "multimax":
            self.pooling = MultiMaxPooling(
                input_dim=input_dim,
                num_heads=num_heads,
                topk_tokens=topk_tokens,
            )
            classifier_input_dim = input_dim
        elif variant == "rolling":
            self.pooling = RollingAttentionPooling(
                input_dim=input_dim,
                num_heads=num_heads,
                window_size=rolling_window,
                stride=rolling_stride,
            )
            classifier_input_dim = input_dim
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        out_dim = 1 if num_classes <= 1 else num_classes
        self.classifier = nn.Linear(classifier_input_dim, out_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L, T, D)
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape (B,L,T,D), got {tuple(x.shape)}")

        selected = []
        masks = []
        for idx, layer in enumerate(self.input_layers):
            x_layer = x[:, layer, :, :]  # (B, T, D)
            selected.append(self.norms[idx](x_layer))
            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    masks.append(attention_mask[:, layer, :])
                else:
                    masks.append(attention_mask)

        x_cat = torch.stack(selected, dim=1)  # (B, Ls, T, D)
        bsz, ls, seq_len, dim = x_cat.shape
        x_cat = x_cat.view(bsz, ls * seq_len, dim)  # (B, Ls*T, D)

        attn_mask_cat = None
        if masks:
            attn_mask_cat = torch.stack(masks, dim=1).view(bsz, ls * seq_len)

        pooled = self.pooling(x_cat, attention_mask=attn_mask_cat)
        return self.classifier(pooled)


class GMHAAttentionProbe(MultiAttentionProbe):
    def __init__(self, input_dim: int, num_heads: int = 8, num_classes: int = 1):
        super().__init__(
            input_dim=input_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            variant="gmha",
        )


class DirectQueryAttentionProbe(MultiAttentionProbe):
    def __init__(self, input_dim: int, num_heads: int = 8, num_classes: int = 1):
        super().__init__(
            input_dim=input_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            variant="direct_q",
        )


class MultiMaxProbe(MultiAttentionProbe):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        num_classes: int = 1,
        topk_tokens: int = 4,
    ):
        super().__init__(
            input_dim=input_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            variant="multimax",
            topk_tokens=topk_tokens,
        )


class RollingAttentionProbe(MultiAttentionProbe):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        num_classes: int = 1,
        rolling_window: int = 64,
        rolling_stride: int = 32,
    ):
        super().__init__(
            input_dim=input_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            variant="rolling",
            rolling_window=rolling_window,
            rolling_stride=rolling_stride,
        )


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
