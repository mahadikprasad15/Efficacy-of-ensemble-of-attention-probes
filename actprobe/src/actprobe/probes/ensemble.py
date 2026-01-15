"""
Layer ensemble logic.
Combines predictions from multiple layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class BaseEnsemble(nn.Module):
    def forward(self, layer_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_outputs: (B, L, 1) - logits/probs from per-layer probes
        Returns:
            (B, 1) - final logit/prob
        """
        raise NotImplementedError

class StaticMeanEnsemble(BaseEnsemble):
    """Simple average of selected layers."""
    def forward(self, layer_outputs: torch.Tensor) -> torch.Tensor:
        return layer_outputs.mean(dim=1)

class StaticWeightedEnsemble(BaseEnsemble):
    """Weighted average with fixed weights (e.g. from validation AUC)."""
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer('weights', weights / weights.sum()) # Normalize
        
    def forward(self, layer_outputs: torch.Tensor) -> torch.Tensor:
        # layer_outputs: (B, L, 1)
        # weights: (L)
        # sum((B, L, 1) * (1, L, 1)) -> (B, 1)
        return (layer_outputs * self.weights.view(1, -1, 1)).sum(dim=1)

class GatedEnsemble(BaseEnsemble):
    """
    Input-dependent gating.
    Computes weights alpha_l(x) based on input features x.
    This requires the input features, not just layer outputs.
    So interface differs slightly.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        # Simple MLP to predict layer weights
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers),
            nn.Softmax(dim=1)
        )
        
    def forward(self, layer_features: torch.Tensor, layer_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_features: (B, L, D) or (B, D_flat) - feature representation to gate on
            layer_logits: (B, L, 1) - predictions from individual probes
        """
        # If features are (B, L, D), maybe avg them to get global context?
        # Or gate_net takes flattened?
        # Let's assume we use a global summary of the input x for gating.
        # For simplicity, mean of layer_features along L.
        
        if layer_features.dim() == 3:
            summary = layer_features.mean(dim=1) # (B, D)
        else:
            summary = layer_features
            
        weights = self.gate_net(summary) # (B, L)
        
        # Weighted sum of logits: (B, L, 1) * (B, L, 1) -> sum
        logits = (layer_logits * weights.unsqueeze(-1)).sum(dim=1)
        return logits
