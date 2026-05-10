"""
Layer ensemble logic.
Combines predictions from multiple layers.
"""

import torch
import torch.nn as nn
from typing import Optional

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


def mean_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the mean categorical entropy across a batch of normalized weights."""
    clipped = torch.clamp(weights, min=eps)
    return -(clipped * torch.log(clipped)).sum(dim=-1).mean()


class ProbeLogitGatedEnsemble(nn.Module):
    """
    Input-dependent gate over a frozen bank of expert logits.

    The default v1 setup uses the same normalized expert-logit vector both as
    the gating feature input and as the values being combined.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.temperature = float(temperature)
        self.gate_net = nn.Sequential(
            nn.Linear(self.num_experts, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_experts),
        )

    def gate_scores(self, expert_features: torch.Tensor) -> torch.Tensor:
        return self.gate_net(expert_features)

    def gate_weights(self, expert_features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate_scores(expert_features) / self.temperature, dim=-1)

    def forward(
        self,
        expert_logits: torch.Tensor,
        expert_features: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        if expert_logits.dim() != 2:
            raise ValueError(f"expert_logits must have shape (B, E), got {tuple(expert_logits.shape)}")
        features = expert_logits if expert_features is None else expert_features
        if features.shape != expert_logits.shape:
            raise ValueError(
                f"expert_features must match expert_logits shape; got {tuple(features.shape)} vs {tuple(expert_logits.shape)}"
            )
        weights = self.gate_weights(features)
        final_logits = (weights * expert_logits).sum(dim=-1, keepdim=True)
        if return_weights:
            return final_logits, weights
        return final_logits
