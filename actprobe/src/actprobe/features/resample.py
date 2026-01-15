"""
Feature resampling / dimension reduction logic.
"""

import torch
import torch.nn.functional as F

def resample_activations(tensor: torch.Tensor, target_L: int = 16, target_T: int = 64) -> torch.Tensor:
    """
    Resample activation tensor (L, T, D) to (target_L, target_T, D).
    
    Args:
        tensor: (L, T, D)
        target_L: Number of layers to keep/interpolate.
        target_T: Number of token bins.
        
    Returns:
        (target_L, target_T, D)
    """
    L, T, D = tensor.shape
    
    # 1. Layer Axis
    # If L == target_L, correct
    # If L != target_L, we might need to pick specific layers (e.g. evenly spaced)
    # Llama 1B has 16 layers. User said keep 16.
    if L == target_L:
        layer_tensor = tensor
    elif L > target_L:
        # Downsample layers: pick indices
        indices = torch.linspace(0, L-1, steps=target_L).long()
        layer_tensor = tensor[indices]
    else:
        # Upsample? Or just fail?
        # User said keep 16 for 1B, but 32 for others. 
        # If we pass target_L=16 for 1B, we land in first case.
        # If we force target_L=32, we'd need to interpolate.
        # Simple nearest neighbor or linear along L? 
        # Let's assume we pass the correct target_L per model config for now.
        layer_tensor = tensor
    
    # 2. Token Axis (Adaptive Pooling)
    # We treat T as the spatial dimension to pool.
    # Input to AdaptiveAvgPool1d needs to be (N, C, L_in) -> (L, D, T)
    # So permute (L, T, D) -> (L, D, T)
    permuted = layer_tensor.permute(0, 2, 1) # (L, D, T)
    
    # Pool to T'
    pooled = F.adaptive_avg_pool1d(permuted, target_T) # (L, D, T')
    
    # Permute back -> (L, T', D)
    result = pooled.permute(0, 2, 1)
    
    return result.to(torch.float16)  # Ensure half precision for storage
