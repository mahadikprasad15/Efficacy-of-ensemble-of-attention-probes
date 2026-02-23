"""
Streaming normalization stats for activations.
Computes per-layer mean/std over token-level activations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import torch


def init_running_stats(num_layers: int, dim: int, dtype: torch.dtype = torch.float64) -> Dict[str, torch.Tensor]:
    return {
        "mean": torch.zeros((num_layers, dim), dtype=dtype),
        "m2": torch.zeros((num_layers, dim), dtype=dtype),
        "count": torch.zeros((num_layers,), dtype=dtype),
    }


def update_running_stats(stats: Dict[str, torch.Tensor], x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Update running stats with a single sample tensor.
    x: (L, T, D) or (L, D)
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (L, 1, D)
    x = x.to(dtype=torch.float64)

    batch_count = x.shape[1]
    batch_mean = x.mean(dim=1)  # (L, D)
    batch_var = x.var(dim=1, unbiased=False)  # (L, D)

    mean = stats["mean"]
    m2 = stats["m2"]
    count = stats["count"]  # (L,)

    count_exp = count.unsqueeze(1)
    batch_count_f = float(batch_count)
    total = count_exp + batch_count_f
    delta = batch_mean - mean

    mean = mean + delta * batch_count_f / total
    m2 = m2 + batch_var * batch_count_f + (delta ** 2) * count_exp * batch_count_f / total
    count = count + batch_count_f

    stats["mean"] = mean
    stats["m2"] = m2
    stats["count"] = count
    return stats


def finalize_running_stats(stats: Dict[str, torch.Tensor], eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    count = stats["count"].clamp(min=1.0)
    var = stats["m2"] / count.unsqueeze(1)
    std = torch.sqrt(var) + eps
    return {
        "mean": stats["mean"],
        "std": std,
        "count": count,
    }


def save_layer_stats(
    stats: Dict[str, torch.Tensor],
    out_dir: str,
    pooling: str,
    source_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    mean = stats["mean"].cpu()
    std = stats["std"].cpu()
    count = stats["count"].cpu()

    num_layers = mean.shape[0]
    dim = mean.shape[1]

    for layer_idx in range(num_layers):
        payload = {
            "layer": layer_idx,
            "shape": [int(dim)],
            "count": int(count[layer_idx].item()),
            "pooling": pooling,
            "source_dir": source_dir,
            "created_at": created_at,
            "mean": mean[layer_idx].tolist(),
            "std": std[layer_idx].tolist(),
        }
        path = os.path.join(out_dir, f"layer_{layer_idx}.json")
        with open(path, "w") as f:
            json.dump(payload, f)


def load_layer_stats(
    stats_dir: str,
    layer_idx: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    path = os.path.join(stats_dir, f"layer_{layer_idx}.json")
    with open(path, "r") as f:
        payload = json.load(f)
    mean = torch.tensor(payload["mean"], dtype=dtype, device=device)
    std = torch.tensor(payload["std"], dtype=dtype, device=device)
    count = int(payload.get("count", 0))
    return mean, std, count


def stats_complete(stats_dir: str, num_layers: int) -> bool:
    if not os.path.isdir(stats_dir):
        return False
    for layer_idx in range(num_layers):
        if not os.path.exists(os.path.join(stats_dir, f"layer_{layer_idx}.json")):
            return False
    return True
