#!/usr/bin/env python3
"""
Utility functions for running GLP denoising on activation tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def ensure_bsd(latents: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor shape is (B, S, D).
    Accepts:
      - (S, D) -> (1, S, D)
      - (B, S, D) -> unchanged
    """
    if latents.ndim == 2:
        return latents.unsqueeze(0)
    if latents.ndim == 3:
        return latents
    raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(latents.shape)}")


@dataclass
class GLPArtifacts:
    model: object
    sample_on_manifold_fn: object
    model_id: str
    checkpoint: str
    device: str


def load_glp_artifacts(
    model_id: str,
    device: str = "cuda:0",
    checkpoint: str = "final",
) -> GLPArtifacts:
    """
    Load GLP model and sampling function from the external `glp` package.
    """
    try:
        from glp.denoiser import load_glp  # type: ignore
        from glp.flow_matching import sample_on_manifold  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import GLP package. Install from "
            "https://github.com/g-luo/generative_latent_prior and ensure `glp` is on PYTHONPATH."
        ) from e

    model = load_glp(model_id, device=device, checkpoint=checkpoint)
    return GLPArtifacts(
        model=model,
        sample_on_manifold_fn=sample_on_manifold,
        model_id=model_id,
        checkpoint=checkpoint,
        device=device,
    )


def resolve_start_timestep_value(
    model: object,
    num_timesteps: int,
    mode: str = "half",
) -> Optional[float]:
    """
    Resolve start_timestep as a scheduler timestep VALUE (not index).
    """
    mode = str(mode).strip().lower()
    if mode in {"none", "null"}:
        return None

    model.scheduler.set_timesteps(num_timesteps)
    ts = model.scheduler.timesteps
    if len(ts) == 0:
        raise ValueError("Scheduler timesteps are empty.")

    if mode == "half":
        idx = len(ts) // 2
        return float(ts[idx].item())

    if mode.startswith("idx:"):
        idx = int(mode.split(":", 1)[1])
        if idx < 0 or idx >= len(ts):
            raise ValueError(f"Start index out of range: {idx} for {len(ts)} timesteps.")
        return float(ts[idx].item())

    if mode.startswith("frac:"):
        frac = float(mode.split(":", 1)[1])
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"frac must be in [0,1], got {frac}")
        idx = int(round(frac * (len(ts) - 1)))
        return float(ts[idx].item())

    raise ValueError(f"Unsupported start timestep mode: {mode}")


def denoise_on_manifold(
    artifacts: GLPArtifacts,
    latents_bsd: torch.Tensor,
    layer_idx: int,
    num_timesteps: int,
    start_timestep_mode: str = "half",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, float | None]:
    """
    Run GLP normalize -> noise injection -> sample_on_manifold -> denormalize.

    Returns:
      denoised latents (B,S,D) on input device and resolved start_timestep value.
    """
    x = ensure_bsd(latents_bsd)
    device = x.device
    dtype = x.dtype

    # GLP's packaged sampler currently assumes effective sequence length S=1
    # in timestep shaping. Flatten (B,S,D) -> (B*S,1,D), denoise, then restore.
    bsz, seqlen, dim = x.shape
    x_norm = artifacts.model.normalizer.normalize(x, layer_idx=layer_idx)
    x_norm_flat = x_norm.reshape(bsz * seqlen, 1, dim)

    if seed is None:
        noise = torch.randn_like(x_norm_flat)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        noise = torch.randn(x_norm_flat.shape, device=device, dtype=x_norm_flat.dtype, generator=gen)
    x_noisy_flat = x_norm_flat + float(noise_scale) * noise

    start_timestep_value = resolve_start_timestep_value(
        model=artifacts.model,
        num_timesteps=int(num_timesteps),
        mode=start_timestep_mode,
    )

    x_denoised_norm = artifacts.sample_on_manifold_fn(
        artifacts.model,
        latents=x_noisy_flat,
        num_timesteps=int(num_timesteps),
        start_timestep=start_timestep_value,
        layer_idx=layer_idx,
    )
    x_denoised = artifacts.model.normalizer.denormalize(x_denoised_norm, layer_idx=layer_idx)
    x_denoised = x_denoised.reshape(bsz, seqlen, dim)
    x_denoised = x_denoised.to(device=device, dtype=dtype)
    return x_denoised, start_timestep_value
