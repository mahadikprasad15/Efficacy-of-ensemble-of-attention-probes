"""PCA subspace projection utilities for probe training experiments."""

import numpy as np


def load_pca_artifact(npz_path: str) -> dict:
    """Load PCA artifact from .npz file.

    Returns dict with keys: mean, components, explained_variance,
    explained_variance_ratio, dim.
    """
    data = np.load(npz_path, allow_pickle=False)
    return {
        "mean": data["mean"].astype(np.float32),
        "components": data["components"].astype(np.float32),
        "explained_variance": data["explained_variance"],
        "explained_variance_ratio": data["explained_variance_ratio"],
        "dim": int(data["dim"][0]) if data["dim"].ndim > 0 else int(data["dim"]),
    }


def get_top_k_components(components: np.ndarray, K: int) -> np.ndarray:
    """Return top-K PC directions (highest eigenvalues). Shape: (K, D)."""
    K_max = components.shape[0]
    if K > K_max:
        raise ValueError(f"K={K} exceeds available components={K_max}")
    return components[:K].astype(np.float32)


def get_bottom_k_components(components: np.ndarray, K: int) -> np.ndarray:
    """Return bottom-K PC directions (smallest eigenvalues in fitted set).

    Shape: (K, D). Uses the last K rows of the components matrix
    (which are sorted by descending eigenvalue).
    """
    K_max = components.shape[0]
    if K > K_max:
        raise ValueError(f"K={K} exceeds available components={K_max}")
    return components[K_max - K :].astype(np.float32)


def generate_random_orthogonal(D: int, K: int, seed: int) -> np.ndarray:
    """Generate K orthonormal vectors in R^D via QR decomposition.

    Returns: (K, D) matrix with orthonormal rows. Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((D, K)).astype(np.float64)
    Q, R = np.linalg.qr(Z)
    # Fix sign ambiguity for reproducibility
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q[:, :K].T.astype(np.float32)  # (K, D)


def project_activations(
    X: np.ndarray,
    mean: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Center and project activations onto a K-dimensional subspace.

    Args:
        X: (N, D) pooled activations.
        mean: (D,) PCA mean for centering.
        V: (K, D) projection directions (rows are basis vectors).

    Returns:
        (N, K) projected features.
    """
    X_centered = X - mean[np.newaxis, :]
    return (X_centered @ V.T).astype(np.float32)
