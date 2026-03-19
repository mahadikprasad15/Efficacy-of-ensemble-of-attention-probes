"""Dataset-fingerprint fitting and residualization utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression


def load_manifest_entries(manifest_path: str | Path) -> List[dict]:
    """Load non-empty JSONL manifest rows."""
    path = Path(manifest_path)
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def pool_layer_tensor(tensor: torch.Tensor, pooling: str) -> np.ndarray:
    """Pool a single activation tensor from shape (T, D) or (D,)."""
    if tensor.ndim == 1:
        return tensor.detach().cpu().float().numpy().astype(np.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected tensor with shape (T, D) or (D,), got {tuple(tensor.shape)}")

    if pooling == "mean":
        pooled = tensor.mean(dim=0)
    elif pooling == "max":
        pooled = tensor.max(dim=0).values
    elif pooling == "last":
        pooled = tensor[-1]
    elif pooling == "none":
        if tensor.shape[0] != 1:
            raise ValueError("Pooling 'none' only supports a single token.")
        pooled = tensor[0]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return pooled.detach().cpu().float().numpy().astype(np.float32)


def load_pooled_layer_features(
    activations_dir: str | Path,
    layer_idx: int,
    pooling: str = "mean",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load pooled layer activations in manifest order.

    Returns:
        X: (N, D) pooled features
        sample_ids: manifest-ordered sample ids
        labels: (N,) binary labels from the manifest
    """
    activations_path = Path(activations_dir)
    manifest_entries = load_manifest_entries(activations_path / "manifest.jsonl")
    if not manifest_entries:
        raise ValueError(f"No manifest entries found in {activations_path}")

    shard_to_entries: MutableMapping[int, List[dict]] = defaultdict(list)
    for entry in manifest_entries:
        shard_idx = int(entry.get("shard", 0))
        shard_to_entries[shard_idx].append(entry)

    pooled_rows: List[np.ndarray] = []
    sample_ids: List[str] = []
    labels: List[int] = []

    for shard_idx in sorted(shard_to_entries):
        shard_path = activations_path / f"shard_{shard_idx:03d}.safetensors"
        if not shard_path.exists():
            alt_path = activations_path / f"shard_{shard_idx}.safetensors"
            shard_path = alt_path if alt_path.exists() else shard_path
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard {shard_idx} in {activations_path}")

        tensors = load_file(str(shard_path))
        for entry in shard_to_entries[shard_idx]:
            sample_id = str(entry["id"])
            if sample_id not in tensors:
                continue
            tensor = tensors[sample_id]
            if tensor.ndim == 3:
                layer_tensor = tensor[layer_idx]
            elif tensor.ndim == 2:
                layer_tensor = tensor[layer_idx]
            else:
                raise ValueError(f"Unexpected tensor shape for {sample_id}: {tuple(tensor.shape)}")
            pooled_rows.append(pool_layer_tensor(layer_tensor, pooling))
            sample_ids.append(sample_id)
            labels.append(int(entry.get("label", -1)))

    if not pooled_rows:
        raise ValueError(f"No pooled rows loaded from {activations_path}")
    return np.stack(pooled_rows).astype(np.float32), sample_ids, np.asarray(labels, dtype=np.int64)


def select_equal_count_indices(
    group_labels: Sequence[str],
    count_per_group: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample equal counts from each group without replacement."""
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(group_labels):
        grouped[str(label)].append(idx)
    if not grouped:
        raise ValueError("No group labels supplied.")

    group_sizes = {group: len(idxs) for group, idxs in grouped.items()}
    effective_count = min(group_sizes.values()) if count_per_group is None else int(count_per_group)
    if effective_count <= 0:
        raise ValueError("count_per_group must be positive.")
    for group, size in group_sizes.items():
        if size < effective_count:
            raise ValueError(
                f"Group '{group}' has {size} examples, fewer than requested {effective_count}."
            )

    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for group in sorted(grouped):
        selected = rng.choice(np.asarray(grouped[group], dtype=np.int64), size=effective_count, replace=False)
        chosen.extend(int(x) for x in selected.tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def sample_balanced_binary_ids(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    per_class_count: int,
    seed: int = 0,
    negative_label: int = 0,
    positive_label: int = 1,
) -> List[str]:
    """Sample a balanced binary subset and return the selected ids."""
    if per_class_count <= 0:
        raise ValueError("per_class_count must be positive.")

    neg_ids = [sample_id for sample_id, label in zip(sample_ids, labels) if int(label) == negative_label]
    pos_ids = [sample_id for sample_id, label in zip(sample_ids, labels) if int(label) == positive_label]

    if len(neg_ids) < per_class_count or len(pos_ids) < per_class_count:
        raise ValueError(
            "Insufficient class counts for balanced sampling: "
            f"need {per_class_count} per class, got negatives={len(neg_ids)} positives={len(pos_ids)}."
        )

    rng = np.random.default_rng(seed)
    chosen_neg = rng.choice(np.asarray(neg_ids, dtype=object), size=per_class_count, replace=False).tolist()
    chosen_pos = rng.choice(np.asarray(pos_ids, dtype=object), size=per_class_count, replace=False).tolist()
    chosen = [str(x) for x in chosen_neg + chosen_pos]
    rng.shuffle(chosen)
    return chosen


def fit_multinomial_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    regularization_c: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
) -> LogisticRegression:
    """Fit multinomial logistic regression for dataset identification."""
    clf = LogisticRegression(
        penalty="l2",
        C=regularization_c,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


def row_center_weights(weights: np.ndarray) -> np.ndarray:
    """Remove the shared offset ambiguity across multinomial class rows."""
    if weights.ndim != 2:
        raise ValueError(f"Expected weights with shape (C, D), got {tuple(weights.shape)}")
    return weights - weights.mean(axis=0, keepdims=True)


def fingerprint_basis_from_weights(
    weights: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute an orthonormal basis spanning the row-centered weight matrix."""
    centered = row_center_weights(weights.astype(np.float64, copy=False))
    rank = int(np.linalg.matrix_rank(centered, tol=tol))
    if rank == 0:
        return np.zeros((0, centered.shape[1]), dtype=np.float32), centered.astype(np.float32), 0
    q, _ = np.linalg.qr(centered.T)
    basis = q[:, :rank].T.astype(np.float32)
    return basis, centered.astype(np.float32), rank


def residualize_vectors(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project the rows of X out of the basis row span."""
    X_arr = np.asarray(X, dtype=np.float32)
    basis_arr = np.asarray(basis, dtype=np.float32)
    if basis_arr.size == 0:
        return X_arr.copy()
    coeffs = X_arr @ basis_arr.T
    projection = coeffs @ basis_arr
    return (X_arr - projection).astype(np.float32)


def residualize_token_matrix(tokens: torch.Tensor, basis: np.ndarray) -> torch.Tensor:
    """Residualize a token matrix with shape (T, D) or a single vector (D,)."""
    orig_dtype = tokens.dtype
    basis_tensor = torch.as_tensor(basis, dtype=torch.float32, device=tokens.device)
    if basis_tensor.numel() == 0:
        return tokens.clone()

    x = tokens.float()
    if x.ndim == 1:
        coeffs = torch.matmul(basis_tensor, x)
        projection = torch.matmul(coeffs, basis_tensor)
        return (x - projection).to(orig_dtype)
    if x.ndim != 2:
        raise ValueError(f"Expected token tensor with shape (T, D) or (D,), got {tuple(tokens.shape)}")

    coeffs = torch.matmul(x, basis_tensor.t())
    projection = torch.matmul(coeffs, basis_tensor)
    return (x - projection).to(orig_dtype)


def residualize_activation_tensor(
    tensor: torch.Tensor,
    basis: np.ndarray,
    layer_idx: int,
) -> torch.Tensor:
    """Residualize a single activation tensor only at one layer index."""
    out = tensor.clone()
    if layer_idx < 0 or layer_idx >= int(out.shape[0]):
        raise ValueError(f"Layer index {layer_idx} out of bounds for tensor shape {tuple(tensor.shape)}")
    if out.ndim == 3:
        out[layer_idx] = residualize_token_matrix(out[layer_idx], basis)
        return out
    if out.ndim == 2:
        out[layer_idx] = residualize_token_matrix(out[layer_idx], basis)
        return out
    raise ValueError(f"Unexpected activation tensor shape: {tuple(tensor.shape)}")


def dataset_name_to_index(dataset_names: Iterable[str]) -> Dict[str, int]:
    """Build a stable dataset-name -> integer index mapping."""
    unique = list(dict.fromkeys(str(name) for name in dataset_names))
    return {name: idx for idx, name in enumerate(unique)}
