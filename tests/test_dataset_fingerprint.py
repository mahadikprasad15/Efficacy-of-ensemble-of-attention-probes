from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "actprobe" / "src"))

from actprobe.features.dataset_fingerprint import (  # noqa: E402
    fingerprint_basis_from_weights,
    residualize_vectors,
    sample_balanced_binary_ids,
    select_equal_count_indices,
)


def projection_matrix(basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return basis.T @ basis


def test_row_centering_removes_shared_multinomial_offset() -> None:
    weights = np.asarray(
        [
            [3.0, 1.0, -2.0],
            [0.5, -1.5, 4.0],
            [-2.0, 0.0, 1.5],
        ],
        dtype=np.float32,
    )
    shifted = weights + np.asarray([10.0, -4.0, 7.0], dtype=np.float32)

    basis_a, centered_a, rank_a = fingerprint_basis_from_weights(weights)
    basis_b, centered_b, rank_b = fingerprint_basis_from_weights(shifted)

    assert rank_a == rank_b
    assert np.allclose(centered_a, centered_b)
    assert np.allclose(projection_matrix(basis_a), projection_matrix(basis_b), atol=1e-6)


def test_residualized_vectors_are_orthogonal_to_fingerprint_basis() -> None:
    weights = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [-2.0, -2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    basis, _, rank = fingerprint_basis_from_weights(weights)
    assert rank == 2
    assert np.allclose(basis @ basis.T, np.eye(rank), atol=1e-6)

    X = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.5, 0.5, 2.0, -3.0],
            [0.0, -4.0, 1.0, 2.5],
        ],
        dtype=np.float32,
    )
    residualized = residualize_vectors(X, basis)

    assert np.allclose(residualized @ basis.T, 0.0, atol=1e-5)


def test_equal_count_and_balanced_binary_sampling_are_reproducible() -> None:
    groups = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    selected_a = select_equal_count_indices(groups, count_per_group=2, seed=7)
    selected_b = select_equal_count_indices(groups, count_per_group=2, seed=7)
    assert selected_a.tolist() == selected_b.tolist()

    selected_groups = [groups[idx] for idx in selected_a.tolist()]
    assert selected_groups.count("a") == 2
    assert selected_groups.count("b") == 2
    assert selected_groups.count("c") == 2

    sample_ids = [f"id_{idx}" for idx in range(12)]
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    chosen_a = sample_balanced_binary_ids(sample_ids, labels, per_class_count=3, seed=11)
    chosen_b = sample_balanced_binary_ids(sample_ids, labels, per_class_count=3, seed=11)

    assert chosen_a == chosen_b
    chosen_labels = [labels[sample_ids.index(sample_id)] for sample_id in chosen_a]
    assert chosen_labels.count(0) == 3
    assert chosen_labels.count(1) == 3
