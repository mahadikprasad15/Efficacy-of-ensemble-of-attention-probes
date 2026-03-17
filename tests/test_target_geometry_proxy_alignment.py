from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_target_geometry_proxy_alignment_tests")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analysis" / "run_target_geometry_proxy_alignment.py"

spec = importlib.util.spec_from_file_location("target_geometry_proxy_alignment", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_metric_cosine_matches_standard_cosine_under_identity_geometry() -> None:
    a = np.asarray([1.0, 2.0, -1.0], dtype=np.float64)
    b = np.asarray([2.0, 0.5, 3.0], dtype=np.float64)
    cov_inv = np.eye(3, dtype=np.float64)

    got = module.metric_cosine(a, b, cov_inv)
    expected = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert np.isclose(got, expected)


def test_invert_covariance_uses_ridge_inverse_by_default() -> None:
    cov = np.asarray([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    ridge_lambda = 0.25

    cov_inv, method = module.invert_covariance(cov, ridge_lambda=ridge_lambda, use_pinv=False)
    expected = np.linalg.inv(cov + np.eye(2, dtype=np.float64) * ridge_lambda)

    assert method == "ridge_inv"
    assert np.allclose(cov_inv, expected)


def test_read_matrix_csv_round_trip(tmp_path: Path) -> None:
    rows = ["row_a", "row_b"]
    cols = ["col_x", "col_y", "col_z"]
    matrix = np.asarray([[0.1, np.nan, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)

    path = tmp_path / "matrix.csv"
    module.write_matrix_csv(path, rows, cols, matrix)

    got_rows, got_cols, got_matrix = module.read_matrix_csv(path)

    assert got_rows == rows
    assert got_cols == cols
    assert np.allclose(np.nan_to_num(got_matrix, nan=-999.0), np.nan_to_num(matrix, nan=-999.0))
