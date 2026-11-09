"""Deterministic algebra fixture; no FL arrays, fit, performance, or TEST access."""
import json
from pathlib import Path

import numpy as np
from scipy.special import polygamma


def normalized_count_geometry(weights):
    matrix = np.diag(weights) - np.outer(weights, weights) / weights.sum()
    return 9 * matrix / np.trace(matrix)


def main():
    projection = np.eye(10) - np.ones((10, 10)) / 10
    rows = []
    for counts in (
        np.array([0, 0, 1, 2, 5, 10, 20, 50, 100, 500]),
        np.full(10, 10),
        np.array([450, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ):
        variance = polygamma(1, counts + 1.0)
        covariance = projection @ np.diag(variance) @ projection
        weights = 1 / variance
        closed = np.diag(weights) - np.outer(weights, weights) / weights.sum()
        pseudoinverse = np.linalg.pinv(covariance, hermitian=True)
        candidate = normalized_count_geometry(weights)
        simple = normalized_count_geometry(counts + 0.5)
        error = float(np.max(np.abs(closed - pseudoinverse)))
        assert error < 1e-9
        assert np.allclose(candidate @ np.ones(10), 0, atol=1e-10)
        assert abs(np.trace(candidate) - 9) < 1e-10
        assert np.linalg.eigvalsh(candidate)[1] > 0
        rows.append({
            "synthetic_counts": counts.tolist(),
            "maximum_absolute_pseudoinverse_error": error,
            "relative_matrix_difference_from_simple_count_ridge": float(
                np.linalg.norm(candidate - simple) / np.linalg.norm(candidate)
            ),
            "trace": float(np.trace(candidate)),
            "minimum_positive_centered_eigenvalue": float(np.linalg.eigvalsh(candidate)[1]),
        })
    report = {
        "status": "PASS",
        "scope": "Synthetic count algebra only, not classifier performance evidence",
        "new_fl_fits": 0,
        "official_test_access": False,
        "numpy_version": np.__version__,
        "rows": rows,
    }
    output = Path(__file__).with_name("GEOMETRY_ALGEBRA_FIXTURE.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}))


if __name__ == "__main__":
    main()
