#!/usr/bin/env python3
"""Command-line interface for the SymNMF project."""

from __future__ import annotations

import argparse
import re
import sys
from typing import Sequence

try:
    import symnmf_c as _symnmf_c  # type: ignore
except ImportError:  # pragma: no cover - extension not built yet
    _symnmf_c = None

import numpy as np

class _ArgumentParser(argparse.ArgumentParser):
    """Argument parser variant that raises ValueError on failures."""

    def error(self, message: str) -> None:
        """Raise ValueError instead of exiting the interpreter."""

        raise ValueError(message)

np.random.seed(1234)

EPSILON = 1e-4
MAX_ITER = 300
_TOKEN_PATTERN = re.compile(r"[\s,]+")
_VALID_K_TYPES = (int, np.integer)


def validate_k(k: int, n_samples: int) -> None:
    """Validate that `k` is an integer satisfying 1 < k < n_samples."""

    if not isinstance(k, _VALID_K_TYPES):
        raise ValueError("k must be an integer")
    if n_samples <= 2:
        raise ValueError("dataset must contain more than two samples")
    if k <= 1 or k >= n_samples:
        raise ValueError("k must satisfy 1 < k < number of samples")


def load_dataset(path: str) -> np.ndarray:
    """Load a dataset from a file into a NumPy array.

    The file is expected to be a text file where each line represents a data point,
    and the values within the line are separated by commas or whitespace.

    Args:
        path: The path to the dataset file.

    Returns:
        A NumPy array of float64 values representing the dataset.

    Raises:
        ValueError: If the dataset is empty, or if the rows have an inconsistent
            number of columns.
    """

    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            tokens = [token for token in _TOKEN_PATTERN.split(raw_line.strip()) if token]
            if not tokens:
                continue
            row = [float(token) for token in tokens]
            if rows and len(row) != len(rows[0]):
                raise ValueError("inconsistent number of columns in dataset")
            rows.append(row)

    if not rows:
        raise ValueError("empty dataset")

    return np.asarray(rows, dtype=np.float64)


def init_H(W: np.ndarray, k: int) -> np.ndarray:
    """Initialize a non-negative matrix H for the SymNMF algorithm.

    The initialization is based on the mean of the similarity matrix W.

    Args:
        W: The similarity matrix (n x n).
        k: The number of clusters.

    Returns:
        A randomly initialized non-negative matrix H (n x k).

    Raises:
        ValueError: If W is not a square matrix, or if k is not within the
            valid range (1 < k < n).
    """

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    n = W.shape[0]
    validate_k(k, n)

    mean_value = float(np.mean(W, dtype=np.float64))
    if mean_value <= 0.0:
        upper = 0.0
    else:
        upper = 2.0 * np.sqrt(mean_value / float(k))
    return np.random.uniform(0.0, upper, size=(n, k)).astype(np.float64, copy=False)


def run(goal: str, k: int, path: str):
    """Execute a specific goal of the SymNMF algorithm.

    This function acts as a dispatcher, loading the dataset and calling the
    appropriate function from the C extension based on the specified goal.

    Args:
        goal: The goal to execute. One of "sym", "ddg", "norm", or "symnmf".
        k: The number of clusters (only used for the "symnmf" goal).
        path: The path to the dataset file.

    Returns:
        The result of the computation, which is a NumPy array.

    Raises:
        RuntimeError: If the C extension is not available.
        ValueError: If k is invalid for the "symnmf" goal.
    """

    if _symnmf_c is None:
        raise RuntimeError("symnmf_c extension is not available")

    points = load_dataset(path)

    if goal == "sym":
        return np.asarray(_symnmf_c.sym(points), dtype=np.float64)
    if goal == "ddg":
        return np.asarray(_symnmf_c.ddg(points), dtype=np.float64)
    if goal == "norm":
        return np.asarray(_symnmf_c.norm(points), dtype=np.float64)

    if goal == "symnmf":
        validate_k(k, points.shape[0])

    W = np.asarray(_symnmf_c.norm(points), dtype=np.float64)
    H0 = init_H(W, k)
    return np.asarray(_symnmf_c.symnmf(H0, W), dtype=np.float64)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Return parsed arguments for the SymNMF CLI."""

    parser = _ArgumentParser(description="SymNMF driver")
    parser.add_argument("k", type=int, help="Number of clusters")
    parser.add_argument("goal", choices=("sym", "ddg", "norm", "symnmf"))
    parser.add_argument("file_name")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the Python CLI wrapper."""

    try:
        args = parse_args(argv if argv is not None else sys.argv[1:])
        matrix = run(args.goal, args.k, args.file_name)
    except Exception:
        print("An Error Has Occurred")
        return 1

    formatted = "\n".join(
        ",".join(f"{float(value):.4f}" for value in row) for row in np.asarray(matrix)
    )
    print(formatted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
