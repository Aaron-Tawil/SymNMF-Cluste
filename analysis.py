#!/usr/bin/env python3
"""Compare SymNMF and KMeans clustering quality using silhouette scores."""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Sequence

import numpy as np
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

import symnmf
import symnmf_c  # type: ignore

EPSILON = 1e-4
MAX_ITER = 300


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments for the analysis driver.

    Args:
        argv: A sequence of command-line arguments.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description="SymNMF vs KMeans analysis")
    parser.add_argument("k", type=int, help="Number of clusters")
    parser.add_argument("file_name", help="Path to dataset (.txt)")
    return parser.parse_args(argv)


def load_points(path: str) -> List[List[float]]:
    """Load a dataset from a file and return it as a list of lists.

    This function uses `symnmf.load_dataset` to load the data and then converts
    the resulting NumPy array to a list of lists of floats.

    Args:
        path: The path to the dataset file.

    Returns:
        A list of lists, where each inner list represents a data point.
    """

    data = symnmf.load_dataset(path)
    return data.tolist()


def run_symnmf(points: List[List[float]], k: int) -> List[int]:
    """Perform SymNMF clustering on a set of points.

    This function takes a list of data points and the number of clusters, k,
    and performs SymNMF clustering. It returns a list of cluster labels for
    each data point.

    Args:
        points: A list of data points, where each point is a list of floats.
        k: The number of clusters to form.

    Returns:
        A list of integers representing the cluster label for each data point.
    """

    symnmf.validate_k(k, len(points))
    dataset = np.asarray(points, dtype=np.float64)
    normalized = np.asarray(symnmf_c.norm(dataset), dtype=np.float64)
    initial_h = symnmf.init_H(normalized, k)
    final_h = np.asarray(symnmf_c.symnmf(initial_h.copy(), normalized), dtype=np.float64)
    return final_h.argmax(axis=1).tolist()


def silhouette_score(points: List[List[float]], labels: List[int]) -> float:
    """Return the mean silhouette score using scikit-learn's implementation.

    Args:
        points: A list of data points, where each point is a list of floats.
        labels: A list of cluster labels for each data point.

    Returns:
        The mean silhouette score for all data points.
    """

    dataset = np.asarray(points, dtype=np.float64)
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= len(points):
        raise RuntimeError("An Error Has Occurred")
    try:
        return float(
            sklearn_silhouette_score(dataset, labels=labels, metric="euclidean")
        )
    except ValueError as exc:
        raise RuntimeError("An Error Has Occurred") from exc


def _euclid(p: List[float], q: List[float]) -> float:
    """Calculate the Euclidean distance between two points.

    Args:
        p: The first point.
        q: The second point.

    Returns:
        The Euclidean distance between p and q.
    """

    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))


def kmeans(points: List[List[float]], k: int) -> List[int]:
    """Perform k-means clustering on a set of points.

    This function implements Lloyd's algorithm for k-means clustering. It takes a
    list of data points and the number of clusters, k, and returns a list of
    cluster labels for each data point.

    Args:
        points: A list of data points, where each point is a list of floats.
        k: The number of clusters to form.

    Returns:
        A list of integers representing the cluster label for each data point.
    """

    symnmf.validate_k(k, len(points))
    centroids = [point[:] for point in points[:k]]
    for _ in range(MAX_ITER):
        clusters = [[] for _ in range(k)]
        for point in points:
            idx = min(range(k), key=lambda i: _euclid(point, centroids[i]))
            clusters[idx].append(point)
        new_centroids = []
        for idx, cluster in enumerate(clusters):
            if cluster:
                acc = [0.0] * len(cluster[0])
                for point in cluster:
                    acc = [a + b for a, b in zip(acc, point)]
                new_centroids.append([a / len(cluster) for a in acc])
            else:
                new_centroids.append(centroids[idx])
        deltas = [
            _euclid(nc, oc)
            for nc, oc in zip(new_centroids, centroids)
        ]
        centroids = new_centroids
        if max(deltas) < EPSILON:
            break
    labels = []
    for point in points:
        labels.append(min(range(k), key=lambda i: _euclid(point, centroids[i])))
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis comparing SymNMF and KMeans clustering.

    This function parses command-line arguments, loads the dataset, runs both
    SymNMF and k-means clustering algorithms, and prints their respective
    silhouette scores.

    Args:
        argv: A sequence of command-line arguments.

    Returns:
        0 on success, 1 on error.
    """

    try:
        args = parse_args(argv if argv is not None else sys.argv[1:])
        points = load_points(args.file_name)
        nmf_labels = run_symnmf(points, args.k)
        kmeans_labels = kmeans(points, args.k)
        nmf_score = silhouette_score(points, nmf_labels)
        kmeans_score = silhouette_score(points, kmeans_labels)
    except Exception:
        print("An Error Has Occurred")
        return 1

    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
