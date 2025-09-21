#!/usr/bin/env python3
"""Compare SymNMF and KMeans clustering quality via silhouette score."""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Sequence

import numpy as np

import symnmf
import symnmf_c  # type: ignore

EPSILON = 1e-4
MAX_ITER = 300


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments for the analysis driver."""

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

    dataset = np.asarray(points, dtype=np.float64)
    normalized = np.asarray(symnmf_c.norm(dataset), dtype=np.float64)
    initial_h = symnmf.init_H(normalized, k)
    final_h = np.asarray(symnmf_c.symnmf(initial_h.copy(), normalized), dtype=np.float64)
    return final_h.argmax(axis=1).tolist()


def silhouette_score(points: List[List[float]], labels: List[int]) -> float:
    """Calculate the mean silhouette score for a clustering.

    The silhouette score measures how similar a data point is to its own cluster
    compared to other clusters. The score ranges from -1 to 1, where a high
    value indicates that the object is well matched to its own cluster and
    poorly matched to neighboring clusters.

    Args:
        points: A list of data points, where each point is a list of floats.
        labels: A list of cluster labels for each data point.

    Returns:
        The mean silhouette score for all data points.
    """

    clusters: List[List[List[float]]] = []
    k = max(labels) + 1
    for _ in range(k):
        clusters.append([])
    for point, label in zip(points, labels):
        clusters[label].append(point)

    scores = []
    for idx, point in enumerate(points):
        a = _mean_intra_distance(point, clusters[labels[idx]])
        b = _nearest_cluster_distance(point, labels[idx], clusters)
        denominator = max(a, b)
        scores.append(0.0 if denominator == 0.0 else (b - a) / denominator)
    return sum(scores) / len(scores)


def _mean_intra_distance(point: List[float], cluster: List[List[float]]) -> float:
    """Return the mean distance from a point to others in its cluster."""

    if len(cluster) <= 1:
        return 0.0
    acc = 0.0
    for other in cluster:
        if other is point:
            continue
        acc += _euclid(point, other)
    return acc / (len(cluster) - 1)


def _nearest_cluster_distance(point: List[float], label: int, clusters: List[List[List[float]]]) -> float:
    """Return the smallest average distance to any other cluster."""

    best = None
    for idx, cluster in enumerate(clusters):
        if idx == label or not cluster:
            continue
        total = sum(_euclid(point, other) for other in cluster)
        dist = total / len(cluster)
        best = dist if best is None or dist < best else best
    return 0.0 if best is None else best


def _euclid(p: List[float], q: List[float]) -> float:
    """Euclidean distance helper."""

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
    """Entry point for the analysis script."""

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
