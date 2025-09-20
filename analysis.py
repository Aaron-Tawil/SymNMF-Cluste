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
    """Load the dataset via symnmf.load_dataset and return Python lists."""

    data = symnmf.load_dataset(path)
    return data.tolist()


def run_symnmf(points: List[List[float]], k: int) -> List[int]:
    """Cluster points using SymNMF and return label assignments."""

    dataset = np.asarray(points, dtype=np.float64)
    normalized = np.asarray(symnmf_c.norm(dataset), dtype=np.float64)
    initial_h = symnmf.init_H(normalized, k)
    final_h = np.asarray(symnmf_c.symnmf(initial_h.copy(), normalized), dtype=np.float64)
    return final_h.argmax(axis=1).tolist()


def silhouette_score(points: List[List[float]], labels: List[int]) -> float:
    """Compute the mean silhouette score for the given clustering."""

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
    """Run a basic Lloyd's k-means loop and return cluster labels."""

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
