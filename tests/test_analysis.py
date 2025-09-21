import pytest
import tempfile
import os
import re
from analysis import silhouette_score, kmeans
from analysis import main as analysis_main

def test_silhouette_score_perfect_clustering():
    """
    Tests the silhouette_score with a perfect clustering.
    The score should be high (close to 1).
    """
    # Two distinct clusters
    points = [
        [1, 1], [1.1, 1.2], [0.9, 0.8],  # Cluster 0
        [10, 10], [10.1, 10.2], [9.9, 9.8]  # Cluster 1
    ]
    # Perfect labels
    labels = [0, 0, 0, 1, 1, 1]
    
    score = silhouette_score(points, labels)
    assert score > 0.9, "Score for a perfect clustering should be close to 1"

def test_silhouette_score_bad_clustering():
    """
    Tests the silhouette_score with a bad clustering.
    The score should be low (close to 0 or negative).
    """
    # Two distinct clusters
    points = [
        [1, 1], [1.1, 1.2], [0.9, 0.8],  # Cluster 0
        [10, 10], [10.1, 10.2], [9.9, 9.8]  # Cluster 1
    ]
    # Bad labels (mixing clusters)
    labels = [0, 1, 0, 1, 0, 1]
    
    score = silhouette_score(points, labels)
    assert score < 0.1, "Score for a bad clustering should be low"

def test_kmeans_on_simple_data():
    """
    Tests the kmeans implementation on a simple, well-defined dataset.
    It should be able to find the correct clusters.
    """
    # Two distinct clusters
    points = [
        [1, 1], [1.1, 1.2], [0.9, 0.8],
        [10, 10], [10.1, 10.2], [9.9, 9.8]
    ]
    k = 2
    
    labels = kmeans(points, k)
    
    # We can't know for sure if cluster 0 will be assigned to the first
    # group of points or the second, so we check for both possibilities.
    # The important thing is that all points in a group get the same label.
    
    label_group1 = labels[0]
    label_group2 = labels[3]
    
    assert label_group1 != label_group2
    assert all(l == label_group1 for l in labels[0:3])
    assert all(l == label_group2 for l in labels[3:6])

def test_silhouette_score_overlapping_clusters():
    """
    Tests the silhouette_score with overlapping clusters.
    The score should be positive but not high.
    """
    # Two overlapping clusters
    points = [
        [1, 1], [1.5, 1.5], [1, 2],  # Mostly cluster 0
        [2, 2], [2.5, 2.5], [2, 3]   # Mostly cluster 1
    ]
    # A reasonable clustering
    labels = [0, 0, 0, 1, 1, 1]
    
    score = silhouette_score(points, labels)
    assert 0 < score < 0.7, "Score for overlapping clusters should be modest"

def test_silhouette_score_single_cluster_data():
    """
    Tests the silhouette_score when data is one cluster but k=2.
    The score should be low.
    """
    # One cluster
    points = [
        [1, 1], [1.1, 1.2], [0.9, 0.8],
        [1.5, 1.5], [1.2, 1.3], [1.4, 1.1]
    ]
    # Forced k=2 clustering
    labels = [0, 0, 0, 1, 1, 1]
    
    score = silhouette_score(points, labels)
    assert score < 0.5, "Score for a forced split of one cluster should be low"

def test_analysis_main_integration(capsys):
    """
    Integration test for the main function of analysis.py.
    """
    # Create a temporary file with test data
    points = [
        "1.0,1.0,1.0",
        "1.1,1.2,1.3",
        "0.9,0.8,0.7",
        "10.0,10.0,10.0",
        "10.1,10.2,10.3",
        "9.9,9.8,9.7"
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as tmp:
        tmp.write("\n".join(points))
        tmp_path = tmp.name

    try:
        # Run the main function from analysis.py
        analysis_main(["2", tmp_path])
        
        # Capture the output
        captured = capsys.readouterr()
        
        # Verify the output format
        assert "nmf: " in captured.out
        assert "kmeans: " in captured.out
        
        # Verify that the scores are floating point numbers
        nmf_score_str = re.search(r"nmf: (-?\d+\.\d{4})", captured.out)
        kmeans_score_str = re.search(r"kmeans: (-?\d+\.\d{4})", captured.out)
        
        assert nmf_score_str is not None, "NMF score not found or not in correct format"
        assert kmeans_score_str is not None, "KMeans score not found or not in correct format"
        
        nmf_score = float(nmf_score_str.group(1))
        kmeans_score = float(kmeans_score_str.group(1))
        
        # For this well-separated data, scores should be high
        assert nmf_score > 0.8
        assert kmeans_score > 0.8

    finally:
        # Clean up the temporary file
        os.remove(tmp_path)