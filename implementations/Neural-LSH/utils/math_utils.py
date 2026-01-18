"""
Mathematical Utilities for Neural LSH

Distance calculations, similarity measures, and metric computations.
"""

import numpy as np
from typing import Tuple, Union
from scipy.spatial.distance import cdist, pdist, squareform


# ============================================================================
# Distance Functions
# ============================================================================

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Distance calculation functions."""
    
@staticmethod
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        x: First vector of shape (D,)
        y: Second vector of shape (D,)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


def euclidean_distances_matrix(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all points.
    
    Args:
        X: Array of shape (N, D)
        Y: Array of shape (M, D), optional. If None, compute pairwise distances within X
        
    Returns:
        Distance matrix of shape (N, M) or (N, N)
    """
    if Y is None:
        Y = X
    
    return cdist(X, Y, metric='euclidean')


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Cosine similarity (in range [-1, 1])
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    
    return np.dot(x, y) / (norm_x * norm_y)


def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """
    Compute Hamming distance between two binary vectors.
    
    Args:
        x: First binary vector
        y: Second binary vector
        
    Returns:
        Hamming distance (number of differing bits)
    """
    return np.sum(x != y)


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Manhattan distance between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Manhattan distance
    """
    return np.sum(np.abs(x - y))


# ============================================================================
# Nearest Neighbor and Search Metrics
# ============================================================================

def compute_nearest_neighbors(query: np.ndarray, dataset: np.ndarray, 
                             k: int = 1, metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors using exact search.
    
    Args:
        query: Query vector of shape (D,)
        dataset: Dataset of shape (N, D)
        k: Number of neighbors to return
        metric: Distance metric
        
    Returns:
        Tuple of (neighbor_indices, distances)
    """
    distances = cdist([query], dataset, metric=metric)[0]
    nearest_indices = np.argsort(distances)[:k]
    nearest_distances = distances[nearest_indices]
    
    return nearest_indices, nearest_distances


def compute_range_neighbors(query: np.ndarray, dataset: np.ndarray, 
                           radius: float, metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Find all neighbors within a given radius.
    
    Args:
        query: Query vector of shape (D,)
        dataset: Dataset of shape (N, D)
        radius: Search radius
        metric: Distance metric
        
    Returns:
        Tuple of (neighbor_indices, distances)
    """
    distances = cdist([query], dataset, metric=metric)[0]
    mask = distances <= radius
    
    indices = np.where(mask)[0]
    dists = distances[mask]
    
    # Sort by distance
    sorted_idx = np.argsort(dists)
    
    return indices[sorted_idx], dists[sorted_idx]


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_recall(predicted_neighbors: np.ndarray, groundtruth_neighbors: np.ndarray) -> float:
    """
    Compute Recall@N metric.
    
    Recall = |predicted ∩ groundtruth| / min(N, |groundtruth|)
    
    Args:
        predicted_neighbors: Array of predicted neighbor indices
        groundtruth_neighbors: Array of groundtruth neighbor indices
        
    Returns:
        Recall score in [0, 1]
    """
    predicted_set = set(predicted_neighbors)
    groundtruth_set = set(groundtruth_neighbors)
    
    intersection = len(predicted_set & groundtruth_set)
    denominator = min(len(predicted_neighbors), len(groundtruth_neighbors))
    
    if denominator == 0:
        return 0.0
    
    return intersection / denominator


def compute_average_accuracy_fraction(approx_distances: np.ndarray, 
                                     true_distances: np.ndarray) -> float:
    """
    Compute Average Accuracy Fraction (AF).
    
    AF = average(true_distance / approx_distance) for top-k neighbors
    
    Args:
        approx_distances: Approximate distances from LSH method
        true_distances: True distances from exact search
        
    Returns:
        Average accuracy fraction
    """
    # Avoid division by zero
    approx_distances = np.asarray(approx_distances, dtype=np.float32)
    true_distances = np.asarray(true_distances, dtype=np.float32)
    
    # Handle potential division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = true_distances / np.maximum(approx_distances, 1e-10)
        ratios = np.minimum(ratios, 1.0)  # Cap at 1.0
    
    return float(np.mean(ratios))


def compute_qps(num_queries: int, total_time_seconds: float) -> float:
    """
    Compute Queries Per Second (QPS).
    
    Args:
        num_queries: Number of queries processed
        total_time_seconds: Total time in seconds
        
    Returns:
        QPS metric
    """
    if total_time_seconds == 0:
        return float('inf')
    
    return num_queries / total_time_seconds


def compute_mrr(predicted_neighbors: np.ndarray, groundtruth_neighbors: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank_of_first_correct
    
    Args:
        predicted_neighbors: Array of predicted neighbor indices
        groundtruth_neighbors: Array of groundtruth neighbor indices
        
    Returns:
        MRR score
    """
    groundtruth_set = set(groundtruth_neighbors)
    
    for rank, neighbor in enumerate(predicted_neighbors, 1):
        if neighbor in groundtruth_set:
            return 1.0 / rank
    
    return 0.0


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_statistics(data: np.ndarray) -> dict:
    """
    Compute basic statistics on array.
    
    Args:
        data: Input array
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }


def compute_distance_distribution(distances: np.ndarray) -> dict:
    """
    Analyze distribution of distances.
    
    Args:
        distances: Array of distances
        
    Returns:
        Dictionary with distribution info
    """
    return {
        'count': len(distances),
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'median': float(np.median(distances))
    }


# ============================================================================
# Normalization and Scaling
# ============================================================================

def normalize_vectors_l2(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length (L2 norm).
    
    Args:
        vectors: Array of shape (N, D)
        
    Returns:
        L2-normalized vectors
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)


def standardize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Standardize vectors (zero mean, unit variance).
    
    Args:
        vectors: Array of shape (N, D)
        
    Returns:
        Standardized vectors
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)
    return (vectors - mean) / (std + 1e-10)