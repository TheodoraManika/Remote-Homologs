"""
Search Metrics Module
=====================
Υπολογισμός μετρικών αξιολόγησης: AF, Recall@N, QPS, true distances
"""

import numpy as np
from typing import Tuple, List
from scipy.spatial.distance import cdist


def compute_true_distances(query: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """
    Υπολογισμός πραγματικών αποστάσεων με brute force (εξαντλητική αναζήτηση).
    
    Args:
        query: Διάνυσμα query (dim,)
        dataset: Σύνολο δεδομένων (n_samples, dim)
        
    Returns:
        Πίνακας αποστάσεων (n_samples,)
    """
    return cdist([query], dataset, metric='euclidean')[0]


def compute_true_knn(true_distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Εύρεση πραγματικών k-NN βάσει αποστάσεων.
    
    Args:
        true_distances: Όλες οι αποστάσεις (n_samples,)
        k: Αριθμός k-NN
        
    Returns:
        (true_knn_indices, true_knn_distances)
    """
    sorted_indices = np.argsort(true_distances)[:k]
    sorted_distances = true_distances[sorted_indices]
    return sorted_indices, sorted_distances


def compute_range_neighbors(query: np.ndarray, dataset: np.ndarray, radius: float) -> np.ndarray:
    """
    Εύρεση όλων των σημείων εντός ακτίνας R (range search).
    
    Args:
        query: Διάνυσμα query
        dataset: Σύνολο δεδομένων
        radius: Ακτίνα αναζήτησης R
        
    Returns:
        Δείκτες σημείων εντός ακτίνας
    """
    distances = cdist([query], dataset, metric='euclidean')[0]
    return np.where(distances <= radius)[0]


def compute_accuracy_fraction(true_distance: float, approx_distance: float) -> float:
    """
    Υπολογισμός Accuracy Fraction (AF) για ένα ζεύγος거리.
    
    AF = true_distance / approx_distance
    
    Args:
        true_distance: Πραγματική απόσταση
        approx_distance: Προσεγγιστική απόσταση
        
    Returns:
        AF (capped at 1.0)
    """
    if approx_distance == 0:
        return 1.0 if true_distance == 0 else 0.0
    
    af = true_distance / approx_distance
    return min(af, 1.0)  # Cap at 1.0


def compute_average_af(approx_distances: np.ndarray, true_distances: np.ndarray) -> float:
    """
    Υπολογισμός Average Accuracy Fraction.
    
    Average AF = mean(AF) για τα k γειτόνια
    
    Args:
        approx_distances: Προσεγγιστικές αποστάσεις k-NN
        true_distances: Πραγματικές αποστάσεις k-NN
        
    Returns:
        Average AF
    """
    if len(approx_distances) == 0:
        return 0.0
    
    af_values = [compute_accuracy_fraction(true_d, approx_d) 
                 for true_d, approx_d in zip(true_distances, approx_distances)]
    
    return np.mean(af_values)


def compute_recall(predicted_indices: np.ndarray, true_indices: np.ndarray) -> float:
    """
    Υπολογισμός Recall@N.
    
    Recall@N = |predicted ∩ true| / min(N, |true|)
    
    Args:
        predicted_indices: Προβλεπόμενοι k-NN δείκτες
        true_indices: Πραγματικοί k-NN δείκτες
        
    Returns:
        Recall score στο [0, 1]
    """
    if len(predicted_indices) == 0 or len(true_indices) == 0:
        return 0.0
    
    predicted_set = set(predicted_indices)
    true_set = set(true_indices)
    
    intersection = len(predicted_set & true_set)
    denominator = min(len(predicted_indices), len(true_indices))
    
    return intersection / denominator if denominator > 0 else 0.0


def compute_qps(num_queries: int, total_time: float) -> float:
    """
    Υπολογισμός Queries Per Second (QPS).
    
    Args:
        num_queries: Αριθμός queries
        total_time: Συνολικός χρόνος σε δευτερόλεπτα
        
    Returns:
        QPS
    """
    if total_time == 0:
        return float('inf')
    return num_queries / total_time
