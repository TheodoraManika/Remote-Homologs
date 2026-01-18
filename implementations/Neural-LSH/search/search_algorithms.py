"""
Search Algorithm Module
=======================
Core search operations: prediction, bucket selection, candidate collection, and exact search.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict
import time
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from build.index_model.mlp_model import NeuralLSH_MLP
from utils.math_utils import compute_nearest_neighbors
from search.search_metrics import (
    compute_true_distances,
    compute_true_knn,
    compute_range_neighbors,
    compute_average_af,
    compute_recall,
    compute_qps
)


def get_brute_force_cache_path(data_type: str, n_queries: int, n_points: int) -> Path:
    """
    Get path to brute force cache file.
    
    Args:
        data_type: 'sift' or 'mnist'
        n_queries: Number of queries
        n_points: Number of data points
        
    Returns:
        Path to cache file
    """
    cache_dir = Path.cwd() / 'cache' / 'brute_force'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"brute_force_{data_type}_q{n_queries}_n{n_points}.npz"


def load_brute_force_cache(data_type: str, n_queries: int, n_points: int) -> Tuple[np.ndarray, bool]:
    """
    Load cached brute force distances if they exist.
    
    Args:
        data_type: 'sift' or 'mnist'
        n_queries: Number of queries
        n_points: Number of data points
        
    Returns:
        Tuple[distances_array, cache_exists]: Pre-computed distances or None, and whether cache was loaded
    """
    cache_path = get_brute_force_cache_path(data_type, n_queries, n_points)
    print(f"  Looking for cache at: {cache_path}")
    
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            distances = cached['distances']
            print(f"  ✓ Loaded cached brute force distances from {cache_path.name}")
            return distances, True
        except Exception as e:
            print(f"  ⚠ Error loading cache: {e}")
            return None, False
    
    return None, False


def save_brute_force_cache(distances: np.ndarray, data_type: str, n_queries: int, n_points: int) -> None:
    """
    Save brute force distances to cache file.
    
    Args:
        distances: Pre-computed distance matrix (n_queries, n_points)
        data_type: 'sift' or 'mnist'
        n_queries: Number of queries
        n_points: Number of data points
    """
    cache_path = get_brute_force_cache_path(data_type, n_queries, n_points)
    
    try:
        np.savez(cache_path, distances=distances)
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Cached brute force distances to {cache_path.name} ({cache_size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ⚠ Error saving cache: {e}")



def predict_probabilities(model: NeuralLSH_MLP, query: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Βήμα 1 - Πρόβλεψη: Υπολογισμός M(q) = softmax(fθ(q))
    
    Args:
        model: Το εκπαιδευμένο μοντέλο MLP
        query: Διάνυσμα query (dim,)
        device: Συσκευή (cpu/cuda)
        
    Returns:
        numpy array με τις πιθανότητες για κάθε bucket (K,)
    """
    # Μετατροπή σε tensor και προσθήκη batch dimension
    query_tensor = torch.FloatTensor(query).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(query_tensor)
        probabilities = F.softmax(logits, dim=1)
    
    return probabilities.cpu().numpy()[0]


def select_top_buckets(probabilities: np.ndarray, T: int) -> List[int]:
    """
    Βήμα 2 - Επιλογή Μερών (Multi-Probe): Εντοπισμός T buckets με υψηλότερη πιθανότητα
    
    Args:
        probabilities: numpy array με τις πιθανότητες για κάθε bucket
        T: Αριθμός top buckets να επιλεγούν
        
    Returns:
        List με τα IDs των top-T buckets
    """
    # Ταξινόμηση κατά φθίνουσα σειρά πιθανότητας
    top_bucket_indices = np.argsort(probabilities)[::-1][:T]
    
    return list(top_bucket_indices)


def collect_candidates(index_data: Dict, bucket_ids: List[int]) -> np.ndarray:
    """
    Βήμα 3 - Συλλογή Υποψηφίων: Συλλογή σημείων που ανήκουν στα T μέρη
    χρησιμοποιώντας το "inverted file"
    
    Args:
        index_data: Δεδομένα ευρετηρίου που περιέχουν buckets
        bucket_ids: List με τα IDs των buckets από τα οποία θα συλλέξουμε υποψηφίους
        
    Returns:
        numpy array με τους δείκτες των υποψηφίων
    """
    candidate_indices = []
    
    # Access inverted index directly
    inverted_index = index_data['inverted_index']['inverted_index']
    
    for bucket_id in bucket_ids:
        if bucket_id in inverted_index:
            indices = inverted_index[bucket_id]
            candidate_indices.extend(indices)
    
    return np.array(candidate_indices)


def exact_search(query: np.ndarray, 
                 candidates: np.ndarray,
                 candidate_vectors: np.ndarray,
                 k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Βήμα 4 - Ακριβής Αναζήτηση: Εξαντλητική αναζήτηση με Ευκλείδεια μετρική
    στο σύνολο των υποψηφίων
    
    Args:
        query: Διάνυσμα query (dim,)
        candidates: numpy array με τους δείκτες των υποψηφίων
        candidate_vectors: numpy array με τα διανύσματα των υποψηφίων (n_candidates, dim)
        k: Αριθμός πλησιέστερων γειτόνων
        
    Returns:
        Tuple[neighbor_indices, distances]: Δείκτες και αποστάσεις των k-NN
    """
    if len(candidate_vectors) == 0:
        # Περίπτωση: Δεν υπάρχουν υποψήφιοι
        return np.array([]), np.array([])
    
    # Limit candidates to speed up search (max 10k candidates per query)
    MAX_CANDIDATES = 10000
    if len(candidate_vectors) > MAX_CANDIDATES:
        sample_indices = np.random.choice(len(candidate_vectors), MAX_CANDIDATES, replace=False)
        candidate_vectors = candidate_vectors[sample_indices]
        candidates = candidates[sample_indices]
    
    # Χρήση utility function για υπολογισμό nearest neighbors
    # από το math_utils module
    neighbor_indices, neighbor_distances = compute_nearest_neighbors(
        query=query,
        dataset=candidate_vectors,
        k=min(k, len(candidate_vectors)),
        metric='euclidean'
    )
    
    # Επιστροφή των δεικτών των πραγματικών σημείων στο αρχικό dataset
    actual_neighbor_indices = candidates[neighbor_indices]
    
    return actual_neighbor_indices, neighbor_distances


def search_single_query(model: NeuralLSH_MLP,
                       query: np.ndarray,
                       index_data: Dict,
                       dataset: np.ndarray,
                       device: torch.device,
                       k: int,
                       T: int,
                       radius: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Αναζήτηση για ένα μοναδικό ερώτημα.
    Εκτελεί όλα τα 4 βήματα: Πρόβλεψη → Επιλογή Μερών → Συλλογή → Ακριβής Αναζήτηση
    Και υπολογίζει πραγματικές αποστάσεις με brute force
    
    Args:
        model: Το εκπαιδευμένο μοντέλο
        query: Διάνυσμα query
        index_data: Δεδομένα ευρετηρίου
        dataset: Πλήρες σύνολο δεδομένων (για πραγματικές αποστάσεις)
        device: Συσκευή (cpu/cuda)
        k: Αριθμός k-NN
        T: Αριθμός top buckets
        radius: Ακτίνα range search
        
    Returns:
        Tuple[neighbor_indices, distances, result_dict]
    """
    # Χρονομέτρηση προσεγγιστικής αναζήτησης
    approx_start = time.time()
    
    # Βήμα 1: Πρόβλεψη
    probabilities = predict_probabilities(model, query, device)
    
    # Βήμα 2: Επιλογή Μερών
    top_buckets = select_top_buckets(probabilities, T)
    
    # Βήμα 3: Συλλογή Υποψηφίων
    candidate_indices = collect_candidates(index_data, top_buckets)
    
    # Λήψη των διανυσμάτων των υποψηφίων
    candidate_vectors = dataset[candidate_indices]
    
    # Βήμα 4: Ακριβής Αναζήτηση (πάνω στους υποψηφίους)
    neighbor_indices, neighbor_distances = exact_search(
        query, candidate_indices, candidate_vectors, k
    )
    
    approx_time = time.time() - approx_start
    
    # =====================================================================
    # Υπολογισμός πραγματικών αποστάσεων με brute force (εξαντλητική αναζήτηση)
    # =====================================================================
    true_start = time.time()
    true_all_distances = compute_true_distances(query, dataset)
    true_knn_indices, true_knn_distances = compute_true_knn(true_all_distances, k)
    true_time = time.time() - true_start
    
    # Εύρεση range neighbors
    range_neighbor_indices = compute_range_neighbors(query, dataset, radius)
    
    # =====================================================================
    # Υπολογισμός μετρικών
    # =====================================================================
    
    # Εύρεση πραγματικών αποστάσεων για τους απόρριμμα neighbors
    true_distances_for_neighbors = true_all_distances[neighbor_indices]
    
    # Average Accuracy Fraction
    avg_af = compute_average_af(neighbor_distances, true_distances_for_neighbors)
    
    # Recall@N
    recall = compute_recall(neighbor_indices, true_knn_indices)
    
    # QPS θα υπολογιστεί αργότερα για όλα τα queries
    
    # Δημιουργία αποτελέσματος με πλήρη πληροφορίες
    result = {
        'query_id': None,  # Θα οριστεί από τη συνάρτηση κλήσης
        'neighbor_indices': neighbor_indices,
        'neighbor_distances': neighbor_distances,
        'true_distances': true_distances_for_neighbors,
        'range_neighbors': range_neighbor_indices,
        'avg_af': avg_af,
        'recall': recall,
        'qps': 0.0,  # Θα υπολογιστεί μετά
        't_approx': approx_time,
        't_true': true_time,
        'n_candidates': len(candidate_indices),
        'top_buckets': top_buckets,
        'k_found': len(neighbor_indices)
    }
    
    return neighbor_indices, neighbor_distances, result


def search_single_query_with_cache(model: NeuralLSH_MLP,
                                  query: np.ndarray,
                                  index_data: Dict,
                                  dataset: np.ndarray,
                                  device: torch.device,
                                  k: int,
                                  T: int,
                                  radius: float,
                                  true_all_distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Αναζήτηση για ένα μοναδικό ερώτημα με προ-υπολογισμένες αποστάσεις (cached).
    Όπως search_single_query αλλά χρησιμοποιεί cached ground truth distances.
    
    Args:
        model: Το εκπαιδευμένο μοντέλο
        query: Διάνυσμα query
        index_data: Δεδομένα ευρετηρίου
        dataset: Πλήρες σύνολο δεδομένων
        device: Συσκευή (cpu/cuda)
        k: Αριθμός nearest neighbors
        T: Αριθμός top buckets να εξερευνηθούν
        radius: Ακτίνα για range search
        true_all_distances: Pre-computed true distances (cached)
        
    Returns:
        Tuple[neighbor_indices, neighbor_distances, result_dict]
    """
    
    # Βήμα 1: Πρόβλεψη
    approx_start = time.time()
    probabilities = predict_probabilities(model, query, device)
    
    # Βήμα 2: Επιλογή Μερών
    top_buckets = select_top_buckets(probabilities, T)
    
    # Βήμα 3: Συλλογή Υποψηφίων
    candidate_indices = collect_candidates(index_data, top_buckets)
    
    # Λήψη των διανυσμάτων των υποψηφίων
    candidate_vectors = dataset[candidate_indices]
    
    # Βήμα 4: Ακριβής Αναζήτηση (πάνω στους υποψηφίους)
    neighbor_indices, neighbor_distances = exact_search(
        query, candidate_indices, candidate_vectors, k
    )
    
    approx_time = time.time() - approx_start
    
    # =====================================================================
    # USE CACHED TRUE DISTANCES (no recomputation!)
    # =====================================================================
    true_start = time.time()
    true_knn_indices, true_knn_distances = compute_true_knn(true_all_distances, k)
    true_time = time.time() - true_start
    
    # Εύρεση range neighbors
    range_neighbor_indices = compute_range_neighbors(query, dataset, radius)
    
    # =====================================================================
    # Υπολογισμός μετρικών
    # =====================================================================
    
    # Εύρεση πραγματικών αποστάσεων για τους απόρριμμα neighbors
    true_distances_for_neighbors = true_all_distances[neighbor_indices]
    
    # Average Accuracy Fraction
    avg_af = compute_average_af(neighbor_distances, true_distances_for_neighbors)
    
    # Recall@N
    recall = compute_recall(neighbor_indices, true_knn_indices)
    
    # QPS θα υπολογιστεί αργότερα για όλα τα queries
    
    # Δημιουργία αποτελέσματος με πλήρη πληροφορίες
    result = {
        'query_id': None,  # Θα οριστεί από τη συνάρτηση κλήσης
        'neighbor_indices': neighbor_indices,
        'neighbor_distances': neighbor_distances,
        'true_distances': true_distances_for_neighbors,
        'range_neighbors': range_neighbor_indices,
        'avg_af': avg_af,
        'recall': recall,
        'qps': 0.0,  # Θα υπολογιστεί μετά
        't_approx': approx_time,
        't_true': true_time,
        'n_candidates': len(candidate_indices),
        'top_buckets': top_buckets,
        'k_found': len(neighbor_indices)
    }
    
    return neighbor_indices, neighbor_distances, result


def search_all_queries(model: NeuralLSH_MLP,
                      queries: np.ndarray,
                      index_data: Dict,
                      dataset: np.ndarray,
                      device: torch.device,
                      k: int,
                      T: int,
                      radius: float,
                      data_type: str = 'sift') -> List[Dict]:
    """
    Αναζήτηση για όλα τα queries με υπολογισμό μετρικών.
    
    Args:
        model: Το εκπαιδευμένο μοντέλο
        queries: numpy array με τα queries (n_queries, dim)
        index_data: Δεδομένα ευρετηρίου
        dataset: Πλήρες σύνολο δεδομένων
        device: Συσκευή
        k: Αριθμός k-NN
        T: Αριθμός top buckets
        radius: Ακτίνα range search
        data_type: 'sift' or 'mnist' for cache identification
        
    Returns:
        List[Dict]: Αποτελέσματα για κάθε query
    """
    results = []
    
    print(f"\nΑναζήτηση {len(queries)} queries...")
    print(f"  - k (neighbors): {k}")
    print(f"  - T (top buckets): {T}")
    print(f"  - R (search radius): {radius}")
    
    # Try to load brute force cache from disk
    print(f"\n  ⏳ Loading/computing ground truth distances...")
    cache_start = time.time()
    
    true_all_distances_cache, cache_loaded = load_brute_force_cache(
        data_type, len(queries), len(dataset)
    )
    
    # If cache not found, compute and save
    if not cache_loaded:
        print(f"  ⏳ Computing ground truth distances (this will be cached for future runs)...")
        true_all_distances_cache = np.zeros((len(queries), len(dataset)), dtype=np.float32)
        for i, query in enumerate(queries):
            true_all_distances_cache[i] = compute_true_distances(query, dataset)
            if (i + 1) % max(1, len(queries) // 10) == 0:
                print(f"    • Computed {i + 1}/{len(queries)} queries")
        
        # Save to disk for future runs
        save_brute_force_cache(true_all_distances_cache, data_type, len(queries), len(dataset))
    
    cache_time = time.time() - cache_start
    print(f"  ✓ Ground truth distances ready in {cache_time:.2f}s")
    
    # Συνολικός χρόνος για QPS
    total_search_time = 0
    
    for i, query in enumerate(queries):
        query_start = time.time()
        
        # Use cached ground truth distances instead of computing them again
        neighbor_indices, neighbor_distances, result = search_single_query_with_cache(
            model, query, index_data, dataset, device, k, T, radius,
            true_all_distances=true_all_distances_cache[i]
        )
        
        query_time = time.time() - query_start
        total_search_time += query_time
        
        result['query_id'] = i
        results.append(result)
        
        # Εκτύπωση προόδου κάθε 10% των queries
        if (i + 1) % max(1, len(queries) // 10) == 0:
            print(f"  Ολοκληρώθηκαν {i + 1}/{len(queries)} queries")
    
    # Υπολογισμός QPS για όλα τα queries
    qps = compute_qps(len(queries), total_search_time)
    for result in results:
        result['qps'] = qps
    
    return results
