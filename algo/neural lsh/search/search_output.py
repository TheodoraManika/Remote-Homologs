"""
Search Output and Statistics Module
====================================
Functions for displaying results, statistics, and logging experiments.
"""

import numpy as np
from typing import List, Dict
import sys
import os
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.math_utils import compute_statistics
from utils.output import log_experiment, write_results_file, write_results_simple, save_partition_assignment


def print_summary(results: List[Dict], build_params: Dict, search_time: float = 0, 
                 num_queries: int = 0) -> None:
    """
    Εκτύπωση σύνοψης των αποτελεσμάτων με στατιστικά και μετρικές.
    
    Args:
        results: List με τα αποτελέσματα
        build_params: Παράμετροι κατασκευής του ευρετηρίου
        search_time: Συνολικός χρόνος αναζήτησης σε δευτερόλεπτα
        num_queries: Συνολικός αριθμός queries
    """
    print("\n" + "=" * 80)
    print("ΣΎΝΟΨΗ ΑΠΟΤΕΛΕΣΜΆΤΩΝ ΑΝΑΖΗΤΗΣΗΣ")
    print("=" * 80)
    
    print("\nΠαράμετροι Κατασκευής Ευρετηρίου:")
    print(f"  - Data type: {build_params.get('data_type', 'N/A')}")
    print(f"  - Buckets (m): {build_params.get('parts', 'N/A')}")
    print(f"  - k-NN: {build_params.get('knn', 'N/A')}")
    print(f"  - MLP layers: {build_params.get('layers', 'N/A')}")
    
    print(f"\nΣτατιστικά Αναζήτησης ({len(results)} queries):")
    
    # Υπολογισμός στατιστικών
    candidate_counts = np.array([r['n_candidates'] for r in results])
    k_found_counts = np.array([r['k_found'] for r in results])
    
    # Χρήση utility function για υπολογισμό στατιστικών
    candidate_stats = compute_statistics(candidate_counts)
    k_found_stats = compute_statistics(k_found_counts)
    
    print(f"\nΥποψήφιοι σημεία που εξετάστηκαν:")
    print(f"  - Μέσος: {candidate_stats['mean']:.1f}")
    print(f"  - Std Dev: {candidate_stats['std']:.1f}")
    print(f"  - Min/Max: {candidate_stats['min']:.0f}/{candidate_stats['max']:.0f}")
    print(f"  - Median: {candidate_stats['median']:.1f}")
    
    print(f"\nk-NN βρέθηκαν:")
    print(f"  - Μέσος: {k_found_stats['mean']:.1f}")
    print(f"  - Std Dev: {k_found_stats['std']:.1f}")
    print(f"  - Min/Max: {k_found_stats['min']:.0f}/{k_found_stats['max']:.0f}")
    
    # Υπολογισμός μέσων μετρικών
    avg_af = np.mean([r.get('avg_af', 0.0) for r in results])
    avg_recall = np.mean([r.get('recall', 0.0) for r in results])
    avg_t_approx = np.mean([r.get('t_approx', 0.0) for r in results])
    avg_t_true = np.mean([r.get('t_true', 0.0) for r in results])
    qps = results[0].get('qps', 0.0) if len(results) > 0 else 0.0
    
    print(f"\nΜέσες Μετρικές:")
    print(f"  - Average AF: {avg_af:.6f}")
    print(f"  - Average Recall@N: {avg_recall:.6f}")
    print(f"  - QPS: {qps:.2f} queries/sec")
    print(f"  - Avg t_approx: {avg_t_approx:.6f}s")
    print(f"  - Avg t_true: {avg_t_true:.6f}s")
    
    if search_time > 0 and num_queries > 0:
        time_per_query = search_time / num_queries * 1000  # σε milliseconds
        print(f"\nΧρονοσταθμία:")
        print(f"  - Συνολικός χρόνος αναζήτησης: {search_time:.3f}s")
        print(f"  - Μέσος χρόνος ανά query: {time_per_query:.3f}ms")
    
    print("\n" + "=" * 80)


def log_search_experiment(args, results: List[Dict], search_time: float) -> None:
    """
    Logging των αποτελεσμάτων της αναζήτησης σε experiment log.
    
    Args:
        args: Command-line arguments
        results: Αποτελέσματα αναζήτησης
        search_time: Χρόνος αναζήτησης σε δευτερόλεπτα
    """
    log_experiment(
        log_path='search_experiments.log',
        experiment_name='Neural-LSH Search',
        parameters={
            'data_type': args.type,
            'index_path': args.index,
            'query_file': args.query,
            'N': args.N,
            'T': args.T,
            'range_search': args.range_search,
            'search_radius': args.R
        },
        metrics={
            'total_queries': len(results),
            'total_search_time': search_time,
            'avg_time_per_query': search_time / len(results) if len(results) > 0 else 0,
            'avg_candidates_examined': np.mean([r['n_candidates'] for r in results]) if len(results) > 0 else 0,
            'avg_k_found': np.mean([r['k_found'] for r in results]) if len(results) > 0 else 0
        }
    )


def save_results(results: list, output_path: str, method_name: str = 'Neural LSH') -> None:
    """
    Αποθήκευση αποτελεσμάτων σε αρχείο με τη μορφή που απαιτεί η εργασία.
    
    Μορφή εξόδου:
    <METHOD NAME> Neural LSH
    Query: image_id_in_query_set
    Nearest neighbor-1: image_id_in_data_set
    distanceApproximate: <double>
    distanceTrue: <double>
    ...
    R-near neighbors:
    image_id_A
    ...
    Average AF: <double>
    Recall@N: <double>
    QPS: <double>
    tApproximateAverage: <double>
    tTrueAverage: <double>
    
    Args:
        results: List με τα αποτελέσματα
        output_path: Διαδρομή αποθήκευσης
        method_name: Όνομα της μεθόδου (default: 'Neural LSH')
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Αποθήκευση σε μορφή εργασίας
    with open(output_path, 'w') as f:
        f.write(f"<METHOD NAME> {method_name}\n")
        
        for result in results:
            query_id = result['query_id']
            f.write(f"Query: {query_id}\n")
            
            # Γραφή k-NN γειτόνων
            for rank in range(len(result.get('neighbor_indices', []))):
                neighbor_id = result['neighbor_indices'][rank]
                approx_dist = result['neighbor_distances'][rank]
                true_dist = result['true_distances'][rank]
                
                f.write(f"Nearest neighbor-{rank + 1}: {neighbor_id}\n")
                f.write(f"distanceApproximate: {approx_dist:.6f}\n")
                f.write(f"distanceTrue: {true_dist:.6f}\n")
            
            # Γραφή range neighbors
            f.write("R-near neighbors:\n")
            for range_neighbor_id in result.get('range_neighbors', []):
                f.write(f"{range_neighbor_id}\n")
            
            # Γραφή μετρικών
            f.write(f"Average AF: {result.get('avg_af', 0.0):.6f}\n")
            f.write(f"Recall@N: {result.get('recall', 0.0):.6f}\n")
            f.write(f"QPS: {result.get('qps', 0.0):.6f}\n")
            f.write(f"tApproximateAverage: {result.get('t_approx', 0.0):.6f}\n")
            f.write(f"tTrueAverage: {result.get('t_true', 0.0):.6f}\n")
            f.write("\n")
    
    # Αποθήκευση ως pickle για ευκολία φόρτωσης στο Python
    with open(output_path + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nΑποτελέσματα αποθηκεύτηκαν:")
    print(f"  - Μορφή εργασίας: {output_path}")
    print(f"  - Pickle format: {output_path}.pkl")

