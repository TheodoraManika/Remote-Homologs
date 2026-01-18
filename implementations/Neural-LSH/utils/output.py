"""
Output Utilities for Neural LSH

Functions for saving results according to assignment output format specifications.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle
import json
from datetime import datetime


def write_results_file(output_path: str, results: Dict[int, Dict], 
                      method_name: str = 'Neural LSH') -> None:
    """
    Write search results to file in assignment-required format.
    
    Output format (for nlsh_search.py -o parameter):
    <METHOD NAME> Neural LSH
    Query: <query_id>
    Nearest neighbor-1: <neighbor_id>
    distanceApproximate: <distance>
    distanceTrue: <distance>
    ...
    Nearest neighbor-N: <neighbor_id>
    distanceApproximate: <distance>
    distanceTrue: <distance>
    R-near neighbors:
    <neighbor_id_1>
    ...
    <neighbor_id_K>
    Average AF: <metric>
    Recall@N: <metric>
    QPS: <metric>
    tApproximateAverage: <time>
    tTrueAverage: <time>
    
    Args:
        output_path: Output file path (e.g., 'results.txt')
        results: Dictionary with query results
        method_name: Method name to write
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            f.write(f"<METHOD NAME> {method_name}\n")
            
            for query_id in sorted(results.keys()):
                query_results = results[query_id]
                
                f.write(f"Query: {query_id}\n")
                
                # Write nearest neighbors
                for rank, neighbor_info in enumerate(query_results.get('neighbors', []), 1):
                    neighbor_id = neighbor_info['id']
                    approx_dist = neighbor_info.get('approx_dist', 0.0)
                    true_dist = neighbor_info.get('true_dist', 0.0)
                    
                    f.write(f"Nearest neighbor-{rank}: {neighbor_id}\n")
                    f.write(f"distanceApproximate: {approx_dist:.6f}\n")
                    f.write(f"distanceTrue: {true_dist:.6f}\n")
                
                # Write range neighbors
                f.write("R-near neighbors:\n")
                for neighbor_id in query_results.get('range_neighbors', []):
                    f.write(f"{neighbor_id}\n")
                
                # Write metrics
                f.write(f"Average AF: {query_results.get('avg_af', 0.0):.6f}\n")
                f.write(f"Recall@N: {query_results.get('recall', 0.0):.6f}\n")
                f.write(f"QPS: {query_results.get('qps', 0.0):.6f}\n")
                f.write(f"tApproximateAverage: {query_results.get('t_approx', 0.0):.6f}\n")
                f.write(f"tTrueAverage: {query_results.get('t_true', 0.0):.6f}\n")
                f.write("\n")
        
        print(f"✓ Results written to {output_file}")
    except Exception as e:
        print(f"✗ Error writing results: {e}")
        raise


def write_results_simple(output_path: str, results_text: str) -> None:
    """
    Write raw results text to file.
    
    Args:
        output_path: Output file path
        results_text: Results as string
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(results_text)
    
    print(f"✓ Results written to {output_file}")


def save_partition_assignment(output_path: str, assignments: np.ndarray, 
                             point_ids: np.ndarray = None) -> None:
    """
    Save partition assignments for debugging (output from nlsh_build.py).
    
    Args:
        output_path: Path to save file
        assignments: Array of partition IDs for each point
        point_ids: Optional array of point IDs (defaults to 0, 1, 2, ...)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        if point_ids is not None:
            for pid, partition in zip(point_ids, assignments):
                f.write(f"{pid}\t{partition}\n")
        else:
            for i, partition in enumerate(assignments):
                f.write(f"{i}\t{partition}\n")
    
    print(f"✓ Partition assignments saved to {output_file}")


def log_experiment(log_path: str, experiment_name: str, parameters: Dict, 
                  metrics: Dict) -> None:
    """
    Log experiment parameters and results.
    
    Args:
        log_path: Path to log file
        experiment_name: Name of experiment
        parameters: Hyperparameters used
        metrics: Results metrics
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"{'='*70}\n")
        
        f.write("Parameters:\n")
        for key, value in parameters.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nMetrics:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\n{'='*70}\n")
    
    print(f"✓ Experiment logged to {log_file}")
