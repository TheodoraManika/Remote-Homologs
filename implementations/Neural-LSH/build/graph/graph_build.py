"""
Graph Building Module for Neural LSH

Constructs k-NN similarity graphs for SIFT and MNIST datasets.
The graphs are prepared for balanced partitioning using KaHIP algorithm.

Graph Construction Steps:
1. Build k-NN graph G = (P, E) where P is the dataset and E are k-nearest edges
2. Symmetrize: {i, j} ∈ E ⇔ j ∈ NNk(i) or i ∈ NNk(j)
3. Assign weights:
   - wij = 2 if i ↔ j (mutual neighbors)
   - wij = 1 if one-sided neighbor
   - wij = 0 otherwise (no edge)

Output format: KaHIP binary graph format suitable for partitioning
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Set, Dict
import struct
from tqdm import tqdm

# Import dataset parsers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataparser.sift_parser import SIFTParser
from dataparser.mnist_parser import MNISTParser
from utils.math_utils import euclidean_distances_matrix


def get_graph_cache_path(data_type: str, k: int) -> Path:
    """
    Get the cache file path for a k-NN graph.
    
    Args:
        data_type: 'sift' or 'mnist'
        k: Number of nearest neighbors
        
    Returns:
        Path to cached graph file
    """
    cache_dir = Path('cache/knn_graphs')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{data_type}_knn{k}.bin"


def graph_cache_exists(data_type: str, k: int) -> bool:
    """Check if a graph with the same type and k value exists."""
    cache_path = get_graph_cache_path(data_type, k)
    return cache_path.exists()


class GraphBuilder:
    """Base class for k-NN graph construction."""
    
    def __init__(self, data: np.ndarray, k: int = 10, metric: str = 'euclidean'):
        """
        Initialize graph builder.
        
        Args:
            data: Dataset array of shape (N, D)
            k: Number of nearest neighbors
            metric: Distance metric (currently only 'euclidean' supported)
        """
        self.data = data
        self.k = k
        self.metric = metric
        self.n_points = data.shape[0]
        
    def build_knn_graph(self) -> Tuple[Dict[int, Set[int]], np.ndarray]:
        """
        Build directed k-NN graph using memory-efficient sklearn implementation.
        
        Returns:
            Tuple of:
            - adjacency_dict: Dictionary mapping point index to set of k-NN indices
            - distances: Pairwise distances matrix (or None if too large)
        """
        from sklearn.neighbors import NearestNeighbors
        
        print(f"Building k-NN graph with k={self.k} using sklearn (memory efficient)...")
        
        # Use sklearn's efficient KNN search (doesn't materialize full distance matrix)
        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(self.data)
        distances_knn, indices = nbrs.kneighbors(self.data)
        
        # Build directed k-NN adjacency (skip self at index 0)
        adjacency_dict = {}
        for i in tqdm(range(self.n_points), desc="k-NN graph"):
            # indices[i, 0] is self, so take indices[i, 1:k+1]
            nearest_indices = indices[i, 1:self.k+1]
            adjacency_dict[i] = set(nearest_indices)
        
        # Return adjacency dict and None for distances (not needed for memory efficiency)
        return adjacency_dict, None
    
    def symmetrize_graph(self, adjacency_dict: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
        """
        Convert directed k-NN graph to undirected via symmetrization.
        
        Symmetrization rule: {i, j} ∈ E ⇔ j ∈ NNk(i) or i ∈ NNk(j)
        
        Args:
            adjacency_dict: Directed k-NN adjacency dictionary
            
        Returns:
            Symmetrized adjacency dictionary
        """
        print("Symmetrizing graph...")
        
        symmetric_adj = {i: set() for i in range(self.n_points)}
        
        for i in tqdm(range(self.n_points), desc="Symmetrization"):
            # Add neighbors of i
            symmetric_adj[i].update(adjacency_dict[i])
            
            # Add points that have i as neighbor
            for j in adjacency_dict[i]:
                symmetric_adj[j].add(i)
        
        return symmetric_adj
    
    def assign_edge_weights(self, 
                           knn_adj: Dict[int, Set[int]], 
                           sym_adj: Dict[int, Set[int]]) -> Dict[Tuple[int, int], int]:
        """
        Assign weights to edges based on relationship type.
        
        Weight assignment:
        - wij = 2 if i ↔ j (mutual neighbors in k-NN)
        - wij = 1 if one-sided (only in symmetrized graph)
        - wij = 0 otherwise
        
        Args:
            knn_adj: Original directed k-NN adjacency
            sym_adj: Symmetrized adjacency
            
        Returns:
            Dictionary mapping (i, j) tuples to weights
        """
        print("Assigning edge weights...")
        
        edge_weights = {}
        
        for i in tqdm(range(self.n_points), desc="Edge weighting"):
            for j in sym_adj[i]:
                # Ensure i < j to avoid duplicate edges
                if i < j:
                    # Check if mutual neighbors (both directions in k-NN)
                    is_mutual = (j in knn_adj[i]) and (i in knn_adj[j])
                    weight = 2 if is_mutual else 1
                    edge_weights[(i, j)] = weight
        
        return edge_weights
    
    def to_csr_format(self,
                     sym_adj: Dict[int, Set[int]], 
                     edge_weights: Dict[Tuple[int, int], int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert graph to CSR (Compressed Sparse Row) format for KaHIP.
        
        Returns:
            Tuple of (xadj, adjncy, adjcwt, vwgt) arrays:
            - xadj: Row pointers (cumulative neighbor counts)
            - adjncy: Neighbor indices
            - adjcwt: Edge weights
            - vwgt: Vertex weights (all 1)
        """
        print("Converting to CSR format...")
        
        # Build adjacency lists with weights for each vertex
        vertex_neighbors = {i: [] for i in range(self.n_points)}
        vertex_weights = {i: [] for i in range(self.n_points)}
        
        for (i, j), weight in edge_weights.items():
            vertex_neighbors[i].append(j)
            vertex_weights[i].append(weight)
            vertex_neighbors[j].append(i)
            vertex_weights[j].append(weight)
        
        # Sort neighbors for consistency
        for i in range(self.n_points):
            if vertex_neighbors[i]:
                sorted_idx = np.argsort(vertex_neighbors[i])
                vertex_neighbors[i] = [vertex_neighbors[i][idx] for idx in sorted_idx]
                vertex_weights[i] = [vertex_weights[i][idx] for idx in sorted_idx]
        
        # Build CSR arrays
        xadj = [0]
        adjncy = []
        adjcwt = []
        
        for i in range(self.n_points):
            neighbors = vertex_neighbors[i]
            weights = vertex_weights[i]
            
            adjncy.extend(neighbors)
            adjcwt.extend(weights)
            xadj.append(len(adjncy))
        
        xadj = np.array(xadj, dtype=np.int32)
        adjncy = np.array(adjncy, dtype=np.int32)
        adjcwt = np.array(adjcwt, dtype=np.int32)
        vwgt = np.ones(self.n_points, dtype=np.int32)
        
        print(f"  ✓ CSR format: xadj={len(xadj)}, adjncy={len(adjncy)}, adjcwt={len(adjcwt)}")
        print(f"    Total edges (undirected): {len(adjcwt) // 2}")
        print(f"    Average degree: {len(adjncy) / self.n_points:.2f}")
        
        return xadj, adjncy, adjcwt, vwgt


def build_sift_graph(dataset_path: str = "dataset/sift_base.fvecs",
                     output_path: str = None,
                     k: int = 10) -> None:
    """
    Build and save k-NN graph for SIFT dataset.
    
    Args:
        dataset_path: Path to SIFT base vectors file
        output_path: Path to save the graph file (uses default if None)
        k: Number of nearest neighbors for graph construction
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = str(get_graph_cache_path('sift', k))
    
    print("\n" + "="*70)
    print(f"Building k-NN Graph for SIFT Dataset (k={k})")
    print("="*70)
    
    # Check if graph already exists
    if graph_cache_exists('sift', k):
        print(f"✓ Graph already exists at {output_path}")
        print("Skipping build.")
        return
    
    # Load SIFT dataset
    print(f"Loading SIFT dataset from {dataset_path}...")
    parser = SIFTParser()
    sift_data = parser.read_fvecs(dataset_path)
    print(f"✓ Loaded SIFT data: shape={sift_data.shape}, dtype={sift_data.dtype}")
    
    # Build graph
    graph_builder = GraphBuilder(sift_data, k=k, metric='euclidean')
    knn_adj, distances = graph_builder.build_knn_graph()
    sym_adj = graph_builder.symmetrize_graph(knn_adj)
    edge_weights = graph_builder.assign_edge_weights(knn_adj, sym_adj)
    graph_builder.save_graph_kahip_format(sym_adj, edge_weights, output_path)
    
    print("="*70 + "\n")


def build_mnist_graph(dataset_path: str = "dataset/train-images.idx3-ubyte",
                      output_path: str = None,
                      k: int = 10) -> None:
    """
    Build and save k-NN graph for MNIST dataset.
    
    Args:
        dataset_path: Path to MNIST training images file
        output_path: Path to save the graph file (uses default if None)
        k: Number of nearest neighbors for graph construction
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = str(get_graph_cache_path('mnist', k))
    
    print("\n" + "="*70)
    print(f"Building k-NN Graph for MNIST Dataset (k={k})")
    print("="*70)
    
    # Check if graph already exists
    if graph_cache_exists('mnist', k):
        print(f"✓ Graph already exists at {output_path}")
        print("Skipping build.")
        return
    print("Building k-NN Graph for MNIST Dataset")
    print("="*70)
    
    # Load MNIST dataset
    print(f"Loading MNIST dataset from {dataset_path}...")
    parser = MNISTParser()
    mnist_images = parser.read_images(dataset_path)
    print(f"✓ Loaded MNIST data: shape={mnist_images.shape}")
    
    # Flatten and normalize images
    mnist_data = parser.flatten_images(mnist_images)
    mnist_data = parser.normalize_images(mnist_data)
    print(f"✓ Preprocessed MNIST data: shape={mnist_data.shape}, dtype={mnist_data.dtype}")
    
    # Build graph
    graph_builder = GraphBuilder(mnist_data, k=k, metric='euclidean')
    knn_adj, distances = graph_builder.build_knn_graph()
    sym_adj = graph_builder.symmetrize_graph(knn_adj)
    edge_weights = graph_builder.assign_edge_weights(knn_adj, sym_adj)
    graph_builder.save_graph_kahip_format(sym_adj, edge_weights, output_path)
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build k-NN graphs for SIFT and MNIST datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python graph_build.py --dataset sift --knn 15
  python graph_build.py --dataset mnist --knn 10
  python graph_build.py --dataset both --knn 12
        """
    )
    
    parser.add_argument('--dataset', type=str, choices=['sift', 'mnist', 'both'],
                       default='both',
                       help='Dataset to build graph for (default: both)')
    parser.add_argument('--knn', type=int, default=10,
                       help='Number of nearest neighbors (default: 10)')
    parser.add_argument('--sift-input', type=str, default='dataset/sift_base.fvecs',
                       help='Path to SIFT base file (default: dataset/sift_base.fvecs)')
    parser.add_argument('--sift-output', type=str, default=None,
                       help='Output path for SIFT graph (default: cache/knn_graphs/sift_knnK.bin where K is --knn)')
    parser.add_argument('--mnist-input', type=str, default='dataset/train-images.idx3-ubyte',
                       help='Path to MNIST images file (default: dataset/train-images.idx3-ubyte)')
    parser.add_argument('--mnist-output', type=str, default=None,
                       help='Output path for MNIST graph (default: cache/knn_graphs/mnist_knnK.bin where K is --knn)')
    
    args = parser.parse_args()
    
    if args.dataset in ['sift', 'both']:
        build_sift_graph(
            dataset_path=args.sift_input,
            output_path=args.sift_output,
            k=args.knn
        )
    
    if args.dataset in ['mnist', 'both']:
        build_mnist_graph(
            dataset_path=args.mnist_input,
            output_path=args.mnist_output,
            k=args.knn
        )
