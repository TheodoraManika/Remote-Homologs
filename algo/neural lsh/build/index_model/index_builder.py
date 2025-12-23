"""
Index Builder Module
====================
Constructs the Neural-LSH index with inverted file structure using KaHIP partitioning.
"""

import numpy as np
from typing import Dict, List, Tuple
import pickle
import kahip


class InvertedIndexBuilder:
    """Builds an inverted index structure for Neural-LSH using KaHIP partitioning."""
    
    def __init__(self, num_buckets: int, num_points: int):
        """
        Initialize the inverted index builder.
        
        Args:
            num_buckets: Number of hash buckets (partitions from KaHIP)
            num_points: Total number of data points
        """
        self.num_buckets = num_buckets
        self.num_points = num_points
        self.inverted_index = {}  # bucket_id -> list of point indices
        
    def partition_graph_with_kahip(self, xadj: np.ndarray, adjncy: np.ndarray, 
                                   adjcwt: np.ndarray = None, vwgt: np.ndarray = None,
                                   imbalance: float = 0.03, mode: int = 2, seed: int = 1) -> np.ndarray:
        """
        Partition k-NN graph using KaHIP balanced graph partitioning algorithm.
        
        Args:
            xadj: CSR format index array (list[int]) - aggregate #neighbors
            adjncy: CSR format adjacency array (list[int]) - flattened neighbor indices (0-based)
            adjcwt: Edge weights array (list[int]) - importance/penalty to cut (optional)
            vwgt: Node weights array (list[int]) - node weights, usually all 1 (optional)
            imbalance: Allowed imbalance ε for balanced partitions (default: 0.03)
            mode: Preconfigure mode: 0=FAST, 1=ECO, 2=STRONG (default: 2)
            seed: Random seed (default: 1)
            
        Returns:
            partition_vector: Array where partition_vector[i] = bucket_id for point i
        """
        n_nodes = len(xadj) - 1  # Number of nodes
        
        # Convert to appropriate types for KaHIP
        xadj = np.array(xadj, dtype=np.int32)
        adjncy = np.array(adjncy, dtype=np.int32)
        
        if adjcwt is None:
            adjcwt = np.ones(len(adjncy), dtype=np.int32)
        else:
            adjcwt = np.array(adjcwt, dtype=np.int32)
        
        if vwgt is None:
            vwgt = np.ones(n_nodes, dtype=np.int32)
        else:
            vwgt = np.array(vwgt, dtype=np.int32)
        
        print(f"\n  - Running KaHIP partitioning:")
        print(f"    • Nodes: {n_nodes}, Edges: {len(adjncy)//2}")
        print(f"    • Target buckets (m): {self.num_buckets}")
        print(f"    • Imbalance (ε): {imbalance}")
        print(f"    • Mode: {['FAST', 'ECO', 'STRONG'][mode]}")
        
        # Call KaHIP partitioning with positional arguments in correct order
        # kaffpa(vwgt, xadj, adjcwt, adjncy, nblocks, imbalance, suppress_output, seed, mode)
        edgecut, partition_vector = kahip.kaffpa(
            vwgt,                   # arg0: vertex weights
            xadj,                   # arg1: CSR index array
            adjcwt,                 # arg2: edge weights
            adjncy,                 # arg3: CSR adjacency array
            self.num_buckets,       # arg4: nblocks
            imbalance,              # arg5: imbalance factor
            False,                  # arg6: suppress_output
            seed,                   # arg7: seed
            mode                    # arg8: mode
        )
        
        partition_vector = np.array(partition_vector, dtype=np.int32)
        
        print(f"  ✓ KaHIP completed: Edge-cut = {edgecut}")
        print(f"    • Partition vector shape: {partition_vector.shape}")
        
        return partition_vector
    
    def add_point(self, bucket_id: int, point_id: int) -> None:
        """Add a point to a bucket in the inverted index."""
        if bucket_id not in self.inverted_index:
            self.inverted_index[bucket_id] = []
        self.inverted_index[bucket_id].append(point_id)
    
    def build_from_partition(self, partition_vector: np.ndarray) -> Dict:
        """
        Build inverted index from partition vector.
        
        Args:
            partition_vector: Array where partition_vector[i] = bucket_id for point i
            
        Returns:
            Dictionary containing inverted index structure with statistics
        """
        self.inverted_index = {}
        
        # Build inverted index from partition
        for point_id, bucket_id in enumerate(partition_vector):
            bucket_id_int = int(bucket_id)
            self.add_point(bucket_id_int, point_id)
        
        # Compute statistics
        bucket_sizes = np.array([len(points) for points in self.inverted_index.values()])
        
        return {
            'inverted_index': self.inverted_index,
            'num_buckets': self.num_buckets,
            'num_points': self.num_points,
            'bucket_stats': {
                'mean_size': float(bucket_sizes.mean()),
                'min_size': int(bucket_sizes.min()),
                'max_size': int(bucket_sizes.max()),
                'std_size': float(bucket_sizes.std())
            }
        }
    
    def get_bucket(self, bucket_id: int) -> List[int]:
        """Get all point indices in a specific bucket."""
        return self.inverted_index.get(bucket_id, [])
    
    def query_buckets(self, query_partition: int, num_top_buckets: int = 5) -> List[int]:
        """
        Get candidate points from buckets (for range queries).
        
        Args:
            query_partition: Bucket ID that query belongs to
            num_top_buckets: Number of buckets to search
            
        Returns:
            List of candidate point indices
        """
        candidates = []
        
        # Start with query's own bucket
        if query_partition in self.inverted_index:
            candidates.extend(self.inverted_index[query_partition])
        
        # Expand to nearby buckets if needed
        if len(candidates) < 100:
            for bucket_id, points in self.inverted_index.items():
                if bucket_id != query_partition and len(candidates) < 1000:
                    candidates.extend(points)
        
        return candidates

class NeuralLSHIndexFile:
    """Manages the complete index file structure (model + inverted file)."""
    
    def __init__(self):
        """Initialize index file structure."""
        self.model_state_dict = None
        self.model_config = None
        self.inverted_index_data = None
        self.data_metadata = None
        self.partition_data = None
        self.kahip_config = None
    
    def save(self, filepath: str) -> None:
        """
        Save the complete index (model + inverted file) to a binary file.
        
        Index file contains:
        - Trained MLP model weights
        - Model configuration
        - Inverted index structure (bucket -> point mappings)
        - Partition assignments from KaHIP
        - Data metadata
        
        Args:
            filepath: Path to save the .bin index file
        """
        index_data = {
            'model_state_dict': self.model_state_dict,
            'model_config': self.model_config,
            'inverted_index': self.inverted_index_data,
            'data_metadata': self.data_metadata,
            'partition_data': self.partition_data,
            'kahip_config': self.kahip_config,
            'version': '1.0',
            'format': 'inverted_file'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n✓ Index saved to: {filepath}")
        print(f"  - Model state dict: {self.model_config}")
        print(f"  - Inverted index buckets: {self.inverted_index_data['num_buckets']}")
        print(f"  - Data points indexed: {self.data_metadata['num_points']}")
    
    def load(self, filepath: str) -> Dict:
        """
        Load the complete index from a binary file.
        
        Args:
            filepath: Path to the .bin index file
            
        Returns:
            Dictionary with all index components
        """
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.model_state_dict = index_data.get('model_state_dict')
        self.model_config = index_data.get('model_config')
        self.inverted_index_data = index_data.get('inverted_index')
        self.data_metadata = index_data.get('data_metadata')
        self.partition_data = index_data.get('partition_data')
        self.kahip_config = index_data.get('kahip_config')
        
        return index_data


def build_index_with_kahip(
    data: np.ndarray,
    xadj: np.ndarray,
    adjncy: np.ndarray,
    adjcwt: np.ndarray = None,
    model_state_dict: Dict = None,
    model_config: Dict = None,
    num_buckets: int = 100,
    imbalance: float = 0.03,
    kahip_mode: int = 2,
    seed: int = 1
) -> NeuralLSHIndexFile:
    """
    Build complete Neural-LSH index using KaHIP graph partitioning.
    
    This function:
    1. Takes the k-NN graph in CSR format (xadj, adjncy, adjcwt)
    2. Partitions it using KaHIP for balanced graph partitioning
    3. Builds inverted index from partition assignments
    4. Packages with trained MLP model into index file
    
    Args:
        data: Data points (N, D)
        xadj: CSR format index array - aggregate #neighbors
        adjncy: CSR format adjacency array - flattened neighbor indices
        adjcwt: Edge weights array (optional) - importance/penalty to cut
        model_state_dict: Trained MLP state dict (optional)
        model_config: MLP configuration (optional)
        num_buckets: Number of target partitions (default: 100)
        imbalance: Allowed imbalance ε for balanced partitions (default: 0.03)
        kahip_mode: KaHIP mode: 0=FAST, 1=ECO, 2=STRONG (default: 2)
        seed: Random seed (default: 1)
        
    Returns:
        NeuralLSHIndexFile object ready to save
    """
    n_points = len(data)
    
    print(f"\n[Building Index with KaHIP]")
    print(f"  - Data points: {n_points}")
    print(f"  - Dimensions: {data.shape[1]}")
    
    # Initialize inverted index builder
    index_builder = InvertedIndexBuilder(num_buckets, n_points)
    
    # Partition graph using KaHIP
    print(f"\n[KaHIP Partitioning]")
    partition_vector = index_builder.partition_graph_with_kahip(
        xadj=xadj,
        adjncy=adjncy,
        adjcwt=adjcwt,
        imbalance=imbalance,
        mode=kahip_mode,
        seed=seed
    )
    
    # Build inverted index from partition
    print(f"\n[Building Inverted Index]")
    inverted_index_data = index_builder.build_from_partition(partition_vector)
    print(f"  ✓ Inverted index built:")
    print(f"    • Buckets: {inverted_index_data['num_buckets']}")
    print(f"    • Points: {inverted_index_data['num_points']}")
    print(f"    • Mean bucket size: {inverted_index_data['bucket_stats']['mean_size']:.1f}")
    print(f"    • Min/Max bucket size: {inverted_index_data['bucket_stats']['min_size']}/{inverted_index_data['bucket_stats']['max_size']}")
    
    # Create index file
    index_file = NeuralLSHIndexFile()
    index_file.model_state_dict = model_state_dict
    index_file.model_config = model_config or {'architecture': 'MLP', 'status': 'not_trained'}
    index_file.inverted_index_data = inverted_index_data
    index_file.partition_data = {
        'partition_vector': partition_vector,
        'num_buckets': num_buckets
    }
    index_file.data_metadata = {
        'num_points': n_points,
        'dimension': data.shape[1],
        'data_type': 'float32'
    }
    index_file.kahip_config = {
        'imbalance': imbalance,
        'mode': ['FAST', 'ECO', 'STRONG'][kahip_mode],
        'mode_int': kahip_mode
    }
    
    return index_file


def build_index(
    data: np.ndarray,
    partition_vector: np.ndarray,
    model_state_dict: Dict = None,
    model_config: Dict = None,
    num_buckets: int = 100
) -> NeuralLSHIndexFile:
    """
    Build complete Neural-LSH index from pre-computed partition vector.
    
    Use this function when you already have partition assignments.
    For KaHIP-based partitioning, use build_index_with_kahip instead.
    
    Args:
        data: Data points (N, D)
        partition_vector: Pre-computed partition assignments
        model_state_dict: Trained MLP state dict (optional)
        model_config: MLP configuration (optional)
        num_buckets: Number of partitions
        
    Returns:
        NeuralLSHIndexFile object ready to save
    """
    # Build inverted index
    index_builder = InvertedIndexBuilder(num_buckets, len(data))
    inverted_index_data = index_builder.build_from_partition(partition_vector)
    
    # Create index file
    index_file = NeuralLSHIndexFile()
    index_file.model_state_dict = model_state_dict
    index_file.model_config = model_config or {'architecture': 'MLP'}
    index_file.inverted_index_data = inverted_index_data
    index_file.partition_data = {
        'partition_vector': partition_vector,
        'num_buckets': num_buckets
    }
    index_file.data_metadata = {
        'num_points': len(data),
        'dimension': data.shape[1],
        'data_type': 'float32'
    }
    
    return index_file

