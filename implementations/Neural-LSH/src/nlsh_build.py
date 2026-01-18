

import os
import sys
import pickle
import struct
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean imports using standard Python paths
from dataparser.sift_parser import SIFTParser  # type: ignore
from dataparser.mnist_parser import MNISTParser  # type: ignore
from utils.args import parse_and_validate_build_args, print_build_summary  # type: ignore
from build.index_model.mlp_model import NeuralLSH_MLP  # type: ignore
from build.index_model.index_builder import build_index_with_kahip  # type: ignore
from build.graph.graph_build import GraphBuilder, graph_cache_exists, get_graph_cache_path  # type: ignore
from search.search_input import load_data  # type: ignore

try:
    import kahip  # type: ignore
    KAHIP_AVAILABLE = True
except ImportError:
    KAHIP_AVAILABLE = False


def build_knn_graph_and_save(data: np.ndarray, k: int = 10, data_type: str = 'sift', skip_build: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build k-NN graph using GraphBuilder from graph_build.py and return CSR format.
    Checks if graph with same type and k already exists - skips build if it does.
    Saves the graph to cache directory.
    
    Args:
        data: Data points (N, D)
        k: Number of nearest neighbors
        data_type: 'sift' or 'mnist' (for naming the output file)
        skip_build: If True, skip building if graph exists
        
    Returns:
        Tuple of (xadj, adjncy, adjcwt, vwgt) arrays in CSR format
    """
    print(f"\n[ΒΗΜΑ 2α] Κατασκευή k-NN Γράφου (k={k})...")
    
    # Check if graph already exists with same type and k
    if graph_cache_exists(data_type, k):
        graph_path = get_graph_cache_path(data_type, k)
        print(f"  ✓ Graph already exists: {graph_path.name}")
        if skip_build:
            print(f"  ⚠ --skip-graph flag set, skipping build")
            return None, None, None, None  # TODO: Load from cache
    
    n_points = len(data)
    graph_path = get_graph_cache_path(data_type, k)
    
    # Use GraphBuilder class to construct graph with all preprocessing
    print(f"  - Using GraphBuilder class from graph_build.py...")
    graph_builder = GraphBuilder(data, k=k, metric='euclidean')
    
    # Step 1: Build directed k-NN graph
    knn_adj, distances = graph_builder.build_knn_graph()
    
    # Step 2: Symmetrize graph
    sym_adj = graph_builder.symmetrize_graph(knn_adj)
    
    # Step 3: Assign edge weights
    edge_weights = graph_builder.assign_edge_weights(knn_adj, sym_adj)
    
    # Step 4: Convert to CSR format
    xadj, adjncy, adjcwt, vwgt = graph_builder.to_csr_format(sym_adj, edge_weights)
    
    # Step 5: Save CSR arrays to cache
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(graph_path), 
             xadj=xadj, adjncy=adjncy, adjcwt=adjcwt, vwgt=vwgt)
    
    print(f"  ✓ Γράφος αποθηκεύτηκε: {graph_path.parent.name}/{graph_path.name}")
    
    return xadj, adjncy, adjcwt, vwgt




def get_kahip_cache_file(data_type: str, m: int) -> Path:
    """Get cache file path for KaHIP partitioning results."""
    cache_dir = Path.cwd() / 'cache' / 'partition_caches'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{data_type}_kahip_m{m}.pkl"


def load_cached_partition(data_type: str, m: int, n_points: int) -> Optional[np.ndarray]:
    """Load cached KaHIP partition if it exists."""
    cache_file = get_kahip_cache_file(data_type, m)
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_partition = pickle.load(f)
            # Verify cache is valid (correct size)
            if len(cached_partition) == n_points:
                print(f"  ✓ Loaded cached KaHIP partition from {cache_file.name}")
                return cached_partition
            else:
                print(f"  ⚠ Cached partition has wrong size, will recompute")
        except Exception as e:
            print(f"  ⚠ Error loading cache: {e}")
    
    return None


def save_partition_cache(partition_vector: np.ndarray, data_type: str, m: int) -> None:
    """Save KaHIP partition to cache."""
    cache_file = get_kahip_cache_file(data_type, m)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(partition_vector, f)
        print(f"  ✓ Cached KaHIP partition to {cache_file.name}")
    except Exception as e:
        print(f"  ⚠ Error saving cache: {e}")


def main():
    """Κύρια συνάρτηση για την κατασκευή του Neural-LSH ευρετηρίου."""
    # Parse and validate arguments using utils.args
    args = parse_and_validate_build_args()
    
    # Ορισμός seed για αναπαραγωγιμότητα
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Δημιουργία φακέλων αν δεν υπάρχουν
    index_dir = os.path.dirname(args.index)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    
    # Print configuration summary
    print_build_summary(args)
    
    # =========================================================================
    # ΒΗΜΑ 1: Φόρτωση και Προεπεξεργασία Δεδομένων
    # =========================================================================
    print("[ΒΗΜΑ 1] Φόρτωση Δεδομένων...")
    data = load_data(args.data)  # load_data infers format from file extension
    
    # DEBUG: Περιορίστε τα δεδομένα σε 1000 σημεία για γρήγορο έλεγχο
    # Αλλάξτε την τιμή 1000 ή σχολιάστε τις γραμμές παρακάτω για να χρησιμοποιήσετε όλα τα δεδομένα
    DEBUG_LIMIT = 1000  # Αλλάξτε σε None για να χρησιμοποιήσετε όλα τα δεδομένα
    if DEBUG_LIMIT is not None and len(data) > DEBUG_LIMIT:
        data = data[:DEBUG_LIMIT]
        print(f"  ⚠ DEBUG MODE: Περιορίστηκε στα πρώτα {DEBUG_LIMIT} σημεία")
    
    n_points, dim = data.shape
    
    # =========================================================================
    # ΒΗΜΑ 2: Κατασκευή k-NN Γράφου και Διαμέριση με KaHIP
    # =========================================================================
    print(f"\n[ΒΗΜΑ 2] Κατασκευή k-NN Γράφου και Διαμέριση με KaHIP...")
    
    # Build k-NN graph and save to file, get CSR format
    # Skip if --skip-graph flag is set
    if args.skip_graph:
        print(f"  ⚠ --skip-graph: Loading existing graph (not rebuilding)...")
    
    xadj, adjncy, adjcwt, vwgt = build_knn_graph_and_save(data, k=args.knn, data_type=args.type, skip_build=args.skip_graph)
    
    # =========================================================================
    # ΒΗΜΑ 3: Διαμέριση Γράφου με KaHIP
    # =========================================================================
    # Try to load cached partition if --skip-kahip is set
    cached_partition = None
    if args.skip_kahip:
        print(f"  ⚠ --skip-kahip: Checking for cached KaHIP results...")
        cached_partition = load_cached_partition(args.type, args.m, len(data))
    
    if cached_partition is not None:
        partition_vector = cached_partition
        print(f"  ✓ Using cached KaHIP partition (m={args.m})")
    elif not KAHIP_AVAILABLE:
        print("\n  ⚠ KaHIP δεν είναι διαθέσιμο - χρησιμοποιώντας τυχαία διαμέριση")
        partition_vector = np.random.randint(0, args.m, size=n_points)
    else:
        print(f"\n  [KaHIP Partitioning]")
        print(f"    • Nodes: {n_points}, Edges: {len(adjncy)//2}")
        print(f"    • Target buckets (m): {args.m}")
        print(f"    • Imbalance (ε): {args.imbalance}")
        print(f"    • Mode: {['FAST', 'ECO', 'STRONG'][args.kahip_mode]}")
        
        try:
            # Call KaHIP for partitioning using positional arguments
            edgecut, partition_vector = kahip.kaffpa(
                vwgt,                       # arg0: vertex weights
                xadj,                       # arg1: CSR index array
                adjcwt,                     # arg2: edge weights
                adjncy,                     # arg3: CSR adjacency array
                args.m,                     # arg4: nblocks
                args.imbalance,             # arg5: imbalance factor
                False,                      # arg6: suppress_output
                args.seed,                  # arg7: seed
                args.kahip_mode             # arg8: mode
            )
            partition_vector = np.array(partition_vector, dtype=np.int32)
            print(f"  ✓ KaHIP completed: Edge-cut = {edgecut}")
            # Cache the partition result for future use
            save_partition_cache(partition_vector, args.type, args.m)
        except Exception as e:
            print(f"  ✗ KaHIP error: {e} - falling back to random partition")
            partition_vector = np.random.randint(0, args.m, size=n_points)
    
    # =========================================================================
    # ΒΗΜΑ 4: Εκπαίδευση MLP
    # =========================================================================
    print(f"\n[ΒΗΜΑ 4] Εκπαίδευση MLP μοντέλου...")
    print(f"  - Αρχιτεκτονική: {args.layers} στρώματα, {args.nodes} κόμβοι")
    print(f"  - Παράμετροι: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - Device: {device}")
    
    # Δημιουργία μοντέλου
    model = NeuralLSH_MLP(
        input_dim=dim,
        num_layers=args.layers,
        nodes_per_layer=args.nodes,
        output_dim=args.m
    ).to(device)
    
    # Ορισμός κριτηρίου απώλειας και βελτιστοποιητή
    criterion = nn.CrossEntropyLoss()
    import torch.optim as optim  # type: ignore
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Μετατροπή δεδομένων σε PyTorch tensors
    X_tensor = torch.FloatTensor(data).to(device)
    y_tensor = torch.LongTensor(partition_vector).to(device)
    
    # Εκπαίδευση
    print("\n  Εκπαίδευση:")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Mini-batch training
        permutation = np.random.permutation(n_points)
        epoch_loss = 0.0
        n_batches = (n_points + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, n_points)
            
            batch_indices = permutation[start_idx:end_idx]
            X_batch = X_tensor[batch_indices]
            y_batch = y_tensor[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}")
    
    print(f"  ✓ Εκπαίδευση ολοκληρώθηκε. Best Loss: {best_loss:.6f}")
    
    # =========================================================================
    # ΒΗΜΑ 5: Αποθήκευση Μοντέλου και Ευρετηρίου (Inverted File)
    # =========================================================================
    print(f"\n[ΒΗΜΑ 5] Αποθήκευση Μοντέλου και Ευρετηρίου...")
    
    # Δημιουργία index με KaHIP
    index_file = build_index_with_kahip(
        data=data,
        xadj=xadj,
        adjncy=adjncy,
        adjcwt=adjcwt,
        model_state_dict=model.state_dict(),
        model_config=model.get_config(),
        num_buckets=args.m,
        imbalance=args.imbalance,
        kahip_mode=args.kahip_mode,
        seed=args.seed
    )
    
    # Αποθήκευση
    index_file.save(args.index)
    
    print("\n" + "=" * 80)
    print("NEURAL-LSH BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nΕπιτυχής δημιουργία ευρετηρίου:")
    print(f"  - Δεδομένα: {n_points} σημεία × {dim} διάστασης")
    print(f"  - k-NN Graph: k={args.knn}")
    print(f"  - Διαμέριση: {args.m} buckets (με KaHIP)")
    print(f"  - MLP: {args.layers} στρώματα × {args.nodes} κόμβοι")
    print(f"  - Ευρετήριο: {args.index}")
    print()


if __name__ == '__main__':
    main()
