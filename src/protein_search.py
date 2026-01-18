import os
import sys
import argparse

# Add project root to path to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from algo_handler.handler import AlgoHandler, compile_cpp
from dataprocessing.load_embeddings import read_dat, load_ids, load_pfam_map
from implementations.blast.blast import Blast
from utils.output import ReportGenerator

def cmd_compile(args):
    """Compile the C++ search algorithms."""
    print("Compiling C++ algorithms...")
    if compile_cpp():
        print("Compilation successful.")
    else:
        print("Compilation failed.")
        sys.exit(1)

def cmd_search(args):
    """Run the protein homolog search pipeline."""
    base_dir = project_root
    
    # Resolve paths (defaulting to project structure if not provided)
    if args.dataset:
        dataset_file = args.dataset
    else:
        dataset_file = os.path.join(base_dir, "embeddings/swissprot_50k.dat")
        
    if args.query:
        query_file = args.query
    else:
        query_file = os.path.join(base_dir, "embeddings/targets.dat")
        
    dataset_ids_file = args.dataset_ids or os.path.join(base_dir, "embeddings/swissprot_50k_ids.txt")
    query_ids_file = args.query_ids or os.path.join(base_dir, "embeddings/targets_ids.txt")
    
    # Check inputs
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found: {dataset_file}")
        return
    if not os.path.exists(query_file):
        print(f"Error: Query file not found: {query_file}")
        return

    # Load IDs
    print(f"Loading IDs for mapping...")
    try:
        dataset_ids = load_ids(dataset_ids_file)
        query_ids = load_ids(query_ids_file)
    except Exception as e:
        print(f"Warning: Could not load IDs ({e}). Using indices.")
        dataset_ids = []
        query_ids = []

    def get_dataset_id(idx):
        if 0 <= idx < len(dataset_ids):
            return dataset_ids[idx]
        return str(idx)
        
    # Load Pfam Map if available
    pfam_map = {}
    pfam_file = os.path.join(base_dir, "data/pfam_map.tsv")
    if os.path.exists(pfam_file):
        print(f"Loading Pfam map from {pfam_file}...")
        try:
             pfam_map = load_pfam_map(pfam_file)
        except Exception as e:
            print(f"Warning loading Pfam map: {e}")

    # Setup Algorithm Handler
    handler = AlgoHandler(
        executable_path=os.path.join(base_dir, "search"),
        output_dir=os.path.join(base_dir, "results")
    )
    
    # Define Methods to Run    
    N_metrics = args.N
    methods = []
    
    # Add methods based on arguments or run all defaults
    if args.method == "all" or args.method == "lsh":
        methods.append({
            "name": "Euclidean LSH", 
            "func": handler.run_lsh, 
            "kwargs": {"k": 4, "L": 5, "w": 400.0, "N": N_metrics, "R": 1.5}
        })
    if args.method == "all" or args.method == "hypercube":
        methods.append({
            "name": "Hypercube", 
            "func": handler.run_hypercube, 
            "kwargs": {"kproj": 14, "M": 1000, "probes": 2, "w": 400.0, "N": N_metrics, "R": 1.5}
        })
    if args.method == "all" or args.method == "ivfflat":
        methods.append({
            "name": "IVF-Flat", 
            "func": handler.run_ivfflat, 
            "kwargs": {"kclusters": 50, "nprobe": 5, "N": N_metrics, "R": 1.5}
        })
    if args.method == "all" or args.method == "ivfpq":
        methods.append({
            "name": "IVF-PQ", 
            "func": handler.run_ivfpq, 
            "kwargs": {"kclusters": 50, "nprobe": 5, "M": 8, "nbits": 8, "N": N_metrics, "R": 1.5}
        })

    all_methods_stats = []
    all_methods_neighbors = {}

    print(f"\n[Running Search Algorithms] Mode: {args.method}")
    
    for method in methods:
        print(f"Running {method['name']}...")
        output_filename = f"{method['name'].replace(' ', '_').lower()}.txt"
        res = method["func"](dataset_file, query_file, output_filename, **method["kwargs"])
        
        if res:
            metrics = handler.parse_metrics(res)
            metrics["name"] = method["name"]
            all_methods_stats.append(metrics)
            
            # Parse Neighbors
            query_results = handler.parse_query_results(res["output_file"])
            for q_idx, neighbors_list in query_results.items():
                start_q_id = query_ids[q_idx] if 0 <= q_idx < len(query_ids) else str(q_idx)
                
                transformed_neighbors = []
                for n in neighbors_list:
                    transformed_neighbors.append({
                        'id': get_dataset_id(n['id']),
                        'distance': n['distance']
                    })
                
                if start_q_id not in all_methods_neighbors:
                    all_methods_neighbors[start_q_id] = {}
                all_methods_neighbors[start_q_id][method['name']] = transformed_neighbors
        else:
            print(f"Failed to run {method['name']}")

    # BLAST Comparison (Ground Truth)
    print("\n[Loading/Running BLAST Ground Truth]")
    blast_db_path = os.path.join(base_dir, "blast_db/swissprot_db")
    blast = Blast("blastp", blast_db_path, "dummy_query.fasta") # Paths don't strictly matter for loading
    
    blast_results_file = os.path.join(base_dir, "blast_results.json")
    if os.path.exists(blast_results_file):
        print(f"Loading BLAST results from {blast_results_file}")
        blast.load_results(blast_results_file)
    else:
        print("Warning: BLAST results not found. Report will lack BLAST identity info.")

    # Add BLAST stats dummy entry for summary table
    blast_time = blast.get_time_per_query() if blast.get_execution_time() else 1.5
    blast_qps = blast.get_queries_per_second() if blast.get_execution_time() else 0.7
    all_methods_stats.append({
        "name": "BLAST (Ref)",
        "time_per_query": blast_time,
        "qps": blast_qps,
        "recall": 1.00 
    })

    # Generate Report
    print(f"\n[Generating Report]")
    num_queries = min(args.limit, len(query_ids)) if query_ids else min(args.limit, 5)
    
    for i in range(num_queries):
        q_id = query_ids[i] if query_ids else str(i)
        
        # Get BLAST info
        blast_set = blast.get_ground_truth_set(q_id, top_n=N_metrics)
        blast_hits_map = {}
        blast_res = blast.get_results(q_id)
        if blast_res:
            for hit in blast_res.hits:
                blast_hits_map[hit.subject_id] = hit.percent_identity

        neighbors_map = all_methods_neighbors.get(q_id, {})
        
        ReportGenerator.generate_report(
            query_id=q_id,
            n=N_metrics,
            methods_stats=all_methods_stats,
            blast_ground_truth=blast_set,
            blast_hits_details=blast_hits_map,
            pfam_map=pfam_map,
            top_neighbors=neighbors_map
        )
        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Remote Homolog Search Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Compile
    parser_compile = subparsers.add_parser("compile", help="Compile C++ backend")
    
    # Search
    parser_search = subparsers.add_parser("search", help="Run search algorithms and comparison")
    parser_search.add_argument("--dataset", help="Path to .dat dataset file")
    parser_search.add_argument("--query", help="Path to .dat query file")
    parser_search.add_argument("--dataset_ids", help="Path to dataset IDs txt")
    parser_search.add_argument("--query_ids", help="Path to query IDs txt")
    parser_search.add_argument("--method", default="all", choices=["all", "lsh", "hypercube", "ivfflat", "ivfpq"], help="Algorithm to run")
    parser_search.add_argument("--N", type=int, default=50, help="Top-N for Recall evaluation")
    parser_search.add_argument("--limit", type=int, default=5, help="Number of queries to report")

    args = parser.parse_args()
    
    if args.command == "compile":
        cmd_compile(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
