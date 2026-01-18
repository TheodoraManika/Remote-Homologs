"""
Argument Validation for Neural LSH

Validates command-line arguments for nlsh_build.py and nlsh_search.py
according to assignment specifications.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys


def validate_build_args(args) -> bool:
    """
    Validate arguments for nlsh_build.py.
    
    Args:
        args: argparse Namespace with parsed arguments
        
    Returns:
        True if all arguments are valid
        
    Raises:
        ValueError: If any argument is invalid
    """
    # Required arguments
    if not args.data:
        raise ValueError("'-d' (data file) is required")
    
    if not args.index:
        raise ValueError("'-i' (index path) is required")
    
    if not args.type:
        raise ValueError("'-type' (data type) is required")
    
    # Validate data file exists (resolve relative paths from project root)
    data_file = Path(args.data)
    if not data_file.is_absolute():
        # Try relative to project root (parent of src directory)
        project_root = Path(__file__).parent.parent
        data_file = project_root / args.data
    
    if not data_file.exists():
        raise ValueError(f"Data file not found: {args.data} (resolved to: {data_file})")
    
    # Validate data type
    if args.type.lower() not in ['sift', 'mnist']:
        raise ValueError(f"Invalid data type: '{args.type}'. Must be 'sift' or 'mnist'")
    
    # Validate numeric parameters
    if args.knn <= 0:
        raise ValueError(f"--knn must be positive, got {args.knn}")
    
    if args.m <= 0:
        raise ValueError(f"-m must be positive, got {args.m}")
    
    if not (0 <= args.imbalance < 1):
        raise ValueError(f"--imbalance must be in [0, 1), got {args.imbalance}")
    
    if args.kahip_mode not in [0, 1, 2]:
        raise ValueError(f"--kahip_mode must be 0, 1, or 2, got {args.kahip_mode}")
    
    if args.layers <= 0:
        raise ValueError(f"--layers must be positive, got {args.layers}")
    
    if args.nodes <= 0:
        raise ValueError(f"--nodes must be positive, got {args.nodes}")
    
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")
    
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    
    if args.seed < 0:
        raise ValueError(f"--seed must be non-negative, got {args.seed}")
    
    return True


def validate_search_args(args) -> bool:
    """
    Validate arguments for nlsh_search.py.
    
    Args:
        args: argparse Namespace with parsed arguments
        
    Returns:
        True if all arguments are valid
        
    Raises:
        ValueError: If any argument is invalid
    """
    # Required arguments
    if not args.data:
        raise ValueError("'-d' (data file) is required")
    
    if not args.query:
        raise ValueError("'-q' (query file) is required")
    
    if not args.index:
        raise ValueError("'-i' (index path) is required")
    
    if not args.output:
        raise ValueError("'-o' (output file) is required")
    
    if not args.type:
        raise ValueError("'-type' (data type) is required")
    
    # Validate files exist
    data_file = Path(args.data)
    if not data_file.exists():
        raise ValueError(f"Data file not found: {args.data}")
    
    query_file = Path(args.query)
    if not query_file.exists():
        raise ValueError(f"Query file not found: {args.query}")
    
    index_path = Path(args.index)
    if not index_path.exists():
        raise ValueError(f"Index path not found: {args.index}")
    
    # Validate data type
    if args.type.lower() not in ['sift', 'mnist']:
        raise ValueError(f"Invalid data type: '{args.type}'. Must be 'sift' or 'mnist'")
    
    # Validate numeric parameters
    if args.N <= 0:
        raise ValueError(f"-N must be positive, got {args.N}")
    
    if args.R <= 0:
        raise ValueError(f"-R must be positive, got {args.R}")
    
    if args.T <= 0:
        raise ValueError(f"-T must be positive, got {args.T}")
    
    # Validate range parameter
    if args.range.lower() not in ['true', 'false']:
        raise ValueError(f"-range must be 'true' or 'false', got '{args.range}'")
    
    # Validate output directory can be created
    output_file = Path(args.output)
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory: {e}")
    
    return True


def create_build_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for nlsh_build.py with validation.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Build Neural LSH Index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlsh_build.py -d sift_base.fvecs -i nlsh_index -type sift --knn 15 -m 100 --epochs 10
  python nlsh_build.py -d train-images.idx3-ubyte -i nlsh_index -type mnist -m 50 --layers 2
        """
    )
    
    # Required arguments
    parser.add_argument('-d', '--data', required=True, metavar='FILE',
                       help='Dataset file (e.g., input.dat, sift_base.fvecs)')
    parser.add_argument('-i', '--index', required=True, metavar='PATH',
                       help='Output index path (directory) (e.g., nlsh_index)')
    parser.add_argument('-type', '--type', required=True, choices=['sift', 'mnist'],
                       help='Dataset type')
    
    # Graph parameters
    parser.add_argument('--knn', type=int, default=10,
                       help='k for k-NN graph (default: 10)')
    parser.add_argument('-m', type=int, default=100,
                       help='Number of partitions for KaHIP (default: 100)')
    parser.add_argument('--imbalance', type=float, default=0.03,
                       help='KaHIP imbalance factor ϵ (default: 0.03)')
    parser.add_argument('--kahip_mode', type=int, default=2, choices=[0, 1, 2],
                       help='KaHIP mode: 0=FAST, 1=ECO, 2=STRONG (default: 2)')
    
    # Skip expensive computations (for experiments)
    parser.add_argument('--skip-graph', action='store_true',
                       help='Skip k-NN graph construction (use existing graph)')
    parser.add_argument('--skip-kahip', action='store_true',
                       help='Skip KaHIP partitioning (use cached result if available)')
    
    # MLP parameters
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of MLP layers (default: 3)')
    parser.add_argument('--nodes', type=int, default=64,
                       help='Hidden layer size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed (default: 1)')
    
    return parser


def create_search_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for nlsh_search.py with validation.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Search with Neural LSH Index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlsh_search.py -d sift_base.fvecs -q sift_query.fvecs -i nlsh_index -o results.txt -type sift -N 10 -T 5
  python nlsh_search.py -d train-images.idx3-ubyte -q test-images.idx3-ubyte -i nlsh_index -o results.txt -type mnist -N 5
        """
    )
    
    # Required arguments
    parser.add_argument('-d', '--data', required=True, metavar='FILE',
                       help='Dataset file (base data for computing true distances)')
    parser.add_argument('-q', '--query', required=True, metavar='FILE',
                       help='Query file (e.g., query.dat, sift_query.fvecs)')
    parser.add_argument('-i', '--index', required=True, metavar='PATH',
                       help='Index path (created by nlsh_build.py)')
    parser.add_argument('-o', '--output', required=True, metavar='FILE',
                       help='Output results file')
    parser.add_argument('-type', '--type', required=True, choices=['sift', 'mnist'],
                       help='Dataset type')
    
    # Search parameters
    parser.add_argument('-N', type=int, default=1,
                       help='Number of nearest neighbors to return (default: 1)')
    parser.add_argument('-R', type=float, default=None,
                       help='Search radius for range query (default: 2000 for MNIST, 2800 for SIFT)')
    parser.add_argument('-T', type=int, default=5,
                       help='Number of partitions to probe (multi-probe) (default: 5)')
    parser.add_argument('-range', '--range', type=str, default='true', choices=['true', 'false'],
                       help='Whether to perform range search (default: true)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    return parser


def parse_and_validate_build_args(argv=None) -> argparse.Namespace:
    """
    Parse and validate nlsh_build.py arguments.
    
    Args:
        argv: Command-line arguments (default: sys.argv[1:])
        
    Returns:
        Validated argument namespace
    """
    parser = create_build_parser()
    args = parser.parse_args(argv)
    
    try:
        validate_build_args(args)
        return args
    except ValueError as e:
        print(f"✗ Argument validation error: {e}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


def parse_and_validate_search_args(argv=None) -> argparse.Namespace:
    """
    Parse and validate nlsh_search.py arguments.
    
    Args:
        argv: Command-line arguments (default: sys.argv[1:])
        
    Returns:
        Validated argument namespace
    """
    parser = create_search_parser()
    args = parser.parse_args(argv)
    
    # Set default radius based on data type
    if args.R is None:
        args.R = 2000.0 if args.type.lower() == 'mnist' else 2800.0
    
    # Convert range string to boolean
    args.range_search = args.range.lower() == 'true'
    
    try:
        validate_search_args(args)
        return args
    except ValueError as e:
        print(f"✗ Argument validation error: {e}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


def print_build_summary(args) -> None:
    """Print summary of nlsh_build.py parameters."""
    print("\n" + "="*70)
    print("Neural LSH Build Configuration")
    print("="*70)
    print(f"\nData:")
    print(f"  File: {args.data}")
    print(f"  Type: {args.type}")
    print(f"\nIndex:")
    print(f"  Output path: {args.index}")
    print(f"\nGraph Construction:")
    print(f"  k (nearest neighbors): {args.knn}")
    print(f"  m (partitions): {args.m}")
    print(f"  KaHIP imbalance: {args.imbalance}")
    print(f"  KaHIP mode: {args.kahip_mode} (0=FAST, 1=ECO, 2=STRONG)")
    print(f"\nMLP Training:")
    print(f"  Layers: {args.layers}")
    print(f"  Hidden nodes: {args.nodes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    print("="*70 + "\n")


def print_search_summary(args) -> None:
    """Print summary of nlsh_search.py parameters."""
    print("\n" + "="*70)
    print("Neural LSH Search Configuration")
    print("="*70)
    print(f"\nFiles:")
    print(f"  Dataset: {args.data}")
    print(f"  Query: {args.query}")
    print(f"  Index: {args.index}")
    print(f"  Output: {args.output}")
    print(f"  Type: {args.type}")
    print(f"\nSearch Parameters:")
    print(f"  N (nearest neighbors): {args.N}")
    print(f"  R (search radius): {args.R}")
    print(f"  T (partitions to probe): {args.T}")
    print(f"  Range search: {args.range}")
    print("="*70 + "\n")
