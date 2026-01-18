#!/usr/bin/env python3
"""
Create Brute Force Cache Files
================================
Pre-computes ground truth distances and saves them to disk for fast reuse.
Run this once to generate cache files for your datasets.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataparser.sift_parser import SIFTParser
from dataparser.mnist_parser import MNISTParser
from search.search_algorithms import save_brute_force_cache
from search.search_metrics import compute_true_distances


def create_sift_cache(num_queries=1000):
    """
    Create brute force cache for SIFT dataset.
    
    Args:
        num_queries: Number of SIFT queries to compute (default: 1000)
    """
    print("\n" + "="*80)
    print("Creating SIFT Brute Force Cache")
    print("="*80)
    
    print("\n[1] Loading SIFT data...")
    print("  Loading SIFT base data...")
    sift_base = SIFTParser.read_fvecs('dataset/sift_base.fvecs')
    print(f"    ✓ Loaded {len(sift_base)} SIFT vectors (shape: {sift_base.shape})")
    
    print("  Loading SIFT queries...")
    sift_queries = SIFTParser.read_fvecs('dataset/sift_query.fvecs')
    print(f"    ✓ Loaded {len(sift_queries)} SIFT queries available")
    
    # Use only num_queries
    sift_queries = sift_queries[:num_queries]
    print(f"    • Using first {len(sift_queries)} queries for cache")
    
    print("\n[2] Computing ground truth distances...")
    print(f"  Computing {len(sift_queries)} × {len(sift_base)} distance matrix...")
    
    sift_distances = np.zeros((len(sift_queries), len(sift_base)), dtype=np.float32)
    
    for i, query in enumerate(sift_queries):
        sift_distances[i] = compute_true_distances(query, sift_base)
        
        # Progress indicator
        progress = (i + 1) / len(sift_queries) * 100
        if (i + 1) % 100 == 0 or (i + 1) == len(sift_queries):
            print(f"    [{progress:5.1f}%] Computed {i + 1}/{len(sift_queries)} queries")
    
    print("\n[3] Saving cache to disk...")
    save_brute_force_cache(sift_distances, 'sift', len(sift_queries), len(sift_base))
    
    print("\n" + "="*80)
    print(f"✓ SIFT cache created: {len(sift_queries)} queries × {len(sift_base)} base vectors")
    print("="*80)


def create_mnist_cache(num_queries=1000):
    """
    Create brute force cache for MNIST dataset.
    
    Args:
        num_queries: Number of MNIST queries to compute (default: 1000)
    """
    print("\n" + "="*80)
    print("Creating MNIST Brute Force Cache")
    print("="*80)
    
    print("\n[1] Loading MNIST data...")
    print("  Loading MNIST training data...")
    mnist_train = MNISTParser.read_images('dataset/train-images.idx3-ubyte')
    mnist_train = MNISTParser.flatten_images(mnist_train)
    print(f"    ✓ Loaded {len(mnist_train)} MNIST training vectors (shape: {mnist_train.shape})")
    
    print("  Loading MNIST test data...")
    mnist_test = MNISTParser.read_images('dataset/t10k-images.idx3-ubyte')
    mnist_test = MNISTParser.flatten_images(mnist_test)
    print(f"    ✓ Loaded {len(mnist_test)} MNIST test queries available")
    
    # Use only num_queries
    mnist_test = mnist_test[:num_queries]
    print(f"    • Using first {len(mnist_test)} queries for cache")
    
    print("\n[2] Computing ground truth distances...")
    print(f"  Computing {len(mnist_test)} × {len(mnist_train)} distance matrix...")
    
    mnist_distances = np.zeros((len(mnist_test), len(mnist_train)), dtype=np.float32)
    
    for i, query in enumerate(mnist_test):
        mnist_distances[i] = compute_true_distances(query, mnist_train)
        
        # Progress indicator
        progress = (i + 1) / len(mnist_test) * 100
        if (i + 1) % 100 == 0 or (i + 1) == len(mnist_test):
            print(f"    [{progress:5.1f}%] Computed {i + 1}/{len(mnist_test)} queries")
    
    print("\n[3] Saving cache to disk...")
    save_brute_force_cache(mnist_distances, 'mnist', len(mnist_test), len(mnist_train))
    
    print("\n" + "="*80)
    print(f"✓ MNIST cache created: {len(mnist_test)} queries × {len(mnist_train)} base vectors")
    print("="*80)


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("BRUTE FORCE CACHE GENERATOR")
    print("="*80)
    print("\nThis script pre-computes ground truth distances and saves them to disk.")
    print("First run takes some time, but subsequent searches will use the cached results.")
    
    import argparse
    parser = argparse.ArgumentParser(description='Create brute force cache files')
    parser.add_argument('--sift', type=int, default=1000,
                       help='Number of SIFT queries (default: 1000)')
    parser.add_argument('--mnist', type=int, default=1000,
                       help='Number of MNIST queries (default: 1000)')
    parser.add_argument('--sift-only', action='store_true',
                       help='Only create SIFT cache')
    parser.add_argument('--mnist-only', action='store_true',
                       help='Only create MNIST cache')
    
    args = parser.parse_args()
    
    try:
        if not args.mnist_only:
            create_sift_cache(num_queries=args.sift)
        
        if not args.sift_only:
            create_mnist_cache(num_queries=args.mnist)
        
        print("\n" + "="*80)
        print("✓ All brute force caches created successfully!")
        print("="*80)
        print("\nCache files location: cache/")
        print("Future searches will automatically load these cached results.")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
