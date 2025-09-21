"""
Main benchmark runner for Lightning Retrieval test suite
"""

import argparse
from pathlib import Path
from typing import List
import sys
import time

from bench.core.benchmark_framework import (
    BenchmarkSuite,
    BenchmarkResult,
    run_benchmark_suite,
    generate_report
)

from bench.tests.bm25_simd_benchmark import BM25SIMDBenchmark
from bench.tests.memmap_benchmark import MemoryMappedBenchmark

def parse_args():
    parser = argparse.ArgumentParser(description='Lightning Retrieval Benchmark Suite')
    
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to store benchmark results')
    
    parser.add_argument('--dataset-size', type=str, default='medium',
                      choices=['small', 'medium', 'large'],
                      help='Size of synthetic datasets for benchmarks')
    
    parser.add_argument('--benchmarks', type=str, nargs='+',
                      choices=['all', 'bm25', 'embeddings', 'memory', 'topk', 'memmap'],
                      default=['all'],
                      help='Specific benchmarks to run')
    
    parser.add_argument('--quick', action='store_true',
                      help='Run quick validation with smaller datasets')
    
    return parser.parse_args()

def get_dataset_sizes(size: str, quick: bool = False) -> dict:
    """Get dataset sizes based on configuration"""
    if quick:
        return {
            'num_docs': 1000,
            'num_queries': 10,
            'embedding_dim': 128,
        }
    
    sizes = {
        'small': {
            'num_docs': 10000,
            'num_queries': 100,
            'embedding_dim': 256,
        },
        'medium': {
            'num_docs': 100000,
            'num_queries': 1000,
            'embedding_dim': 768,
        },
        'large': {
            'num_docs': 1000000,
            'num_queries': 10000,
            'embedding_dim': 1024,
        }
    }
    
    return sizes.get(size, sizes['medium'])

def create_benchmark_suites(args) -> List[BenchmarkSuite]:
    """Create benchmark suite instances based on configuration"""
    sizes = get_dataset_sizes(args.dataset_size, args.quick)
    suites = []
    
    if 'all' in args.benchmarks or 'bm25' in args.benchmarks:
        suites.append(BM25SIMDBenchmark(
            num_docs=sizes['num_docs'],
            num_queries=sizes['num_queries']
        ))
    
    if 'all' in args.benchmarks or 'memmap' in args.benchmarks:
        # Run memory-mapped benchmarks on FIQA dataset
        suites.append(MemoryMappedBenchmark(
            dataset_name='fiqa',
            config={
                'buffer_size': 8 * 1024 * 1024,  # 8MB buffer
                'cache_size': 50000,
                'compression_level': 1,
                'alignment': 16
            }
        ))
    
    # TODO: Add other benchmark suites as they're implemented
    # if 'all' in args.benchmarks or 'embeddings' in args.benchmarks:
    #     suites.append(EmbeddingQuantizationBenchmark(...))
    
    return suites

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Lightning Retrieval Benchmark Suite")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset Size: {args.dataset_size}")
    print(f"  Quick Mode: {args.quick}")
    print(f"  Selected Benchmarks: {', '.join(args.benchmarks)}")
    print(f"  Output Directory: {output_dir}")
    
    # Create and run benchmark suites
    suites = create_benchmark_suites(args)
    results: List[BenchmarkResult] = []
    
    start_time = time.time()
    
    for suite in suites:
        try:
            print(f"\nRunning {suite.name} benchmark suite...")
            run_benchmark_suite(suite)
            results.append(suite.run())
        except Exception as e:
            print(f"❌ Failed to run {suite.name}: {e}", file=sys.stderr)
    
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    print("\nGenerating benchmark report...")
    generate_report(results, output_dir)
    
    print(f"\n✅ Benchmarking completed in {total_time:.2f}s")
    print(f"📊 Results and reports saved to: {output_dir}")

if __name__ == "__main__":
    main()