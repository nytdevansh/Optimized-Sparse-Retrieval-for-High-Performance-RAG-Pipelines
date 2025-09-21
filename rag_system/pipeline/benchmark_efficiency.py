#!/usr/bin/env python3
"""
Benchmarks for index build time, memory, and query latency.
"""
import time
import psutil
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Callable


def measure_memory():
    """Get current process memory usage in bytes."""
    process = psutil.Process()
    return process.memory_info().rss


def measure_build_performance(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Measure function execution time and memory usage."""
    mem_before = measure_memory()
    start_time = time.perf_counter()
    
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    end_time = time.perf_counter()
    mem_after = measure_memory()
    
    return {
        'execution_time_s': end_time - start_time,
        'memory_before_mb': mem_before / (1024 * 1024),
        'memory_after_mb': mem_after / (1024 * 1024),
        'memory_delta_mb': (mem_after - mem_before) / (1024 * 1024),
        'peak_memory_mb': mem_after / (1024 * 1024),
        'success': success,
        'error': error,
        'result': result
    }


def benchmark_query_latency(search_func: Callable, queries: List[str], 
                          warmup_runs: int = 5, benchmark_runs: int = 20) -> Dict[str, Any]:
    """Benchmark query search latency with warmup."""
    if not queries:
        return {'error': 'No queries provided'}
    
    print(f"Running query latency benchmark with {len(queries)} queries...")
    
    # Warmup runs
    print(f"Warming up with {warmup_runs} runs...")
    for i in range(min(warmup_runs, len(queries))):
        try:
            search_func(queries[i])
        except Exception as e:
            print(f"Warmup query {i} failed: {e}")
    
    # Benchmark runs
    print(f"Running {benchmark_runs} benchmark queries...")
    latencies = []
    successful_queries = 0
    failed_queries = 0
    
    for i in range(benchmark_runs):
        query = queries[i % len(queries)]  # Cycle through queries
        
        start_time = time.perf_counter()
        try:
            search_func(query)
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            successful_queries += 1
        except Exception as e:
            failed_queries += 1
            print(f"Query {i} failed: {e}")
    
    if not latencies:
        return {'error': 'All queries failed'}
    
    return {
        'total_queries': benchmark_runs,
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
        'mean_latency_ms': statistics.mean(latencies),
        'median_latency_ms': statistics.median(latencies),
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
        'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'p95_latency_ms': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        'p99_latency_ms': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
    }


def benchmark_index_scaling(build_func: Callable, corpus_sizes: List[int], 
                          full_corpus: Dict) -> Dict[str, Any]:
    """Benchmark index build time scaling with corpus size."""
    results = {}
    corpus_items = list(full_corpus.items())
    
    for size in corpus_sizes:
        if size > len(corpus_items):
            print(f"Skipping size {size} (larger than corpus size {len(corpus_items)})")
            continue
        
        print(f"Benchmarking index build with {size} documents...")
        
        # Create subset corpus
        subset_corpus = dict(corpus_items[:size])
        
        # Benchmark the build function
        perf_metrics = measure_build_performance(build_func, subset_corpus)
        
        results[str(size)] = {
            'corpus_size': size,
            'build_time_s': perf_metrics['execution_time_s'],
            'memory_usage_mb': perf_metrics['memory_delta_mb'],
            'success': perf_metrics['success'],
            'error': perf_metrics['error']
        }
        
        if perf_metrics['success']:
            print(f"  ✓ {size} docs: {perf_metrics['execution_time_s']:.2f}s, "
                  f"{perf_metrics['memory_delta_mb']:.1f}MB")
        else:
            print(f"  ✗ {size} docs: {perf_metrics['error']}")
    
    return results


def run_comprehensive_benchmark(retriever, corpus: Dict, queries: List[str], 
                               output_dir: Path) -> Dict[str, Any]:
    """Run comprehensive benchmarking suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE RAG BENCHMARKING SUITE")
    print("="*60)
    
    results = {
        'corpus_size': len(corpus),
        'num_queries': len(queries),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 1. Index build benchmark
    print("\n1. Index Build Performance")
    print("-" * 30)
    
    def build_index():
        retriever.build_index_from_corpus(corpus)
    
    build_metrics = measure_build_performance(build_index)
    results['index_build'] = build_metrics
    
    if build_metrics['success']:
        print(f"✓ Index built successfully")
        print(f"  Build time: {build_metrics['execution_time_s']:.2f}s")
        print(f"  Memory usage: {build_metrics['memory_delta_mb']:.1f}MB")
    else:
        print(f"✗ Index build failed: {build_metrics['error']}")
        return results
    
    # 2. Query latency benchmark
    print("\n2. Query Latency Benchmark")
    print("-" * 30)
    
    def search_query(query_text):
        return retriever.search({'test': query_text}, top_k=10)
    
    latency_results = benchmark_query_latency(search_query, queries)
    results['query_latency'] = latency_results
    
    if 'error' in latency_results:
        print(f"✗ Query benchmark failed: {latency_results['error']}")
    else:
        print(f"✓ Query latency benchmark completed")
        print(f"  Mean latency: {latency_results['mean_latency_ms']:.2f}ms")
        print(f"  P95 latency: {latency_results['p95_latency_ms']:.2f}ms")
        print(f"  Success rate: {latency_results['successful_queries']}/{latency_results['total_queries']}")
    
    # 3. Scaling benchmark
    print("\n3. Index Scaling Benchmark")
    print("-" * 30)
    
    # Test with different corpus sizes
    corpus_sizes = [100, 500, 1000, 5000, len(corpus)]
    corpus_sizes = [size for size in corpus_sizes if size <= len(corpus)]
    
    def build_subset_index(subset_corpus):
        # Create new retriever instance for subset
        from copy import deepcopy
        subset_retriever = deepcopy(retriever)
        subset_retriever.build_index_from_corpus(subset_corpus)
        return subset_retriever
    
    scaling_results = benchmark_index_scaling(build_subset_index, corpus_sizes, corpus)
    results['scaling'] = scaling_results
    
    # 4. Memory profiling
    print("\n4. Memory Profile")
    print("-" * 30)
    
    current_memory = measure_memory() / (1024 * 1024)
    results['final_memory_mb'] = current_memory
    print(f"Current memory usage: {current_memory:.1f}MB")
    
    # Save results
    output_file = output_dir / 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Benchmark results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark RAG system performance")
    parser.add_argument('--corpus', type=str, required=True,
                       help='Path to corpus JSONL file')
    parser.add_argument('--queries', type=str, required=True,
                       help='Path to queries JSONL file')
    parser.add_argument('--retriever-config', type=str, required=True,
                       help='Retriever configuration (JSON)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("RAG Benchmarking Suite")
    print("To use this tool, integrate it with your retriever system:")
    print("1. Import measure_build_performance, benchmark_query_latency")
    print("2. Pass your retriever instance to run_comprehensive_benchmark")
    print("3. Results will be saved as JSON for analysis")
    
    # Example usage would require actual retriever implementation
    print(f"\nExample: benchmark your retriever with corpus {args.corpus}")
