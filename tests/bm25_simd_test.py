"""
BM25 SIMD Test Suite for Lightning Retrieval.
Tests vectorized vs scalar BM25 scoring implementations.
"""

import numpy as np
import time
import numba as nb
from numba import float32, float64, int32, int64, boolean, jit, prange
from typing import List, Dict, Tuple
import psutil
import memory_profiler
from collections import defaultdict

# Constants for BM25 parameters
K1 = 1.2
B = 0.75

class BM25Implementations:
    @staticmethod
    def scalar_bm25(tf: np.ndarray, df: np.ndarray, doc_len: float, 
                    avg_doc_len: float, total_docs: int) -> np.ndarray:
        """Scalar implementation of BM25 scoring."""
        idf = np.log((total_docs - df + 0.5) / (df + 0.5))
        numerator = tf * (K1 + 1)
        denominator = tf + K1 * (1 - B + B * doc_len / avg_doc_len)
        return idf * numerator / denominator

    @staticmethod
    @nb.vectorize([float64(float64, float64, float64, float64, float64)])
    def numba_bm25_score(tf, df, doc_len, avg_doc_len, total_docs):
        """Numba JIT-compiled BM25 scoring."""
        idf = np.log((total_docs - df + 0.5) / (df + 0.5))
        numerator = tf * (K1 + 1)
        denominator = tf + K1 * (1 - B + B * doc_len / avg_doc_len)
        return idf * numerator / denominator

    @staticmethod
    @jit(nopython=True, parallel=True)
    def parallel_bm25(tf: np.ndarray, df: np.ndarray, doc_len: float, 
                      avg_doc_len: float, total_docs: int) -> np.ndarray:
        """Parallel implementation using Numba."""
        result = np.zeros_like(tf)
        for i in prange(len(tf)):
            idf = np.log((total_docs - df[i] + 0.5) / (df[i] + 0.5))
            numerator = tf[i] * (K1 + 1)
            denominator = tf[i] + K1 * (1 - B + B * doc_len / avg_doc_len)
            result[i] = idf * numerator / denominator
        return result

class BM25TestSuite:
    def __init__(self, num_terms: int = 100000, num_docs: int = 1000):
        """Initialize test suite with synthetic data."""
        self.num_terms = num_terms
        self.num_docs = num_docs
        self.implementations = BM25Implementations()
        self.generate_test_data()

    def generate_test_data(self):
        """Generate synthetic test data."""
        np.random.seed(42)
        self.tf = np.random.poisson(2, self.num_terms).astype(np.float64)
        self.df = np.random.randint(1, self.num_docs, self.num_terms).astype(np.float64)
        self.doc_len = np.random.normal(500, 100)
        self.avg_doc_len = 500.0
        self.total_docs = self.num_docs

    def measure_memory(self, func, *args) -> Tuple[float, float]:
        """Measure memory usage of a function."""
        process = psutil.Process()
        mem_before = process.memory_info().rss
        result = func(*args)
        mem_after = process.memory_info().rss
        mem_used = (mem_after - mem_before) / 1024 / 1024  # MB
        return result, mem_used

    def measure_performance(self, func, num_runs: int = 10, warmup: int = 3) -> Dict:
        """Measure performance metrics of an implementation."""
        args = (self.tf, self.df, self.doc_len, self.avg_doc_len, self.total_docs)
        times = []
        
        # Warmup runs
        for _ in range(warmup):
            func(*args)
            
        # Actual measurement runs
        for _ in range(num_runs):
            start = time.perf_counter()
            result, mem_used = self.measure_memory(func, *args)
            duration = time.perf_counter() - start
            times.append(duration)
            
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'memory_mb': mem_used,
            'result_checksum': np.sum(result)  # For correctness validation
        }

    def validate_correctness(self) -> Dict:
        """Validate correctness of all implementations against scalar baseline."""
        scalar_result = self.implementations.scalar_bm25(
            self.tf, self.df, self.doc_len, self.avg_doc_len, self.total_docs
        )
        
        results = {
            'numba': self.implementations.numba_bm25_score(
                self.tf, self.df, self.doc_len, self.avg_doc_len, self.total_docs
            ),
            'parallel': self.implementations.parallel_bm25(
                self.tf, self.df, self.doc_len, self.avg_doc_len, self.total_docs
            )
        }
        
        validation = {}
        for name, result in results.items():
            max_diff = np.max(np.abs(result - scalar_result))
            relative_diff = np.mean(np.abs((result - scalar_result) / scalar_result))
            validation[name] = {
                'max_absolute_diff': max_diff,
                'mean_relative_diff': relative_diff,
                'passes_tolerance': max_diff < 1e-10
            }
            
        return validation

    def run_benchmark(self) -> Dict:
        """Run comprehensive benchmark of all implementations."""
        implementations = {
            'scalar': self.implementations.scalar_bm25,
            'numba': self.implementations.numba_bm25_score,
            'parallel': self.implementations.parallel_bm25
        }
        
        results = {}
        for name, impl in implementations.items():
            print(f"Benchmarking {name} implementation...")
            results[name] = self.measure_performance(impl)
            
        # Calculate speedups relative to scalar
        scalar_time = results['scalar']['mean_time']
        for name in results:
            if name != 'scalar':
                results[name]['speedup'] = scalar_time / results[name]['mean_time']
                
        return results

    def generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        benchmark_results = self.run_benchmark()
        correctness_results = self.validate_correctness()
        
        # Determine if implementations meet the 8x speedup target
        target_speedup = 8.0
        speedup_assessment = {}
        for name, result in benchmark_results.items():
            if name != 'scalar':
                speedup = result.get('speedup', 0)
                speedup_assessment[name] = {
                    'meets_target': speedup >= target_speedup,
                    'speedup': speedup,
                    'percentage_of_target': (speedup / target_speedup) * 100
                }
        
        return {
            'performance': benchmark_results,
            'correctness': correctness_results,
            'speedup_assessment': speedup_assessment,
            'test_parameters': {
                'num_terms': self.num_terms,
                'num_docs': self.num_docs,
                'target_speedup': target_speedup
            }
        }

def main():
    """Run the BM25 test suite and print results."""
    print("\n=== BM25 SIMD Test Suite ===")
    
    # Initialize and run test suite
    test_suite = BM25TestSuite()
    report = test_suite.generate_report()
    
    # Print performance results
    print("\nPerformance Results:")
    for impl, results in report['performance'].items():
        print(f"\n{impl.capitalize()} Implementation:")
        print(f"  Mean time: {results['mean_time']*1000:.2f} ms")
        print(f"  Memory used: {results['memory_mb']:.2f} MB")
        if 'speedup' in results:
            print(f"  Speedup vs scalar: {results['speedup']:.2f}x")
            
    # Print correctness validation
    print("\nCorrectness Validation:")
    for impl, results in report['correctness'].items():
        print(f"\n{impl.capitalize()} Implementation:")
        print(f"  Max absolute difference: {results['max_absolute_diff']:.2e}")
        print(f"  Passes tolerance: {'✓' if results['passes_tolerance'] else '✗'}")
        
    # Print speedup assessment
    print("\nSpeedup Assessment (Target: 8x):")
    for impl, assessment in report['speedup_assessment'].items():
        print(f"\n{impl.capitalize()} Implementation:")
        print(f"  Meets target: {'✓' if assessment['meets_target'] else '✗'}")
        print(f"  Achieved: {assessment['speedup']:.2f}x")
        print(f"  Percentage of target: {assessment['percentage_of_target']:.1f}%")

if __name__ == '__main__':
    main()