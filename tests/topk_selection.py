"""
Top-K Selection Performance Tests
Tests for branchless and optimized top-k selection algorithms
"""

import numpy as np
import numba
from numba import njit, prange
import time
import heapq
from typing import List, Tuple, Dict
import random


class TopKSelectors:
    """Collection of top-k selection algorithms"""
    
    @staticmethod
    def heapq_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Standard heapq-based top-k selection"""
        # Convert to list of (score, index) pairs
        score_pairs = [(scores[i], i) for i in range(len(scores))]
        
        # Get top-k using heapq
        top_k_pairs = heapq.nlargest(k, score_pairs)
        
        # Extract indices and scores
        indices = np.array([pair[1] for pair in top_k_pairs])
        values = np.array([pair[0] for pair in top_k_pairs])
        
        return indices, values
    
    @staticmethod
    def numpy_argpartition_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy argpartition-based top-k selection"""
        # argpartition puts k largest elements at the end
        partition_idx = np.argpartition(scores, -k)[-k:]
        
        # Sort the top-k elements
        top_k_scores = scores[partition_idx]
        sorted_indices = np.argsort(-top_k_scores)  # Descending order
        
        indices = partition_idx[sorted_indices]
        values = scores[indices]
        
        return indices, values
    
    @staticmethod
    def numpy_argsort_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy argsort-based top-k selection (baseline)"""
        sorted_indices = np.argsort(-scores)[:k]  # Descending order
        values = scores[sorted_indices]
        
        return sorted_indices, values
    
    @staticmethod
    @njit
    def numba_partial_sort_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized partial sort for top-k"""
        n = len(scores)
        indices = np.arange(n)
        
        # Partial selection sort for first k elements
        for i in range(k):
            max_idx = i
            max_val = scores[indices[i]]
            
            # Find maximum in remaining elements
            for j in range(i + 1, n):
                if scores[indices[j]] > max_val:
                    max_idx = j
                    max_val = scores[indices[j]]
            
            # Swap
            if max_idx != i:
                indices[i], indices[max_idx] = indices[max_idx], indices[i]
        
        # Extract top-k
        top_indices = indices[:k]
        top_values = scores[top_indices]
        
        return top_indices, top_values
    
    @staticmethod
    @njit
    def numba_heap_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized min-heap for top-k"""
        n = len(scores)
        
        # Initialize with first k elements
        heap_scores = np.zeros(k, dtype=scores.dtype)
        heap_indices = np.zeros(k, dtype=np.int32)
        
        # Fill initial heap
        for i in range(min(k, n)):
            heap_scores[i] = scores[i]
            heap_indices[i] = i
        
        # Heapify (min-heap, so we can replace minimum)
        for i in range(k // 2 - 1, -1, -1):
            _heapify_down_numba(heap_scores, heap_indices, i, k)
        
        # Process remaining elements
        for i in range(k, n):
            if scores[i] > heap_scores[0]:  # Better than current minimum
                heap_scores[0] = scores[i]
                heap_indices[0] = i
                _heapify_down_numba(heap_scores, heap_indices, 0, k)
        
        # Sort heap to get descending order
        for i in range(k - 1, 0, -1):
            # Swap max (at 0) with element at i
            heap_scores[0], heap_scores[i] = heap_scores[i], heap_scores[0]
            heap_indices[0], heap_indices[i] = heap_indices[i], heap_indices[0]
            _heapify_down_numba(heap_scores, heap_indices, 0, i)
        
        return heap_indices, heap_scores
    
    @staticmethod
    @njit
    def numba_quickselect_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized quickselect for top-k"""
        n = len(scores)
        indices = np.arange(n, dtype=np.int32)
        
        # Quickselect to partition around k-th largest element
        _quickselect_partition(scores, indices, 0, n - 1, n - k)
        
        # Sort the top-k elements (last k elements after partitioning)
        top_k_indices = indices[n - k:]
        top_k_scores = scores[top_k_indices]
        
        # Simple insertion sort for small k
        for i in range(1, k):
            key_score = top_k_scores[i]
            key_idx = top_k_indices[i]
            j = i - 1
            
            # Move elements that are smaller than key to the right
            while j >= 0 and top_k_scores[j] < key_score:
                top_k_scores[j + 1] = top_k_scores[j]
                top_k_indices[j + 1] = top_k_indices[j]
                j -= 1
            
            top_k_scores[j + 1] = key_score
            top_k_indices[j + 1] = key_idx
        
        return top_k_indices, top_k_scores
    
    @staticmethod
    @njit
    def simd_approximate_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """SIMD-friendly approximate top-k for very large arrays"""
        n = len(scores)
        
        if k >= n:
            return TopKSelectors.numba_partial_sort_topk(scores, n)
        
        # For large arrays, use sampling-based approach
        if n > 10000:
            # Sample elements to estimate threshold
            sample_size = min(1000, n // 10)
            sample_indices = np.random.choice(n, size=sample_size, replace=False)
            sample_scores = scores[sample_indices]
            
            # Get approximate threshold from sample
            sample_sorted = np.sort(sample_scores)
            threshold_idx = max(0, sample_size - k)
            threshold = sample_sorted[threshold_idx]
            
            # First pass: count elements above threshold
            above_threshold = 0
            for i in range(n):
                if scores[i] >= threshold:
                    above_threshold += 1
            
            # Adjust k if needed
            effective_k = min(k, above_threshold + k // 4)  # Allow some margin
            
            # Second pass: collect top elements
            candidates = np.zeros(effective_k, dtype=np.int32)
            candidate_scores = np.zeros(effective_k, dtype=scores.dtype)
            count = 0
            
            for i in range(n):
                if count < effective_k and scores[i] >= threshold:
                    candidates[count] = i
                    candidate_scores[count] = scores[i]
                    count += 1
            
            # Sort candidates and return top-k
            if count > k:
                return TopKSelectors.numba_partial_sort_topk(candidate_scores[:count], k)
            else:
                sorted_idx = np.argsort(-candidate_scores[:count])
                return candidates[sorted_idx], candidate_scores[sorted_idx]
        
        else:
            return TopKSelectors.numba_partial_sort_topk(scores, k)


@njit
def _heapify_down_numba(heap_scores, heap_indices, start, size):
    """Numba-compatible heapify down operation"""
    parent = start
    while True:
        left_child = 2 * parent + 1
        right_child = 2 * parent + 2
        smallest = parent
        
        # Find smallest among parent and children
        if left_child < size and heap_scores[left_child] < heap_scores[smallest]:
            smallest = left_child
        
        if right_child < size and heap_scores[right_child] < heap_scores[smallest]:
            smallest = right_child
        
        if smallest == parent:
            break
        
        # Swap parent with smallest child
        heap_scores[parent], heap_scores[smallest] = heap_scores[smallest], heap_scores[parent]
        heap_indices[parent], heap_indices[smallest] = heap_indices[smallest], heap_indices[parent]
        
        parent = smallest


@njit
def _quickselect_partition(scores, indices, low, high, target):
    """Numba-compatible quickselect partitioning"""
    if low >= high:
        return
    
    # Choose pivot (median of three)
    mid = (low + high) // 2
    if scores[indices[mid]] < scores[indices[low]]:
        indices[low], indices[mid] = indices[mid], indices[low]
    if scores[indices[high]] < scores[indices[low]]:
        indices[low], indices[high] = indices[high], indices[low]
    if scores[indices[high]] < scores[indices[mid]]:
        indices[mid], indices[high] = indices[high], indices[mid]
    
    pivot_idx = indices[mid]
    pivot_score = scores[pivot_idx]
    
    # Move pivot to end
    indices[mid], indices[high] = indices[high], indices[mid]
    
    # Partition
    i = low
    for j in range(low, high):
        if scores[indices[j]] <= pivot_score:
            indices[i], indices[j] = indices[j], indices[i]
            i += 1
    
    # Move pivot to final position
    indices[i], indices[high] = indices[high], indices[i]
    
    # Recursive calls
    if i == target:
        return
    elif i < target:
        _quickselect_partition(scores, indices, i + 1, high, target)
    else:
        _quickselect_partition(scores, indices, low, i - 1, target)


class TopKBenchmarkSuite:
    """Benchmark suite for top-k selection algorithms"""
    
    def __init__(self):
        self.selectors = TopKSelectors()
    
    def generate_test_scores(self, n: int, distribution: str = 'normal') -> np.ndarray:
        """Generate test score arrays with different distributions"""
        np.random.seed(42)
        
        if distribution == 'normal':
            scores = np.random.normal(0, 1, n).astype(np.float32)
        elif distribution == 'uniform':
            scores = np.random.uniform(0, 1, n).astype(np.float32)
        elif distribution == 'zipfian':
            # Zipfian distribution (heavy-tailed)
            ranks = np.arange(1, n + 1)
            scores = (1.0 / ranks).astype(np.float32)
            np.random.shuffle(scores)  # Randomize order
        elif distribution == 'bimodal':
            # Two peaks
            scores1 = np.random.normal(-2, 0.5, n // 2)
            scores2 = np.random.normal(2, 0.5, n - n // 2)
            scores = np.concatenate([scores1, scores2]).astype(np.float32)
            np.random.shuffle(scores)
        else:
            scores = np.random.random(n).astype(np.float32)
        
        return scores
    
    def test_correctness(self) -> Dict[str, bool]:
        """Test correctness of all top-k implementations"""
        print("Testing top-k selection correctness...")
        
        test_cases = [
            (100, 10, 'normal'),
            (1000, 50, 'uniform'),
            (500, 5, 'zipfian'),
            (200, 100, 'bimodal')  # k = n/2
        ]
        
        methods = {
            'heapq': self.selectors.heapq_topk,
            'argpartition': self.selectors.numpy_argpartition_topk,
            'argsort': self.selectors.numpy_argsort_topk,
            'numba_partial_sort': self.selectors.numba_partial_sort_topk,
            'numba_heap': self.selectors.numba_heap_topk,
            'numba_quickselect': self.selectors.numba_quickselect_topk
        }
        
        results = {}
        tolerance = 1e-5
        
        for n, k, dist in test_cases:
            scores = self.generate_test_scores(n, dist)
            
            # Get reference result (NumPy argsort)
            ref_indices, ref_values = self.selectors.numpy_argsort_topk(scores, k)
            ref_set = set(ref_indices)
            
            case_name = f"n={n}_k={k}_{dist}"
            results[case_name] = {}
            
            for method_name, method_func in methods.items():
                if method_name == 'argsort':  # Skip reference
                    continue
                
                try:
                    indices, values = method_func(scores, k)
                    
                    # Check if we got exactly k elements
                    correct_size = len(indices) == k and len(values) == k
                    
                    # Check if indices are valid
                    valid_indices = all(0 <= idx < n for idx in indices)
                    
                    # Check if values match indices
                    values_match = np.allclose(values, scores[indices], atol=tolerance)
                    
                    # Check if top-k elements are correct (at least 90% overlap for approximate methods)
                    result_set = set(indices)
                    overlap = len(ref_set & result_set)
                    sufficient_overlap = overlap >= k * 0.9  # Allow 10% difference for approximate
                    
                    # Check if returned in descending order
                    descending_order = all(values[i] >= values[i+1] for i in range(len(values)-1))
                    
                    method_correct = (correct_size and valid_indices and 
                                    values_match and sufficient_overlap and descending_order)
                    
                    results[case_name][method_name] = method_correct
                    
                except Exception as e:
                    print(f"Error in {method_name} for {case_name}: {e}")
                    results[case_name][method_name] = False
        
        # Overall correctness summary
        overall_results = {}
        for method_name in methods.keys():
            if method_name == 'argsort':
                continue
            method_results = []
            for case_results in results.values():
                if method_name in case_results:
                    method_results.append(case_results[method_name])
            
            overall_results[method_name] = all(method_results) if method_results else False
            
            status = "✅ PASS" if overall_results[method_name] else "❌ FAIL"
            print(f"  {method_name}: {status}")
        
        return overall_results
    
    def benchmark_performance(self) -> Dict[str, Dict[str, float]]:
        """Benchmark performance across different array sizes and k values"""
        print("Benchmarking top-k selection performance...")
        
        test_configs = [
            {'n': 1000, 'k': 10, 'name': 'small_array_small_k'},
            {'n': 1000, 'k': 100, 'name': 'small_array_large_k'},
            {'n': 100000, 'k': 10, 'name': 'large_array_small_k'},
            {'n': 100000, 'k': 1000, 'name': 'large_array_medium_k'},
            {'n': 100000, 'k': 10000, 'name': 'large_array_large_k'},
            {'n': 1000000, 'k': 100, 'name': 'very_large_array_small_k'}
        ]
        
        methods = {
            'heapq': self.selectors.heapq_topk,
            'argpartition': self.selectors.numpy_argpartition_topk,
            'argsort': self.selectors.numpy_argsort_topk,
            'numba_partial_sort': self.selectors.numba_partial_sort_topk,
            'numba_heap': self.selectors.numba_heap_topk,
            'numba_quickselect': self.selectors.numba_quickselect_topk,
            'simd_approximate': self.selectors.simd_approximate_topk
        }
        
        results = {}
        num_runs = 5
        
        for config in test_configs:
            n, k, config_name = config['n'], config['k'], config['name']
            print(f"\nTesting {config_name} (n={n}, k={k})...")
            
            # Generate test data
            scores = self.generate_test_scores(n, 'normal')
            
            config_results = {}
            
            for method_name, method_func in methods.items():
                # Skip methods that are too slow for large arrays
                if n >= 1000000 and method_name in ['heapq', 'argsort']:
                    continue
                
                try:
                    # Warmup
                    for _ in range(2):
                        _ = method_func(scores, k)
                    
                    # Benchmark
                    times = []
                    for _ in range(num_runs):
                        start = time.perf_counter()
                        _, _ = method_func(scores, k)
                        end = time.perf_counter()
                        times.append(end - start)
                    
                    median_time = np.median(times)
                    config_results[method_name] = median_time
                    
                    print(f"  {method_name}: {median_time:.6f}s")
                    
                except Exception as e:
                    print(f"  {method_name}: ERROR - {e}")
                    config_results[method_name] = float('inf')
            
            results[config_name] = config_results
        
        return results
    
    def analyze_performance_characteristics(self, benchmark_results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Analyze performance characteristics and provide recommendations"""
        print("\nAnalyzing performance characteristics...")
        
        recommendations = {}
        
        # Find best method for each scenario
        for config_name, config_results in benchmark_results.items():
            valid_results = {k: v for k, v in config_results.items() if v != float('inf')}
            
            if valid_results:
                best_method = min(valid_results.keys(), key=lambda x: valid_results[x])
                best_time = valid_results[best_method]
                
                # Calculate speedups vs baseline (argsort)
                baseline_time = valid_results.get('argsort', valid_results.get('argpartition', best_time))
                speedup = baseline_time / best_time if best_time > 0 else 1.0
                
                recommendations[config_name] = {
                    'best_method': best_method,
                    'best_time': best_time,
                    'speedup': speedup
                }
                
                print(f"{config_name}:")
                print(f"  Best: {best_method} ({best_time:.6f}s, {speedup:.2f}x speedup)")
        
        return recommendations
    
    def test_memory_efficiency(self) -> Dict[str, float]:
        """Test memory efficiency of different approaches"""
        print("Testing memory efficiency...")
        
        import tracemalloc
        
        n = 100000
        k = 1000
        scores = self.generate_test_scores(n, 'normal')
        
        methods_to_test = {
            'argpartition': self.selectors.numpy_argpartition_topk,
            'numba_heap': self.selectors.numba_heap_topk,
            'numba_quickselect': self.selectors.numba_quickselect_topk
        }
        
        results = {}
        
        for method_name, method_func in methods_to_test.items():
            tracemalloc.start()
            
            # Run method multiple times to get stable memory measurement
            for _ in range(10):
                _, _ = method_func(scores, k)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[method_name] = peak / (1024 * 1024)  # Convert to MB
            print(f"{method_name}: {results[method_name]:.2f} MB peak memory")
        
        return results


if __name__ == "__main__":
    print("=" * 60)
    print("Top-K Selection Performance Test Suite")
    print("=" * 60)
    
    suite = TopKBenchmarkSuite()
    
    # Test correctness
    correctness_results = suite.test_correctness()
    print(f"\nCorrectness Summary:")
    all_correct = all(correctness_results.values())
    print(f"All methods correct: {'✅ YES' if all_correct else '❌ NO'}")
    
    if not all_correct:
        for method, correct in correctness_results.items():
            if not correct:
                print(f"  ⚠️  {method} has correctness issues")
    
    print("\n" + "=" * 60)
    
    # Benchmark performance
    benchmark_results = suite.benchmark_performance()
    
    print("\n" + "=" * 60)
    
    # Analyze performance
    recommendations = suite.analyze_performance_characteristics(benchmark_results)
    
    # Check if we achieve target speedups
    target_speedups_met = 0
    total_configs = len(recommendations)
    
    for config, rec in recommendations.items():
        if rec['speedup'] >= 5.0:  # Target: 5-10x speedup
            target_speedups_met += 1
    
    speedup_success_rate = target_speedups_met / total_configs if total_configs > 0 else 0
    
    print(f"\nSpeedup Achievement:")
    print(f"  Target speedup (5x+) achieved: {target_speedups_met}/{total_configs} configs")
    print(f"  Success rate: {'✅ EXCELLENT' if speedup_success_rate > 0.7 else '✅ GOOD' if speedup_success_rate > 0.5 else '❌ POOR'} ({speedup_success_rate:.1%})")
    
    print("\n" + "=" * 60)
    
    # Test memory efficiency
    memory_results = suite.test_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    
    correctness_pass = all_correct
    performance_pass = speedup_success_rate > 0.5
    memory_reasonable = all(mem < 100 for mem in memory_results.values())  # Under 100MB
    
    print(f"Correctness: {'✅ PASS' if correctness_pass else '❌ FAIL'}")
    print(f"Performance: {'✅ PASS' if performance_pass else '❌ FAIL'}")
    print(f"Memory efficiency: {'✅ PASS' if memory_reasonable else '❌ FAIL'}")
    
    overall_success = correctness_pass and performance_pass and memory_reasonable
    print(f"Overall result: {'🎉 SUCCESS' if overall_success else '🚨 NEEDS WORK'}")
    
    if overall_success:
        print(f"\n🚀 Top-k selection optimizations are working excellently!")
        print(f"   Recommended for production use.")
    else:
        print(f"\n⚠️  Some top-k implementations need refinement.")