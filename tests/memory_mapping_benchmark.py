"""
Memory Mapping Benchmarks on Real Datasets
Tests memory mapping performance on FIQA and NQ datasets
"""

import os
import json
import time
from pathlib import Path
import numpy as np
from memory_mapping import (
    MemoryMappingTestSuite,
    BinaryCorpusBuilder,
    StandardCorpusReader,
    MemoryMappedCorpus
)

class DatasetBenchmark:
    """Benchmark memory mapping on real datasets"""
    
    def __init__(self, datasets_dir: str):
        self.datasets_dir = Path(datasets_dir)
        self.test_suite = MemoryMappingTestSuite()
        self.results = {}
    
    def load_jsonl_documents(self, jsonl_path: str) -> list:
        """Load documents from a JSONL file"""
        documents = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                doc = json.loads(line.strip())
                documents.append(doc)
        return documents
    
    def benchmark_dataset(self, dataset_name: str) -> dict:
        """Run benchmarks on a specific dataset"""
        dataset_path = self.datasets_dir / dataset_name
        corpus_path = dataset_path / 'corpus.jsonl'
        
        if not corpus_path.exists():
            raise ValueError(f"Corpus file not found: {corpus_path}")
        
        print(f"\n{'='*60}")
        print(f"Benchmarking {dataset_name} dataset")
        print(f"{'='*60}")
        
        # Load dataset stats
        doc_count = sum(1 for _ in open(corpus_path))
        file_size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
        print(f"Dataset size: {file_size_mb:.1f}MB, {doc_count} documents")
        
        # Create binary format
        binary_path = dataset_path / 'corpus.bin'
        start_time = time.perf_counter()
        documents = self.load_jsonl_documents(corpus_path)
        load_time = time.perf_counter() - start_time
        
        # Set paths in test suite
        self.test_suite.corpus_path = str(corpus_path)
        self.test_suite.binary_path = str(binary_path)
        
        print(f"Loading time: {load_time:.2f}s")
        
        # Build binary corpus
        builder = BinaryCorpusBuilder()
        start_time = time.perf_counter()
        stats = builder.build_binary_corpus(documents, str(binary_path))
        build_time = time.perf_counter() - start_time
        
        # Store paths for memory testing
        self.test_suite.corpus_path = str(corpus_path)
        self.test_suite.binary_path = str(binary_path)
        
        binary_size = os.path.getsize(binary_path)
        index_size = os.path.getsize(binary_path.with_suffix('.idx'))
        total_binary_size_mb = (binary_size + index_size) / (1024 * 1024)
        
        print(f"Binary conversion time: {build_time:.2f}s")
        print(f"Binary format size: {total_binary_size_mb:.1f}MB")
        print(f"Compression ratio: {file_size_mb/total_binary_size_mb:.2f}x")
        
        # Run performance tests
        results = {
            'dataset': dataset_name,
            'doc_count': doc_count,
            'original_size_mb': file_size_mb,
            'binary_size_mb': total_binary_size_mb,
            'compression_ratio': file_size_mb/total_binary_size_mb,
            'load_time': load_time,
            'build_time': build_time
        }
        
        # Memory usage test
        memory_results = self.test_suite.test_memory_usage(documents)
        results.update({
            'standard_memory_mb': memory_results['standard_memory_mb'],
            'mmap_memory_mb': memory_results['mmap_memory_mb'],
            'memory_efficiency': memory_results['memory_efficiency']
        })
        
        # Random access test
        random_results = self.test_suite.test_random_access_performance(documents)
        results.update({
            'standard_latency_ms': random_results['standard_avg_latency_ms'],
            'mmap_latency_ms': random_results['memory_mapped_avg_latency_ms'],
            'random_access_speedup': random_results['mmap_speedup']
        })
        
        # Sequential access test
        seq_results = self.test_suite.test_sequential_access_performance(documents)
        results.update({
            'standard_seq_time': seq_results['standard_sequential_time'],
            'mmap_seq_time': seq_results['memory_mapped_sequential_time'],
            'sequential_speedup': seq_results['mmap_sequential_speedup']
        })
        
        # Cold start test
        cold_results = self.test_suite.test_cold_start_performance(documents)
        results.update({
            'cold_start_latency_ms': cold_results['memory_mapped_cold_start_ms'],
            'cold_start_std_ms': cold_results['memory_mapped_cold_start_std']
        })
        
        print("\nPerformance Summary:")
        print(f"Memory efficiency: {results['memory_efficiency']:.2f}x")
        print(f"Random access speedup: {results['random_access_speedup']:.2f}x")
        print(f"Sequential access speedup: {results['sequential_speedup']:.2f}x")
        print(f"Cold start latency: {results['cold_start_latency_ms']:.2f}±{results['cold_start_std_ms']:.2f}ms")
        
        return results
    
    def run_all_benchmarks(self) -> dict:
        """Run benchmarks on all available datasets"""
        datasets = ['fiqa', 'nq']
        all_results = {}
        
        for dataset in datasets:
            try:
                results = self.benchmark_dataset(dataset)
                all_results[dataset] = results
            except Exception as e:
                print(f"Error benchmarking {dataset}: {e}")
        
        return all_results
    
    def save_results(self, results: dict, output_path: str):
        """Save benchmark results to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    datasets_dir = Path(__file__).parent.parent / "datasets"
    output_path = Path(__file__).parent.parent / "results" / "memory_mapping_benchmarks.json"
    
    benchmark = DatasetBenchmark(datasets_dir)
    results = benchmark.run_all_benchmarks()
    benchmark.save_results(results, output_path)