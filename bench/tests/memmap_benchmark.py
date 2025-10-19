"""
Memory-mapped corpus benchmark implementation
"""

import os
from pathlib import Path
import time
import gc
import psutil
from typing import Dict, Any, List
from bench.core.benchmark_framework import BenchmarkSuite, BenchmarkResult
from bench.core.memmap_retrieval import MemoryMappedRetrieval
from beir.datasets.data_loader import GenericDataLoader

class MemoryMappedBenchmark(BenchmarkSuite):
    def __init__(self, dataset_name: str = "fiqa", config: Dict[str, Any] = None):
        name = f"memmap_{dataset_name}"
        category = "memory_mapping"
        super().__init__(name=name, category=category)
        
        self.dataset_name = dataset_name
        self.config = config or {}
        
        # Default configuration
        self.buffer_size = self.config.get('buffer_size', 8 * 1024 * 1024)
        self.cache_size = self.config.get('cache_size', 50000)
        self.compression_level = self.config.get('compression_level', 1)
        self.alignment = self.config.get('alignment', 16)
        
    def setup(self) -> None:
        """Set up memory-mapped corpus and load dataset."""
        try:
            # Load dataset
            data_path = Path("datasets") / self.dataset_name
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found: {data_path}")
            
            # Check required files
            if not (data_path / "corpus.jsonl").exists():
                raise FileNotFoundError(f"Corpus file not found: {data_path}/corpus.jsonl")
            if not (data_path / "queries.jsonl").exists():
                raise FileNotFoundError(f"Queries file not found: {data_path}/queries.jsonl")
                
            self.data_loader = GenericDataLoader(str(data_path))
            # Load corpus and queries, ignore qrels for benchmarking
            corpus_data, queries_data, qrels_data = self.data_loader.load(split="test")
            
            # Validate dataset format
            if not isinstance(corpus_data, dict) or not corpus_data:
                raise ValueError(f"Invalid corpus format or empty corpus in {self.dataset_name}")
            if not isinstance(queries_data, dict) or not queries_data:
                raise ValueError(f"Invalid queries format or empty queries in {self.dataset_name}")
            
            self.corpus = corpus_data
            self.queries = queries_data
            self.qrels = qrels_data
            
            # Initialize retriever
            self.retriever = MemoryMappedRetrieval(
                buffer_size=self.buffer_size,
                cache_size=self.cache_size,
                compression_level=self.compression_level,
                alignment=self.alignment
            )
            
            # Index corpus
            self.stats = self.retriever.index(self.corpus)
            
        except Exception as e:
            print(f"\n⚠️  Error setting up {self.dataset_name} dataset: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        if hasattr(self, 'retriever'):
            self.retriever.cleanup()
        
        # Clear references to help with garbage collection
        self.data_loader = None
        self.corpus = None
        self.queries = None
        self.qrels = None
        self.retriever = None
        self.stats = None

    def run(self) -> BenchmarkResult:
        """Run memory mapping benchmarks and collect metrics"""
        try:
            # Initialize metrics containers
            metrics: Dict[str, float] = {}
            timings: Dict[str, float] = {}
            memory_stats: Dict[str, float] = {}
            
            if not hasattr(self, 'corpus') or not self.corpus:
                raise RuntimeError("No corpus data available. Setup may have failed.")
            
            if not hasattr(self, 'queries') or not self.queries:
                raise RuntimeError("No query data available. Setup may have failed.")
                
            if not hasattr(self, 'stats'):
                raise RuntimeError("Missing corpus statistics. Setup may have failed.")
            
            # Get dataset metrics
            metrics["num_documents"] = len(self.corpus)
            metrics["num_queries"] = len(self.queries)
            metrics["compression_ratio"] = self.stats.get('compression_ratio', 1.0)
            
            # Performance testing
            process = psutil.Process()
            gc.collect()  # Clear caches before memory test
            
            # Memory baseline
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_stats["baseline_mb"] = baseline_memory
            
            # Random access performance
            doc_indices = list(range(0, len(self.corpus), max(1, len(self.corpus) // 100)))  # Sample 1%
            start_time = time.perf_counter()
            for idx in doc_indices:
                _ = self.retriever.corpus[idx]
            access_time = time.perf_counter() - start_time
            timings["random_access"] = access_time
            metrics["docs_per_second"] = len(doc_indices) / access_time
            
            # Memory peak
            peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_stats["peak_mb"] = peak_memory
            memory_stats["overhead_mb"] = peak_memory - baseline_memory
            
            # Sequential access performance
            start_time = time.perf_counter()
            batch_size = 100
            total_docs = 0
            for i in range(0, len(self.corpus), batch_size):
                batch = self.retriever.corpus.get_batch(i, batch_size)
                total_docs += len(batch)
            seq_time = time.perf_counter() - start_time
            timings["sequential_access"] = seq_time
            metrics["sequential_docs_per_second"] = total_docs / seq_time
            
            # System info
            hardware_info = {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024 * 1024),  # MB
                'platform': str(os.uname())
            }
            
            # Configuration parameters
            parameters = {
                'buffer_size': self.buffer_size,
                'cache_size': self.cache_size,
                'compression_level': self.compression_level,
                'alignment': self.alignment,
                'dataset': self.dataset_name
            }
            
            return BenchmarkResult(
                name=self.name,
                category=self.category,
                metrics=metrics,
                timings=timings,
                memory=memory_stats,
                hardware_info=hardware_info,
                parameters=parameters,
                success=True
            )
            
        except Exception as e:
            print(f"\n⚠️  Benchmark error in {self.name}: {str(e)}")
            return BenchmarkResult(
                name=self.name,
                category=self.category,
                metrics={},
                timings={},
                memory={},
                hardware_info={},
                parameters={},
                success=False,
                error=str(e)
            )