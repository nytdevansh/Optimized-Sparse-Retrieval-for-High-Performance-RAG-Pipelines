"""
BM25 SIMD Benchmark Implementation using the core framework
"""

import numpy as np
from typing import Dict, Any, List
import numba as nb
from numba import float32, float64, int32, boolean, jit, prange
from pathlib import Path
import time
from scipy.stats import spearmanr
from rank_bm25 import BM25Okapi

from bench.core.benchmark_framework import (
    BenchmarkSuite,
    BenchmarkResult,
    TimingContext,
    MemoryMonitor
)

class BM25SIMDBenchmark(BenchmarkSuite):
    """BM25 SIMD optimization benchmark suite"""
    
    def __init__(self, num_docs: int = 10000, num_queries: int = 100):
        super().__init__(name="BM25-SIMD", category="retrieval")
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.test_corpus = []
        self.test_queries = []
        
    def setup(self) -> None:
        """Generate synthetic test data with realistic distributions"""
        np.random.seed(42)
        
        # Generate vocabulary with Zipfian distribution
        vocab_size = 5000
        vocab = [f"term_{i}" for i in range(vocab_size)]
        zipf_probs = 1.0 / np.arange(1, vocab_size + 1)
        zipf_probs /= zipf_probs.sum()
        
        # Generate documents
        print(f"Generating {self.num_docs} test documents...")
        self.test_corpus = []
        for _ in range(self.num_docs):
            doc_length = np.random.gamma(2, 25)  # Average ~50 words
            doc_length = max(5, min(200, int(doc_length)))
            doc_terms = np.random.choice(vocab, size=doc_length, p=zipf_probs)
            self.test_corpus.append(doc_terms.tolist())
        
        # Generate queries
        print(f"Generating {self.num_queries} test queries...")
        self.test_queries = []
        for _ in range(self.num_queries):
            query_length = np.random.gamma(1, 3)  # Average ~3 words
            query_length = max(1, min(10, int(query_length)))
            query_probs = zipf_probs[:vocab_size//4]  # Queries use more frequent terms
            query_probs /= query_probs.sum()
            query_terms = np.random.choice(vocab[:vocab_size//4], size=query_length, p=query_probs)
            self.test_queries.append(query_terms.tolist())
    
    @staticmethod
    @jit(nopython=True)
    def _numba_bm25_score(tf: np.ndarray, df: np.ndarray, doc_len: float, 
                         avg_doc_len: float, total_docs: int, k1: float = 1.2, b: float = 0.75) -> float:
        """Numba-optimized BM25 scoring"""
        score = 0.0
        for i in range(len(tf)):
            if tf[i] > 0:
                idf = np.log((total_docs - df[i] + 0.5) / (df[i] + 0.5))
                numerator = tf[i] * (k1 + 1)
                denominator = tf[i] + k1 * (1 - b + b * doc_len / avg_doc_len)
                score += idf * numerator / denominator
        return score

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _numba_bm25_batch(query_tf: np.ndarray, doc_tfs: np.ndarray, 
                         doc_lens: np.ndarray, df: np.ndarray, avg_doc_len: float, 
                         total_docs: int, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        """Parallel BM25 scoring for multiple documents"""
        scores = np.zeros(len(doc_tfs))
        for i in prange(len(doc_tfs)):
            scores[i] = BM25SIMDBenchmark._numba_bm25_score(
                doc_tfs[i], df, doc_lens[i], avg_doc_len, total_docs, k1, b
            )
        return scores

    def run(self) -> BenchmarkResult:
        """Run the BM25 SIMD benchmark suite"""
        memory_monitor = MemoryMonitor()
        timings = {}
        metrics = {}
        memory = {}
        
        try:
            # Initialize implementations
            print("Initializing BM25 implementations...")
            memory_monitor.start()
            
            # Standard BM25 (rank_bm25)
            baseline_bm25 = BM25Okapi(self.test_corpus)
            memory_monitor.sample()
            baseline_mem = memory_monitor.get_statistics()
            
            # Prepare vectorized data
            vocab = sorted(set(word for doc in self.test_corpus for word in doc))
            vocab_map = {word: idx for idx, word in enumerate(vocab)}
            doc_tfs = np.zeros((len(self.test_corpus), len(vocab)))
            doc_lens = np.array([len(doc) for doc in self.test_corpus])
            
            for doc_idx, doc in enumerate(self.test_corpus):
                for word in doc:
                    doc_tfs[doc_idx, vocab_map[word]] += 1
            
            df = np.sum(doc_tfs > 0, axis=0)
            avg_doc_len = np.mean(doc_lens)
            memory_monitor.sample()
            vectorized_mem = memory_monitor.get_statistics()
            
            # Benchmark search performance
            print("\nBenchmarking search performance...")
            
            # Baseline performance
            with TimingContext(warmup_runs=3) as baseline_timer:
                for query in self.test_queries:
                    _ = baseline_bm25.get_scores(query)
            timings['baseline'] = baseline_timer.statistics
            
            # Vectorized performance
            with TimingContext(warmup_runs=3) as vec_timer:
                for query in self.test_queries:
                    query_tf = np.zeros(len(vocab))
                    for word in query:
                        if word in vocab_map:
                            query_tf[vocab_map[word]] += 1
                            
                    # Vectorized BM25 computation
                    _ = self._numba_bm25_batch(
                        query_tf, doc_tfs, doc_lens, df, 
                        avg_doc_len, len(self.test_corpus)
                    )
            timings['vectorized'] = vec_timer.statistics
            
            # Calculate metrics
            metrics['speedup'] = timings['baseline']['mean'] / timings['vectorized']['mean']
            metrics['memory_ratio'] = vectorized_mem['peak_mb'] / baseline_mem['peak_mb']
            
            # Validate correctness
            print("\nValidating correctness...")
            query_sample = self.test_queries[:10]  # Use subset for validation
            
            baseline_rankings = []
            vectorized_rankings = []
            
            for query in query_sample:
                # Baseline rankings
                baseline_scores = baseline_bm25.get_scores(query)
                baseline_ranks = np.argsort(-np.array(baseline_scores))
                baseline_rankings.append(baseline_ranks)
                
                # Vectorized rankings
                query_tf = np.zeros(len(vocab))
                for word in query:
                    if word in vocab_map:
                        query_tf[vocab_map[word]] += 1
                
                vec_scores = self._numba_bm25_batch(
                    query_tf, doc_tfs, doc_lens, df, 
                    avg_doc_len, len(self.test_corpus)
                )
                vec_ranks = np.argsort(-vec_scores)
                vectorized_rankings.append(vec_ranks)
            
            # Calculate rank correlation
            correlations = []
            for b_ranks, v_ranks in zip(baseline_rankings, vectorized_rankings):
                corr, _ = spearmanr(b_ranks[:100], v_ranks[:100])  # Top-100 correlation
                correlations.append(corr)
            
            metrics['rank_correlation'] = np.mean(correlations)
            
            # Prepare final metrics
            memory.update({
                'baseline_peak_mb': baseline_mem['peak_mb'],
                'vectorized_peak_mb': vectorized_mem['peak_mb'],
                'memory_reduction_factor': baseline_mem['peak_mb'] / vectorized_mem['peak_mb']
            })
            
            return BenchmarkResult(
                name=self.name,
                category=self.category,
                metrics=metrics,
                timings={k: v['mean'] for k, v in timings.items()},
                memory=memory,
                hardware_info=self.get_hardware_info(),
                parameters={
                    'num_docs': self.num_docs,
                    'num_queries': self.num_queries,
                    'vocab_size': len(vocab)
                },
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                name=self.name,
                category=self.category,
                metrics={},
                timings={},
                memory={},
                hardware_info=self.get_hardware_info(),
                parameters={},
                success=False,
                error=str(e)
            )
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.test_corpus = []
        self.test_queries = []