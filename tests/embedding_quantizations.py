"""
Embedding Quantization Performance Tests
Tests for int8 quantization of dense embeddings
"""

import numpy as np
import numba
from numba import njit, prange
import time
import faiss
from typing import Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class EmbeddingQuantizer:
    """High-performance embedding quantization utilities"""
    
    @staticmethod
    @njit
    def quantize_symmetric(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric quantization: [-scale, scale] -> [-127, 127]"""
        num_vectors, dim = embeddings.shape
        quantized = np.zeros((num_vectors, dim), dtype=np.int8)
        scales = np.zeros(num_vectors, dtype=np.float32)
        
        for i in range(num_vectors):
            # Find max absolute value
            max_val = 0.0
            for j in range(dim):
                abs_val = abs(embeddings[i, j])
                if abs_val > max_val:
                    max_val = abs_val
            
            # Calculate scale
            scale = max_val / 127.0 if max_val > 0 else 1.0
            scales[i] = scale
            
            # Quantize
            for j in range(dim):
                quantized[i, j] = np.int8(embeddings[i, j] / scale)
        
        return quantized, scales
    
    @staticmethod
    @njit
    def dequantize_symmetric(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize symmetric quantized embeddings"""
        num_vectors, dim = quantized.shape
        embeddings = np.zeros((num_vectors, dim), dtype=np.float32)
        
        for i in range(num_vectors):
            scale = scales[i]
            for j in range(dim):
                embeddings[i, j] = quantized[i, j] * scale
        
        return embeddings
    
    @staticmethod
    @njit
    def quantize_asymmetric(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asymmetric quantization: [min, max] -> [0, 255]"""
        num_vectors, dim = embeddings.shape
        quantized = np.zeros((num_vectors, dim), dtype=np.uint8)
        scales = np.zeros(num_vectors, dtype=np.float32)
        zero_points = np.zeros(num_vectors, dtype=np.float32)
        
        for i in range(num_vectors):
            # Find min and max
            min_val = embeddings[i, 0]
            max_val = embeddings[i, 0]
            
            for j in range(1, dim):
                if embeddings[i, j] < min_val:
                    min_val = embeddings[i, j]
                if embeddings[i, j] > max_val:
                    max_val = embeddings[i, j]
            
            # Calculate scale and zero point
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            zero_point = min_val
            
            scales[i] = scale
            zero_points[i] = zero_point
            
            # Quantize
            for j in range(dim):
                quantized[i, j] = np.uint8((embeddings[i, j] - zero_point) / scale)
        
        return quantized, scales, zero_points
    
    @staticmethod
    @njit
    def dequantize_asymmetric(quantized: np.ndarray, scales: np.ndarray, 
                            zero_points: np.ndarray) -> np.ndarray:
        """Dequantize asymmetric quantized embeddings"""
        num_vectors, dim = quantized.shape
        embeddings = np.zeros((num_vectors, dim), dtype=np.float32)
        
        for i in range(num_vectors):
            scale = scales[i]
            zero_point = zero_points[i]
            for j in range(dim):
                embeddings[i, j] = quantized[i, j] * scale + zero_point
        
        return embeddings


class SIMDDotProduct:
    """SIMD-optimized dot product implementations"""
    
    @staticmethod
    @njit(parallel=True)
    def int8_dot_product_batch(queries_int8: np.ndarray, 
                              corpus_int8: np.ndarray,
                              query_scales: np.ndarray,
                              corpus_scales: np.ndarray) -> np.ndarray:
        """Batch int8 dot product with scaling"""
        num_queries, dim = queries_int8.shape
        num_corpus, _ = corpus_int8.shape
        
        scores = np.zeros((num_queries, num_corpus), dtype=np.float32)
        
        for q in prange(num_queries):
            query_scale = query_scales[q]
            for c in range(num_corpus):
                corpus_scale = corpus_scales[c]
                combined_scale = query_scale * corpus_scale
                
                # Int8 dot product
                dot_product = 0
                for d in range(dim):
                    dot_product += int(queries_int8[q, d]) * int(corpus_int8[c, d])
                
                scores[q, c] = dot_product * combined_scale
        
        return scores
    
    @staticmethod
    def float32_dot_product_batch(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Standard float32 dot product using NumPy"""
        return np.dot(queries, corpus.T)
    
    @staticmethod
    @njit
    def simd_cosine_similarity_int8(vec_a_int8: np.ndarray, vec_b_int8: np.ndarray,
                                   scale_a: float, scale_b: float) -> float:
        """SIMD-optimized cosine similarity for int8 vectors"""
        dim = len(vec_a_int8)
        
        # Compute dot product
        dot_product = 0
        norm_a_sq = 0
        norm_b_sq = 0
        
        for i in range(dim):
            a_val = int(vec_a_int8[i])
            b_val = int(vec_b_int8[i])
            
            dot_product += a_val * b_val
            norm_a_sq += a_val * a_val
            norm_b_sq += b_val * b_val
        
        # Apply scaling
        scaled_dot = dot_product * scale_a * scale_b
        scaled_norm_a = np.sqrt(norm_a_sq) * scale_a
        scaled_norm_b = np.sqrt(norm_b_sq) * scale_b
        
        if scaled_norm_a == 0 or scaled_norm_b == 0:
            return 0.0
        
        return scaled_dot / (scaled_norm_a * scaled_norm_b)


class QuantizationTestSuite:
    """Test suite for embedding quantization"""
    
    def __init__(self):
        self.quantizer = EmbeddingQuantizer()
        self.simd_ops = SIMDDotProduct()
    
    def generate_test_embeddings(self, num_vectors: int = 10000, 
                                dim: int = 768, seed: int = 42) -> np.ndarray:
        """Generate realistic test embeddings"""
        np.random.seed(seed)
        
        # Generate embeddings with realistic distribution
        # Mix of Gaussian clusters (simulating sentence embeddings)
        num_clusters = 50
        cluster_assignments = np.random.choice(num_clusters, size=num_vectors)
        
        embeddings = np.zeros((num_vectors, dim), dtype=np.float32)
        
        # Generate cluster centers
        cluster_centers = np.random.normal(0, 0.5, size=(num_clusters, dim))
        
        for i in range(num_vectors):
            cluster_id = cluster_assignments[i]
            # Add noise around cluster center
            embedding = cluster_centers[cluster_id] + np.random.normal(0, 0.2, size=dim)
            
            # Normalize to unit length (common for sentence embeddings)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
            
            embeddings[i] = embedding.astype(np.float32)
        
        return embeddings
    
    def test_quantization_quality(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Test quality preservation after quantization"""
        print("Testing quantization quality...")
        
        # Test symmetric quantization
        quant_sym, scales_sym = self.quantizer.quantize_symmetric(embeddings)
        dequant_sym = self.quantizer.dequantize_symmetric(quant_sym, scales_sym)
        
        # Test asymmetric quantization
        quant_asym, scales_asym, zero_points = self.quantizer.quantize_asymmetric(embeddings)
        dequant_asym = self.quantizer.dequantize_asymmetric(quant_asym, scales_asym, zero_points)
        
        # Calculate reconstruction errors
        mse_sym = np.mean((embeddings - dequant_sym) ** 2)
        mse_asym = np.mean((embeddings - dequant_asym) ** 2)
        
        mae_sym = np.mean(np.abs(embeddings - dequant_sym))
        mae_asym = np.mean(np.abs(embeddings - dequant_asym))
        
        # Test cosine similarity preservation
        # Sample pairs for cosine similarity test
        num_pairs = 1000
        indices = np.random.choice(len(embeddings), size=(num_pairs, 2), replace=False)
        
        cos_sim_original = []
        cos_sim_sym = []
        cos_sim_asym = []
        
        for i, j in indices:
            # Original
            cos_orig = np.dot(embeddings[i], embeddings[j])
            cos_sim_original.append(cos_orig)
            
            # Symmetric quantization
            cos_s = np.dot(dequant_sym[i], dequant_sym[j])
            cos_sim_sym.append(cos_s)
            
            # Asymmetric quantization
            cos_a = np.dot(dequant_asym[i], dequant_asym[j])
            cos_sim_asym.append(cos_a)
        
        # Calculate correlation
        cos_corr_sym = np.corrcoef(cos_sim_original, cos_sim_sym)[0, 1]
        cos_corr_asym = np.corrcoef(cos_sim_original, cos_sim_asym)[0, 1]
        
        results = {
            'symmetric_mse': float(mse_sym),
            'asymmetric_mse': float(mse_asym),
            'symmetric_mae': float(mae_sym),
            'asymmetric_mae': float(mae_asym),
            'symmetric_cosine_correlation': float(cos_corr_sym),
            'asymmetric_cosine_correlation': float(cos_corr_asym),
            'compression_ratio': 4.0  # float32 to int8 = 4x compression
        }
        
        print(f"Symmetric quantization:")
        print(f"  MSE: {mse_sym:.6f}")
        print(f"  MAE: {mae_sym:.6f}")
        print(f"  Cosine correlation: {cos_corr_sym:.4f}")
        
        print(f"Asymmetric quantization:")
        print(f"  MSE: {mse_asym:.6f}")
        print(f"  MAE: {mae_asym:.6f}")
        print(f"  Cosine correlation: {cos_corr_asym:.4f}")
        
        return results
    
    def benchmark_quantization_speed(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Benchmark quantization and dequantization speed"""
        print("Benchmarking quantization speed...")
        
        num_runs = 10
        
        # Benchmark symmetric quantization
        print("Warming up symmetric quantization...")
        for _ in range(3):
            _ = self.quantizer.quantize_symmetric(embeddings)
        
        print("Benchmarking symmetric quantization...")
        sym_quant_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            quant_sym, scales_sym = self.quantizer.quantize_symmetric(embeddings)
            end = time.perf_counter()
            sym_quant_times.append(end - start)
        
        sym_dequant_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.quantizer.dequantize_symmetric(quant_sym, scales_sym)
            end = time.perf_counter()
            sym_dequant_times.append(end - start)
        
        # Benchmark asymmetric quantization
        print("Benchmarking asymmetric quantization...")
        asym_quant_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            quant_asym, scales_asym, zero_points = self.quantizer.quantize_asymmetric(embeddings)
            end = time.perf_counter()
            asym_quant_times.append(end - start)
        
        asym_dequant_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.quantizer.dequantize_asymmetric(quant_asym, scales_asym, zero_points)
            end = time.perf_counter()
            asym_dequant_times.append(end - start)
        
        results = {
            'symmetric_quantize_time': np.median(sym_quant_times),
            'symmetric_dequantize_time': np.median(sym_dequant_times),
            'asymmetric_quantize_time': np.median(asym_quant_times),
            'asymmetric_dequantize_time': np.median(asym_dequant_times)
        }
        
        print(f"Symmetric quantization: {results['symmetric_quantize_time']:.4f}s")
        print(f"Symmetric dequantization: {results['symmetric_dequantize_time']:.4f}s")
        print(f"Asymmetric quantization: {results['asymmetric_quantize_time']:.4f}s")
        print(f"Asymmetric dequantization: {results['asymmetric_dequantize_time']:.4f}s")
        
        return results
    
    def benchmark_dot_product_performance(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Benchmark int8 vs float32 dot product performance"""
        print("Benchmarking dot product performance...")
        
        # Prepare data
        num_queries = 100
        query_embeddings = embeddings[:num_queries]
        corpus_embeddings = embeddings[num_queries:num_queries+5000]
        
        # Quantize embeddings
        query_int8, query_scales = self.quantizer.quantize_symmetric(query_embeddings)
        corpus_int8, corpus_scales = self.quantizer.quantize_symmetric(corpus_embeddings)
        
        num_runs = 5
        
        # Benchmark float32 dot product
        print("Warming up float32 dot product...")
        for _ in range(3):
            _ = self.simd_ops.float32_dot_product_batch(query_embeddings, corpus_embeddings)
        
        print("Benchmarking float32 dot product...")
        float32_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            float32_scores = self.simd_ops.float32_dot_product_batch(query_embeddings, corpus_embeddings)
            end = time.perf_counter()
            float32_times.append(end - start)
        
        # Benchmark int8 dot product
        print("Warming up int8 dot product...")
        for _ in range(3):
            _ = self.simd_ops.int8_dot_product_batch(query_int8, corpus_int8, query_scales, corpus_scales)
        
        print("Benchmarking int8 dot product...")
        int8_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            int8_scores = self.simd_ops.int8_dot_product_batch(query_int8, corpus_int8, query_scales, corpus_scales)
            end = time.perf_counter()
            int8_times.append(end - start)
        
        float32_time = np.median(float32_times)
        int8_time = np.median(int8_times)
        speedup = float32_time / int8_time
        
        # Test score correlation
        correlation = np.corrcoef(float32_scores.flatten(), int8_scores.flatten())[0, 1]
        
        results = {
            'float32_time': float32_time,
            'int8_time': int8_time,
            'speedup': speedup,
            'score_correlation': correlation,
            'memory_reduction': 4.0  # float32 to int8
        }
        
        print(f"Float32 dot product time: {float32_time:.4f}s")
        print(f"Int8 dot product time: {int8_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Score correlation: {correlation:.4f}")
        
        return results
    
    def test_retrieval_quality_impact(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Test impact of quantization on retrieval quality"""
        print("Testing retrieval quality impact...")
        
        # Prepare query and corpus
        num_queries = 50
        query_embeddings = embeddings[:num_queries]
        corpus_embeddings = embeddings[num_queries:num_queries+2000]
        
        # Quantize
        query_int8, query_scales = self.quantizer.quantize_symmetric(query_embeddings)
        corpus_int8, corpus_scales = self.quantizer.quantize_symmetric(corpus_embeddings)
        
        # Calculate similarities
        float32_scores = np.dot(query_embeddings, corpus_embeddings.T)
        int8_scores = self.simd_ops.int8_dot_product_batch(query_int8, corpus_int8, query_scales, corpus_scales)
        
        # Calculate ranking metrics
        k_values = [1, 5, 10, 20, 50]
        metrics = {}
        
        for k in k_values:
            precision_at_k = []
            
            for i in range(num_queries):
                # Get top-k from float32
                float32_topk = np.argsort(-float32_scores[i])[:k]
                
                # Get top-k from int8
                int8_topk = np.argsort(-int8_scores[i])[:k]
                
                # Calculate precision@k
                overlap = len(set(float32_topk) & set(int8_topk))
                precision = overlap / k
                precision_at_k.append(precision)
            
            metrics[f'precision_at_{k}'] = np.mean(precision_at_k)
        
        # Calculate ranking correlation
        ranking_correlations = []
        for i in range(num_queries):
            float32_ranking = np.argsort(-float32_scores[i])
            int8_ranking = np.argsort(-int8_scores[i])
            
            # Spearman correlation on top-100
            top_n = min(100, len(float32_ranking))
            float32_ranks = {doc_id: rank for rank, doc_id in enumerate(float32_ranking[:top_n])}
            int8_ranks = {doc_id: rank for rank, doc_id in enumerate(int8_ranking[:top_n])}
            
            common_docs = set(float32_ranks.keys()) & set(int8_ranks.keys())
            if len(common_docs) > 10:
                float32_common = [float32_ranks[doc] for doc in common_docs]
                int8_common = [int8_ranks[doc] for doc in common_docs]
                
                from scipy.stats import spearmanr
                corr, _ = spearmanr(float32_common, int8_common)
                ranking_correlations.append(corr)
        
        metrics['ranking_correlation'] = np.mean(ranking_correlations)
        
        print(f"Retrieval quality metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Quantization Test Suite")
    print("=" * 60)
    
    suite = QuantizationTestSuite()
    
    # Generate test embeddings
    print("Generating test embeddings...")
    embeddings = suite.generate_test_embeddings(num_vectors=10000, dim=768)
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    print("\n" + "=" * 60)
    
    # Test quantization quality
    quality_results = suite.test_quantization_quality(embeddings)
    print("\nQuality Assessment:")
    sym_quality = quality_results['symmetric_cosine_correlation']
    asym_quality = quality_results['asymmetric_cosine_correlation']
    print(f"  Symmetric quantization: {'✅ GOOD' if sym_quality > 0.95 else '❌ POOR'} ({sym_quality:.4f})")
    print(f"  Asymmetric quantization: {'✅ GOOD' if asym_quality > 0.95 else '❌ POOR'} ({asym_quality:.4f})")
    
    print("\n" + "=" * 60)
    
    # Benchmark quantization speed
    speed_results = suite.benchmark_quantization_speed(embeddings)
    
    print("\n" + "=" * 60)
    
    # Benchmark dot product performance
    dot_product_results = suite.benchmark_dot_product_performance(embeddings)
    speedup_achieved = dot_product_results['speedup']
    print(f"\nDot Product Performance:")
    print(f"  Target speedup (3x): {'✅ ACHIEVED' if speedup_achieved >= 3.0 else '❌ NOT ACHIEVED'}")
    print(f"  Actual speedup: {speedup_achieved:.2f}x")
    
    print("\n" + "=" * 60)
    
    # Test retrieval quality impact
    retrieval_results = suite.test_retrieval_quality_impact(embeddings)
    p_at_10 = retrieval_results['precision_at_10']
    ranking_corr = retrieval_results['ranking_correlation']
    
    print(f"\nRetrieval Quality Impact:")
    print(f"  Precision@10 preservation: {'✅ GOOD' if p_at_10 > 0.90 else '❌ DEGRADED'} ({p_at_10:.4f})")
    print(f"  Ranking correlation: {'✅ GOOD' if ranking_corr > 0.95 else '❌ POOR'} ({ranking_corr:.4f})")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    
    quality_pass = sym_quality > 0.95 and asym_quality > 0.95
    performance_pass = speedup_achieved >= 3.0
    retrieval_pass = p_at_10 > 0.90 and ranking_corr > 0.95
    
    print(f"Quality preservation: {'✅ PASS' if quality_pass else '❌ FAIL'}")
    print(f"Performance improvement: {'✅ PASS' if performance_pass else '❌ FAIL'}")
    print(f"Retrieval quality: {'✅ PASS' if retrieval_pass else '❌ FAIL'}")
    
    overall_success = quality_pass and performance_pass and retrieval_pass
    print(f"Overall result: {'🎉 SUCCESS' if overall_success else '🚨 NEEDS WORK'}")