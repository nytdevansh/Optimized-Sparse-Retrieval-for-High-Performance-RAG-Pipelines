# Lightning Retrieval Integration Test Results

Generated: 2025-09-19 11:03:52

## Summary

| Test Suite | Status | Key Metrics |
|------------|--------|-------------|
| bm25 | ✅ | vectorized_correct: True, numba_correct: True, ranking_correlation: True |
| quantization | ✅ | symmetric_mse: 0.0000, asymmetric_mse: 0.0000, symmetric_mae: 0.0005 |
| memory_mapping | ✅ | jsonl_creation_time: 0.1284, binary_creation_time: 0.0913, jsonl_size_mb: 29.2057 |
| topk | ✅ | heapq: True, argpartition: True, numba_partial_sort: True |

## bm25

### Metrics

- vectorized_correct: True
- numba_correct: True
- ranking_correlation: True
- reference_time: 2.0980
- vectorized_time: 0.0595
- numba_time: 0.0595
- baseline_time: 0.3169
- vectorized_speedup: 35.2518
- numba_speedup: 35.2856
- vs_baseline_vectorized: 5.3243
- vs_baseline_numba: 5.3294
- vectorized_memory_mb: 26.2819
- baseline_memory_mb: 11.0019
- memory_ratio: 2.3888

### Hardware Metrics

- current_memory_mb: 0.0MB
- peak_memory_mb: 0.6MB

## quantization

### Metrics

- symmetric_mse: 0.0000
- asymmetric_mse: 0.0000
- symmetric_mae: 0.0005
- asymmetric_mae: 0.0004
- symmetric_cosine_correlation: 1.0000
- asymmetric_cosine_correlation: 1.0000
- compression_ratio: 4.0000
- symmetric_quantize_time: 0.0124
- symmetric_dequantize_time: 0.0017
- asymmetric_quantize_time: 0.0123
- asymmetric_dequantize_time: 0.0019
- float32_time: 0.0047
- int8_time: 0.0252
- speedup: 0.1857
- score_correlation: 1.0000
- memory_reduction: 4.0000
- precision_at_1: 0.8200
- precision_at_5: 0.9240
- precision_at_10: 0.9360
- precision_at_20: 0.9550
- precision_at_50: 0.9940
- ranking_correlation: 0.9979

### Hardware Metrics

- current_memory_mb: 0.0MB
- peak_memory_mb: 161.3MB

## memory_mapping

### Metrics

- jsonl_creation_time: 0.1284
- binary_creation_time: 0.0913
- jsonl_size_mb: 29.2057
- binary_size_mb: 29.2543
- index_size_mb: 0.1526
- total_binary_size_mb: 29.4069
- compression_ratio: 0.9932
- standard_total_time: 0.0008
- standard_avg_latency_ms: 0.0008
- lazy_jsonl_total_time: 0.0235
- lazy_jsonl_avg_latency_ms: 0.0235
- memory_mapped_total_time: 0.0041
- memory_mapped_avg_latency_ms: 0.0041
- lazy_speedup: 0.0348
- mmap_speedup: 0.1987
- standard_sequential_time: 0.0010
- standard_docs_per_sec: 9617308.0809
- lazy_jsonl_sequential_time: 0.0674
- lazy_jsonl_docs_per_sec: 148303.5923
- memory_mapped_sequential_time: 0.0116
- memory_mapped_docs_per_sec: 861270.8051
- lazy_sequential_speedup: 0.0154
- mmap_sequential_speedup: 0.0896
- standard_memory_mb: 8.2188
- mmap_memory_mb: -7.8281
- memory_efficiency: -1.0499
- lazy_jsonl_cold_start_ms: 0.0926
- lazy_jsonl_cold_start_std: 0.0083
- memory_mapped_cold_start_ms: 0.0088
- memory_mapped_cold_start_std: 0.0039

## topk

### Metrics

- heapq: True
- argpartition: True
- numba_partial_sort: True
- numba_heap: True
- numba_quickselect: True

### Hardware Metrics

- current_memory_mb: 1.6MB
- peak_memory_mb: 8.6MB
