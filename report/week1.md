# Week 1 Results

## Overview

This report contains baseline results for retrieval methods on the Natural Questions dataset.

## Results


| Dataset  | Method       | nDCG@10 | Recall@100 | Index Size (MB) | P50 Latency (ms) | P95 Latency (ms) |
|----------|-------------|---------|-------------|-----------------|------------------|------------------|


## Notes

- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **Recall@100**: Recall at rank 100  
- **Index Size**: Storage size of the retrieval index in MB
- **P50/P95 Latency**: 50th and 95th percentile query latencies in milliseconds
- **N/A**: Metric not available (e.g., missing qrels for evaluation)

## Method Details

- **BM25**: Classic sparse retrieval using Elasticsearch/Lucene
- **DPR**: Dense Passage Retrieval using BERT-based encoders
- **Contriever**: Contrastive pre-training for dense retrieval  
- **SPLADE**: Sparse Learned Dense Retrieval with BERT

## Next Steps

1. Add more datasets (MS MARCO, TREC-COVID, etc.)
2. Implement hybrid retrieval (sparse + dense)
3. Add re-ranking models
4. Optimize latency and index sizes
