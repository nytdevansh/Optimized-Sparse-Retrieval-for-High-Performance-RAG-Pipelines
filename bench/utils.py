import os
import time
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

def measure_size(path):
    """Rough index size in MB"""
    if not path or not os.path.exists(path):
        return 0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)

def log_latency(start, end, n_queries):
    """Return avg latency in ms"""
    if n_queries == 0:
        return 0
    return round((end - start) * 1000 / n_queries, 2)

def measure_query_latencies(retriever, corpus, queries, sample_size=100):
    """
    Measure per-query latencies and return P50/P95 metrics
    
    Args:
        retriever: BEIR retriever object
        corpus: Document corpus
        queries: Query dictionary
        sample_size: Number of queries to sample for latency measurement
    
    Returns:
        Dict with P50, P95, and mean latencies in ms
    """
    query_ids = list(queries.keys())
    
    # Sample queries if we have too many
    if len(query_ids) > sample_size:
        import random
        query_ids = random.sample(query_ids, sample_size)
    
    latencies = []
    
    print(f"Measuring latency on {len(query_ids)} queries...")
    
    for i, qid in enumerate(query_ids):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(query_ids)}")
            
        # Create single query dict
        single_query = {qid: queries[qid]}
        
        # Time single query
        start = time.time()
        _ = retriever.retrieve(corpus, single_query)
        end = time.time()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
    
    latencies = np.array(latencies)
    
    return {
        "mean_latency_ms": round(np.mean(latencies), 2),
        "p50_latency_ms": round(np.percentile(latencies, 50), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "queries_measured": len(latencies)
    }

def estimate_index_build_time(method: str, num_docs: int) -> str:
    """Estimate index build time based on method and corpus size"""
    estimates = {
        "bm25": num_docs / 10000,  # ~10k docs per second
        "dpr": num_docs / 100,     # ~100 docs per second (encoding)  
        "contriever": num_docs / 100,
        "splade": num_docs / 50    # Slower due to sparse encoding
    }
    
    minutes = estimates.get(method, num_docs / 100) / 60
    
    if minutes < 1:
        return f"~{int(minutes * 60)}s"
    elif minutes < 60:
        return f"~{int(minutes)}m"
    else:
        return f"~{int(minutes/60)}h{int(minutes%60)}m"

def format_results_table(results_dir: Path) -> str:
    """
    Generate markdown table from JSON results
    """
    import json
    
    table_header = """
| Dataset  | Method       | nDCG@10 | Recall@100 | Index Size (MB) | P50 Latency (ms) | P95 Latency (ms) |
|----------|-------------|---------|-------------|-----------------|------------------|------------------|"""
    
    rows = []
    
    # Load all result files
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
            
            dataset = data.get("dataset", "Unknown")
            method = data.get("method", "Unknown").upper()
            ndcg = f"{data.get('nDCG@10', 0):.4f}" if data.get('nDCG@10') else "N/A"
            recall = f"{data.get('Recall@100', 0):.4f}" if data.get('Recall@100') else "N/A"
            size = data.get("index_size_mb", 0)
            p50 = data.get("p50_latency_ms", "N/A")
            p95 = data.get("p95_latency_ms", "N/A")
            
            row = f"| {dataset.upper():<8} | {method:<11} | {ndcg:>7} | {recall:>10} | {size:>15} | {p50:>16} | {p95:>16} |"
            rows.append(row)
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
    
    return table_header + "\n" + "\n".join(sorted(rows))

def create_report_template(results_dir: Path, output_file: Path):
    """Create week1.md report with results"""
    
    table = format_results_table(results_dir)
    
    report = f"""# Week 1 Results

## Overview

This report contains baseline results for retrieval methods on the Natural Questions dataset.

## Results

{table}

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
"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report generated: {output_file}")