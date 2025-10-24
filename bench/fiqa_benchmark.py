#!/usr/bin/env python3
"""
FiQA Dataset Benchmark Runner
Runs comprehensive benchmarks on the FiQA dataset using multiple retrieval methods.
Features:
- Automatic dataset download and validation
- Multiple retrieval methods (BM25, DPR, Contriever, SPLADE)
- CPU-optimized for reproducibility
- Detailed performance metrics and reporting
"""

import json
import time
import statistics
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Force CPU for reproducibility
import torch
torch.backends.mps.enabled = False
device = torch.device("cpu")

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

try:
    from rank_bm25 import BM25Okapi
    print("✅ rank_bm25 imported successfully")
except ImportError:
    print("❌ Please install rank_bm25: pip install rank_bm25")
    BM25Okapi = None

# Constants
RESULTS_DIR = Path("results")
DATA_DIR = Path("datasets/fiqa")
DOWNLOAD_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"

# Available retrieval methods
METHODS = [
    {"name": "bm25_custom", "model": None},
    {"name": "dpr", "model": "sentence-transformers/facebook-dpr-question_encoder-single-nq-base"},
    {"name": "contriever", "model": "facebook/contriever"},
    {"name": "splade", "model": "naver/splade-v3-distil"}
]

class DatasetManager:
    """Handles dataset download, extraction, and validation"""
    
    @staticmethod
    def download_dataset():
        """Download and extract FiQA dataset if not available"""
        if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
            print(f"✅ Dataset already exists at {DATA_DIR}")
            return True
        
        print(f"📥 Downloading dataset from {DOWNLOAD_URL}")
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        zip_path = datasets_dir / "fiqa.zip"
        
        try:
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size / total_size) * 100)
                    print(f"\r📥 Downloading: {percent:.1f}%", end="", flush=True)
            
            urllib.request.urlretrieve(DOWNLOAD_URL, zip_path, reporthook=show_progress)
            print("\n✅ Download complete")
            
            print("📂 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(datasets_dir)
            
            zip_path.unlink()
            print(f"✅ Extraction complete")
            return True
            
        except Exception as e:
            print(f"❌ Download/extraction failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return False
    
    @staticmethod
    def load_dataset():
        """Load the FiQA dataset"""
        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=str(DATA_DIR)).load(split="test")
            print(f"✅ Dataset loaded successfully:")
            print(f"   📄 Documents: {len(corpus):,}")
            print(f"   ❓ Queries: {len(queries):,}")
            print(f"   📋 Relevance judgments: {len(qrels):,}")
            return corpus, queries, qrels
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            return None, None, None

class CustomBM25Model:
    """BM25 implementation compatible with BEIR interface"""
    
    def __init__(self, corpus):
        print("🔧 Building BM25 index...")
        self.corpus = corpus
        self.corpus_ids = list(corpus.keys())
        
        # Prepare and tokenize corpus
        corpus_texts = []
        for doc_id in self.corpus_ids:
            doc = corpus[doc_id]
            title = doc.get('title', '')
            text = doc.get('text', '')
            combined = f"{title} {text}".strip()
            tokens = combined.lower().split()
            corpus_texts.append(tokens)
        
        self.bm25 = BM25Okapi(corpus_texts)
    
    def encode_queries(self, queries, batch_size=32, **kwargs):
        """Process queries for BM25"""
        return queries
    
    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        """Process corpus for BM25"""
        return None

class CustomBM25Search:
    """Search interface for BM25"""
    
    def __init__(self, model):
        self.model = model
    
    def search(self, queries, corpus_embeddings=None, query_embeddings=None, top_k=10):
        results = {}
        for query_id, query in tqdm(queries.items(), desc="Searching"):
            query_tokens = query.lower().split()
            doc_scores = self.model.bm25.get_scores(query_tokens)
            
            # Get top-k results
            top_k_idx = np.argsort(doc_scores)[-top_k:][::-1]
            results[query_id] = {
                self.model.corpus_ids[idx]: float(doc_scores[idx])
                for idx in top_k_idx if doc_scores[idx] > 0
            }
        return results

def initialize_method(method, corpus=None):
    """Initialize a retrieval method"""
    name = method["name"]
    
    if name == "bm25_custom":
        if BM25Okapi is None or corpus is None:
            return None
        return CustomBM25Search(CustomBM25Model(corpus))
    
    try:
        print(f"Initializing {method['model']} on CPU...")
        model = models.SentenceBERT(method["model"], device=str(device))
        return DRES(model, batch_size=8, score_function="dot")  # Add score_function parameter
    except Exception as e:
        print(f"❌ Initialization failed for {name}: {e}")
        return None

def run_benchmark(corpus, queries, qrels, method_cfg):
    """Run benchmark for a single method"""
    name = method_cfg["name"]
    print(f"\n🔍 Testing {name}...")
    
    # Initialize model
    retriever = initialize_method(method_cfg, corpus)
    if retriever is None:
        return None
    
    # Run retrieval
    start = time.time()
    if name == "bm25_custom":
        results = retriever.search(queries, corpus, top_k=100)
    else:
        # For dense retrievers (DPR, Contriever, SPLADE)
        results = retriever.search(queries, corpus, top_k=100, score_function="dot")
    end = time.time()
    
    n_queries = len(queries)
    avg_latency = (end - start) / max(1, n_queries)
    
    # Evaluate results
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [10, 100])
    
    # Save detailed results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"fiqa_{name}_results.json", "w") as f:
        json.dump(results, f)
    
    # Prepare summary
    summary = {
        "method": name,
        "n_queries": n_queries,
        "n_docs": len(corpus),
        "total_time_s": round(end - start, 3),
        "avg_latency_s": round(avg_latency, 4),
        "nDCG@10": round(ndcg["NDCG@10"], 4),
        "nDCG@100": round(ndcg["NDCG@100"], 4),
        "MAP@10": round(_map["MAP@10"], 4),
        "MAP@100": round(_map["MAP@100"], 4),
        "Recall@10": round(recall["Recall@10"], 4),
        "Recall@100": round(recall["Recall@100"], 4),
        "P@10": round(precision["P@10"], 4),
        "device": "cpu"
    }
    
    with open(RESULTS_DIR / f"fiqa_{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

def generate_report(summaries, corpus, queries):
    """Generate comprehensive benchmark report"""
    
    # Console summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"{'Method':<15} {'nDCG@10':<8} {'Recall@100':<11} {'MAP@100':<9} {'Latency(s)':<10}")
    print("-" * 60)
    for s in summaries:
        print(f"{s['method']:<15} {s['nDCG@10']:<8.4f} {s['Recall@100']:<11.4f} {s['MAP@100']:<9.4f} {s['avg_latency_s']:<10.4f}")
    
    # Markdown report
    md = "# FiQA Benchmark Results\n\n"
    md += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += f"**Dataset**: FiQA-2018 ({len(corpus):,} docs, {len(queries)} queries)\n"
    md += f"**Environment**: CPU-only (for reproducibility)\n"
    md += f"**Methods Tested**: {len(summaries)}\n\n"
    
    md += "## Results\n\n"
    md += "|Method|nDCG@10|nDCG@100|MAP@10|MAP@100|Recall@10|Recall@100|P@10|Latency(s)|\n"
    md += "|------|--------|---------|-------|--------|----------|-----------|-----|----------|\n"
    
    for s in summaries:
        md += f"|{s['method']}|{s['nDCG@10']:.4f}|{s['nDCG@100']:.4f}|{s['MAP@10']:.4f}|{s['MAP@100']:.4f}|{s['Recall@10']:.4f}|{s['Recall@100']:.4f}|{s['P@10']:.4f}|{s['avg_latency_s']:.4f}|\n"
    
    md += "\n## Expected Performance Ranges\n\n"
    md += "Based on BEIR paper and community results:\n\n"
    md += "- **BM25**: nDCG@10 ~0.23-0.26, Recall@100 ~0.64-0.70\n"
    md += "- **DPR**: nDCG@10 ~0.22-0.28, Recall@100 ~0.60-0.75\n"
    md += "- **Contriever**: nDCG@10 ~0.25-0.30, Recall@100 ~0.65-0.80\n"
    md += "- **SPLADE**: nDCG@10 ~0.27-0.32, Recall@100 ~0.70-0.85\n"
    
    with open(RESULTS_DIR / "fiqa_benchmark_report.md", "w") as f:
        f.write(md)
    
    # CSV for analysis
    csv_lines = ["method,ndcg_10,recall_100,map_100,latency_s,total_time_s"]
    for s in summaries:
        csv_lines.append(f"{s['method']},{s['nDCG@10']},{s['Recall@100']},{s['MAP@100']},{s['avg_latency_s']},{s['total_time_s']}")
    
    with open(RESULTS_DIR / "fiqa_results.csv", "w") as f:
        f.write("\n".join(csv_lines))

def main():
    """Main benchmark runner"""
    print("🚀 FiQA Benchmark Runner")
    print("=" * 70)
    print(f"PyTorch Device: {device}")
    print("=" * 70)
    
    # Prepare dataset
    if not DatasetManager.download_dataset():
        return
    
    corpus, queries, qrels = DatasetManager.load_dataset()
    if not all([corpus, queries, qrels]):
        return
    
    # Run benchmarks
    summaries = []
    total_methods = len(METHODS)
    
    for i, method in enumerate(METHODS, 1):
        print(f"\n📍 Progress: {i}/{total_methods}")
        summary = run_benchmark(corpus, queries, qrels, method)
        if summary:
            summaries.append(summary)
    
    if not summaries:
        print("\n❌ All methods failed. Check dependencies and try again.")
        return
    
    # Generate reports
    generate_report(summaries, corpus, queries)
    
    print("\n✅ Benchmark complete!")
    print(f"📊 Results saved in {RESULTS_DIR}/")
    print("Files generated:")
    print("- Individual results: fiqa_*_results.json")
    print("- Method summaries: fiqa_*_summary.json")
    print("- Full report: fiqa_benchmark_report.md")
    print("- CSV data: fiqa_results.csv")

if __name__ == "__main__":
    main()