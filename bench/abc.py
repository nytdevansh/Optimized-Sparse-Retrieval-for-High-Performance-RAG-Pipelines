#!/usr/bin/env python3
"""
Run FiQA retrieval experiments - M1 PROOF VERSION 🚫🍏
Forces CPU for everything. MPS is YEETED into the trash bin.
Auto-downloads FiQA dataset if not available.
Saves results JSON and comparison tables.
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

# 🚫 NUKE MPS FROM ORBIT - IT'S THE ONLY WAY TO BE SURE
import torch
torch.backends.mps.enabled = False
device = torch.device("cpu") 
print(f"🚫 MPS DISABLED. Using {device} for ALL models. No more tensor dimension bullsh*t!")

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

# Import rank_bm25 for the fixed BM25 implementation
try:
    from rank_bm25 import BM25Okapi
    print("✅ rank_bm25 imported successfully")
except ImportError:
    print("❌ Please install rank_bm25: pip install rank_bm25")
    BM25Okapi = None

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("datasets/fiqa")
DOWNLOAD_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"

# All methods - now that MPS is dead, we can try everything
METHODS = [
    {"name": "bm25_custom", "model": None},
    {"name": "dpr", "model": "sentence-transformers/facebook-dpr-question_encoder-single-nq-base"},
    {"name": "contriever", "model": "facebook/contriever"},
    {"name": "splade", "model": "naver/splade-v3-distil"},  # Why not try it now?
]

def download_fiqa_dataset():
    """Download and extract FiQA dataset if not available"""
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"✅ FiQA dataset already exists at {DATA_DIR}")
        return True
    
    print(f"📥 Downloading FiQA dataset from {DOWNLOAD_URL}")
    
    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = datasets_dir / "fiqa.zip"
    
    try:
        # Download with progress bar
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size / total_size) * 100)
                print(f"\r📥 Downloading: {percent:.1f}%", end="", flush=True)
        
        print("🌐 Starting download...")
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path, reporthook=show_progress)
        print("\n✅ Download completed!")
        
        # Extract zip file
        print("📂 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        # Verify extraction
        if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
            print(f"✅ FiQA dataset extracted successfully to {DATA_DIR}")
            return True
        else:
            print(f"❌ Dataset extraction failed - {DATA_DIR} is empty or missing")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

def verify_dataset_structure():
    """Verify that the dataset has the expected structure"""
    expected_files = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
    
    for file_path in expected_files:
        full_path = DATA_DIR / file_path
        if not full_path.exists():
            print(f"❌ Missing expected file: {full_path}")
            return False
    
    print("✅ Dataset structure verified")
    return True

def install_missing_dependencies():
    """Check and install missing dependencies"""
    missing_deps = []
    
    # Check for required packages
    try:
        import beir
    except ImportError:
        missing_deps.append("beir")
    
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        missing_deps.append("rank_bm25")
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    if missing_deps:
        print("❌ Missing dependencies detected:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ All required dependencies available")
    return True

# Custom BM25 model that mimics BEIR's SentenceBERT interface
class CustomBM25Model:
    def __init__(self, corpus):
        print("🔧 Building BM25 index...")
        self.corpus = corpus
        self.corpus_ids = list(corpus.keys())
        
        # Prepare corpus texts for BM25
        corpus_texts = []
        for doc_id in self.corpus_ids:
            doc = corpus[doc_id]
            # Combine title and text, handle missing fields gracefully
            title = doc.get('title', '')
            text = doc.get('text', '')
            combined_text = f"{title} {text}".strip()
            corpus_texts.append(combined_text)
        
        # Tokenize corpus (simple whitespace tokenization + lowercasing)
        print("📝 Tokenizing corpus...")
        tokenized_corpus = []
        for text in tqdm(corpus_texts, desc="Tokenizing"):
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)
        
        # Build BM25 index
        print("🏗️ Creating BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25 index ready!")
    
    def encode_queries(self, queries, batch_size=32, **kwargs):
        """Encode queries - for BM25, we just return the queries as-is"""
        return list(queries.values())
    
    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        """Encode corpus - for BM25, we don't need to encode, return dummy embeddings"""
        return np.zeros((len(corpus), 1))

# Custom search class that uses rank_bm25
class CustomBM25Search:
    def __init__(self, model, batch_size=128, corpus_chunk_size=50000):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.results = {}
    
    def search(self, corpus, queries, top_k, score_function, **kwargs):
        """Search method that mimics BEIR's search interface"""
        results = {}
        
        print(f"🔍 Processing {len(queries)} queries...")
        for query_id, query_text in tqdm(queries.items(), desc="Searching"):
            # Tokenize query
            tokenized_query = query_text.lower().split()
            
            # Get BM25 scores
            scores = self.model.bm25.get_scores(tokenized_query)
            
            # Get top-k results
            if len(scores) > 0:
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                # Format results as expected by BEIR evaluation
                query_results = {}
                for idx in top_indices:
                    if idx < len(self.model.corpus_ids) and scores[idx] > 0:
                        doc_id = self.model.corpus_ids[idx]
                        query_results[doc_id] = float(scores[idx])
                
                results[query_id] = query_results
            else:
                results[query_id] = {}
        
        return results

def safe_init_model(method, corpus=None):
    name = method["name"]
    
    if name == "bm25_custom":
        if BM25Okapi is None:
            print("❌ rank_bm25 not available. Please install: pip install rank_bm25")
            return None
        if corpus is None:
            print("❌ Corpus required for BM25 initialization")
            return None
        
        # Create custom BM25 model and search
        bm25_model = CustomBM25Model(corpus)
        return CustomBM25Search(bm25_model)
        
    else:
        model_name = method.get("model")
        try:
            print(f"Initializing SentenceBERT wrapper for {model_name} on CPU...")
            print("💪 CPU POWER ENGAGED - No MPS BS here!")
            
            # Force device to CPU - this is the key fix
            model = models.SentenceBERT(model_name, device=str(device))
            return DRES(model, batch_size=8)  # Smaller batch for CPU
            
        except Exception as e:
            print(f"❌ Failed to init model {model_name}: {e}")
            return None

def run_single(corpus, queries, qrels, method_cfg):
    name = method_cfg["name"]
    print(f"\n" + "="*50)
    print(f"=== Running method: {name} ===")
    print("="*50)
    
    model = safe_init_model(method_cfg, corpus=corpus)
    if model is None:
        print(f"💀 Skipping {name} due to init failure.")
        return None

    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    print("Running retrieval (full query set)...")
    t0 = time.time()
    try:
        results = retriever.retrieve(corpus, queries)
        print("✅ No MPS errors! CPU FTW!")
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        if "MPS" in str(e) or "expand" in str(e):
            print("🍎 This looks like an MPS issue, but MPS should be disabled...")
        return None
        
    t1 = time.time()
    total_time = t1 - t0
    n_queries = len(queries)
    avg_latency = total_time / max(1, n_queries)

    # Evaluate
    print("📊 Evaluating performance...")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[10, 100])

    # Save raw results for later analysis
    out_json = RESULTS_DIR / f"fiqa_{name}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    summary = {
        "method": name,
        "n_queries": n_queries,
        "n_docs": len(corpus),
        "total_time_s": round(total_time, 3),
        "avg_latency_s": round(avg_latency, 4),
        "nDCG@10": round(ndcg.get("NDCG@10", 0), 4),
        "nDCG@100": round(ndcg.get("NDCG@100", 0), 4),
        "MAP@10": round(_map.get("MAP@10", 0), 4),
        "MAP@100": round(_map.get("MAP@100", 0), 4),
        "Recall@10": round(recall.get("Recall@10", 0), 4),
        "Recall@100": round(recall.get("Recall@100", 0), 4),
        "P@10": round(precision.get("P@10", 0), 4),
        "device": "cpu",
        "notes": "MPS disabled - CPU only"
    }

    # Try to collect per-query latency samples
    try:
        per_query_times = []
        sample_qids = list(queries.keys())[:min(20, n_queries)]
        print(f"📊 Collecting latency from {len(sample_qids)} sample queries...")
        
        for qid in sample_qids:
            qt = queries[qid]
            start = time.time()
            _ = retriever.retrieve(corpus, {qid: qt})
            per_query_times.append(time.time() - start)
            
        if per_query_times:
            summary["p50_latency_s"] = round(statistics.median(per_query_times), 4)
            summary["p95_latency_s"] = round(sorted(per_query_times)[int(0.95 * len(per_query_times)) - 1], 4)
        else:
            summary["p50_latency_s"] = None
            summary["p95_latency_s"] = None
    except Exception as e:
        summary["notes"] += f"; Latency timing failed: {str(e)[:50]}"
        summary["p50_latency_s"] = None
        summary["p95_latency_s"] = None

    # Save summary
    out_summary = RESULTS_DIR / f"fiqa_{name}_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to {out_summary}")
    
    # Print quick results
    print(f"🎯 Quick Results:")
    print(f"   nDCG@10: {summary['nDCG@10']:.4f}")
    print(f"   Recall@100: {summary['Recall@100']:.4f}") 
    print(f"   Avg Latency: {summary['avg_latency_s']:.4f}s")
    print(f"   Total Time: {summary['total_time_s']:.1f}s")
    
    return summary

def main():
    print("🚫🍏 MPS-FREE FiQA BEIR Benchmark - CPU ENFORCED VERSION")
    print("=" * 70)
    print(f"PyTorch Device: {device}")
    print(f"MPS Enabled: {torch.backends.mps.enabled}")
    print("=" * 70)
    
    # Check dependencies first
    print("🔍 Checking dependencies...")
    if not install_missing_dependencies():
        print("❌ Please install missing dependencies and run again.")
        return
    
    # Download dataset if needed
    print("📥 Checking FiQA dataset...")
    if not download_fiqa_dataset():
        print("❌ Failed to download dataset. Please check your internet connection.")
        return
    
    # Verify dataset structure
    if not verify_dataset_structure():
        print("❌ Dataset structure verification failed. Removing corrupted data...")
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        print("💡 Please run the script again to re-download the dataset.")
        return
    
    print("Loading FiQA dataset via BEIR GenericDataLoader...")
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=str(DATA_DIR)).load(split="test")
        print(f"✅ Loaded successfully:")
        print(f"   📄 Corpus: {len(corpus):,} documents")
        print(f"   ❓ Queries: {len(queries):,} queries") 
        print(f"   📋 Qrels: {len(qrels):,} relevance judgments")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print(f"💡 Try removing {DATA_DIR} and running again to re-download")
        return

    summaries = []
    total_methods = len(METHODS)
    
    for i, method in enumerate(METHODS):
        print(f"\n📍 Progress: {i+1}/{total_methods} methods")
        print(f"🚀 Starting {method['name']}...")
        
        s = run_single(corpus, queries, qrels, method)
        if s:
            summaries.append(s)
            print(f"✅ {method['name']} completed successfully!")
        else:
            print(f"💀 {method['name']} failed - moving on...")

    if not summaries:
        print("💀 All methods failed. Check your setup and dependencies.")
        return

    # Create comprehensive results summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    # Console table
    print(f"{'Method':<15} {'nDCG@10':<8} {'Recall@100':<11} {'MAP@100':<9} {'Latency(s)':<10}")
    print("-" * 60)
    for s in summaries:
        print(f"{s['method']:<15} {s['nDCG@10']:<8.4f} {s['Recall@100']:<11.4f} {s['MAP@100']:<9.4f} {s['avg_latency_s']:<10.4f}")

    # Save comprehensive markdown table
    md = "# FiQA Model Comparison (CPU-Only, MPS Disabled)\n\n"
    md += "Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
    md += f"**Dataset**: FiQA-2018 ({len(corpus):,} docs, {len(queries)} queries)\n"
    md += f"**Device**: CPU (MPS disabled)\n"
    md += f"**Methods Tested**: {len(summaries)}/{total_methods}\n\n"
    
    md += "## Results\n\n"
    md += "|Method|nDCG@10|nDCG@100|MAP@10|MAP@100|Recall@10|Recall@100|P@10|Avg Latency (s)|P50 Latency (s)|P95 Latency (s)|Total Time (s)|Device|Notes|\n"
    md += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    
    for s in summaries:
        p50 = s.get('p50_latency_s', 'N/A')
        p95 = s.get('p95_latency_s', 'N/A') 
        p50_str = f"{p50:.4f}" if p50 != 'N/A' and p50 is not None else 'N/A'
        p95_str = f"{p95:.4f}" if p95 != 'N/A' and p95 is not None else 'N/A'
        
        md += f"|{s['method']}|{s['nDCG@10']:.4f}|{s['nDCG@100']:.4f}|{s['MAP@10']:.4f}|{s['MAP@100']:.4f}|{s['Recall@10']:.4f}|{s['Recall@100']:.4f}|{s['P@10']:.4f}|{s['avg_latency_s']:.4f}|{p50_str}|{p95_str}|{s['total_time_s']:.1f}|{s.get('device', 'cpu')}|{s.get('notes', '')}|\n"

    # Add benchmark context
    md += "\n## Expected FiQA Performance Ranges\n\n"
    md += "Based on BEIR paper and community results:\n\n"
    md += "- **BM25**: nDCG@10 ~0.23-0.26, Recall@100 ~0.64-0.70\n"
    md += "- **DPR**: nDCG@10 ~0.22-0.28, Recall@100 ~0.60-0.75\n" 
    md += "- **Contriever**: nDCG@10 ~0.25-0.30, Recall@100 ~0.65-0.80\n"
    md += "- **SPLADE**: nDCG@10 ~0.27-0.32, Recall@100 ~0.70-0.85\n"
    
    md += "\n## Setup Information\n\n"
    md += f"- **Dataset Source**: {DOWNLOAD_URL}\n"
    md += f"- **Dataset Path**: {DATA_DIR}\n"
    md += f"- **Results Path**: {RESULTS_DIR}\n"
    md += "- **Auto-downloaded**: Yes\n"
    md += "- **CPU-only mode**: Enabled (MPS disabled)\n"

    with open(RESULTS_DIR / "fiqa_comparison_cpu_only.md", "w") as f:
        f.write(md)

    # Save simple CSV for easy analysis
    csv_lines = ["method,ndcg_10,recall_100,map_100,avg_latency_s,total_time_s,device"]
    for s in summaries:
        csv_lines.append(f"{s['method']},{s['nDCG@10']},{s['Recall@100']},{s['MAP@100']},{s['avg_latency_s']},{s['total_time_s']},{s.get('device', 'cpu')}")
    
    with open(RESULTS_DIR / "fiqa_results.csv", "w") as f:
        f.write("\n".join(csv_lines))

    print(f"\n✅ All results saved to {RESULTS_DIR}/")
    print("📄 Individual JSON files: fiqa_*_results.json, fiqa_*_summary.json")
    print("📊 Comparison table: fiqa_comparison_cpu_only.md") 
    print("📈 CSV: fiqa_results.csv")
    print("\n🚫🍏 MPS successfully avoided. No tensor dimension errors!")
    print("💪 CPU-only results are stable and reproducible.")
    print(f"🎯 Successfully tested {len(summaries)}/{total_methods} methods")
    print(f"\n📁 Dataset location: {DATA_DIR}")
    print(f"📊 Results location: {RESULTS_DIR}")

if __name__ == "__main__":
    main()