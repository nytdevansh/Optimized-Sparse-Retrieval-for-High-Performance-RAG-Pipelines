import json
import time
import requests
import zipfile
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util
from urllib.parse import urlparse
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Import rank_bm25 instead of beir models
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("❌ Please install rank_bm25: pip install rank_bm25")
    exit(1)

# ==== Paths ====
DATASET = "fiqa"
DATA_DIR = Path(f"datasets/{DATASET}")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def download_with_retry(url, destination, max_retries=3, chunk_size=8192):
    """Download file with retry mechanism and resume capability"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Check if partial file exists
            resume_header = {}
            initial_pos = 0
            if destination.exists():
                initial_pos = destination.stat().st_size
                resume_header['Range'] = f'bytes={initial_pos}-'
                print(f"📁 Resuming download from byte {initial_pos:,}")
            
            # Start download with resume support
            response = requests.get(url, headers=resume_header, stream=True, timeout=30)
            
            # Get total file size
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            elif 'content-length' in response.headers:
                total_size = int(response.headers['content-length']) + initial_pos
            else:
                total_size = None
            
            # Open file in appropriate mode
            mode = 'ab' if initial_pos > 0 else 'wb'
            with open(destination, mode) as f:
                with tqdm(total=total_size, initial=initial_pos, unit='B', unit_scale=True, desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Download completed: {destination}")
            return str(destination)
            
        except Exception as e:
            print(f"❌ Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"❌ All download attempts failed")
                raise e

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

# Custom BM25 retrieval class compatible with BEIR evaluation
class BM25Retriever:
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
    
    def search(self, queries, top_k=1000):
        """Search method compatible with BEIR evaluation"""
        results = {}
        
        print(f"🔍 Processing {len(queries)} queries...")
        for query_id, query_text in tqdm(queries.items(), desc="Searching"):
            # Tokenize query
            tokenized_query = query_text.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k results
            if len(scores) > 0:
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                # Format results as expected by BEIR evaluation
                query_results = {}
                for idx in top_indices:
                    if idx < len(self.corpus_ids) and scores[idx] > 0:
                        doc_id = self.corpus_ids[idx]
                        query_results[doc_id] = float(scores[idx])
                
                results[query_id] = query_results
            else:
                results[query_id] = {}
        
        return results

print("=== FiQA-2018 BEIR Pipeline Test (Robust Version with rank_bm25) ===")

# ==== Step 1: Download FiQA dataset with retry ====
if not DATA_DIR.exists():
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
    zip_path = Path(f"datasets/{DATASET}.zip")
    
    print(f"📥 Downloading {DATASET} dataset...")
    try:
        download_with_retry(url, zip_path, max_retries=5)
        print(f"📦 Extracting {DATASET}.zip...")
        if extract_zip(zip_path, "datasets/"):
            # Clean up zip file
            zip_path.unlink()
            print(f"🧹 Cleaned up zip file")
        else:
            print(f"❌ Extraction failed, keeping zip file for manual extraction")
            exit(1)
    except Exception as e:
        print(f"❌ Download/extraction failed: {e}")
        print(f"💡 Try downloading manually from: {url}")
        exit(1)
else:
    print(f"✅ Dataset already exists at {DATA_DIR}")

# ==== Step 2: Load dataset ====
print(f"📂 Loading {DATASET}...")
try:
    corpus, queries, qrels = GenericDataLoader(data_folder=str(DATA_DIR)).load(split="test")
    
    print(f"✅ Loaded successfully:")
    print(f"   📄 Corpus: {len(corpus):,} documents")
    print(f"   ❓ Queries: {len(queries):,} queries")
    print(f"   📋 Qrels: {len(qrels):,} relevance judgments")
    
    # Show sample data
    sample_query_id = list(queries.keys())[0]
    sample_doc_id = list(corpus.keys())[0]
    print(f"   🔍 Sample query: {queries[sample_query_id][:100]}...")
    print(f"   📑 Sample doc: {corpus[sample_doc_id].get('title', 'No title')[:50]}...")
    
except Exception as e:
    print(f"❌ Loading failed: {e}")
    print(f"💡 Check if the dataset was extracted properly in {DATA_DIR}")
    exit(1)

# ==== Step 3: Run BM25 retrieval ====
print(f"🔍 Running BM25 retrieval...")
try:
    # Create custom BM25 retriever
    bm25_retriever = BM25Retriever(corpus)
    
    # Run retrieval
    start = time.time()
    results = bm25_retriever.search(queries, top_k=1000)
    end = time.time()
    
    print(f"✅ Retrieval completed in {end-start:.2f}s")
    print(f"   📊 Retrieved results for {len(results)} queries")
    
    # Show sample results
    if sample_query_id in results and results[sample_query_id]:
        sample_results = results[sample_query_id]
        top_score = max(sample_results.values())
        print(f"   🎯 Top result for sample query: score={top_score:.4f}")
    else:
        print(f"   ⚠️ No results found for sample query")
        
except Exception as e:
    print(f"❌ Retrieval failed: {e}")
    exit(1)

# ==== Step 4: Evaluate ====
print(f"📈 Evaluating results...")
try:
    # Use BEIR's evaluation directly on our results
    from beir.retrieval.evaluation import EvaluateRetrieval
    
    # Create a dummy model for evaluation (we already have results)
    class DummyModel:
        def __init__(self, results):
            self.results = results
        def search(self, corpus, queries, top_k, **kwargs):
            return self.results
    
    dummy_model = DummyModel(results)
    evaluator = EvaluateRetrieval(dummy_model)
    
    # Evaluate using BEIR's metrics
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[1, 5, 10, 100])
    
    print("🎯 Evaluation Results:")
    print(f"   nDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"   nDCG@100: {ndcg['NDCG@100']:.4f}")
    print(f"   MAP@10: {_map['MAP@10']:.4f}")
    print(f"   MAP@100: {_map['MAP@100']:.4f}")
    print(f"   Recall@10: {recall['Recall@10']:.4f}")
    print(f"   Recall@100: {recall['Recall@100']:.4f}")
    print(f"   P@10: {precision['P@10']:.4f}")
    
    # Quality check
    if ndcg['NDCG@10'] > 0.1:
        print("✅ Results look reasonable!")
    else:
        print("⚠️ Results seem low - check data quality")
        
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    exit(1)

# ==== Step 5: Save results ====
print(f"💾 Saving results...")
try:
    # Save detailed results
    results_file = RESULTS_DIR / f"{DATASET}_bm25_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary metrics
    summary_file = RESULTS_DIR / f"{DATASET}_bm25_summary.json"
    summary = {
        "dataset": DATASET,
        "method": "BM25 (rank_bm25)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "nDCG@10": ndcg['NDCG@10'],
            "nDCG@100": ndcg['NDCG@100'],
            "MAP@10": _map['MAP@10'],
            "MAP@100": _map['MAP@100'],
            "Recall@10": recall['Recall@10'],
            "Recall@100": recall['Recall@100'],
            "P@10": precision['P@10']
        },
        "stats": {
            "num_queries": len(queries),
            "num_docs": len(corpus),
            "num_qrels": len(qrels),
            "retrieval_time_s": round(end-start, 2)
        }
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Results saved:")
    print(f"   📄 Full results: {results_file}")
    print(f"   📊 Summary: {summary_file}")
    
except Exception as e:
    print(f"❌ Save failed: {e}")

print("\n🎉 FiQA test completed successfully!")
print("📈 Expected FiQA BM25 performance (approximate):")
print("   • nDCG@10: ~0.23-0.26")
print("   • MAP@100: ~0.18-0.21")
print("   • Recall@100: ~0.64-0.70")
print("\n💡 If this worked, your BEIR setup is solid!")
print("\n🔧 Note: This version uses rank_bm25 instead of BEIR's native BM25")
print("   Install with: pip install rank_bm25")
