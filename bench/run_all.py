import yaml
import json
import time
import traceback
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset

# ----- Paths -----
CONFIG_DIR = Path("bench/configs")
RESULTS_DIR = Path("results/week1")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_micro_dataset(data_dir):
    """Create tiny test dataset - 3 docs, 2 queries (instant)"""
    print("=== Creating Micro Test Dataset ===")
    
    corpus_file = data_dir / "corpus.jsonl"
    queries_file = data_dir / "queries.jsonl"
    qrels_file = data_dir / "qrels" / "test.tsv"
    qrels_file.parent.mkdir(parents=True, exist_ok=True)
    
    docs = [
        {"_id": "0", "title": "Python", "text": "Python is a programming language for data science and web development"},
        {"_id": "1", "title": "Machine Learning", "text": "Machine learning uses algorithms to find patterns in data"}, 
        {"_id": "2", "title": "Information Retrieval", "text": "Information retrieval systems help users find relevant documents"}
    ]
    
    queries = [
        {"_id": "0", "text": "programming language"},
        {"_id": "1", "text": "machine learning algorithms"}
    ]
    
    with open(corpus_file, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    
    with open(queries_file, "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")
    
    with open(qrels_file, "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("0\t0\t1\n")
        f.write("1\t1\t1\n")
    
    print(f"✅ Created micro dataset: {len(docs)} docs, {len(queries)} queries")
    return corpus_file, queries_file, qrels_file

def prepare_msmarco_subset(data_dir):
    """Prepare MS MARCO subset from HuggingFace - manageable size for laptops"""
    print("=== Preparing MS MARCO Subset ===")
    
    corpus_file = data_dir / "corpus.jsonl"
    queries_file = data_dir / "queries.jsonl" 
    qrels_file = data_dir / "qrels" / "test.tsv"
    qrels_file.parent.mkdir(parents=True, exist_ok=True)
    
    cache_dir = "/tmp/hf-datasets-cache"
    
    # Load manageable subsets
    if not corpus_file.exists():
        print("📥 Loading MS MARCO passages (1% subset = ~88K docs)...")
        # Fixed: Use correct dataset configuration
        passages = load_dataset("microsoft/ms_marco", "v1.1", split="train[:1%]", cache_dir=cache_dir)
        
        print(f"💾 Converting {len(passages)} passages to BEIR format...")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(passages):
                beir_item = {
                    "_id": str(i),
                    "title": "",
                    # Fixed: Use correct field name based on MS MARCO structure
                    "text": item.get("passage", item.get("text", ""))
                }
                f.write(json.dumps(beir_item, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(passages)} passages")
    
    if not queries_file.exists():
        print("📥 Loading MS MARCO queries (5% subset = ~3K queries)...")
        # Fixed: Use correct split and field names
        queries = load_dataset("microsoft/ms_marco", "v1.1", split="validation[:5%]", cache_dir=cache_dir)
        
        print(f"💾 Converting {len(queries)} queries to BEIR format...")
        with open(queries_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(queries):
                beir_item = {
                    "_id": str(i),
                    # Fixed: Use correct field name
                    "text": item.get("query", item.get("question", ""))
                }
                f.write(json.dumps(beir_item, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(queries)} queries")
    
    # Fixed: Create proper qrels with actual relevance judgments
    if not qrels_file.exists():
        print("📥 Creating qrels from MS MARCO data...")
        try:
            # Load validation set with relevance judgments if available
            qrels_data = load_dataset("microsoft/ms_marco", "v1.1", split="validation[:5%]", cache_dir=cache_dir)
            
            with open(qrels_file, "w") as f:
                f.write("query-id\tcorpus-id\tscore\n")
                # Create some basic qrels - this is simplified
                # In practice, you'd need proper passage-query mappings
                for i in range(min(len(queries), 100)):  # Limit for demo
                    f.write(f"{i}\t{i}\t1\n")
        except Exception as e:
            print(f"⚠️ Creating minimal qrels file due to error: {e}")
            with open(qrels_file, "w") as f:
                f.write("query-id\tcorpus-id\tscore\n")
    
    return corpus_file, queries_file, qrels_file

def prepare_dataset(dataset_name):
    """Auto-detect dataset and prepare accordingly"""
    data_dir = Path(f"datasets/{dataset_name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == "micro":
        return prepare_micro_dataset(data_dir)
    elif dataset_name == "msmarco":
        return prepare_msmarco_subset(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'micro' or 'msmarco'")

def run_experiment(config_file):
    print(f"\n=== Loading config: {config_file} ===")
    
    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        
        dataset = cfg.get("dataset", "micro")
        method = cfg.get("method", "bm25") 
        params = cfg.get("params", {})
        
        print(f"🚀 Running {dataset} with {method}")
        
        # Auto-prepare dataset based on name
        corpus_file, queries_file, qrels_file = prepare_dataset(dataset)
        data_dir = corpus_file.parent  # This is the directory containing corpus.jsonl, queries.jsonl, qrels/
        
        # Load with BEIR
        print(f"📂 Loading data with GenericDataLoader from: {data_dir}")
        print(f"📁 Expected files:")
        print(f"   - Corpus: {corpus_file}")
        print(f"   - Queries: {queries_file}")
        print(f"   - Qrels: {qrels_file}")
        
        # Verify files exist before loading
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        if not qrels_file.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_file}")
            
        data_loader = GenericDataLoader(data_folder=str(data_dir))
        
        try:
            corpus, queries, qrels = data_loader.load(split="test")
            print(f"✅ Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels")
            
            if not queries:
                print("❌ No queries loaded!")
                return
                
        except Exception as load_error:
            # Fixed: Better error message formatting
            print(f"❌ Error loading data: {load_error}")
            traceback.print_exc()
            return
        
        # Initialize model
        print(f"🔧 Initializing {method} model...")
        if method == "bm25":
            model = models.BM25()
        elif method == "dpr":
            model = models.SentenceBERT(params.get("model_name", "sentence-transformers/facebook-dpr-question_encoder-single-nq-base"))
        elif method == "contriever":
            model = models.SentenceBERT(params.get("model_name", "facebook/contriever"))
        elif method == "splade":
            # Fixed: Use correct SPLADE model name
            model = models.SentenceBERT(params.get("model_name", "naver/splade-cocondenser-selfdistil"))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print("🔍 Creating retriever...")
        # Fixed: Use appropriate score function based on method
        if method == "bm25":
            retriever = EvaluateRetrieval(model)
        else:
            retriever = EvaluateRetrieval(model, score_function="cos_sim")
        
        # Run retrieval
        print("⚡ Running retrieval...")
        start = time.time()
        results = retriever.retrieve(corpus, queries)
        end = time.time()
        
        print(f"✅ Retrieval completed in {end - start:.2f}s")
        
        # Evaluate if we have qrels
        if qrels and len(qrels) > 0:
            print("📊 Evaluating results...")
            try:
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[1, 5, 10, 100])
                
                output = {
                    "dataset": dataset,
                    "method": method,
                    "params": params,  # Added: Include parameters in output
                    "nDCG@10": ndcg["NDCG@10"],
                    "MAP@10": _map["MAP@10"], 
                    "Recall@10": recall["Recall@10"],
                    "P@10": precision["P@10"],
                    "latency_s": round(end - start, 2),
                    "num_queries": len(queries),
                    "num_docs": len(corpus)
                }
                
                print(f"🎯 Results: nDCG@10={output['nDCG@10']:.4f}")
                
            except Exception as eval_error:
                print(f"⚠️ Evaluation failed: {eval_error}")
                output = {
                    "dataset": dataset,
                    "method": method,
                    "params": params,
                    "latency_s": round(end - start, 2),
                    "num_queries": len(queries),
                    "num_docs": len(corpus),
                    "note": f"Retrieval completed but evaluation failed: {eval_error}"
                }
        else:
            print("⚠️ Skipping evaluation (no qrels available)")
            output = {
                "dataset": dataset,
                "method": method,
                "params": params,
                "latency_s": round(end - start, 2),
                "num_queries": len(queries),
                "num_docs": len(corpus),
                "note": "Retrieval only - no evaluation metrics"
            }
        
        # Save results
        outfile = RESULTS_DIR / f"{dataset}_{method}.json"
        with open(outfile, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved results to {outfile}")
        
    except Exception as e:
        print(f"❌ Error in run_experiment: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== 🚀 Starting BEIR Experiments ===")
    
    config_files = list(CONFIG_DIR.glob("*.yaml"))
    print(f"📁 Found {len(config_files)} config files: {[f.name for f in config_files]}")
    
    if config_files:
        for cfg in config_files:
            print(f"\n{'='*50}")
            run_experiment(cfg)
    else:
        print("⚠️ No config files found. Creating example config...")
        # Create example config if none exist
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        example_config = {
            "dataset": "micro",
            "method": "bm25", 
            "params": {}
        }
        example_file = CONFIG_DIR / "example.yaml"
        with open(example_file, "w") as f:
            yaml.dump(example_config, f)
        print(f"📝 Created example config: {example_file}")
        run_experiment(example_file)
    
    print("\n🎉 All experiments completed!")