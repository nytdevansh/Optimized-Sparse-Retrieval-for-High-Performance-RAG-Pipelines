#!/usr/bin/env python3
"""
Simple error analysis notebook-like script: groups failures by type.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_preds(path):
    """Load predictions from JSON file."""
    pred_path = Path(path)
    if not pred_path.exists():
        print(f"Error: Predictions file not found: {pred_path}")
        sys.exit(1)
    
    try:
        return json.loads(pred_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error parsing predictions JSON: {e}")
        sys.exit(1)


def load_qrels(path):
    """Load qrels from TSV file with header detection."""
    if not path:
        return {}
    
    qrels_path = Path(path)
    if not qrels_path.exists():
        print(f"Warning: Qrels file not found: {qrels_path}")
        return {}
    
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Skip header row
            if line_no == 1 and any(h in line.lower() for h in ['query', 'doc', 'rel', 'score']):
                continue
            
            try:
                parts = line.split('\t')
                if len(parts) < 3:
                    parts = line.split()
                
                if len(parts) >= 3:
                    qid, docid, rel = parts[:3]
                    qrels.setdefault(qid, {})[docid] = int(rel)
            except (ValueError, IndexError):
                continue
    
    return qrels


def load_corpus(path):
    """Load corpus from JSONL file."""
    if not path:
        return {}
    
    corpus_path = Path(path)
    if not corpus_path.exists():
        print(f"Warning: Corpus file not found: {corpus_path}")
        return {}
    
    corpus = {}
    with open(corpus_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
                # Try different ID field names
                doc_id = None
                for id_field in ['id', '_id', 'doc_id', 'docid']:
                    if id_field in doc:
                        doc_id = doc[id_field]
                        break
                
                if doc_id:
                    corpus[str(doc_id)] = doc
            except json.JSONDecodeError:
                continue
    
    return corpus


def categorize_errors(predictions, qrels, corpus):
    """Categorize prediction errors by type."""
    categories = Counter()
    samples = defaultdict(list)
    
    for pred in predictions:
        qid = pred.get('qid')
        if not qid:
            continue
        
        # Get gold standard documents for this query
        gold_docs = set(qrels.get(qid, {}).keys()) if qrels else set()
        
        # Get retrieved documents
        retrieved_docs = set()
        contexts = pred.get('contexts', [])
        if isinstance(contexts, list):
            for ctx in contexts:
                if isinstance(ctx, dict) and 'docid' in ctx:
                    retrieved_docs.add(ctx['docid'])
        
        # Categorize the prediction
        if not gold_docs:
            categories['no_gold_standard'] += 1
            samples['no_gold_standard'].append((qid, pred))
        elif not retrieved_docs:
            categories['no_retrieval'] += 1
            samples['no_retrieval'].append((qid, pred))
        elif not gold_docs.intersection(retrieved_docs):
            categories['retriever_miss'] += 1
            samples['retriever_miss'].append((qid, pred))
        else:
            # Some relevant documents were retrieved
            categories['reader_issue'] += 1
            samples['reader_issue'].append((qid, pred))
    
    return categories, dict(samples)


def analyze_retrieval_performance(predictions, qrels):
    """Analyze retrieval performance metrics."""
    if not qrels:
        return {}
    
    total_queries = 0
    total_recall = 0.0
    total_precision = 0.0
    
    for pred in predictions:
        qid = pred.get('qid')
        if not qid or qid not in qrels:
            continue
        
        gold_docs = set(qrels[qid].keys())
        retrieved_docs = set()
        
        contexts = pred.get('contexts', [])
        if isinstance(contexts, list):
            for ctx in contexts:
                if isinstance(ctx, dict) and 'docid' in ctx:
                    retrieved_docs.add(ctx['docid'])
        
        if gold_docs and retrieved_docs:
            overlap = gold_docs.intersection(retrieved_docs)
            recall = len(overlap) / len(gold_docs)
            precision = len(overlap) / len(retrieved_docs)
            
            total_recall += recall
            total_precision += precision
            total_queries += 1
    
    if total_queries > 0:
        return {
            'avg_recall': total_recall / total_queries,
            'avg_precision': total_precision / total_queries,
            'total_queries_analyzed': total_queries
        }
    else:
        return {}


def print_analysis_report(categories, samples, retrieval_stats, predictions):
    """Print comprehensive error analysis report."""
    print("=" * 60)
    print("RAG ERROR ANALYSIS REPORT")
    print("=" * 60)
    
    total_predictions = len(predictions)
    print(f"Total predictions analyzed: {total_predictions}")
    print()
    
    # Error categories
    print("ERROR CATEGORIES:")
    print("-" * 30)
    for category, count in categories.most_common():
        percentage = (count / total_predictions) * 100
        print(f"{category:20} {count:5d} ({percentage:5.1f}%)")
    print()
    
    # Retrieval performance
    if retrieval_stats:
        print("RETRIEVAL PERFORMANCE:")
        print("-" * 30)
        print(f"Average Recall:    {retrieval_stats.get('avg_recall', 0):.3f}")
        print(f"Average Precision: {retrieval_stats.get('avg_precision', 0):.3f}")
        print(f"Queries Analyzed:  {retrieval_stats.get('total_queries_analyzed', 0)}")
        print()
    
    # Sample errors
    print("SAMPLE ERRORS (first 3 per category):")
    print("-" * 40)
    for category, sample_list in samples.items():
        print(f"\n{category.upper()}:")
        for i, (qid, pred) in enumerate(sample_list[:3]):
            query_text = pred.get('query', 'N/A')[:100]
            answer_text = pred.get('answer', 'N/A')[:100]
            print(f"  {i+1}. Query ID: {qid}")
            print(f"     Query: {query_text}...")
            print(f"     Answer: {answer_text}...")
            print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze RAG prediction errors")
    parser.add_argument('--preds', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--qrels', type=str,
                       help='Path to qrels TSV file')
    parser.add_argument('--corpus', type=str,
                       help='Path to corpus JSONL file')
    parser.add_argument('--output', type=str,
                       help='Save analysis results to JSON file')
    
    args = parser.parse_args()
    
    # Load data
    predictions = load_preds(args.preds)
    qrels = load_qrels(args.qrels) if args.qrels else {}
    corpus = load_corpus(args.corpus) if args.corpus else {}
    
    # Perform analysis
    categories, samples = categorize_errors(predictions, qrels, corpus)
    retrieval_stats = analyze_retrieval_performance(predictions, qrels)
    
    # Print report
    print_analysis_report(categories, samples, retrieval_stats, predictions)
    
    # Save results if requested
    if args.output:
        results = {
            'total_predictions': len(predictions),
            'error_categories': dict(categories),
            'retrieval_performance': retrieval_stats,
            'sample_errors': {k: v[:5] for k, v in samples.items()}  # Limit samples
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {args.output}")
