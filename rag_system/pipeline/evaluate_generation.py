#!/usr/bin/env python3
"""
Evaluate generation outputs (ROUGE, BERTScore, simple faithfulness checks).
"""
import json
import re
import string
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


def load_predictions(path: str) -> List[Dict]:
    """Load predictions from JSON file."""
    pred_path = Path(path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing predictions JSON: {e}")


def normalize_text(text: str) -> str:
    """Normalize text for evaluation."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return normalize_text(text).split()


def compute_rouge_l(pred: str, ref: str) -> Dict[str, float]:
    """Compute ROUGE-L using longest common subsequence."""
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    
    if not pred_tokens and not ref_tokens:
        return {"rouge_l_f1": 1.0, "rouge_l_precision": 1.0, "rouge_l_recall": 1.0}
    
    if not pred_tokens or not ref_tokens:
        return {"rouge_l_f1": 0.0, "rouge_l_precision": 0.0, "rouge_l_recall": 0.0}
    
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "rouge_l_f1": f1,
        "rouge_l_precision": precision,
        "rouge_l_recall": recall
    }


def compute_rouge_n(pred: str, ref: str, n: int = 1) -> Dict[str, float]:
    """Compute ROUGE-N scores."""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Extract n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    
    if not pred_tokens and not ref_tokens:
        return {f"rouge_{n}_f1": 1.0, f"rouge_{n}_precision": 1.0, f"rouge_{n}_recall": 1.0}
    
    if not pred_tokens or not ref_tokens:
        return {f"rouge_{n}_f1": 0.0, f"rouge_{n}_precision": 0.0, f"rouge_{n}_recall": 0.0}
    
    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    
    if not pred_ngrams or not ref_ngrams:
        return {f"rouge_{n}_f1": 0.0, f"rouge_{n}_precision": 0.0, f"rouge_{n}_recall": 0.0}
    
    # Calculate overlap
    overlap = sum((pred_ngrams & ref_ngrams).values())
    
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        f"rouge_{n}_f1": f1,
        f"rouge_{n}_precision": precision,
        f"rouge_{n}_recall": recall
    }


def compute_bleu(pred: str, ref: str, max_n: int = 4) -> float:
    """Compute sentence-level BLEU score."""
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Brevity penalty
    bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0.0
    
    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter()
        ref_ngrams = Counter()
        
        for i in range(len(pred_tokens) - n + 1):
            pred_ngrams[tuple(pred_tokens[i:i+n])] += 1
        
        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams[tuple(ref_tokens[i:i+n])] += 1
        
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        precision = overlap / sum(pred_ngrams.values())
        precisions.append(precision)
    
    if not any(precisions):
        return 0.0
    
    # Geometric mean of precisions
    import math
    log_sum = sum(math.log(p) if p > 0 else float('-inf') for p in precisions)
    geo_mean = math.exp(log_sum / len(precisions)) if log_sum > float('-inf') else 0.0
    
    return bp * geo_mean


def simple_faithfulness_check(pred: str, contexts: List[str]) -> Dict[str, Any]:
    """Simple faithfulness check based on token overlap."""
    if not pred or not contexts:
        return {"faithfulness_score": 0.0, "supporting_contexts": 0}
    
    pred_tokens = set(tokenize(pred))
    supporting_contexts = 0
    total_overlap = 0
    
    for context in contexts:
        context_tokens = set(tokenize(context))
        overlap = len(pred_tokens & context_tokens)
        
        if overlap > 0:
            supporting_contexts += 1
            total_overlap += overlap
    
    faithfulness_score = total_overlap / len(pred_tokens) if pred_tokens else 0.0
    
    return {
        "faithfulness_score": min(1.0, faithfulness_score),
        "supporting_contexts": supporting_contexts,
        "total_contexts": len(contexts)
    }


def evaluate(predictions: List[Dict], references: Dict[str, str]) -> Dict[str, Any]:
    """Comprehensive evaluation of RAG predictions."""
    if not predictions:
        return {"error": "No predictions provided"}
    
    if not references:
        return {"error": "No references provided"}
    
    rouge_l_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    bleu_scores = []
    faithfulness_scores = []
    
    evaluated_count = 0
    missing_refs = 0
    
    for pred in predictions:
        qid = pred.get('qid')
        if not qid:
            continue
        
        if qid not in references:
            missing_refs += 1
            continue
        
        pred_text = pred.get('answer', '')
        ref_text = references[qid]
        contexts = pred.get('contexts', [])
        
        if not pred_text or not ref_text:
            continue
        
        # ROUGE scores
        rouge_l = compute_rouge_l(pred_text, ref_text)
        rouge_1 = compute_rouge_n(pred_text, ref_text, n=1)
        rouge_2 = compute_rouge_n(pred_text, ref_text, n=2)
        
        rouge_l_scores.append(rouge_l['rouge_l_f1'])
        rouge_1_scores.append(rouge_1['rouge_1_f1'])
        rouge_2_scores.append(rouge_2['rouge_2_f1'])
        
        # BLEU score
        bleu = compute_bleu(pred_text, ref_text)
        bleu_scores.append(bleu)
        
        # Faithfulness
        context_texts = []
        if isinstance(contexts, list):
            for ctx in contexts:
                if isinstance(ctx, dict) and 'text' in ctx:
                    context_texts.append(ctx['text'])
        
        faith = simple_faithfulness_check(pred_text, context_texts)
        faithfulness_scores.append(faith['faithfulness_score'])
        
        evaluated_count += 1
    
    if evaluated_count == 0:
        return {"error": "No valid prediction-reference pairs found"}
    
    # Calculate averages
    def safe_mean(scores):
        return sum(scores) / len(scores) if scores else 0.0
    
    results = {
        "total_predictions": len(predictions),
        "evaluated_predictions": evaluated_count,
        "missing_references": missing_refs,
        "rouge_l": safe_mean(rouge_l_scores),
        "rouge_1": safe_mean(rouge_1_scores),
        "rouge_2": safe_mean(rouge_2_scores),
        "bleu": safe_mean(bleu_scores),
        "faithfulness": safe_mean(faithfulness_scores),
        "individual_scores": {
            "rouge_l": rouge_l_scores,
            "rouge_1": rouge_1_scores,
            "rouge_2": rouge_2_scores,
            "bleu": bleu_scores,
            "faithfulness": faithfulness_scores
        }
    }
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG generation quality")
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--references', type=str, required=True,
                       help='Path to references JSON file')
    parser.add_argument('--output', type=str,
                       help='Save evaluation results to JSON file')
    
    args = parser.parse_args()
    
    # Load data
    predictions = load_predictions(args.predictions)
    
    with open(args.references, 'r') as f:
        references = json.load(f)
    
    # Run evaluation
    results = evaluate(predictions, references)
    
    # Print results
    print("="*50)
    print("RAG GENERATION EVALUATION RESULTS")
    print("="*50)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Total predictions: {results['total_predictions']}")
        print(f"Evaluated predictions: {results['evaluated_predictions']}")
        print(f"Missing references: {results['missing_references']}")
        print()
        print("SCORES:")
        print("-"*20)
        print(f"ROUGE-L: {results['rouge_l']:.4f}")
        print(f"ROUGE-1: {results['rouge_1']:.4f}")
        print(f"ROUGE-2: {results['rouge_2']:.4f}")
        print(f"BLEU:    {results['bleu']:.4f}")
        print(f"Faithfulness: {results['faithfulness']:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
