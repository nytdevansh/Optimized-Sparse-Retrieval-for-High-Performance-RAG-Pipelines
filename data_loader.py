"""
Dataset loading, validation and preparation utilities for Lightning Retrieval
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("datasets")
SUPPORTED_DATASETS = ["nq", "fiqa", "msmarco", "micro"]

@dataclass
class DatasetStats:
    """Statistics about a validated dataset"""
    num_docs: int
    num_queries: int
    num_qrels: int
    avg_doc_length: float
    avg_query_length: float
    vocab_size: int
    missing_docs: List[str]
    missing_queries: List[str]

def validate_jsonl(path: Path) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """Validate a JSONL file and return parsed contents"""
    contents = []
    errors = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        errors.append(f"Line {i}: Expected JSON object, got {type(item)}")
                        continue
                    contents.append(item)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON - {str(e)}")
    except Exception as e:
        errors.append(f"Failed to read {path}: {str(e)}")
        return False, [], errors
        
    return len(errors) == 0, contents, errors

def validate_corpus(corpus_path: Path) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """Validate corpus.jsonl format"""
    success, items, errors = validate_jsonl(corpus_path)
    if not success:
        return False, [], errors
        
    for i, item in enumerate(items, 1):
        if '_id' not in item:
            errors.append(f"Document {i}: Missing '_id' field")
        if not isinstance(item.get('_id', ''), str):
            errors.append(f"Document {i}: '_id' must be string")
        if 'text' not in item:
            errors.append(f"Document {i}: Missing 'text' field")
        if not isinstance(item.get('text', ''), str):
            errors.append(f"Document {i}: 'text' must be string")
            
    return len(errors) == 0, items, errors

def validate_queries(queries_path: Path) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """Validate queries.jsonl format"""
    success, items, errors = validate_jsonl(queries_path)
    if not success:
        return False, [], errors
        
    for i, item in enumerate(items, 1):
        if '_id' not in item:
            errors.append(f"Query {i}: Missing '_id' field")
        if not isinstance(item.get('_id', ''), str):
            errors.append(f"Query {i}: '_id' must be string")
        if 'text' not in item:
            errors.append(f"Query {i}: Missing 'text' field")
        if not isinstance(item.get('text', ''), str):
            errors.append(f"Query {i}: 'text' must be string")
            
    return len(errors) == 0, items, errors

def validate_qrels(qrels_path: Path, valid_qids: set, valid_docids: set) -> Tuple[bool, List[str]]:
    """Validate qrels TSV format and ID references"""
    errors = []
    
    try:
        with open(qrels_path, 'r', encoding='utf-8') as f:
            header = next(f, None)
            if header and not header.strip() == 'query-id\tcorpus-id\tscore':
                errors.append(f"Invalid header: {header.strip()}")
                
            for i, line in enumerate(f, 2):
                try:
                    qid, docid, score = line.strip().split('\t')
                    if qid not in valid_qids:
                        errors.append(f"Line {i}: Unknown query ID: {qid}")
                    if docid not in valid_docids:
                        errors.append(f"Line {i}: Unknown document ID: {docid}")
                    try:
                        score = int(score)
                        if score not in {0, 1}:  # Assuming binary relevance
                            errors.append(f"Line {i}: Invalid score (expected 0 or 1): {score}")
                    except ValueError:
                        errors.append(f"Line {i}: Invalid score format: {score}")
                except ValueError:
                    errors.append(f"Line {i}: Invalid format (expected 3 tab-separated values)")
                    
    except Exception as e:
        errors.append(f"Failed to read {qrels_path}: {str(e)}")
        return False, errors
        
    return len(errors) == 0, errors

def compute_stats(corpus: List[Dict[str, Any]], queries: List[Dict[str, Any]], 
                 qrels_path: Optional[Path] = None) -> DatasetStats:
    """Compute dataset statistics"""
    # Get document stats
    doc_lengths = [len(doc['text'].split()) for doc in corpus]
    vocab = set()
    for doc in corpus:
        vocab.update(doc['text'].lower().split())
        
    # Get query stats
    query_lengths = [len(q['text'].split()) for q in queries]
    
    # Count qrels if available
    num_qrels = 0
    if qrels_path and qrels_path.exists():
        with open(qrels_path, 'r') as f:
            next(f)  # Skip header
            num_qrels = sum(1 for _ in f)
    
    # Find missing references
    doc_ids = {doc['_id'] for doc in corpus}
    query_ids = {q['_id'] for q in queries}
    
    missing_docs = []
    missing_queries = []
    
    if qrels_path and qrels_path.exists():
        with open(qrels_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                qid, docid, _ = line.strip().split('\t')
                if docid not in doc_ids:
                    missing_docs.append(docid)
                if qid not in query_ids:
                    missing_queries.append(qid)
    
    return DatasetStats(
        num_docs=len(corpus),
        num_queries=len(queries),
        num_qrels=num_qrels,
        avg_doc_length=sum(doc_lengths) / len(doc_lengths),
        avg_query_length=sum(query_lengths) / len(query_lengths),
        vocab_size=len(vocab),
        missing_docs=list(set(missing_docs)),
        missing_queries=list(set(missing_queries))
    )

def validate_dataset(dataset_path: Path) -> Tuple[bool, DatasetStats, List[str]]:
    """Validate entire dataset directory structure and contents"""
    errors = []
    dataset_path = Path(dataset_path)
    
    # Check required files
    corpus_path = dataset_path / 'corpus.jsonl'
    queries_path = dataset_path / 'queries.jsonl'
    qrels_dir = dataset_path / 'qrels'
    
    if not corpus_path.exists():
        errors.append(f"Missing corpus.jsonl in {dataset_path}")
        return False, None, errors
    
    if not queries_path.exists():
        errors.append(f"Missing queries.jsonl in {dataset_path}")
        return False, None, errors
    
    if not qrels_dir.exists() or not qrels_dir.is_dir():
        errors.append(f"Missing qrels directory in {dataset_path}")
        return False, None, errors
    
    # Validate corpus
    logger.info("Validating corpus.jsonl...")
    corpus_valid, corpus, corpus_errors = validate_corpus(corpus_path)
    errors.extend(corpus_errors)
    
    # Validate queries
    logger.info("Validating queries.jsonl...")
    queries_valid, queries, query_errors = validate_queries(queries_path)
    errors.extend(query_errors)
    
    if not (corpus_valid and queries_valid):
        return False, None, errors
    
    # Get valid IDs
    doc_ids = {doc['_id'] for doc in corpus}
    query_ids = {q['_id'] for q in queries}
    
    # Validate qrels files
    for qrels_file in qrels_dir.glob('*.tsv'):
        logger.info(f"Validating {qrels_file.name}...")
        qrels_valid, qrel_errors = validate_qrels(qrels_file, query_ids, doc_ids)
        errors.extend([f"{qrels_file.name}: {err}" for err in qrel_errors])
    
    # Compute statistics
    stats = compute_stats(corpus, queries, next(qrels_dir.glob('*.tsv'), None))
    
    success = len(errors) == 0
    if success:
        logger.info("✅ Dataset validation successful!")
        logger.info(f"Stats:\n{stats}")
    else:
        logger.error("❌ Dataset validation failed!")
        for error in errors:
            logger.error(f"  - {error}")
    
    return success, stats, errors

def prepare_dataset(dataset_name: str = "nq", subset_size: Optional[int] = None) -> Tuple[Path, Path, Path]:
    """Prepare dataset for benchmarking"""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    data_dir = DEFAULT_DATA_DIR / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_file = data_dir / "corpus.jsonl"
    queries_file = data_dir / "queries.jsonl"
    qrels_file = data_dir / "qrels" / "test.tsv"
    qrels_file.parent.mkdir(parents=True, exist_ok=True)

    if not corpus_file.exists():
        logger.info(f"Preparing {dataset_name} dataset...")
        
        if dataset_name == "nq":
            # Natural Questions dataset
            ds = load_dataset("BeIR/nq", "corpus")
            if subset_size:
                ds["corpus"] = ds["corpus"].select(range(min(subset_size, len(ds["corpus"]))))
            ds["corpus"].to_json(corpus_file)

            ds = load_dataset("BeIR/nq", "queries")
            ds["queries"].to_json(queries_file)

            ds = load_dataset("BeIR/nq", "qrels")
            ds["qrels"].to_csv(qrels_file, sep="\t", index=False)
            
        # Add other dataset preparations here
            
    else:
        logger.info(f"Dataset {dataset_name} already exists at {data_dir}")

    # Validate the dataset
    success, stats, errors = validate_dataset(data_dir)
    if not success:
        raise ValueError(f"Dataset validation failed:\n" + "\n".join(errors))
        
    return corpus_file, queries_file, qrels_file

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset preparation and validation tool")
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS, help="Dataset to prepare/validate")
    parser.add_argument("--subset-size", type=int, help="Number of documents to use (for testing)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't prepare")
    
    args = parser.parse_args()
    
    try:
        if args.validate_only:
            success, stats, errors = validate_dataset(DEFAULT_DATA_DIR / args.dataset)
            if not success:
                sys.exit(1)
        else:
            prepare_dataset(args.dataset, args.subset_size)
            logger.info("✅ Dataset preparation completed successfully")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)