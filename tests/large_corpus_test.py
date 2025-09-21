"""Test memory mapping with large corpora"""

import os
import json 
import mmap
from pathlib import Path
from memory_mapping import MemoryMappedCorpus, BinaryCorpusBuilder

def test_large_corpus(corpus_path: str):
    """Test memory mapping with large corpus files"""
    
    builder = BinaryCorpusBuilder(compression_level=1)  # Use minimal compression to start
    binary_path = str(Path(corpus_path).with_suffix('.bin'))
    
    # Process JSONL file directly to avoid BEIR issues
    print("Loading corpus data...")
    corpus_data = {}
    doc_count = 0
    
    with open(corpus_path) as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                doc_id = str(doc_count)  # Use sequential IDs
                corpus_data[doc_id] = {
                    "text": doc.get("text", ""),
                    "title": doc.get("title", "")
                }
                doc_count += 1
                if doc_count % 100000 == 0:
                    print(f"Processed {doc_count} documents...")
            except Exception as e:
                print(f"Error processing document: {e}")
                continue
    
    print(f"Loaded {doc_count} documents")
    
    # Build binary corpus with larger buffer
    print("Converting to binary format...")
    builder.BUFFER_SIZE = 64 * 1024 * 1024  # 64MB buffer
    stats = builder.build_binary_corpus([
        {"_id": doc_id, "text": doc.get("text", ""), "title": doc.get("title", "")} 
        for doc_id, doc in corpus_data.items()
    ], binary_path)
    
    print(f"Built binary corpus with stats: {stats}")
    
    # Test reading with larger buffer
    print("Testing memory-mapped reading...")
    corpus = MemoryMappedCorpus(
        binary_path,
        cache_size=10000,  # Reduced cache for large corpus
        buffer_size=64 * 1024 * 1024  # 64MB read buffer
    )
    
    # Try reading first and last documents
    first_doc = corpus[0]
    last_doc = corpus[len(corpus)-1]
    
    print(f"Successfully read first and last documents")
    print(f"First doc ID: {first_doc['_id']}")
    print(f"Last doc ID: {last_doc['_id']}")
    
    # Test batch reading
    print("Testing batch reading...")
    batch = corpus.get_batch(0, 100)
    print(f"Successfully read batch of {len(batch)} documents")
    
    return corpus

if __name__ == "__main__":
    corpus_path = Path("datasets/nq/corpus.jsonl")
    print(f"Testing large corpus: {corpus_path}")
    test_large_corpus(str(corpus_path))