"""Memory-mapped corpus integration with BEIR framework"""

import os
from pathlib import Path
from typing import Dict, List, Union, Any
from beir.retrieval.search.base import BaseSearch

# Import our implementation
from tests.memory_mapping import BinaryCorpusBuilder, MemoryMappedCorpus, LRUCache
from typing import Optional, Dict, List

class MemoryMappedRetrieval(BaseSearch):
    """BEIR-compatible wrapper for memory-mapped corpus retrieval with BEIR API"""
    
    def __init__(self, buffer_size: int = 8*1024*1024, cache_size: int = 50000,
                 compression_level: int = 1, alignment: int = 16):
        self.buffer_size = buffer_size
        self.cache_size = cache_size
        self.compression_level = compression_level
        self.alignment = alignment
        self.corpus_builder = BinaryCorpusBuilder(compression_level=compression_level)
        self.corpus = None
        self._working_dir = None
    
    def index(self, corpus: Dict[str, Dict[str, str]], *args, **kwargs):
        """Build binary corpus index from documents"""
        # Create temporary directory for binary corpus
        self._working_dir = Path(os.environ.get('TEMP', '/tmp')) / 'memmap_bench'
        self._working_dir.mkdir(parents=True, exist_ok=True)
        
        binary_path = self._working_dir / 'corpus.bin'
        
        # Convert corpus format
        documents = [
            {"_id": doc_id, "text": doc.get("text", ""), "title": doc.get("title", "")}
            for doc_id, doc in corpus.items()
        ]
        
        # Build binary corpus
        stats = self.corpus_builder.build_binary_corpus(documents, str(binary_path))
        
        # Initialize memory-mapped reader
        self.corpus = MemoryMappedCorpus(
            str(binary_path),
            cache_size=self.cache_size,
            buffer_size=self.buffer_size
        )
        
        return stats
    
    def search(self, 
               queries: Dict[str, str],
               query_ids: List[str],
               top_k: int,
               *args, **kwargs) -> Dict[str, Dict[str, float]]:
        """Perform retrieval - in this case just return exact matches
        since we're testing memory mapping performance, not ranking"""
        
        if self.corpus is None:
            raise RuntimeError("Must index corpus before searching")
        
        results = {}
        
        # For each query
        for query_id in query_ids:
            query = queries[query_id]
            query_results = {}
            
            # For demo purposes, just return first top_k documents
            # In practice, you'd implement actual ranking here
            for idx in range(min(top_k, len(self.corpus))):
                try:
                    doc = self.corpus[idx]
                    query_results[doc["_id"]] = 1.0 / (idx + 1)  # Simple reciprocal rank scoring
                except:
                    continue
                    
            results[query_id] = query_results
            
        return results
    
    def encode(self, texts: List[str], batch_size: int = 16, **kwargs) -> List[str]:
        """Text encoding is not needed for memory mapping benchmarks"""
        # We simply return the text as-is since we're testing storage/retrieval
        # not semantic search capabilities
        return texts
    
    def search_from_files(self, queries: List[str], top_k: int, **kwargs) -> Dict[str, Dict[str, float]]:
        """Implement if needed - for now this is not used in benchmarks"""
        raise NotImplementedError("File-based search not implemented")
    
    def cleanup(self):
        """Cleanup resources - called by benchmark framework"""
        if self.corpus is not None:
            self.corpus.close()
        
        if self._working_dir is not None and self._working_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._working_dir)
            except:
                pass
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()