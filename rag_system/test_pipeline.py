"""
Test script for validating the RAG system pipeline components.
"""

import logging
import sys
from pathlib import Path
import time

from core.data_processor import CorpusProcessor
from core.memory_index import MemoryIndex
from core.retrieval import RetrievalService
from core.monitoring import StatsMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the complete RAG pipeline"""
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "datasets" / "fiqa"
    output_dir = base_dir / "results" / "pipeline_test"
    
    corpus_path = data_dir / "corpus.jsonl"
    processed_path = output_dir / "processed_corpus.jsonl"
    index_path = output_dir / "document_index"
    stats_dir = output_dir / "stats"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize monitoring
    monitor = StatsMonitor(stats_dir, log_interval=10)
    
    try:
        # 1. Test corpus processing
        logger.info("Testing corpus processing...")
        processor = CorpusProcessor(
            input_path=corpus_path,
            output_path=processed_path,
            num_workers=4
        )
        documents = processor.process()
        logger.info(f"Processed corpus stats: {processor.get_corpus_stats()}")
        
        # 2. Test indexing
        logger.info("\nTesting document indexing...")
        with MemoryIndex(index_path, create=True) as index:
            index.add_documents(documents[:1000])  # Test with subset first
            
            # Test random access
            test_doc = index.get_document(documents[0].id)
            logger.info(f"Retrieved test document: {test_doc.id}")
            
            # Test batch retrieval
            test_ids = [doc.id for doc in documents[10:20]]
            batch_docs = index.get_documents(test_ids)
            logger.info(f"Retrieved {len([d for d in batch_docs if d])} documents in batch")
        
        # 3. Test retrieval service
        logger.info("\nTesting retrieval service...")
        with RetrievalService(index_path) as service:
            # Test single document retrieval
            query_stats = monitor.start_query("test_single")
            doc = service.get_document(documents[0].id)
            monitor.end_query(query_stats, num_results=1 if doc else 0)
            
            # Test batch retrieval
            query_stats = monitor.start_query("test_batch")
            docs = service.get_documents(test_ids)
            monitor.end_query(query_stats, num_results=len([d for d in docs if d]))
            
            # Test cache hit
            query_stats = monitor.start_query("test_cache")
            cached_doc = service.get_document(documents[0].id)
            monitor.end_query(query_stats, num_results=1 if cached_doc else 0)
        
        # Print final stats
        logger.info("\nFinal Statistics:")
        stats = monitor.get_current_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
            
        logger.info("\nPipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during pipeline test: {str(e)}")
        raise

if __name__ == "__main__":
    test_pipeline()