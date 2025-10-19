"""
Core data processing functionality for corpus management.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Union
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class Document:
    """Document representation with validation"""
    id: str
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def validate(self) -> bool:
        """Validate document fields"""
        return bool(self.id and self.text)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        doc_dict = {
            "id": self.id,
            "text": self.text
        }
        if self.title:
            doc_dict["title"] = self.title
        if self.metadata:
            doc_dict["metadata"] = self.metadata
        return doc_dict
    
    @staticmethod
    def from_dict(data: Dict) -> 'Document':
        """Create Document from dictionary"""
        return Document(
            id=str(data.get("id", data.get("_id", ""))),
            text=str(data.get("text", "")),
            title=str(data.get("title", "")) or None,
            metadata=data.get("metadata")
        )

class CorpusProcessor:
    """Production-grade corpus processing with validation and error handling"""
    
    def __init__(self, 
                 input_path: Union[str, Path], 
                 output_path: Optional[Union[str, Path]] = None,
                 num_workers: int = 4,
                 chunk_size: int = 10000):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else None
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe counter and lock
        self.processed_count = 0
        self.lock = threading.Lock()
        
        # Initialize stats with thread-safe counters
        self.stats = {
            "total_documents": 0,
            "valid_documents": 0,
            "invalid_documents": 0,
            "total_tokens": 0,
            "avg_doc_length": 0,
            "errors": {
                "validation": 0,
                "json": 0,
                "other": 0
            }
        }
    
    def process_document(self, line: str) -> Optional[Document]:
        """Process a single document line with validation"""
        try:
            data = json.loads(line)
            doc = Document.from_dict(data)
            
            if not doc.validate():
                with self.lock:
                    self.processed_count += 1
                    self.stats["invalid_documents"] += 1
                    self.stats["errors"]["validation"] += 1
                self.logger.warning(f"Invalid document: {doc.id}")
                return None
                
            # Update stats
            with self.lock:
                self.processed_count += 1
                self.stats["valid_documents"] += 1
                self.stats["total_tokens"] += len(doc.text.split())
                if self.processed_count % 10000 == 0:
                    self.logger.info(f"Processed {self.processed_count} documents")
            
            return doc
            
        except json.JSONDecodeError:
            with self.lock:
                self.processed_count += 1
                self.stats["invalid_documents"] += 1
                self.stats["errors"]["json"] += 1
            self.logger.error(f"Invalid JSON: {line[:100]}...")
            return None
        except Exception as e:
            with self.lock:
                self.processed_count += 1
                self.stats["invalid_documents"] += 1
                self.stats["errors"]["other"] += 1
            self.logger.error(f"Error processing document: {str(e)}")
            return None
    
    def process_chunk(self, chunk: List[str]) -> List[Document]:
        """Process a chunk of documents in parallel"""
        valid_docs = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = executor.map(self.process_document, chunk)
            
            for doc in results:
                if doc:
                    valid_docs.append(doc)
                    
        return valid_docs
    
    def read_corpus(self) -> Iterator[Document]:
        """Read and validate corpus documents"""
        current_chunk: List[str] = []
        
        with open(self.input_path, 'r') as f:
            for line in f:
                current_chunk.append(line.strip())
                
                if len(current_chunk) >= self.chunk_size:
                    for doc in self.process_chunk(current_chunk):
                        yield doc
                    current_chunk = []
            
            # Process remaining documents
            if current_chunk:
                for doc in self.process_chunk(current_chunk):
                    yield doc
    
    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of corpus file"""
        sha256_hash = hashlib.sha256()
        
        with open(self.input_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
    
    def get_corpus_stats(self) -> Dict:
        """Get corpus statistics"""
        if not self.stats["total_documents"]:
            self.logger.warning("No statistics available - process corpus first")
            
        return self.stats.copy()
    
    def save_processed_corpus(self, documents: List[Document]):
        """Save processed and validated corpus"""
        if not self.output_path:
            raise ValueError("No output path specified")
            
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict()) + "\n")
                
        self.logger.info(f"Saved processed corpus to {self.output_path}")
        
    def process(self) -> List[Document]:
        """Process entire corpus with validation and stats collection"""
        self.logger.info(f"Processing corpus: {self.input_path}")
        
        # Get corpus checksum
        checksum = self.compute_checksum()
        self.logger.info(f"Corpus checksum: {checksum}")
        
        # Process documents
        documents = list(self.read_corpus())
        
        # Update final stats
        self.stats["total_documents"] = self.processed_count
        
        # Calculate average document length
        if self.stats["valid_documents"] > 0:
            self.stats["avg_doc_length"] = self.stats["total_tokens"] / self.stats["valid_documents"]
        
        # Save processed corpus if output path specified
        if self.output_path:
            self.save_processed_corpus(documents)
        
        # Log final stats with error breakdown
        self.logger.info(f"Completed processing with stats: {self.stats}")
        if self.stats["invalid_documents"] > 0:
            self.logger.warning(
                f"Found {self.stats['invalid_documents']} invalid documents - "
                f"Validation: {self.stats['errors']['validation']}, "
                f"JSON: {self.stats['errors']['json']}, "
                f"Other: {self.stats['errors']['other']}"
            )
        
        return documents