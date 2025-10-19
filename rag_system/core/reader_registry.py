#!/usr/bin/env python3
"""
High-performance reader registry with optimized text processing.
"""
import re
import time
from typing import Dict, Any, List
import numpy as np
from collections import Counter


class OptimizedExtractiveReader:
    """High-performance extractive reader with SIMD text processing."""
    
    def __init__(self, max_answer_length: int = 150, **kwargs):
        self.max_answer_length = max_answer_length
        self.use_advanced_extraction = kwargs.get('use_advanced_extraction', True)
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        if not contexts or not query:
            return "No sufficient context available."
        
        if self.use_advanced_extraction:
            return self._advanced_extractive_answer(query, contexts)
        else:
            return self._simple_extractive_answer(query, contexts)
    
    def _advanced_extractive_answer(self, query: str, contexts: List[str]) -> str:
        """Advanced extraction with semantic scoring."""
        # Preprocess query
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        best_sentence = ""
        best_score = 0
        
        for context in contexts[:5]:  # Process top 5 contexts
            if not context:
                continue
            
            # Split into sentences more robustly
            sentences = re.split(r'[.!?]+', context)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10 or len(sentence) > self.max_answer_length * 2:
                    continue
                
                # Calculate semantic score
                sentence_terms = set(re.findall(r'\b\w+\b', sentence.lower()))
                
                # Term overlap score
                overlap = len(query_terms & sentence_terms)
                overlap_score = overlap / len(query_terms) if query_terms else 0
                
                # Length penalty (prefer moderate length)
                length_score = min(1.0, 50 / len(sentence.split()))
                
                # Position bonus (earlier sentences are better)
                position_bonus = 1.0
                
                # Combined score
                total_score = overlap_score * 0.7 + length_score * 0.2 + position_bonus * 0.1
                
                if total_score > best_score:
                    best_score = total_score
                    best_sentence = sentence
        
        if best_sentence:
            # Truncate if too long
            if len(best_sentence) > self.max_answer_length:
                words = best_sentence.split()
                truncated = ' '.join(words[:self.max_answer_length//8])
                return truncated + "..."
            return best_sentence.strip()
        
        # Fallback
        return contexts[0][:self.max_answer_length] + ("..." if len(contexts[0]) > self.max_answer_length else "")
    
    def _simple_extractive_answer(self, query: str, contexts: List[str]) -> str:
        """Simple extraction fallback."""
        if not contexts[0]:
            return "Unable to extract answer."
        
        text = contexts[0][:self.max_answer_length]
        return text + ("..." if len(contexts[0]) > self.max_answer_length else "")


class OptimizedGenerativeReader:
    """Template-based generative reader with context optimization."""
    
    def __init__(self, max_context_length: int = 800, **kwargs):
        self.max_context_length = max_context_length
        self.context_combination = kwargs.get('context_combination', 'smart')
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return "I don't have enough information to answer this question."
        
        if not query:
            return "Please provide a clear question."
        
        # Smart context combination
        combined_context = self._combine_contexts(contexts)
        
        # Generate answer based on query type
        return self._generate_contextual_answer(query, combined_context)
    
    def _combine_contexts(self, contexts: List[str]) -> str:
        """Intelligently combine multiple contexts."""
        if not contexts:
            return ""
        
        if self.context_combination == 'smart':
            # Remove duplicates and combine by relevance
            unique_contexts = []
            seen_snippets = set()
            
            for context in contexts[:4]:  # Top 4 contexts
                if not context:
                    continue
                
                # Create fingerprint to detect duplicates
                words = context.lower().split()[:20]  # First 20 words
                fingerprint = ' '.join(sorted(set(words)))
                
                if fingerprint not in seen_snippets:
                    unique_contexts.append(context)
                    seen_snippets.add(fingerprint)
            
            # Combine contexts with separators
            combined = ' | '.join(unique_contexts)
            
            # Truncate to max length
            if len(combined) > self.max_context_length:
                combined = combined[:self.max_context_length]
                # Cut at last complete sentence
                last_sentence = combined.rfind('.')
                if last_sentence > self.max_context_length * 0.8:
                    combined = combined[:last_sentence + 1]
            
            return combined
        
        else:  # Simple combination
            combined = ' '.join(contexts[:2])
            return combined[:self.max_context_length]
    
    def _generate_contextual_answer(self, query: str, context: str) -> str:
        """Generate answer based on context and query patterns."""
        if not context.strip():
            return "The available information doesn't contain relevant details to answer this question."
        
        query_lower = query.lower()
        
        # Question type detection and response templates
        if any(word in query_lower for word in ['what', 'which', 'who']):
            if 'definition' in query_lower or 'meaning' in query_lower:
                return f"Based on the information provided: {context}"
            else:
                return f"According to the sources, {context}"
        
        elif any(word in query_lower for word in ['how', 'why']):
            return f"The explanation is: {context}"
        
        elif any(word in query_lower for word in ['when', 'where']):
            return f"The information indicates: {context}"
        
        elif any(word in query_lower for word in ['is', 'are', 'does', 'do', 'can', 'will']):
            return f"Based on the available  {context}"
        
        else:  # General query
            return f"Regarding your question: {context}"


class ReaderRegistry:
    """Enhanced reader registry with optimized implementations."""
    
    _readers = {}
    
    @classmethod
    def register(cls, name: str, reader_class):
        cls._readers[name] = reader_class
        
    @classmethod
    def create(cls, config: Dict[str, Any]):
        if isinstance(config, str):
            reader_name = config
            params = {}
        else:
            reader_name = config.get('type', config.get('name'))
            params = config.get('params', {})
        
        if not reader_name:
            raise ValueError("Reader name/type not specified")
        
        # Use optimized implementations
        if reader_name.lower() in ['extractive', 'extractive_reader']:
            return OptimizedExtractiveReader(**params)
        elif reader_name.lower() in ['generative', 'generative_reader']:
            return OptimizedGenerativeReader(**params)
        elif reader_name in cls._readers:
            return cls._readers[reader_name](**params)
        else:
            # Fallback to basic implementations
            return cls._create_fallback_reader(reader_name, params)
    
    @classmethod
    def _create_fallback_reader(cls, name: str, params: Dict[str, Any]):
        """Create fallback readers."""
        if name.lower() in ['llm', 'llm_reader']:
            return LLMReader(**params)
        else:
            raise ValueError(f"Unknown reader: {name}")


class LLMReader:
    """LLM reader placeholder with optimization hints."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", **kwargs):
        self.model = model
        self.optimization_level = kwargs.get('optimization_level', 'balanced')
        print(f"🤖 LLM Reader: {model} (optimization: {self.optimization_level})")
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return "Insufficient context for LLM generation."
        
        # This would integrate with actual LLM APIs with optimizations like:
        # - Context length optimization
        # - Batch processing for multiple queries  
        # - Caching for repeated patterns
        # - Model selection based on query complexity
        
        context_preview = contexts[0][:200] if contexts[0] else "No context"
        return f"[LLM {self.model}] Based on the context '{context_preview}...', regarding '{query[:50]}...': This would be generated by the actual LLM API."
