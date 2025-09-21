"""
Statistics and monitoring for RAG system performance.
"""

import logging
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import psutil
import numpy as np
from collections import deque

@dataclass
class QueryStats:
    """Statistics for a single query"""
    query_id: str
    start_time: float
    end_time: float = field(default=0.0)
    num_results: int = field(default=0)
    latency_ms: float = field(default=0.0)
    error: Optional[str] = field(default=None)

@dataclass
class SystemStats:
    """System-wide performance statistics"""
    total_queries: int = field(default=0)
    successful_queries: int = field(default=0)
    failed_queries: int = field(default=0)
    avg_latency_ms: float = field(default=0.0)
    min_latency_ms: float = field(default=float('inf'))
    max_latency_ms: float = field(default=0.0)
    total_results: int = field(default=0)
    memory_usage_mb: float = field(default=0.0)
    
    # Rolling window stats
    window_size: int = field(default=100)
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, query_stats: QueryStats):
        """Update statistics with new query results"""
        self.total_queries += 1
        
        if query_stats.error:
            self.failed_queries += 1
        else:
            self.successful_queries += 1
            self.total_results += query_stats.num_results
            
            # Update latency stats
            self._latencies.append(query_stats.latency_ms)
            self.min_latency_ms = min(self.min_latency_ms, query_stats.latency_ms)
            self.max_latency_ms = max(self.max_latency_ms, query_stats.latency_ms)
            self.avg_latency_ms = np.mean(self._latencies)
        
        # Update memory usage
        self.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary"""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "total_results": self.total_results,
            "memory_usage_mb": self.memory_usage_mb,
            "success_rate": (self.successful_queries / self.total_queries 
                           if self.total_queries > 0 else 0),
            "avg_results_per_query": (self.total_results / self.successful_queries 
                                    if self.successful_queries > 0 else 0)
        }

class StatsMonitor:
    """Monitor and log system performance"""
    
    def __init__(self, 
                 log_dir: Union[str, Path],
                 log_interval: int = 60,
                 window_size: int = 100):
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        
        self.stats = SystemStats(window_size=window_size)
        self.last_log_time = time.time()
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def start_query(self, query_id: str) -> QueryStats:
        """Start tracking a new query"""
        return QueryStats(
            query_id=query_id,
            start_time=time.time()
        )
    
    def end_query(self, query_stats: QueryStats, 
                 num_results: int = 0,
                 error: Optional[str] = None):
        """Complete query tracking and update stats"""
        query_stats.end_time = time.time()
        query_stats.num_results = num_results
        query_stats.error = error
        query_stats.latency_ms = (query_stats.end_time - query_stats.start_time) * 1000
        
        # Update system stats
        self.stats.update(query_stats)
        
        # Log if interval elapsed
        if time.time() - self.last_log_time > self.log_interval:
            self._log_stats()
    
    def _log_stats(self):
        """Log current statistics to file"""
        stats_dict = self.stats.to_dict()
        timestamp = datetime.now().isoformat()
        
        # Add timestamp to stats
        stats_dict["timestamp"] = timestamp
        
        # Log to file
        log_file = self.log_dir / f"stats_{datetime.now():%Y%m%d}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(stats_dict) + "\n")
            
        self.last_log_time = time.time()
        
        # Log summary to console
        self.logger.info(
            f"Stats update - Queries: {stats_dict['total_queries']}, "
            f"Avg latency: {stats_dict['avg_latency_ms']:.2f}ms, "
            f"Success rate: {stats_dict['success_rate']*100:.1f}%, "
            f"Memory: {stats_dict['memory_usage_mb']:.1f}MB"
        )
    
    def get_current_stats(self) -> Dict:
        """Get current statistics"""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset all statistics"""
        window_size = self.stats.window_size
        self.stats = SystemStats(window_size=window_size)
        self.last_log_time = time.time()