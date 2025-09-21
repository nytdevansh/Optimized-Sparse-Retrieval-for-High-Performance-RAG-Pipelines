"""
Core Benchmarking Framework for Lightning Retrieval
Provides standardized tools for performance measurement and validation
"""

import time
import json
import numpy as np
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import statistics

@dataclass
class BenchmarkResult:
    """Structured container for benchmark results"""
    name: str
    category: str
    metrics: Dict[str, float]
    timings: Dict[str, float]
    memory: Dict[str, float]
    hardware_info: Dict[str, Any]
    parameters: Dict[str, Any]
    success: bool
    error: Optional[str] = None

class BenchmarkSuite(ABC):
    """Abstract base class for benchmark suites"""
    
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    @abstractmethod
    def setup(self) -> None:
        """Prepare the benchmark environment"""
        pass
    
    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Run the benchmark suite"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up after benchmark execution"""
        pass

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get system hardware information"""
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }
        
        memory = psutil.virtual_memory()
        memory_info = {
            'total': memory.total,
            'available': memory.available,
            'percent_used': memory.percent
        }
        
        return {
            'cpu': cpu_info,
            'memory': memory_info,
            'platform': os.uname()._asdict()
        }

class TimingContext:
    """Context manager for precise timing measurements"""
    
    def __init__(self, warmup_runs: int = 3):
        self.warmup_runs = warmup_runs
        self.times = []
        self.current_start = None
    
    def __enter__(self):
        # Run warmup iterations
        for _ in range(self.warmup_runs):
            start = time.perf_counter_ns()
            end = time.perf_counter_ns()
            _ = end - start  # Force CPU to stay active
        
        self.current_start = time.perf_counter_ns()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            end = time.perf_counter_ns()
            self.times.append((end - self.current_start) / 1e9)  # Convert to seconds
    
    @property
    def elapsed(self) -> float:
        """Get the last measured time"""
        return self.times[-1] if self.times else 0.0
    
    @property
    def statistics(self) -> Dict[str, float]:
        """Get timing statistics"""
        if not self.times:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': statistics.mean(self.times),
            'std': statistics.stdev(self.times) if len(self.times) > 1 else 0.0,
            'min': min(self.times),
            'max': max(self.times)
        }

class MemoryMonitor:
    """Monitor memory usage during benchmark execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = None
        self.peak = 0
        self.samples = []
    
    def start(self):
        """Start memory monitoring"""
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.samples = [0.0]  # Initial relative usage
    
    def sample(self):
        """Take a memory sample"""
        current = self.process.memory_info().rss
        relative_usage = (current - self.baseline) / (1024 * 1024)  # MB
        self.samples.append(relative_usage)
        self.peak = max(self.peak, current)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if not self.samples:
            return {'peak_mb': 0.0, 'mean_mb': 0.0, 'std_mb': 0.0}
        
        return {
            'peak_mb': (self.peak - self.baseline) / (1024 * 1024),
            'mean_mb': statistics.mean(self.samples),
            'std_mb': statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0
        }

def run_benchmark_suite(suite: BenchmarkSuite) -> None:
    """Execute a benchmark suite and save results"""
    print(f"\n{'='*60}")
    print(f"Running Benchmark Suite: {suite.name}")
    print(f"Category: {suite.category}")
    print('='*60)
    
    try:
        # Setup phase
        print("\nSetting up benchmark environment...")
        suite.setup()
        
        # Run benchmarks
        print("\nExecuting benchmarks...")
        result = suite.run()
        
        # Save results
        results_file = suite.results_dir / f"{suite.category}_{suite.name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Print summary
        print("\nBenchmark Summary:")
        print("-" * 40)
        for metric, value in result.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nTiming Summary:")
        print("-" * 40)
        for timing, value in result.timings.items():
            print(f"{timing}: {value:.4f}s")
        
        print("\nMemory Summary:")
        print("-" * 40)
        for mem_metric, value in result.memory.items():
            print(f"{mem_metric}: {value:.2f}MB")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        suite.cleanup()
        
    print("\nBenchmark completed successfully!")

def grade_performance(
    metric_value: float,
    target: float,
    tolerance: float = 0.1
) -> tuple[str, str]:
    """
    Grade a performance metric against a target.
    Returns (grade, description)
    """
    ratio = metric_value / target
    
    if ratio >= 1.0 + tolerance:
        return "A+", "Exceeds target significantly"
    elif ratio >= 1.0:
        return "A", "Meets target"
    elif ratio >= 0.9:
        return "B", "Close to target"
    elif ratio >= 0.75:
        return "C", "Below target"
    elif ratio >= 0.5:
        return "D", "Significantly below target"
    else:
        return "F", "Far below target"

def generate_report(results: List[BenchmarkResult], output_dir: Path) -> None:
    """Generate comprehensive benchmark report"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_benchmarks': len(results),
            'successful': sum(1 for r in results if r.success),
            'failed': sum(1 for r in results if not r.success)
        },
        'results': [r.__dict__ for r in results]
    }
    
    # Save JSON report
    with open(output_dir / 'benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate Markdown report
    md = f"# Lightning Retrieval Benchmark Report\n\n"
    md += f"Generated: {report['timestamp']}\n\n"
    
    md += "## Summary\n\n"
    md += f"- Total benchmarks: {report['summary']['total_benchmarks']}\n"
    md += f"- Successful: {report['summary']['successful']}\n"
    md += f"- Failed: {report['summary']['failed']}\n\n"
    
    for result in results:
        md += f"## {result.name} ({result.category})\n\n"
        
        if result.success:
            md += "### Metrics\n\n"
            md += "| Metric | Value | Target | Grade |\n"
            md += "|--------|--------|--------|-------|\n"
            
            for metric, value in result.metrics.items():
                # Example target mapping - customize based on your metrics
                targets = {
                    'speedup': 8.0,
                    'memory_reduction': 4.0,
                    'throughput': 10000,
                }
                target = targets.get(metric, None)
                
                if target:
                    grade, desc = grade_performance(value, target)
                    md += f"| {metric} | {value:.2f} | {target:.2f} | {grade} ({desc}) |\n"
                else:
                    md += f"| {metric} | {value:.2f} | N/A | N/A |\n"
            
            md += "\n### Timings\n\n"
            for timing, value in result.timings.items():
                md += f"- {timing}: {value:.4f}s\n"
            
            md += "\n### Memory Usage\n\n"
            for metric, value in result.memory.items():
                md += f"- {metric}: {value:.2f}MB\n"
        else:
            md += f"❌ Failed: {result.error}\n"
        
        md += "\n---\n\n"
    
    with open(output_dir / 'benchmark_report.md', 'w') as f:
        f.write(md)