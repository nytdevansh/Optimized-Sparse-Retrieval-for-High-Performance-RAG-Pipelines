"""
Core benchmark runner for Lightning Retrieval test suite
"""

import os
import time
import json
import yaml
import logging
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import torch

# Force CPU usage and disable GPU
torch.backends.mps.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    seed: int = 42
    device: str = "cpu"
    num_threads: int = -1
    dataset: Dict[str, Any] = None
    benchmark: Dict[str, Any] = None
    metrics: List[str] = None
    targets: Dict[str, Any] = None
    methods: Dict[str, Any] = None
    output: Dict[str, Any] = None

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    method: str
    dataset: str
    metrics: Dict[str, float]
    latency: Dict[str, float]
    memory: Dict[str, float]
    config: Dict[str, Any]
    hardware_info: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    
class PerformanceMonitor:
    """Monitor system performance during benchmark execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.samples = []
        self.peak_memory = 0
        self.start_time = None
        
    def start(self):
        """Start monitoring"""
        self.baseline_memory = self.process.memory_info().rss
        self.peak_memory = self.baseline_memory
        self.samples = []
        self.start_time = time.perf_counter()
        
    def sample(self):
        """Take a performance sample"""
        current = self.process.memory_info().rss
        cpu_percent = self.process.cpu_percent()
        
        sample = {
            'timestamp': time.perf_counter() - self.start_time,
            'memory_mb': (current - self.baseline_memory) / (1024 * 1024),
            'cpu_percent': cpu_percent
        }
        
        self.samples.append(sample)
        self.peak_memory = max(self.peak_memory, current)
        
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.samples:
            return {}
            
        memory_usage = [s['memory_mb'] for s in self.samples]
        cpu_usage = [s['cpu_percent'] for s in self.samples]
        
        return {
            'peak_memory_mb': (self.peak_memory - self.baseline_memory) / (1024 * 1024),
            'mean_memory_mb': np.mean(memory_usage),
            'std_memory_mb': np.std(memory_usage),
            'mean_cpu_percent': np.mean(cpu_usage),
            'p95_memory_mb': np.percentile(memory_usage, 95)
        }

class BenchmarkRunner:
    """Main benchmark orchestrator"""
    
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.monitor = PerformanceMonitor()
        self.results_dir = Path(self.config.output['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: Path) -> BenchmarkConfig:
        """Load and validate configuration"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            
        # Set environment variables
        if config_dict.get('num_threads', -1) > 0:
            os.environ["OMP_NUM_THREADS"] = str(config_dict['num_threads'])
            os.environ["MKL_NUM_THREADS"] = str(config_dict['num_threads'])
            
        return BenchmarkConfig(**config_dict)
    
    def _get_hardware_info(self) -> Dict[str, Any]:
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
            'device': self.config.device,
            'num_threads': self.config.num_threads
        }
    
    def _format_results(self, results: List[BenchmarkResult]) -> str:
        """Generate markdown report from results"""
        md = "# Lightning Retrieval Benchmark Results\n\n"
        md += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary table
        md += "## Summary\n\n"
        md += "| Method | Dataset | Status | nDCG@10 | Latency (ms) | Memory (MB) |\n"
        md += "|--------|----------|--------|----------|--------------|-------------|\n"
        
        for r in results:
            status = "✅" if r.success else "❌"
            ndcg = r.metrics.get('nDCG@10', 'N/A')
            latency = r.latency.get('mean_query_time', 'N/A')
            memory = r.memory.get('peak_memory_mb', 'N/A')
            
            md += f"| {r.method} | {r.dataset} | {status} | {ndcg:.4f} | {latency:.2f} | {memory:.1f} |\n"
        
        # Detailed results
        for r in results:
            md += f"\n## {r.method} on {r.dataset}\n\n"
            
            if not r.success:
                md += f"❌ Failed: {r.error}\n\n"
                continue
                
            md += "### Metrics\n\n"
            for metric, value in r.metrics.items():
                md += f"- {metric}: {value:.4f}\n"
            
            md += "\n### Performance\n\n"
            for metric, value in r.latency.items():
                md += f"- {metric}: {value:.4f}ms\n"
                
            md += "\n### Memory Usage\n\n"
            for metric, value in r.memory.items():
                md += f"- {metric}: {value:.1f}MB\n"
        
        return md
    
    def run(self) -> List[BenchmarkResult]:
        """Run all benchmarks according to configuration"""
        logger.info("Starting benchmark suite")
        logger.info(f"Configuration: {self.config}")
        
        # Set random seeds
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        results = []
        hardware_info = self._get_hardware_info()
        
        # Run benchmarks for each method
        for method, method_config in self.config.methods.items():
            try:
                logger.info(f"\nRunning benchmark for {method}")
                
                # Initialize method-specific components
                # TODO: Add method initialization logic
                
                # Run benchmark
                self.monitor.start()
                
                # TODO: Add benchmark execution logic
                
                # Collect results
                perf_stats = self.monitor.get_stats()
                
                result = BenchmarkResult(
                    method=method,
                    dataset=self.config.dataset['name'],
                    metrics={},  # TODO: Add actual metrics
                    latency={},  # TODO: Add latency measurements
                    memory=perf_stats,
                    config=method_config,
                    hardware_info=hardware_info,
                    success=True
                )
                
            except Exception as e:
                logger.error(f"Benchmark failed for {method}: {str(e)}")
                result = BenchmarkResult(
                    method=method,
                    dataset=self.config.dataset['name'],
                    metrics={},
                    latency={},
                    memory={},
                    config=method_config,
                    hardware_info=hardware_info,
                    success=False,
                    error=str(e)
                )
                
            results.append(result)
            
            # Save individual result
            result_path = self.results_dir / f"{method}_{result.dataset}_result.json"
            with open(result_path, 'w') as f:
                json.dump(asdict(result), f, indent=2)
        
        # Generate report
        report = self._format_results(results)
        report_path = self.results_dir / "benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
            
        logger.info(f"\nBenchmark completed. Results saved to {self.results_dir}")
        return results

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightning Retrieval Benchmark Runner")
    parser.add_argument('config', type=Path, help='Path to benchmark configuration YAML')
    parser.add_argument('--output-dir', type=Path, default='results',
                       help='Directory to save results (default: results/)')
    
    args = parser.parse_args()
    
    try:
        runner = BenchmarkRunner(args.config)
        results = runner.run()
        
        # Exit with error if any benchmark failed
        if any(not r.success for r in results):
            exit(1)
            
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()