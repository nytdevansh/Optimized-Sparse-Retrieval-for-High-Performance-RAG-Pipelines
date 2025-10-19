"""
Integration framework for Lightning Retrieval benchmarks
"""

import os
import sys
import yaml
import json
import time
import logging
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import test suites
from tests.bm25_performance import BM25TestSuite
from tests.embedding_quantizations import QuantizationTestSuite as QuantizationSuite
from tests.memory_mapping import MemoryMappingTestSuite
from tests.topk_selection import TopKBenchmarkSuite as TopKSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    test_suites: List[str]
    parameters: Dict[str, Any]
    output_dir: str
    hardware_info: bool = True
    memory_profiling: bool = True

@dataclass
class TestResult:
    """Results from a test suite run"""
    suite_name: str
    success: bool
    metrics: Dict[str, float]
    error: Optional[str] = None
    hardware_metrics: Optional[Dict[str, Any]] = None

class BenchmarkRunner:
    """Main benchmark orchestrator"""
    
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test suites
        self.test_suites = {
            'bm25': BM25TestSuite(),
            'quantization': QuantizationSuite(),
            'memory_mapping': MemoryMappingTestSuite(),
            'topk': TopKSuite()
        }
    
    def _load_config(self, config_path: Path) -> BenchmarkConfig:
        """Load benchmark configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return BenchmarkConfig(**config)
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get system hardware information"""
        try:
            import psutil
            import cpuinfo
            
            cpu_info = cpuinfo.get_cpu_info()
            memory = psutil.virtual_memory()
            
            return {
                'cpu': {
                    'brand': cpu_info.get('brand_raw', 'Unknown'),
                    'architecture': cpu_info.get('arch', 'Unknown'),
                    'cores': {
                        'physical': psutil.cpu_count(logical=False),
                        'logical': psutil.cpu_count(logical=True)
                    },
                    'features': cpu_info.get('flags', [])
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent_used': memory.percent
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get hardware info: {e}")
            return {}
    
    def _run_memory_profiling(self, suite_name: str) -> Dict[str, float]:
        """Run memory profiling for a test suite"""
        try:
            import tracemalloc
            import gc
            
            gc.collect()
            tracemalloc.start()
            
            # Run a small subset of tests
            suite = self.test_suites[suite_name]
            if suite_name == 'bm25':
                suite.test_correctness()
            elif suite_name == 'quantization':
                suite.test_quantization_quality(suite.generate_test_embeddings())
            elif suite_name == 'memory_mapping':
                raise NotImplementedError("Memory mapping tests not implemented yet")
            elif suite_name == 'topk':
                suite.test_correctness()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return {
                'current_memory_mb': current / (1024 * 1024),
                'peak_memory_mb': peak / (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Memory profiling failed: {e}")
            return {}
    
    def _format_results(self, results: List[TestResult]) -> str:
        """Generate markdown report"""
        md = "# Lightning Retrieval Integration Test Results\n\n"
        md += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary table
        md += "## Summary\n\n"
        md += "| Test Suite | Status | Key Metrics |\n"
        md += "|------------|--------|-------------|\n"
        
        for result in results:
            status = "✅" if result.success else "❌"
            metrics = []
            
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    metrics.append(f"{metric}: {value:.4f}")
                else:
                    metrics.append(f"{metric}: {value}")
            
            md += f"| {result.suite_name} | {status} | {', '.join(metrics[:3])} |\n"
        
        # Detailed results
        for result in results:
            md += f"\n## {result.suite_name}\n\n"
            
            if not result.success:
                md += f"❌ Failed: {result.error}\n\n"
                continue
            
            md += "### Metrics\n\n"
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    md += f"- {metric}: {value:.4f}\n"
                else:
                    md += f"- {metric}: {value}\n"
            
            if result.hardware_metrics:
                md += "\n### Hardware Metrics\n\n"
                for metric, value in result.hardware_metrics.items():
                    if isinstance(value, float):
                        md += f"- {metric}: {value:.1f}MB\n"
                    else:
                        md += f"- {metric}: {value}\n"
        
        return md
    
    def run(self) -> List[TestResult]:
        """Run all configured test suites"""
        logger.info("Starting integration test suite")
        
        results = []
        hardware_info = self._get_hardware_info() if self.config.hardware_info else None
        
        for suite_name in self.config.test_suites:
            if suite_name not in self.test_suites:
                logger.warning(f"Unknown test suite: {suite_name}")
                continue
            
            logger.info(f"\nRunning {suite_name} test suite...")
            suite = self.test_suites[suite_name]
            
            try:
                # Run tests
                if suite_name == 'bm25':
                    # Generate test data first
                    suite.generate_test_data(num_docs=10000, num_queries=100)
                    # Run the tests
                    correctness = suite.test_correctness()
                    performance = suite.benchmark_performance()
                    memory = suite.test_memory_usage()
                    metrics = {**correctness, **performance, **memory}
                elif suite_name == 'quantization':
                    quality = suite.test_quantization_quality(suite.generate_test_embeddings())
                    speed = suite.benchmark_quantization_speed(suite.generate_test_embeddings())
                    dotprod = suite.benchmark_dot_product_performance(suite.generate_test_embeddings())
                    retrieval = suite.test_retrieval_quality_impact(suite.generate_test_embeddings())
                    metrics = {**quality, **speed, **dotprod, **retrieval}
                elif suite_name == 'memory_mapping':
                    # Generate test corpus first
                    test_docs = suite.generate_test_corpus(num_docs=20000, avg_doc_length=150)
                    # Run all memory mapping tests
                    creation_results = suite.test_corpus_creation_speed(test_docs)
                    random_results = suite.test_random_access_performance(test_docs)
                    sequential_results = suite.test_sequential_access_performance(test_docs)
                    memory_results = suite.test_memory_usage(test_docs)
                    cold_start_results = suite.test_cold_start_performance(test_docs)
                    # Combine all results
                    metrics = {
                        **creation_results,
                        **random_results,
                        **sequential_results,
                        **memory_results,
                        **cold_start_results
                    }
                elif suite_name == 'topk':
                    metrics = suite.test_correctness()
                
                # Run memory profiling if enabled
                memory_metrics = None
                if self.config.memory_profiling:
                    memory_metrics = self._run_memory_profiling(suite_name)
                
                result = TestResult(
                    suite_name=suite_name,
                    success=True,
                    metrics=metrics,
                    hardware_metrics=memory_metrics
                )
                
            except Exception as e:
                logger.error(f"Test suite {suite_name} failed: {str(e)}")
                result = TestResult(
                    suite_name=suite_name,
                    success=False,
                    metrics={},
                    error=str(e)
                )
            
            results.append(result)
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy(x) for x in obj]
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Save individual result
            result_path = self.results_dir / f"{suite_name}_results.json"
            result_dict = asdict(result)
            result_dict = convert_numpy(result_dict)
            with open(result_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
        
        # Generate report
        report = self._format_results(results)
        report_path = self.results_dir / "integration_test_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save hardware info if available
        if hardware_info:
            hw_path = self.results_dir / "hardware_info.json"
            with open(hw_path, 'w') as f:
                json.dump(hardware_info, f, indent=2)
        
        logger.info(f"\nTests completed. Results saved to {self.results_dir}")
        return results

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightning Retrieval Integration Tests")
    parser.add_argument('config', type=Path, help='Path to benchmark configuration YAML')
    parser.add_argument('--output-dir', type=Path, default='test_results',
                       help='Directory to save results (default: test_results/)')
    
    args = parser.parse_args()
    
    try:
        runner = BenchmarkRunner(args.config)
        results = runner.run()
        
        # Exit with error if any suite failed
        if any(not r.success for r in results):
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()