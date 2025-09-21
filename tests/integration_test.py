#!/usr/bin/env python3
"""
Lightning Retrieval Integration Test Runner
Orchestrates all performance tests and generates comprehensive reports
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import argparse

# Import test modules
sys.path.append(str(Path(__file__).parent))

try:
    from test_framework import RetrievalTestSuite, BenchmarkResult, HardwareProfiler
    from bm25_tests import BM25TestSuite
    from quantization_tests import QuantizationTestSuite
    from memory_mapping_tests import MemoryMappingTestSuite
    from topk_tests import TopKBenchmarkSuite
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Make sure all test files are in the same directory.")
    sys.exit(1)


@dataclass
class TestConfiguration:
    """Test configuration parameters"""
    small_corpus_size: int = 1000
    medium_corpus_size: int = 10000
    large_corpus_size: int = 50000
    embedding_dim: int = 768
    num_queries: int = 100
    enable_large_tests: bool = False
    output_dir: str = "test_results"
    generate_plots: bool = True


class IntegrationTestRunner:
    """Main test runner that orchestrates all performance tests"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.profiler = HardwareProfiler()
        self.all_results = []
        self.test_summary = {
            'timestamp': time.time(),
            'config': asdict(config),
            'hardware': self.profiler.cpu_info,
            'test_results': {},
            'overall_assessment': {}
        }
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀" * 20)
        print("LIGHTNING RETRIEVAL INTEGRATION TEST SUITE")
        print("🚀" * 20)
        
        print(f"\nHardware Configuration:")
        print(f"  CPU: {self.profiler.cpu_info.get('brand', 'Unknown')}")
        print(f"  Cores: {self.profiler.cpu_info.get('cores', 'Unknown')}")
        print(f"  Threads: {self.profiler.cpu_info.get('threads', 'Unknown')}")
        print(f"  AVX2: {self.profiler.cpu_info.get('avx2', False)}")
        print(f"  AVX512: {self.profiler.cpu_info.get('avx512f', False)}")
        
        print(f"\nTest Configuration:")
        print(f"  Small corpus: {self.config.small_corpus_size} docs")
        print(f"  Medium corpus: {self.config.medium_corpus_size} docs")
        print(f"  Large corpus: {self.config.large_corpus_size} docs")
        print(f"  Large tests enabled: {self.config.enable_large_tests}")
        
        # Run individual test suites
        self._run_bm25_tests()
        self._run_quantization_tests()
        self._run_memory_mapping_tests()
        self._run_topk_tests()
        
        # Generate comprehensive report
        self._generate_integration_report()
        self._generate_recommendations()
        
        print(f"\n{'=' * 60}")
        print("INTEGRATION TEST COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'=' * 60}")
    
    def _run_bm25_tests(self):
        """Run BM25 SIMD performance tests"""
        print(f"\n{'=' * 60}")
        print("🔥 BM25 SIMD PERFORMANCE TESTS")
        print(f"{'=' * 60}")
        
        try:
            suite = BM25TestSuite()
            suite.generate_test_data(
                num_docs=self.config.medium_corpus_size,
                num_queries=self.config.num_queries
            )
            
            # Test correctness
            print("\n📋 Testing BM25 correctness...")
            correctness = suite.test_correctness()
            
            # Benchmark performance
            print(f"\n⚡ Benchmarking BM25 performance...")
            performance = suite.benchmark_performance()
            
            # Test memory usage
            print(f"\n💾 Testing BM25 memory usage...")
            memory = suite.test_memory_usage()
            
            # Store results
            bm25_results = {
                'correctness': correctness,
                'performance': performance,
                'memory': memory,
                'target_speedup_achieved': performance.get('vectorized_speedup', 0) >= 8.0,
                'all_tests_correct': all(correctness.values())
            }
            
            self.test_summary['test_results']['bm25'] = bm25_results
            
            # Print summary
            speedup = performance.get('vectorized_speedup', 0)
            print(f"\n📊 BM25 Test Summary:")
            print(f"  Correctness: {'✅ PASS' if bm25_results['all_tests_correct'] else '❌ FAIL'}")
            print(f"  Target speedup (8x): {'✅ ACHIEVED' if bm25_results['target_speedup_achieved'] else '❌ NOT ACHIEVED'}")
            print(f"  Actual speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"❌ BM25 tests failed: {e}")
            self.test_summary['test_results']['bm25'] = {'error': str(e)}
    
    def _run_quantization_tests(self):
        """Run embedding quantization tests"""
        print(f"\n{'=' * 60}")
        print("🔢 EMBEDDING QUANTIZATION TESTS")
        print(f"{'=' * 60}")
        
        try:
            suite = QuantizationTestSuite()
            
            # Generate test embeddings
            embeddings = suite.generate_test_embeddings(
                num_vectors=self.config.medium_corpus_size,
                dim=self.config.embedding_dim
            )
            
            # Test quantization quality
            print(f"\n📋 Testing quantization quality...")
            quality = suite.test_quantization_quality(embeddings)
            
            # Benchmark quantization speed
            print(f"\n⚡ Benchmarking quantization speed...")
            speed = suite.benchmark_quantization_speed(embeddings)
            
            # Benchmark dot product performance
            print(f"\n💨 Benchmarking dot product performance...")
            dot_product = suite.benchmark_dot_product_performance(embeddings)
            
            # Test retrieval quality impact
            print(f"\n🎯 Testing retrieval quality impact...")
            retrieval_quality = suite.test_retrieval_quality_impact(embeddings)
            
            # Store results
            quant_results = {
                'quality': quality,
                'speed': speed,
                'dot_product': dot_product,
                'retrieval_quality': retrieval_quality,
                'speedup_achieved': dot_product.get('speedup', 0) >= 3.0,
                'quality_preserved': quality.get('symmetric_cosine_correlation', 0) > 0.95
            }
            
            self.test_summary['test_results']['quantization'] = quant_results
            
            # Print summary
            speedup = dot_product.get('speedup', 0)
            correlation = quality.get('symmetric_cosine_correlation', 0)
            print(f"\n📊 Quantization Test Summary:")
            print(f"  Quality preservation: {'✅ PASS' if quant_results['quality_preserved'] else '❌ FAIL'}")
            print(f"  Target speedup (3x): {'✅ ACHIEVED' if quant_results['speedup_achieved'] else '❌ NOT ACHIEVED'}")
            print(f"  Actual speedup: {speedup:.2f}x")
            print(f"  Cosine correlation: {correlation:.4f}")
            
        except Exception as e:
            print(f"❌ Quantization tests failed: {e}")
            self.test_summary['test_results']['quantization'] = {'error': str(e)}
    
    def _run_memory_mapping_tests(self):
        """Run memory mapping performance tests"""
        print(f"\n{'=' * 60}")
        print("💾 MEMORY MAPPING PERFORMANCE TESTS")
        print(f"{'=' * 60}")
        
        try:
            suite = MemoryMappingTestSuite()
            
            # Generate test corpus
            corpus_size = self.config.large_corpus_size if self.config.enable_large_tests else self.config.medium_corpus_size
            documents = suite.generate_test_corpus(num_docs=corpus_size)
            
            # Test corpus creation speed
            print(f"\n🏗️  Testing corpus creation speed...")
            creation = suite.test_corpus_creation_speed(documents)
            
            # Test random access performance
            print(f"\n🎲 Testing random access performance...")
            random_access = suite.test_random_access_performance(documents)
            
            # Test sequential access performance
            print(f"\n📜 Testing sequential access performance...")
            sequential = suite.test_sequential_access_performance(documents)
            
            # Test memory usage
            print(f"\n🧠 Testing memory usage...")
            memory = suite.test_memory_usage(documents)
            
            # Store results
            mmap_results = {
                'creation': creation,
                'random_access': random_access,
                'sequential': sequential,
                'memory': memory,
                'random_speedup_achieved': random_access.get('mmap_speedup', 0) >= 2.0,
                'memory_efficient': memory.get('memory_efficiency', 0) >= 2.0
            }
            
            self.test_summary['test_results']['memory_mapping'] = mmap_results
            
            # Print summary
            random_speedup = random_access.get('mmap_speedup', 0)
            memory_efficiency = memory.get('memory_efficiency', 0)
            print(f"\n📊 Memory Mapping Test Summary:")
            print(f"  Random access speedup (2x): {'✅ ACHIEVED' if mmap_results['random_speedup_achieved'] else '❌ NOT ACHIEVED'}")
            print(f"  Memory efficiency (2x): {'✅ ACHIEVED' if mmap_results['memory_efficient'] else '❌ NOT ACHIEVED'}")
            print(f"  Actual random speedup: {random_speedup:.2f}x")
            print(f"  Actual memory efficiency: {memory_efficiency:.2f}x")
            
        except Exception as e:
            print(f"❌ Memory mapping tests failed: {e}")
            self.test_summary['test_results']['memory_mapping'] = {'error': str(e)}
    
    def _run_topk_tests(self):
        """Run top-k selection performance tests"""
        print(f"\n{'=' * 60}")
        print("🔝 TOP-K SELECTION PERFORMANCE TESTS")
        print(f"{'=' * 60}")
        
        try:
            suite = TopKBenchmarkSuite()
            
            # Test correctness
            print(f"\n📋 Testing top-k correctness...")
            correctness = suite.test_correctness()
            
            # Benchmark performance
            print(f"\n⚡ Benchmarking top-k performance...")
            performance = suite.benchmark_performance()
            
            # Test memory efficiency
            print(f"\n🧠 Testing top-k memory efficiency...")
            memory = suite.test_memory_efficiency()
            
            # Analyze performance characteristics
            recommendations = suite.analyze_performance_characteristics(performance)
            
            # Calculate overall speedup achievement
            speedups = [rec.get('speedup', 1.0) for rec in recommendations.values()]
            avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
            good_speedups = sum(1 for s in speedups if s >= 5.0)
            speedup_success_rate = good_speedups / len(speedups) if speedups else 0.0
            
            # Store results
            topk_results = {
                'correctness': correctness,
                'performance': performance,
                'memory': memory,
                'recommendations': recommendations,
                'avg_speedup': avg_speedup,
                'speedup_success_rate': speedup_success_rate,
                'all_correct': all(correctness.values()),
                'target_speedup_achieved': speedup_success_rate > 0.5
            }
            
            self.test_summary['test_results']['topk'] = topk_results
            
            # Print summary
            print(f"\n📊 Top-K Test Summary:")
            print(f"  Correctness: {'✅ PASS' if topk_results['all_correct'] else '❌ FAIL'}")
            print(f"  Target speedup success: {'✅ GOOD' if topk_results['target_speedup_achieved'] else '❌ POOR'}")
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Success rate (5x+ speedup): {speedup_success_rate:.1%}")
            
        except Exception as e:
            print(f"❌ Top-k tests failed: {e}")
            self.test_summary['test_results']['topk'] = {'error': str(e)}
    
    def _generate_integration_report(self):
        """Generate comprehensive integration test report"""
        print(f"\n{'=' * 60}")
        print("📝 GENERATING INTEGRATION REPORT")
        print(f"{'=' * 60}")
        
        # Calculate overall assessment
        self._calculate_overall_assessment()
        
        # Save JSON report
        json_report_path = self.output_dir / "integration_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.test_summary, f, indent=2, default=str)
        
        # Generate Markdown report
        self._generate_markdown_report()
        
        # Generate performance comparison charts (if enabled)
        if self.config.generate_plots:
            self._generate_performance_plots()
        
        print(f"📄 JSON report saved: {json_report_path}")
        print(f"📄 Markdown report saved: {self.output_dir / 'integration_report.md'}")
    
    def _calculate_overall_assessment(self):
        """Calculate overall test assessment"""
        results = self.test_summary['test_results']
        assessment = {
            'tests_run': len([k for k, v in results.items() if 'error' not in v]),
            'tests_failed': len([k for k, v in results.items() if 'error' in v]),
            'component_scores': {}
        }
        
        # BM25 Assessment
        if 'bm25' in results and 'error' not in results['bm25']:
            bm25 = results['bm25']
            bm25_score = (
                (1.0 if bm25.get('all_tests_correct', False) else 0.0) +
                (1.0 if bm25.get('target_speedup_achieved', False) else 0.0)
            ) / 2.0
            assessment['component_scores']['bm25'] = bm25_score
        
        # Quantization Assessment
        if 'quantization' in results and 'error' not in results['quantization']:
            quant = results['quantization']
            quant_score = (
                (1.0 if quant.get('quality_preserved', False) else 0.0) +
                (1.0 if quant.get('speedup_achieved', False) else 0.0)
            ) / 2.0
            assessment['component_scores']['quantization'] = quant_score
        
        # Memory Mapping Assessment
        if 'memory_mapping' in results and 'error' not in results['memory_mapping']:
            mmap = results['memory_mapping']
            mmap_score = (
                (1.0 if mmap.get('random_speedup_achieved', False) else 0.0) +
                (1.0 if mmap.get('memory_efficient', False) else 0.0)
            ) / 2.0
            assessment['component_scores']['memory_mapping'] = mmap_score
        
        # Top-K Assessment
        if 'topk' in results and 'error' not in results['topk']:
            topk = results['topk']
            topk_score = (
                (1.0 if topk.get('all_correct', False) else 0.0) +
                (1.0 if topk.get('target_speedup_achieved', False) else 0.0)
            ) / 2.0
            assessment['component_scores']['topk'] = topk_score
        
        # Overall Score
        if assessment['component_scores']:
            assessment['overall_score'] = sum(assessment['component_scores'].values()) / len(assessment['component_scores'])
        else:
            assessment['overall_score'] = 0.0
        
        # Grade assignment
        score = assessment['overall_score']
        if score >= 0.9:
            assessment['grade'] = 'A+ (Excellent)'
        elif score >= 0.8:
            assessment['grade'] = 'A (Very Good)'
        elif score >= 0.7:
            assessment['grade'] = 'B+ (Good)'
        elif score >= 0.6:
            assessment['grade'] = 'B (Acceptable)'
        elif score >= 0.5:
            assessment['grade'] = 'C (Needs Improvement)'
        else:
            assessment['grade'] = 'F (Major Issues)'
        
        self.test_summary['overall_assessment'] = assessment
    
    def _generate_markdown_report(self):
        """Generate markdown report"""
        report_lines = [
            "# Lightning Retrieval Integration Test Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.test_summary['timestamp']))}",
            f"**Overall Grade:** {self.test_summary['overall_assessment']['grade']}",
            f"**Overall Score:** {self.test_summary['overall_assessment']['overall_score']:.2f}/1.00",
            "",
            "## Hardware Configuration",
            "",
            f"- **CPU:** {self.profiler.cpu_info.get('brand', 'Unknown')}",
            f"- **Architecture:** {self.profiler.cpu_info.get('arch', 'Unknown')}",
            f"- **Cores:** {self.profiler.cpu_info.get('cores', 'Unknown')}",
            f"- **Threads:** {self.profiler.cpu_info.get('threads', 'Unknown')}",
            f"- **AVX2 Support:** {'✅' if self.profiler.cpu_info.get('avx2') else '❌'}",
            f"- **AVX512 Support:** {'✅' if self.profiler.cpu_info.get('avx512f') else '❌'}",
            "",
            "## Test Configuration",
            "",
            f"- **Small Corpus:** {self.config.small_corpus_size:,} documents",
            f"- **Medium Corpus:** {self.config.medium_corpus_size:,} documents",
            f"- **Large Corpus:** {self.config.large_corpus_size:,} documents",
            f"- **Embedding Dimension:** {self.config.embedding_dim}",
            f"- **Number of Queries:** {self.config.num_queries}",
            f"- **Large Tests Enabled:** {'Yes' if self.config.enable_large_tests else 'No'}",
            "",
            "## Component Test Results",
            ""
        ]
        
        # Add results for each component
        results = self.test_summary['test_results']
        
        if 'bm25' in results:
            report_lines.extend(self._format_bm25_results(results['bm25']))
        
        if 'quantization' in results:
            report_lines.extend(self._format_quantization_results(results['quantization']))
        
        if 'memory_mapping' in results:
            report_lines.extend(self._format_memory_mapping_results(results['memory_mapping']))
        
        if 'topk' in results:
            report_lines.extend(self._format_topk_results(results['topk']))
        
        # Add overall assessment
        report_lines.extend([
            "",
            "## Overall Assessment",
            "",
            f"**Final Grade:** {self.test_summary['overall_assessment']['grade']}",
            f"**Overall Score:** {self.test_summary['overall_assessment']['overall_score']:.2f}/1.00",
            "",
            "### Component Scores",
            ""
        ])
        
        for component, score in self.test_summary['overall_assessment']['component_scores'].items():
            status = "✅ PASS" if score >= 0.7 else "⚠️ PARTIAL" if score >= 0.5 else "❌ FAIL"
            report_lines.append(f"- **{component.replace('_', ' ').title()}:** {score:.2f} {status}")
        
        # Save markdown report
        markdown_path = self.output_dir / "integration_report.md"
        with open(markdown_path, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _format_bm25_results(self, results: Dict) -> List[str]:
        """Format BM25 test results for markdown"""
        if 'error' in results:
            return [
                "### 🔥 BM25 SIMD Performance Tests",
                "",
                f"❌ **Status:** Failed with error: {results['error']}",
                ""
            ]
        
        perf = results.get('performance', {})
        speedup = perf.get('vectorized_speedup', 0)
        
        return [
            "### 🔥 BM25 SIMD Performance Tests",
            "",
            f"✅ **Status:** {'PASS' if results.get('all_tests_correct') and results.get('target_speedup_achieved') else 'PARTIAL'}",
            f"**Vectorized Speedup:** {speedup:.2f}x (Target: 8.0x)",
            f"**Correctness:** {'All tests passed' if results.get('all_tests_correct') else 'Some tests failed'}",
            f"**Memory Usage:** {results.get('memory', {}).get('vectorized_memory_mb', 0):.1f} MB",
            ""
        ]
    
    def _format_quantization_results(self, results: Dict) -> List[str]:
        """Format quantization test results for markdown"""
        if 'error' in results:
            return [
                "### 🔢 Embedding Quantization Tests",
                "",
                f"❌ **Status:** Failed with error: {results['error']}",
                ""
            ]
        
        speedup = results.get('dot_product', {}).get('speedup', 0)
        correlation = results.get('quality', {}).get('symmetric_cosine_correlation', 0)
        
        return [
            "### 🔢 Embedding Quantization Tests",
            "",
            f"✅ **Status:** {'PASS' if results.get('speedup_achieved') and results.get('quality_preserved') else 'PARTIAL'}",
            f"**Dot Product Speedup:** {speedup:.2f}x (Target: 3.0x)",
            f"**Quality Preservation:** {correlation:.4f} cosine correlation",
            f"**Compression Ratio:** 4.0x (float32 → int8)",
            ""
        ]
    
    def _format_memory_mapping_results(self, results: Dict) -> List[str]:
        """Format memory mapping test results for markdown"""
        if 'error' in results:
            return [
                "### 💾 Memory Mapping Performance Tests",
                "",
                f"❌ **Status:** Failed with error: {results['error']}",
                ""
            ]
        
        random_speedup = results.get('random_access', {}).get('mmap_speedup', 0)
        memory_efficiency = results.get('memory', {}).get('memory_efficiency', 0)
        
        return [
            "### 💾 Memory Mapping Performance Tests",
            "",
            f"✅ **Status:** {'PASS' if results.get('random_speedup_achieved') and results.get('memory_efficient') else 'PARTIAL'}",
            f"**Random Access Speedup:** {random_speedup:.2f}x (Target: 2.0x)",
            f"**Memory Efficiency:** {memory_efficiency:.2f}x reduction",
            f"**Compression Ratio:** {results.get('creation', {}).get('compression_ratio', 1.0):.2f}x",
            ""
        ]
    
    def _format_topk_results(self, results: Dict) -> List[str]:
        """Format top-k test results for markdown"""
        if 'error' in results:
            return [
                "### 🔝 Top-K Selection Performance Tests",
                "",
                f"❌ **Status:** Failed with error: {results['error']}",
                ""
            ]
        
        avg_speedup = results.get('avg_speedup', 0)
        success_rate = results.get('speedup_success_rate', 0)
        
        return [
            "### 🔝 Top-K Selection Performance Tests",
            "",
            f"✅ **Status:** {'PASS' if results.get('all_correct') and results.get('target_speedup_achieved') else 'PARTIAL'}",
            f"**Average Speedup:** {avg_speedup:.2f}x",
            f"**Success Rate (5x+ speedup):** {success_rate:.1%}",
            f"**Correctness:** {'All algorithms correct' if results.get('all_correct') else 'Some algorithms failed'}",
            ""
        ]
    
    def _generate_performance_plots(self):
        """Generate performance comparison plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Lightning Retrieval Performance Results', fontsize=16)
            
            results = self.test_summary['test_results']
            
            # BM25 Speedup Chart
            if 'bm25' in results and 'performance' in results['bm25']:
                perf = results['bm25']['performance']
                speedups = [
                    perf.get('vectorized_speedup', 0),
                    perf.get('numba_speedup', 0),
                    perf.get('vs_baseline_vectorized', 0)
                ]
                labels = ['Vectorized vs\nReference', 'Numba vs\nReference', 'Vectorized vs\nBaseline']
                
                axes[0, 0].bar(labels, speedups, color=['#2E86AB', '#A23B72', '#F18F01'])
                axes[0, 0].axhline(y=8, color='red', linestyle='--', label='Target (8x)')
                axes[0, 0].set_title('BM25 Performance Speedups')
                axes[0, 0].set_ylabel('Speedup Factor')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Quantization Quality vs Performance
            if 'quantization' in results:
                quant = results['quantization']
                quality = quant.get('quality', {})
                dot_product = quant.get('dot_product', {})
                
                correlation = quality.get('symmetric_cosine_correlation', 0)
                speedup = dot_product.get('speedup', 0)
                
                axes[0, 1].scatter([speedup], [correlation], s=200, color='#2E86AB', alpha=0.7)
                axes[0, 1].axhline(y=0.95, color='red', linestyle='--', label='Quality Threshold')
                axes[0, 1].axvline(x=3.0, color='red', linestyle='--', label='Speed Target')
                axes[0, 1].set_xlabel('Speedup Factor')
                axes[0, 1].set_ylabel('Cosine Correlation')
                axes[0, 1].set_title('Quantization: Quality vs Performance')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_xlim(0, max(5, speedup + 1))
                axes[0, 1].set_ylim(0.9, 1.0)
            
            # Memory Mapping Performance
            if 'memory_mapping' in results:
                mmap = results['memory_mapping']
                random_access = mmap.get('random_access', {})
                
                methods = ['Standard', 'Lazy JSONL', 'Memory Mapped']
                latencies = [
                    random_access.get('standard_avg_latency_ms', 0),
                    random_access.get('lazy_jsonl_avg_latency_ms', 0),
                    random_access.get('memory_mapped_avg_latency_ms', 0)
                ]
                
                bars = axes[1, 0].bar(methods, latencies, color=['#A23B72', '#F18F01', '#2E86AB'])
                axes[1, 0].set_title('Memory Mapping: Random Access Latency')
                axes[1, 0].set_ylabel('Latency (ms)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add values on bars
                for bar, latency in zip(bars, latencies):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{latency:.2f}ms', ha='center', va='bottom')
            
            # Top-K Performance Comparison
            if 'topk' in results and 'recommendations' in results['topk']:
                recommendations = results['topk']['recommendations']
                
                configs = list(recommendations.keys())[:5]  # Show top 5 configs
                speedups = [recommendations[config]['speedup'] for config in configs]
                
                bars = axes[1, 1].bar(range(len(configs)), speedups, color='#2E86AB')
                axes[1, 1].axhline(y=5, color='red', linestyle='--', label='Target (5x)')
                axes[1, 1].set_title('Top-K Selection Speedups')
                axes[1, 1].set_ylabel('Speedup Factor')
                axes[1, 1].set_xticks(range(len(configs)))
                axes[1, 1].set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / "performance_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance plots saved: {plot_path}")
            
        except ImportError:
            print("⚠️  matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        print(f"\n{'=' * 60}")
        print("🎯 RECOMMENDATIONS")
        print(f"{'=' * 60}")
        
        assessment = self.test_summary['overall_assessment']
        results = self.test_summary['test_results']
        
        print(f"\n**Overall Grade: {assessment['grade']}**")
        print(f"**Overall Score: {assessment['overall_score']:.2f}/1.00**")
        
        recommendations = []
        
        # BM25 Recommendations
        if 'bm25' in results and 'error' not in results['bm25']:
            bm25 = results['bm25']
            if not bm25.get('target_speedup_achieved', False):
                recommendations.append("🔥 BM25: Consider implementing true SIMD instructions (AVX2/AVX512) for better vectorization")
            if not bm25.get('all_tests_correct', True):
                recommendations.append("🔥 BM25: Fix correctness issues in optimized implementations")
        
        # Quantization Recommendations
        if 'quantization' in results and 'error' not in results['quantization']:
            quant = results['quantization']
            if not quant.get('speedup_achieved', False):
                recommendations.append("🔢 Quantization: Implement hardware-accelerated int8 operations for better performance")
            if not quant.get('quality_preserved', True):
                recommendations.append("🔢 Quantization: Tune quantization parameters to preserve embedding quality")
        
        # Memory Mapping Recommendations
        if 'memory_mapping' in results and 'error' not in results['memory_mapping']:
            mmap = results['memory_mapping']
            if not mmap.get('random_speedup_achieved', False):
                recommendations.append("💾 Memory Mapping: Optimize file format layout and add prefetching")
            if not mmap.get('memory_efficient', True):
                recommendations.append("💾 Memory Mapping: Investigate memory usage patterns and add lazy loading")
        
        # Top-K Recommendations
        if 'topk' in results and 'error' not in results['topk']:
            topk = results['topk']
            if not topk.get('target_speedup_achieved', False):
                recommendations.append("🔝 Top-K: Implement specialized algorithms for different k ranges")
            if not topk.get('all_correct', True):
                recommendations.append("🔝 Top-K: Fix correctness issues in optimized selection algorithms")
        
        # General Recommendations
        if assessment['overall_score'] < 0.8:
            recommendations.append("⚡ General: Focus on the lowest-scoring components first for maximum impact")
        
        if not self.profiler.cpu_info.get('avx2', False):
            recommendations.append("🖥️  Hardware: Consider testing on AVX2-capable hardware for full SIMD benefits")
        
        if recommendations:
            print(f"\n📋 Specific Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print(f"\n🎉 Excellent! All optimizations are working as expected.")
            print(f"   The lightning retrieval system is ready for production deployment.")
        
        # Save recommendations
        rec_path = self.output_dir / "recommendations.txt"
        with open(rec_path, 'w') as f:
            f.write(f"Lightning Retrieval Test Recommendations\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Overall Grade: {assessment['grade']}\n")
            f.write(f"Overall Score: {assessment['overall_score']:.2f}/1.00\n\n")
            
            if recommendations:
                f.write("Recommendations:\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("No recommendations needed - all tests passed!\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Lightning Retrieval Integration Test Runner")
    parser.add_argument("--small-corpus", type=int, default=1000, help="Small corpus size")
    parser.add_argument("--medium-corpus", type=int, default=10000, help="Medium corpus size")
    parser.add_argument("--large-corpus", type=int, default=50000, help="Large corpus size")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--enable-large-tests", action="store_true", help="Enable large-scale tests")
    parser.add_argument("--output-dir", default="test_results", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestConfiguration(
        small_corpus_size=args.small_corpus,
        medium_corpus_size=args.medium_corpus,
        large_corpus_size=args.large_corpus,
        embedding_dim=args.embedding_dim,
        num_queries=args.num_queries,
        enable_large_tests=args.enable_large_tests,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots
    )
    
    # Run tests
    runner = IntegrationTestRunner(config)
    
    try:
        runner.run_all_tests()
        return 0
    except KeyboardInterrupt:
        print(f"\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())