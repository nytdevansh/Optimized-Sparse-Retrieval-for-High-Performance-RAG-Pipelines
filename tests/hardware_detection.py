"""
Hardware Detection Module for Lightning Retrieval Test Suite.
Detects and validates system capabilities including SIMD support.
"""

import platform
import sys
import cpuinfo
import psutil
import numpy as np

class HardwareCapabilities:
    def __init__(self):
        self.cpu_info = cpuinfo.get_cpu_info()
        self.memory_info = psutil.virtual_memory()
        self.platform_info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': sys.version,
        }
        
    def get_simd_support(self):
        """Detect available SIMD instruction sets."""
        flags = self.cpu_info.get('flags', [])
        return {
            'avx2': 'avx2' in flags,
            'avx512': any('avx512' in flag for flag in flags),
            'neon': platform.machine().startswith('aarch64'),
            'sse2': 'sse2' in flags
        }
    
    def validate_numpy_simd(self):
        """Validate that NumPy is using SIMD optimizations."""
        # Create large arrays to force SIMD usage
        size = 10000000
        a = np.random.rand(size)
        b = np.random.rand(size)
        
        # Time the operation to detect if SIMD is being used
        import time
        start = time.perf_counter()
        _ = a * b
        duration = time.perf_counter() - start
        
        # Typical SIMD operations should be significantly faster
        return {
            'operation_time': duration,
            'size': size,
            'simd_likely': duration < 0.1  # Threshold based on typical SIMD performance
        }
    
    def get_memory_stats(self):
        """Get detailed memory statistics."""
        return {
            'total': self.memory_info.total,
            'available': self.memory_info.available,
            'percent_used': self.memory_info.percent,
            'page_size': psutil.PAGESIZE
        }
    
    def get_cpu_stats(self):
        """Get detailed CPU statistics."""
        return {
            'architecture': self.cpu_info['arch'],
            'model': self.cpu_info['brand_raw'],
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
    
    def run_capability_checks(self):
        """Run all capability checks and return comprehensive results."""
        return {
            'simd_support': self.get_simd_support(),
            'numpy_simd': self.validate_numpy_simd(),
            'memory_stats': self.get_memory_stats(),
            'cpu_stats': self.get_cpu_stats(),
            'platform': self.platform_info
        }
    
    def get_optimization_recommendations(self):
        """Generate optimization recommendations based on hardware capabilities."""
        recommendations = []
        simd = self.get_simd_support()
        
        if simd['avx512']:
            recommendations.append({
                'feature': 'AVX-512',
                'priority': 'high',
                'description': 'Use AVX-512 for maximum SIMD performance'
            })
        elif simd['avx2']:
            recommendations.append({
                'feature': 'AVX2',
                'priority': 'high',
                'description': 'Use AVX2 for improved SIMD performance'
            })
        elif simd['neon']:
            recommendations.append({
                'feature': 'NEON',
                'priority': 'high',
                'description': 'Use NEON SIMD instructions for ARM architecture'
            })
            
        mem_stats = self.get_memory_stats()
        if mem_stats['available'] > 32 * (1024 ** 3):  # 32GB
            recommendations.append({
                'feature': 'Large Memory',
                'priority': 'medium',
                'description': 'Consider memory mapping for large datasets'
            })
            
        return recommendations

def main():
    """Run hardware detection and print results."""
    hw = HardwareCapabilities()
    results = hw.run_capability_checks()
    recommendations = hw.get_optimization_recommendations()
    
    print("\n=== Hardware Capability Report ===")
    print("\nSIMD Support:")
    for inst, supported in results['simd_support'].items():
        print(f"  {inst.upper()}: {'✓' if supported else '✗'}")
    
    print("\nCPU Information:")
    cpu_stats = results['cpu_stats']
    print(f"  Model: {cpu_stats['model']}")
    print(f"  Physical cores: {cpu_stats['cores_physical']}")
    print(f"  Logical cores: {cpu_stats['cores_logical']}")
    
    print("\nMemory Information:")
    mem_stats = results['memory_stats']
    print(f"  Total: {mem_stats['total'] / (1024**3):.1f} GB")
    print(f"  Available: {mem_stats['available'] / (1024**3):.1f} GB")
    print(f"  Used: {mem_stats['percent_used']}%")
    
    print("\nOptimization Recommendations:")
    for rec in recommendations:
        print(f"\n  {rec['feature']} (Priority: {rec['priority']})")
        print(f"  → {rec['description']}")

if __name__ == '__main__':
    main()