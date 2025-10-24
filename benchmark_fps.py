#!/usr/bin/env python3
"""
Benchmark Farthest Point Sampling implementations
Run this to verify the speedup from optimization
"""

import numpy as np
import time
import sys
import os

# Add Point-BERT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/Point-BERT'))

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Original slow implementation
def fps_original(point, npoint):
    """Original implementation (slow)"""
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@jit(nopython=True)
def fps_numba(points, npoint):
    """Numba-optimized implementation"""
    N = points.shape[0]
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :3]
        dist = np.sum((points[:, :3] - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return centroids


def fps_vectorized(point, npoint):
    """Vectorized NumPy implementation"""
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.full(N, 1e10, dtype=np.float32)
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest:farthest+1, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    
    return point[centroids]


def benchmark_fps(n_points=8192, n_sample=1024, n_trials=10):
    """Benchmark different FPS implementations"""
    
    print("="*70)
    print(f"Farthest Point Sampling Benchmark")
    print(f"Input points: {n_points}, Sample points: {n_sample}, Trials: {n_trials}")
    print("="*70)
    print()
    
    # Generate random point cloud
    np.random.seed(42)
    points = np.random.randn(n_points, 3).astype(np.float32)
    
    results = {}
    
    # Benchmark original
    print("Testing ORIGINAL implementation...")
    times = []
    for i in range(n_trials):
        start = time.time()
        sampled = fps_original(points.copy(), n_sample)
        elapsed = time.time() - start
        times.append(elapsed)
        if i == 0:
            print(f"  First run: {elapsed:.4f}s")
    results['original'] = times
    print(f"  Average: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
    print()
    
    # Benchmark vectorized
    print("Testing VECTORIZED NumPy implementation...")
    times = []
    for i in range(n_trials):
        start = time.time()
        sampled = fps_vectorized(points.copy(), n_sample)
        elapsed = time.time() - start
        times.append(elapsed)
        if i == 0:
            print(f"  First run: {elapsed:.4f}s")
    results['vectorized'] = times
    print(f"  Average: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
    speedup = np.mean(results['original']) / np.mean(times)
    print(f"  Speedup vs original: {speedup:.1f}x")
    print()
    
    # Benchmark Numba
    if NUMBA_AVAILABLE:
        print("Testing NUMBA JIT implementation...")
        times = []
        # Warm up Numba (compilation happens on first call)
        print("  Warming up Numba (first call compiles)...")
        start = time.time()
        _ = fps_numba(points, n_sample)
        compile_time = time.time() - start
        print(f"  Compilation time: {compile_time:.4f}s")
        
        for i in range(n_trials):
            start = time.time()
            indices = fps_numba(points, n_sample)
            sampled = points[indices]
            elapsed = time.time() - start
            times.append(elapsed)
            if i == 0:
                print(f"  First run (after compilation): {elapsed:.4f}s")
        results['numba'] = times
        print(f"  Average: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
        speedup = np.mean(results['original']) / np.mean(times)
        print(f"  Speedup vs original: {speedup:.1f}x")
        print()
    else:
        print("NUMBA not available (install with: pip install numba)")
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Time (s)':<15} {'Speedup':<10}")
    print("-"*70)
    
    original_time = np.mean(results['original'])
    print(f"{'Original':<20} {original_time:<15.4f} {'1.0x':<10}")
    
    vec_time = np.mean(results['vectorized'])
    vec_speedup = original_time / vec_time
    print(f"{'Vectorized NumPy':<20} {vec_time:<15.4f} {f'{vec_speedup:.1f}x':<10}")
    
    if NUMBA_AVAILABLE:
        numba_time = np.mean(results['numba'])
        numba_speedup = original_time / numba_time
        print(f"{'Numba JIT':<20} {numba_time:<15.4f} {f'{numba_speedup:.1f}x':<10}")
    
    print()
    print("Recommendation: Use Numba JIT for best performance!" if NUMBA_AVAILABLE else 
          "Recommendation: Install Numba for 10-100x speedup!")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark FPS implementations')
    parser.add_argument('--n_points', type=int, default=8192, 
                        help='Number of input points (default: 8192)')
    parser.add_argument('--n_sample', type=int, default=1024,
                        help='Number of points to sample (default: 1024)')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trials (default: 10)')
    
    args = parser.parse_args()
    
    benchmark_fps(args.n_points, args.n_sample, args.n_trials)
