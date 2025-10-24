"""
Preprocessing script for dental IOS dataset
Processes STL files from source datasets and saves preprocessed point clouds

Usage:
    python scripts/preprocess_dataset.py --source_dir <path> --output_dir <path> --num_samples <N>
    
This script will:
1. Load STL files from source directory
2. Normalize to unit sphere
3. Flip z-axis if upper arch
4. Apply Farthest Point Sampling (FPS) multiple times with different random seeds
5. Save preprocessed point clouds to disk
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
import json
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Install with: pip install numba")
    print("Continuing with standard NumPy (slower)...")


def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def load_stl(file_path):
    """Load STL file and return point cloud"""
    mesh = o3d.io.read_triangle_mesh(file_path)
    # Sample points from the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=65536*2)
    ptcloud = np.array(pcd.points, dtype=np.float32)
    return ptcloud


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=False, fastmath=True)
    def farthest_point_sample_core(point, npoint, initial_idx):
        """
        Numba-optimized FPS core algorithm
        """
        N = point.shape[0]
        centroids = np.zeros(npoint, dtype=np.int32)
        distance = np.ones(N, dtype=np.float32) * 1e10
        farthest = initial_idx
        
        for i in range(npoint):
            centroids[i] = farthest
            centroid = point[farthest, :]
            
            # Calculate distances
            for j in range(N):
                dist = 0.0
                for k in range(3):
                    diff = point[j, k] - centroid[k]
                    dist += diff * diff
                
                if dist < distance[j]:
                    distance[j] = dist
            
            # Find farthest point
            farthest = 0
            max_dist = distance[0]
            for j in range(1, N):
                if distance[j] > max_dist:
                    max_dist = distance[j]
                    farthest = j
        
        return centroids


def farthest_point_sample_numpy(point, npoint):
    """
    Farthest Point Sampling using NumPy or Numba (CPU only, multiprocessing-safe)
    Args:
        point: numpy array of shape (N, 3)
        npoint: number of points to sample
    Returns:
        sampled points as numpy array of shape (npoint, 3)
    """
    N = point.shape[0]
    initial_idx = np.random.randint(0, N)
    
    if NUMBA_AVAILABLE:
        # Use Numba-optimized version
        centroids = farthest_point_sample_core(point, npoint, initial_idx)
    else:
        # Fallback to pure NumPy
        centroids = np.zeros(npoint, dtype=np.int32)
        distance = np.ones(N, dtype=np.float32) * 1e10
        farthest = initial_idx
        
        for i in range(npoint):
            centroids[i] = farthest
            centroid = point[farthest, :]
            dist = np.sum((point - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
    
    return point[centroids].astype(np.float32)


def process_stl_file(stl_path, arch_type, num_samples, npoints=65536):
    """
    Process a single STL file
    Args:
        stl_path: path to STL file
        arch_type: 'upper' or 'lower'
        num_samples: number of different FPS samples to generate
        npoints: number of points to sample via FPS
    Returns:
        List of processed point clouds
    """
    # Load STL
    point_cloud = load_stl(stl_path)
    
    # Normalize to unit sphere
    point_cloud = pc_normalize(point_cloud)
    
    # Flip z if upper arch
    if arch_type == 'upper':
        point_cloud[:, 2] = -point_cloud[:, 2]
    
    # Generate multiple FPS samples with different random starting points
    samples = []
    for i in range(num_samples):
        # Use different random seed for each sample
        np.random.seed(None)  # Reseed for randomness
        sampled_pc = farthest_point_sample_numpy(point_cloud, npoints)
        samples.append(sampled_pc)
    
    return samples


def save_point_cloud(point_cloud, output_path):
    """Save point cloud as .ply file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Save as PLY
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)


def process_single_stl_task(task_info):
    """
    Process a single STL file task (wrapper for multiprocessing)
    Args:
        task_info: dict with keys: stl_file, arch_type, output_dir, unique_id, num_samples, npoints
    Returns:
        dict with processing results
    """
    stl_file = task_info['stl_file']
    arch_type = task_info['arch_type']
    output_dir = task_info['output_dir']
    unique_id = task_info['unique_id']
    num_samples = task_info['num_samples']
    npoints = task_info['npoints']
    
    try:
        # Process STL file
        samples = process_stl_file(stl_file, arch_type, num_samples, npoints)
        
        # Save each sample with unified naming: {unique_id}_{arch}_{sample_idx}.ply
        for sample_idx, sample_pc in enumerate(samples):
            os.makedirs(output_dir, exist_ok=True)
            
            # Unified naming format: {unique_id}_{arch}_{sample_idx}.ply
            output_file = os.path.join(output_dir, f'{unique_id}_{arch_type}_{sample_idx:03d}.ply')
            save_point_cloud(sample_pc, output_file)
        
        return {
            'success': True,
            'unique_id': unique_id,
            'arch': arch_type,
            'num_samples': num_samples,
            'source': stl_file
        }
    except Exception as e:
        return {
            'success': False,
            'unique_id': unique_id,
            'arch': arch_type,
            'source': stl_file,
            'error': str(e)
        }



def process_3dteethland_dataset(source_dir, output_dir, num_samples, npoints=65536, n_workers=None):
    """
    Process 3DTeethLand dataset with multiprocessing
    Structure: Patient_XXX/stl/ios_{upper|lower}_processed.stl
    Output: Unified flat structure with {unique_id}_{arch}_{sample_idx}.ply
    """
    print(f"Processing 3DTeethLand dataset from {source_dir}")
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"Using {n_workers} worker processes")
    
    patient_dirs = sorted([d for d in os.listdir(source_dir) 
                          if os.path.isdir(os.path.join(source_dir, d)) 
                          and d.startswith('Patient')])
    
    # Build list of tasks with unique IDs
    tasks = []
    for patient_id in patient_dirs:
        patient_path = os.path.join(source_dir, patient_id)
        stl_dir = os.path.join(patient_path, 'stl')
        
        if not os.path.exists(stl_dir):
            continue
        
        # Extract patient number as unique ID (e.g., Patient 01328DDN_3747 -> 3DTeeth_3747)
        # Use the patient_id as the unique identifier with "3DTeeth_" prefix
        unique_id = f"3DTeeth_{patient_id.replace('Patient ', '').replace(' ', '_')}"
        
        # Process upper and lower arches
        for arch_type in ['upper', 'lower']:
            stl_file = os.path.join(stl_dir, f'ios_{arch_type}_processed.stl')
            
            if not os.path.exists(stl_file):
                continue
            
            tasks.append({
                'stl_file': stl_file,
                'arch_type': arch_type,
                'output_dir': output_dir,
                'unique_id': unique_id,
                'num_samples': num_samples,
                'npoints': npoints
            })
    
    print(f"Found {len(tasks)} STL files to process")
    
    # Process tasks in parallel
    stats = {
        'total_stl_files': len(tasks),
        'total_samples_generated': 0,
        'scans': {}
    }
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_stl_task, tasks),
            total=len(tasks),
            desc="Processing STL files"
        ))
    
    # Collect statistics
    for result in results:
        if result['success']:
            unique_id = result['unique_id']
            if unique_id not in stats['scans']:
                stats['scans'][unique_id] = {'scan_id': unique_id, 'files': []}
            
            stats['scans'][unique_id]['files'].append({
                'arch': result['arch'],
                'num_samples': result['num_samples'],
                'source': result['source']
            })
            stats['total_samples_generated'] += result['num_samples']
        else:
            print(f"\nError processing {result['source']}: {result['error']}")
    
    # Convert scans dict to list
    stats['scans'] = list(stats['scans'].values())
    
    return stats


def process_ferraradump_dataset(source_dir, output_dir, num_samples, npoints=65536, n_workers=None):
    """
    Process FerraraDump dataset with multiprocessing
    Structure: sample_id/ios/{upper|lower}.stl
    Output: Unified flat structure with {unique_id}_{arch}_{sample_idx}.ply
    """
    print(f"Processing FerraraDump dataset from {source_dir}")
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"Using {n_workers} worker processes")
    
    sample_dirs = sorted([d for d in os.listdir(source_dir) 
                         if os.path.isdir(os.path.join(source_dir, d))])
    
    # Build list of tasks with unique IDs
    tasks = []
    for sample_id in sample_dirs:
        sample_path = os.path.join(source_dir, sample_id)
        ios_dir = os.path.join(sample_path, 'ios')
        
        if not os.path.exists(ios_dir):
            continue
        
        # Use sample_id directly as unique ID with "Ferrara_" prefix
        unique_id = f"Ferrara_{sample_id}"
        
        # Process upper and lower arches
        for arch_type in ['upper', 'lower']:
            stl_file = os.path.join(ios_dir, f'{arch_type}.stl')
            
            if not os.path.exists(stl_file):
                continue
            
            tasks.append({
                'stl_file': stl_file,
                'arch_type': arch_type,
                'output_dir': output_dir,
                'unique_id': unique_id,
                'num_samples': num_samples,
                'npoints': npoints
            })
    
    print(f"Found {len(tasks)} STL files to process")
    
    # Process tasks in parallel
    stats = {
        'total_stl_files': len(tasks),
        'total_samples_generated': 0,
        'scans': {}
    }
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_stl_task, tasks),
            total=len(tasks),
            desc="Processing STL files"
        ))
    
    # Collect statistics
    for result in results:
        if result['success']:
            unique_id = result['unique_id']
            if unique_id not in stats['scans']:
                stats['scans'][unique_id] = {'scan_id': unique_id, 'files': []}
            
            stats['scans'][unique_id]['files'].append({
                'arch': result['arch'],
                'num_samples': result['num_samples'],
                'source': result['source']
            })
            stats['total_samples_generated'] += result['num_samples']
        else:
            print(f"\nError processing {result['source']}: {result['error']}")
    
    # Convert scans dict to list
    stats['scans'] = list(stats['scans'].values())
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess dental IOS dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Path to source dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory for preprocessed data')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of different FPS samples to generate per STL file')
    parser.add_argument('--npoints', type=int, default=65536,
                       help='Number of points to sample via FPS')
    parser.add_argument('--dataset_type', type=str, choices=['3dteethland', 'ferraradump'], 
                       default='ferraradump',
                       help='Type of dataset structure')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    n_workers = args.n_workers if args.n_workers else max(1, cpu_count() - 1)
    
    print("="*80)
    print("Dataset Preprocessing")
    print("="*80)
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Samples per STL: {args.num_samples}")
    print(f"Points per sample: {args.npoints}")
    print(f"Worker processes: {n_workers}")
    print("="*80)
    
    # Process dataset based on type
    if args.dataset_type == '3dteethland':
        stats = process_3dteethland_dataset(
            args.source_dir, 
            args.output_dir, 
            args.num_samples,
            args.npoints,
            n_workers
        )
    else:  # ferraradump
        stats = process_ferraradump_dataset(
            args.source_dir, 
            args.output_dir, 
            args.num_samples,
            args.npoints,
            n_workers
        )
    
    # Save statistics
    stats_file = os.path.join(args.output_dir, 'preprocessing_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("Preprocessing Complete!")
    print("="*80)
    print(f"Total STL files processed: {stats['total_stl_files']}")
    print(f"Total samples generated: {stats['total_samples_generated']}")
    print(f"Statistics saved to: {stats_file}")
    print("="*80)


if __name__ == '__main__':
    main()
