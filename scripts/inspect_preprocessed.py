"""
Utility script to inspect preprocessed dataset
Shows statistics and samples from preprocessed .npy files
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from collections import defaultdict


def load_preprocessing_stats(output_dir):
    """Load preprocessing statistics JSON file"""
    stats_file = os.path.join(output_dir, 'preprocessing_stats.json')
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return None


def count_files(output_dir):
    """Count number of .npy files in output directory"""
    npy_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def analyze_point_cloud(file_path):
    """Analyze a single point cloud file"""
    pc = np.load(file_path)
    
    analysis = {
        'shape': pc.shape,
        'dtype': str(pc.dtype),
        'min': float(pc.min()),
        'max': float(pc.max()),
        'mean': pc.mean(axis=0).tolist(),
        'std': float(pc.std()),
        'centroid': pc.mean(axis=0).tolist(),
        'max_dist_from_origin': float(np.max(np.sqrt(np.sum(pc**2, axis=1)))),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Inspect preprocessed dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to preprocessed dataset directory')
    parser.add_argument('--sample_files', type=int, default=5,
                       help='Number of sample files to analyze in detail')
    parser.add_argument('--show_tree', action='store_true',
                       help='Show directory tree structure')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory {args.output_dir} does not exist")
        sys.exit(1)
    
    print("="*80)
    print("PREPROCESSED DATASET INSPECTION")
    print("="*80)
    print(f"Directory: {args.output_dir}")
    print()
    
    # Load preprocessing stats if available
    stats = load_preprocessing_stats(args.output_dir)
    if stats:
        print("Preprocessing Statistics:")
        print(f"  Total STL files processed: {stats['total_stl_files']}")
        print(f"  Total samples generated: {stats['total_samples_generated']}")
        if 'patients' in stats:
            print(f"  Total patients: {len(stats['patients'])}")
        elif 'samples' in stats:
            print(f"  Total samples: {len(stats['samples'])}")
        print()
    
    # Count files
    print("Counting .npy files...")
    npy_files = count_files(args.output_dir)
    print(f"  Found {len(npy_files)} .npy files")
    print()
    
    # Analyze file distribution
    print("File Distribution:")
    by_arch = defaultdict(int)
    by_sample_idx = defaultdict(int)
    total_size = 0
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        if filename.startswith('upper'):
            by_arch['upper'] += 1
        elif filename.startswith('lower'):
            by_arch['lower'] += 1
        
        # Extract sample index
        if '_sample_' in filename:
            idx = filename.split('_sample_')[1].split('.')[0]
            by_sample_idx[idx] += 1
        
        total_size += os.path.getsize(file_path)
    
    print(f"  Upper arch samples: {by_arch['upper']}")
    print(f"  Lower arch samples: {by_arch['lower']}")
    print(f"  Total disk usage: {total_size / (1024**3):.2f} GB")
    print(f"  Average file size: {total_size / len(npy_files) / (1024**2):.2f} MB")
    print()
    
    if by_sample_idx:
        print("Sample indices distribution:")
        for idx in sorted(by_sample_idx.keys()):
            print(f"  Sample {idx}: {by_sample_idx[idx]} files")
        print()
    
    # Directory structure
    if args.show_tree:
        print("Directory Structure:")
        sample_dirs = sorted([d for d in os.listdir(args.output_dir) 
                            if os.path.isdir(os.path.join(args.output_dir, d))])
        for i, sample_dir in enumerate(sample_dirs[:10]):  # Show first 10
            ios_dir = os.path.join(args.output_dir, sample_dir, 'ios')
            if os.path.exists(ios_dir):
                files = sorted([f for f in os.listdir(ios_dir) if f.endswith('.npy')])
                print(f"  {sample_dir}/")
                print(f"    ios/")
                for f in files[:3]:  # Show first 3 files
                    print(f"      {f}")
                if len(files) > 3:
                    print(f"      ... ({len(files) - 3} more files)")
        if len(sample_dirs) > 10:
            print(f"  ... ({len(sample_dirs) - 10} more directories)")
        print()
    
    # Detailed analysis of sample files
    print(f"Detailed Analysis of {args.sample_files} Sample Files:")
    print("-"*80)
    
    sample_files = np.random.choice(npy_files, min(args.sample_files, len(npy_files)), replace=False)
    
    all_analyses = []
    for i, file_path in enumerate(sample_files):
        print(f"\nFile {i+1}: {os.path.relpath(file_path, args.output_dir)}")
        analysis = analyze_point_cloud(file_path)
        all_analyses.append(analysis)
        
        print(f"  Shape: {analysis['shape']}")
        print(f"  Data type: {analysis['dtype']}")
        print(f"  Value range: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
        print(f"  Centroid: [{analysis['centroid'][0]:.4f}, {analysis['centroid'][1]:.4f}, {analysis['centroid'][2]:.4f}]")
        print(f"  Std deviation: {analysis['std']:.4f}")
        print(f"  Max distance from origin: {analysis['max_dist_from_origin']:.4f}")
        print(f"  File size: {analysis['file_size_mb']:.2f} MB")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    shapes = [a['shape'] for a in all_analyses]
    max_dists = [a['max_dist_from_origin'] for a in all_analyses]
    stds = [a['std'] for a in all_analyses]
    file_sizes = [a['file_size_mb'] for a in all_analyses]
    
    print(f"Point cloud shapes: {set(shapes)}")
    print(f"Max distance from origin: mean={np.mean(max_dists):.4f}, std={np.std(max_dists):.4f}")
    print(f"Point cloud std dev: mean={np.mean(stds):.4f}, std={np.std(stds):.4f}")
    print(f"File sizes: mean={np.mean(file_sizes):.2f} MB, std={np.std(file_sizes):.2f} MB")
    
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    # Validation checks
    checks_passed = 0
    checks_total = 0
    
    # Check 1: All files have same shape
    checks_total += 1
    if len(set(shapes)) == 1:
        print("✓ All point clouds have consistent shape")
        checks_passed += 1
    else:
        print("✗ WARNING: Point clouds have different shapes!")
        print(f"  Found shapes: {set(shapes)}")
    
    # Check 2: Point clouds are normalized (max distance ~1.0)
    checks_total += 1
    mean_max_dist = np.mean(max_dists)
    if 0.8 < mean_max_dist < 1.2:
        print(f"✓ Point clouds appear normalized (mean max distance: {mean_max_dist:.4f})")
        checks_passed += 1
    else:
        print(f"✗ WARNING: Point clouds may not be properly normalized!")
        print(f"  Mean max distance from origin: {mean_max_dist:.4f}")
    
    # Check 3: Centroids are near origin
    checks_total += 1
    centroids = np.array([a['centroid'] for a in all_analyses])
    mean_centroid_dist = np.mean(np.sqrt(np.sum(centroids**2, axis=1)))
    if mean_centroid_dist < 0.1:
        print(f"✓ Point clouds are centered (mean centroid distance: {mean_centroid_dist:.4f})")
        checks_passed += 1
    else:
        print(f"⚠ Point clouds may not be perfectly centered")
        print(f"  Mean centroid distance from origin: {mean_centroid_dist:.4f}")
    
    # Check 4: File sizes are reasonable
    checks_total += 1
    mean_size = np.mean(file_sizes)
    if 0.1 < mean_size < 10:
        print(f"✓ File sizes are reasonable (mean: {mean_size:.2f} MB)")
        checks_passed += 1
    else:
        print(f"⚠ File sizes seem unusual (mean: {mean_size:.2f} MB)")
    
    print()
    print(f"Validation: {checks_passed}/{checks_total} checks passed")
    
    if checks_passed == checks_total:
        print("\n✓ Dataset looks good! Ready for training.")
    else:
        print("\n⚠ Some validation checks failed. Please review the warnings above.")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
