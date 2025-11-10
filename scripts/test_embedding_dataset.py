#!/usr/bin/env python3
"""
Test script for the IOSEmbeddingDataset.

This script tests that the dataset can:
1. Find valid samples with all required files
2. Load a single sample correctly
3. Return the expected data structure
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "Qwen3-IOS"))

from data.embedding_dataset import IOSEmbeddingDataset


def test_dataset_initialization():
    """Test that dataset initializes and finds samples."""
    print("=" * 80)
    print("Test 1: Dataset Initialization")
    print("=" * 80)
    
    dataset = IOSEmbeddingDataset(
        point_cloud_dir="/work/grana_maxillo/IOS-DraftReport/_data/ios-iop-text",
        captions_dir="/work/grana_maxillo/IOS-DraftReport/src/auto-captioning/real_captions",
        embeddings_dir="/work/grana_maxillo/IOS-DraftReport/_data/iop_qwen_embeddings",
        num_points=32768
        normalize=True,
        augment=False,
        require_all_views=True
    )
    
    print(f"✓ Dataset initialized successfully")
    print(f"  Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("  ⚠ Warning: No samples found!")
        return None
    
    return dataset


def test_single_sample(dataset):
    """Test loading a single sample."""
    if dataset is None or len(dataset) == 0:
        print("  ✗ Cannot test sample loading (no samples available)")
        return
    
    print("\n" + "=" * 80)
    print("Test 2: Load Single Sample")
    print("=" * 80)
    
    try:
        sample = dataset[0]
        
        print(f"✓ Sample loaded successfully")
        print(f"  Patient ID: {sample['patient_id']}")
        print(f"  Point cloud shape: {sample['point_cloud'].shape}")
        print(f"  Description length: {len(sample['description'])} chars")
        print(f"  Available embeddings: {list(sample['embeddings'].keys())}")
        
        # Check embedding shapes
        for view_name, embedding in sample['embeddings'].items():
            print(f"    - {view_name}: {embedding.shape}")
        
        # Show first 100 chars of description
        print(f"\n  Description preview:")
        print(f"    {sample['description'][:100]}...")
        
        return sample
        
    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_samples(dataset, n=3):
    """Test loading multiple samples."""
    if dataset is None or len(dataset) == 0:
        print("  ✗ Cannot test multiple samples (no samples available)")
        return
    
    print("\n" + "=" * 80)
    print(f"Test 3: Load Multiple Samples (n={n})")
    print("=" * 80)
    
    n = min(n, len(dataset))
    
    for i in range(n):
        try:
            sample = dataset[i]
            print(f"✓ Sample {i+1}/{n}: {sample['patient_id']}")
            print(f"    PC: {sample['point_cloud'].shape}, Embeddings: {len(sample['embeddings'])}")
        except Exception as e:
            print(f"  ✗ Error loading sample {i}: {e}")


def test_dataset_statistics(dataset):
    """Show some statistics about the dataset."""
    if dataset is None or len(dataset) == 0:
        print("  ✗ Cannot compute statistics (no samples available)")
        return
    
    print("\n" + "=" * 80)
    print("Test 4: Dataset Statistics")
    print("=" * 80)
    
    # Count samples per view
    view_counts = {view: 0 for view in ["center", "down", "left", "right", "up"]}
    
    for sample_info in dataset.samples[:min(10, len(dataset.samples))]:
        for view in sample_info['embedding_paths'].keys():
            view_counts[view] += 1
    
    print(f"  View availability (first 10 samples):")
    for view, count in view_counts.items():
        print(f"    {view}: {count}/10")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("IOSEmbeddingDataset Test Suite")
    print("=" * 80 + "\n")
    
    # Test 1: Initialization
    dataset = test_dataset_initialization()
    
    if dataset is None or len(dataset) == 0:
        print("\n⚠ Dataset is empty. Check that:")
        print("  1. STL files exist in ios-iop-text/<patient_id>/ios/")
        print("  2. Caption JSON files exist in auto-captioning/real_captions/")
        print("  3. Embedding .pt files exist in iop_qwen_embeddings/<patient_id>/")
        return
    
    # Test 2: Single sample
    sample = test_single_sample(dataset)
    
    # Test 3: Multiple samples
    test_multiple_samples(dataset, n=5)
    
    # Test 4: Statistics
    test_dataset_statistics(dataset)
    
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
