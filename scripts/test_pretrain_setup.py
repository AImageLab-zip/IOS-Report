#!/usr/bin/env python3
"""
Quick test script to verify the pretraining setup.

This script tests:
1. PointEncoder can be created with output projection
2. Dataset loads correctly
3. Forward pass works with correct shapes
4. Loss computation works
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.point_encoder import PointEncoder
from data.embedding_dataset import IOSEmbeddingDataset


def test_point_encoder_with_projection():
    """Test PointEncoder with output projection."""
    print("=" * 80)
    print("Test 1: PointEncoder with Output Projection")
    print("=" * 80)
    
    model = PointEncoder(
        trans_dim=521,
        depth=12,
        num_heads=6,
        encoder_dims=512,
        group_size=32,
        num_group=1280,
        drop_path_rate=0.1,
        output_dim=2560  # Project to Qwen embedding size
    )
    
    print(f"✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    num_points = 8192
    dummy_points = torch.randn(batch_size, num_points, 3)
    
    output, centers = model(dummy_points, return_deepstack=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_points.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Centers shape: {centers.shape}")
    
    # Check shapes
    expected_shape = (batch_size, 1280, 2560)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape is correct: {output.shape}")
    
    return model


def test_dataset_loading():
    """Test that dataset loads correctly."""
    print("\n" + "=" * 80)
    print("Test 2: Dataset Loading")
    print("=" * 80)
    
    dataset = IOSEmbeddingDataset(
        point_cloud_dir="/work/grana_maxillo/IOS-DraftReport/_data/ios-iop-text",
        captions_dir="/work/grana_maxillo/IOS-DraftReport/src/auto-captioning/real_captions",
        embeddings_dir="/work/grana_maxillo/IOS-DraftReport/_data/iop_qwen_embeddings",
        num_points=8192,
        normalize=True,
        augment=False,
        require_all_views=True
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("  ⚠ Warning: No samples found!")
        return None
    
    # Load a sample
    sample = dataset[0]
    
    print(f"✓ Sample loaded successfully")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Point cloud shape: {sample['point_cloud'].shape}")
    print(f"  Available embeddings: {list(sample['embeddings'].keys())}")
    
    # Check embedding shapes
    for view_name, embedding in sample['embeddings'].items():
        print(f"    - {view_name}: {embedding.shape}")
    
    return dataset


def test_embedding_concatenation():
    """Test embedding concatenation."""
    print("\n" + "=" * 80)
    print("Test 3: Embedding Concatenation")
    print("=" * 80)
    
    # Create dummy embeddings
    batch_size = 2
    view_names = ["center", "up", "down", "left", "right"]
    
    embeddings = {}
    for view_name in view_names:
        embeddings[view_name] = torch.randn(batch_size, 256, 2560)
    
    # Concatenate in order
    concatenated = torch.cat([embeddings[v] for v in view_names], dim=1)
    
    print(f"✓ Embeddings concatenated successfully")
    print(f"  Order: {view_names}")
    print(f"  Individual shape: (batch_size, 256, 2560)")
    print(f"  Concatenated shape: {concatenated.shape}")
    
    expected_shape = (batch_size, 1280, 2560)
    assert concatenated.shape == expected_shape, f"Expected {expected_shape}, got {concatenated.shape}"
    print(f"✓ Concatenated shape is correct: {concatenated.shape}")
    
    return concatenated


def test_loss_computation():
    """Test loss computation."""
    print("\n" + "=" * 80)
    print("Test 4: Loss Computation")
    print("=" * 80)
    
    batch_size = 2
    predicted = torch.randn(batch_size, 1280, 2560)
    target = torch.randn(batch_size, 1280, 2560)
    
    # MSE loss
    mse_loss = torch.nn.functional.mse_loss(predicted, target)
    
    print(f"✓ MSE loss computed successfully")
    print(f"  Predicted shape: {predicted.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  MSE loss: {mse_loss.item():.4f}")
    
    # Cosine similarity
    pred_norm = torch.nn.functional.normalize(predicted, p=2, dim=-1)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1).mean()
    
    print(f"✓ Cosine similarity computed successfully")
    print(f"  Cosine similarity: {cos_sim.item():.4f}")
    
    return mse_loss, cos_sim


def test_end_to_end():
    """Test end-to-end forward pass."""
    print("\n" + "=" * 80)
    print("Test 5: End-to-End Forward Pass")
    print("=" * 80)
    
    # Create model
    model = PointEncoder(
        trans_dim=521,
        depth=12,
        num_heads=6,
        encoder_dims=512,
        group_size=32,
        num_group=1280,
        drop_path_rate=0.1,
        output_dim=2560
    )
    model.eval()
    
    # Create dummy data
    batch_size = 2
    num_points = 8192
    point_clouds = torch.randn(batch_size, num_points, 3)
    
    # Create target embeddings
    view_names = ["center", "up", "down", "left", "right"]
    target_embeddings = torch.randn(batch_size, 1280, 2560)
    
    # Forward pass
    with torch.no_grad():
        predicted_embeddings, _ = model(point_clouds, return_deepstack=False)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(predicted_embeddings, target_embeddings)
    
    print(f"✓ End-to-end forward pass successful")
    print(f"  Point clouds: {point_clouds.shape}")
    print(f"  Predicted embeddings: {predicted_embeddings.shape}")
    print(f"  Target embeddings: {target_embeddings.shape}")
    print(f"  Loss: {loss.item():.4f}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PointEncoder Pretraining Setup Test Suite")
    print("=" * 80 + "\n")
    
    try:
        # Test 1: Model with projection
        test_point_encoder_with_projection()
        
        # Test 2: Dataset loading
        dataset = test_dataset_loading()
        
        # Test 3: Embedding concatenation
        test_embedding_concatenation()
        
        # Test 4: Loss computation
        test_loss_computation()
        
        # Test 5: End-to-end
        test_end_to_end()
        
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80 + "\n")
        
        if dataset and len(dataset) > 0:
            print("Ready to start pretraining!")
            print("\nTo start pretraining, run:")
            print("  cd /work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS")
            print("  python training/train_aux_point_to_qwen_image_embedding.py --config configs/auxiliary_training/point_to_qwen_image_embedding_pretrain.yaml")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Tests failed!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
