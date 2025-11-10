import torch
import sys
from pathlib import Path

# Add necessary paths
qwen_path = Path(__file__).parent.parent / "src" / "Qwen3-IOS"
sys.path.insert(0, str(qwen_path))

# Import from the correct location
sys.path.insert(0, str(qwen_path / "models"))
from point_encoder import PointEncoder

# Initialize the point encoder with the same config
point_encoder = PointEncoder(
    trans_dim=384,
    depth=12,
    num_heads=6,
    encoder_dims=256,
    group_size=32,
    num_group=512,
    drop_path_rate=0.1,
)

print("Point encoder initialized")

# Load the checkpoint using the built-in method
checkpoint_path = '/work/grana_maxillo/IOS-DraftReport/_checkpoints/checkpoints/point_bert_v1.2.pt'
print(f"\nLoading pretrained checkpoint from:")
print(f"  {checkpoint_path}")
print()

point_encoder.load_pretrained(checkpoint_path)

# Test the encoder with a dummy input
print("\nTesting encoder with dummy input...")
dummy_input = torch.randn(2, 2048, 3)  # Batch size 2, 2048 points, 3 coords (x,y,z)
with torch.no_grad():
    output, centers = point_encoder(dummy_input)
    print(f"✓ Forward pass successful!")
    print(f"  Input shape (XYZ): {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Centers shape: {centers.shape}")
    print(f"\n✓ The point_encoder is now ready to use with XYZ-only point clouds!")

