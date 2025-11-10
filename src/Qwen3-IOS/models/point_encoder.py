"""
PointTransformer encoder adapted for Qwen3-VL integration.
Reuses components from Point-BERT project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add Point-BERT to path for imports
POINT_BERT_PATH = Path(__file__).parent.parent.parent / "Point-BERT"
sys.path.insert(0, str(POINT_BERT_PATH))

from models.dvae import Group, Encoder
from timm.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    """MLP module for transformer blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder without hierarchical structure."""
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos, return_intermediate=False):
        """
        Forward pass through transformer blocks.
        
        Args:
            x: Input features
            pos: Positional embeddings
            return_intermediate: If True, return intermediate layer outputs (DeepStack)
        
        Returns:
            If return_intermediate=False: final output
            If return_intermediate=True: (final_output, list of intermediate outputs)
        """
        intermediate_outputs = []
        
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            
            # Store intermediate outputs for DeepStack
            if return_intermediate:
                intermediate_outputs.append(x)
        
        if return_intermediate:
            return x, intermediate_outputs
        return x


class PointEncoder(nn.Module):
    """
    Point cloud encoder adapted from Point-BERT.
    Encodes point clouds into token sequences.
    """
    def __init__(
        self,
        trans_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        encoder_dims: int = 384,
        group_size: int = 32,
        num_group: int = 512,
        drop_path_rate: float = 0.1,
        output_dim: Optional[int] = None,  # Optional projection to match Qwen embeddings
        projection_dropout: float = 0.1,  # Dropout for output projection layer
        **kwargs
    ):
        super().__init__()
        
        self.trans_dim = trans_dim
        self.depth = depth
        self.num_heads = num_heads
        self.encoder_dims = encoder_dims
        self.group_size = group_size
        self.num_group = num_group
        self.output_dim = output_dim
        self.projection_dropout = projection_dropout
        
        # Group point cloud into patches
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        # Encode groups
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        
        # Project to transformer dimension
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        
        self.norm = nn.LayerNorm(self.trans_dim)
        
        # Optional projection layer for embedding matching (pretraining)
        if self.output_dim is not None and self.output_dim != self.trans_dim:
            self.output_projection = nn.Sequential(
                nn.Dropout(self.projection_dropout),
                nn.Linear(self.trans_dim, self.output_dim)
            )
        else:
            self.output_projection = None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, pts: torch.Tensor, return_deepstack: bool = False):
        """
        Forward pass of point encoder.
        
        Args:
            pts: Point cloud tensor of shape (B, N, 3) or (B, N, C)
            return_deepstack: If True, return intermediate features (DeepStack style)
        
        Returns:
            If return_deepstack=False:
                Tuple of (group_tokens, center_positions)
                - group_tokens: (B, num_groups, trans_dim)
                - center_positions: (B, num_groups, 3)
            
            If return_deepstack=True:
                Tuple of (final_tokens, intermediate_tokens, center_positions)
                - final_tokens: (B, num_groups, trans_dim)
                - intermediate_tokens: List of tensors from selected layers
                - center_positions: (B, num_groups, 3)
        """
        # Group points
        neighborhood, center = self.group_divider(pts)
        # neighborhood: (B, num_group, group_size, 3)
        # center: (B, num_group, 3)
        
        # Encode each group
        group_input_tokens = self.encoder(neighborhood)  # (B, num_group, encoder_dims)
        
        # Project to transformer dimension
        group_tokens = self.reduce_dim(group_input_tokens)  # (B, num_group, trans_dim)
        
        # Add positional encoding
        pos = self.pos_embed(center)  # (B, num_group, trans_dim)
        
        # Pass through transformer
        if return_deepstack:
            x, all_intermediate = self.blocks(group_tokens, pos, return_intermediate=True)
            
            # Select intermediate layers (similar to Qwen3-VL's approach)
            # For depth=12: use layers 3, 6, 9 (indices 2, 5, 8)
            # For depth=6: use layers 2, 4 (indices 1, 3)
            if self.depth >= 12:
                layer_indices = [2, 5, 8, 11]  # 3rd, 6th, 9th, 12th layers
            elif self.depth >= 6:
                layer_indices = [1, 3, 5]  # 2nd, 4th, 6th layers
            else:
                layer_indices = [i for i in range(self.depth)]
            
            selected_intermediate = [all_intermediate[i] for i in layer_indices if i < len(all_intermediate)]
        else:
            x = self.blocks(group_tokens, pos, return_intermediate=False)
            selected_intermediate = None
        
        # Normalize
        x = self.norm(x)
        
        # Apply output projection if available
        if self.output_projection is not None:
            x = self.output_projection(x)
        
        if return_deepstack:
            # Normalize intermediate features too
            if selected_intermediate is not None:
                normalized_intermediate = [self.norm(feat) for feat in selected_intermediate]
                if self.output_projection is not None:
                    normalized_intermediate = [self.output_projection(feat) for feat in normalized_intermediate]
            else:
                normalized_intermediate = []
            return x, normalized_intermediate, center
        else:
            return x, center
    
    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained Point-BERT weights.
        
        Handles checkpoints with:
        - 'module.point_encoder.' prefix (DataParallel)
        - 'state_dict' wrapper
        - XYZ+RGB input (6 channels) adapted to XYZ-only (3 channels)
        - cls_token and cls_pos parameters (ignored)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'base_model' in checkpoint:
            state_dict = checkpoint['base_model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.point_encoder.' or 'module.' prefix from keys
        # Also discard classification_head parameters
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Skip classification head parameters
            if key.startswith('classification_head') or key.startswith('module.classification_head'):
                continue
            
            if key.startswith('module.point_encoder.'):
                new_key = key.replace('module.point_encoder.', '')
                cleaned_state_dict[new_key] = value
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
                cleaned_state_dict[new_key] = value
            elif key.startswith('encoder.encoder.'):
                new_key = key.replace('encoder.encoder.', 'encoder.')
                cleaned_state_dict[new_key] = value
            elif key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        
        # Get model's state dict
        model_dict = self.state_dict()
        
        # Filter to only keep matching keys with matching shapes
        matched_dict = {}
        unmatched_keys = []
        for key, value in cleaned_state_dict.items():
            if key in model_dict:
                if value.shape == model_dict[key].shape:
                    matched_dict[key] = value
                elif key == 'encoder.first_conv.0.weight':
                    # Special case: checkpoint has 6 channels (XYZ+RGB), model has 3 (XYZ only)
                    # Take only the XYZ weights (first 3 channels)
                    if value.shape[1] == 6 and model_dict[key].shape[1] == 3:
                        matched_dict[key] = value[:, :3, :]
                        print(f"  Adapted '{key}': used only XYZ channels (discarded RGB)")
                    else:
                        print(f"  Warning: Shape mismatch for '{key}': checkpoint {value.shape} vs model {model_dict[key].shape}")
                else:
                    print(f"  Warning: Shape mismatch for '{key}': checkpoint {value.shape} vs model {model_dict[key].shape}")
            else:
                unmatched_keys.append(key)
        # Load the weights
        self.load_state_dict(matched_dict, strict=True)
        
        # Report loading statistics
        skipped_keys = set(cleaned_state_dict.keys()) - set(model_dict.keys())
        if skipped_keys:
            print(f"  Skipped {len(skipped_keys)} parameters not in model: {sorted(skipped_keys)}")
        
        print(f"✓ Loaded pretrained weights from {checkpoint_path}")
        print(f"  Matched {len(matched_dict)}/{len(model_dict)} parameters")
