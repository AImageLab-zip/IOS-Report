"""
Projection layers to align point cloud tokens with Qwen's visual token space.
"""

import torch
import torch.nn as nn
from typing import Optional


class PointToQwenProjection(nn.Module):
    """
    Projects point cloud embeddings to Qwen's visual token space.
    """
    def __init__(
        self,
        point_dim: int,
        qwen_visual_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        **kwargs
    ):
        """
        Args:
            point_dim: Dimension of point encoder output
            qwen_visual_dim: Dimension of Qwen's visual tokens
            hidden_dim: Hidden dimension for intermediate layers
            num_layers: Number of projection layers
            dropout: Dropout rate
            use_layernorm: Whether to use layer normalization
        """
        super().__init__()
        
        self.point_dim = point_dim
        self.qwen_visual_dim = qwen_visual_dim
        
        # Learnable gate alpha, initialized to ~0.1
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        if hidden_dim is None:
            hidden_dim = (point_dim + qwen_visual_dim) // 2
        
        # Build projection layers
        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(point_dim, qwen_visual_dim))
        else:
            # Input layer
            layers.append(nn.Linear(point_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, qwen_visual_dim))
            layers.append(nn.LayerNorm(qwen_visual_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Final layer norm
        if use_layernorm:
            self.output_norm = nn.LayerNorm(qwen_visual_dim)
        else:
            self.output_norm = nn.Identity()
    
    def forward(self, point_tokens: torch.Tensor) -> torch.Tensor:
        """
        Project point tokens to Qwen visual token space.
        
        Args:
            point_tokens: (B, N, point_dim) tensor of point embeddings
        
        Returns:
            (B, N, qwen_visual_dim) tensor of projected visual tokens
        """
        x = self.projection(point_tokens)
        x = self.output_norm(x)
        # Apply learnable gate
        x = self.alpha * x
        return x