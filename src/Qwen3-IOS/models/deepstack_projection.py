"""
DeepStack projection aligned with Qwen3-VL architecture.

In Qwen3-VL, DeepStack extracts visual features from multiple intermediate 
ViT layers and injects them into corresponding LLM layers via residual connections.
This provides multi-resolution visual information at different depths of the LLM.

Architecture:
1. Extract features from multiple PointTransformer layers (e.g., layers 3, 6, 9, 12)
2. Each intermediate layer has its own projection module
3. During LLM forward pass, inject these features at corresponding LLM layers
4. Features are added via residual connection

Reference: Qwen3-VL implementation (qwen3_vl.py:1114-1120, 1611-1650)

This implementation uses PyTorch forward hooks to inject features at specific
LLM layers during the forward pass, exactly as in Qwen3-VL.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Callable


class DeepStackProjection(nn.Module):
    """
    DeepStack projection with TRUE layer-wise injection.
    
    This implementation injects intermediate features at corresponding LLM layers
    via residual connections, exactly as in Qwen3-VL architecture.
    
    Example:
        Point Layer 3  → Projection → Inject at LLM Layer 8  (1/4 depth)
        Point Layer 6  → Projection → Inject at LLM Layer 16 (1/2 depth)
        Point Layer 9  → Projection → Inject at LLM Layer 24 (3/4 depth)
        Point Layer 12 → Projection → Main input to LLM
    """
    
    def __init__(
        self,
        point_dim: int,
        qwen_visual_dim: int,
        num_deepstack_levels: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            point_dim: Dimension of point encoder output (trans_dim)
            qwen_visual_dim: Dimension of Qwen's visual tokens (text_hidden_size)
            num_deepstack_levels: Number of intermediate layers to use
            dropout: Dropout rate
        """
        super().__init__()
        
        self.point_dim = point_dim
        self.qwen_visual_dim = qwen_visual_dim
        self.num_deepstack_levels = num_deepstack_levels
        
        # Main projection for final features (injected at input)
        self.main_projection = nn.Sequential(
            nn.LayerNorm(point_dim),
            nn.Linear(point_dim, point_dim * 4),
            nn.GELU(),
            nn.Linear(point_dim * 4, qwen_visual_dim),
        )
        
        # DeepStack projections for intermediate features
        # Each level gets its own projection module (injected at LLM layers)
        self.deepstack_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(point_dim),
                nn.Linear(point_dim, point_dim * 4),
                nn.GELU(),
                nn.Linear(point_dim * 4, qwen_visual_dim),
            )
            for _ in range(num_deepstack_levels)
        ])
    
    def forward(
        self,
        final_features: torch.Tensor,
        intermediate_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Project multi-level features for layer-wise injection.
        
        Args:
            final_features: Final layer output (B, N, point_dim)
            intermediate_features: List of intermediate outputs (B, N, point_dim) each
        
        Returns:
            main_visual_tokens: Visual tokens for input injection (B, N, qwen_visual_dim)
            deepstack_embeds: List of intermediate features to inject at LLM layers
        """
        # Project main features (injected at input)
        main_visual_tokens = self.main_projection(final_features)
        
        # Project deepstack features (injected at intermediate LLM layers)
        deepstack_embeds = []
        for i, feat in enumerate(intermediate_features[:self.num_deepstack_levels]):
            projected = self.deepstack_projections[i](feat)
            deepstack_embeds.append(projected)
        
        return main_visual_tokens, deepstack_embeds


class DeepStackInjector:
    """
    Handles injection of DeepStack features into LLM layers using forward hooks.
    
    This mimics Qwen3-VL's layer-wise injection mechanism where intermediate
    visual features are added to LLM hidden states at specific layer depths.
    """
    
    def __init__(
        self,
        qwen_model,
        injection_layer_indices: Optional[List[int]] = None
    ):
        """
        Args:
            qwen_model: The Qwen language model
            injection_layer_indices: Which LLM layers to inject features into
                Example: [8, 16, 24] for ~1/4, 1/2, 3/4 depth
                If None, automatically determined based on model depth
        """
        self.qwen_model = qwen_model
        
        self.llm_layers = qwen_model.model.language_model.layers
        num_layers = len(self.llm_layers)
        
        if injection_layer_indices is None:
            injection_layer_indices = [
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4
            ]
        
        self.injection_layer_indices = injection_layer_indices
        self.num_injection_points = len(injection_layer_indices)
        self.deepstack_buffer: Optional[List[torch.Tensor]] = None
        self.num_visual_tokens: int = 0
        
        self.hook_handles = []
        
        print(f"  DeepStack injection at LLM layers: {injection_layer_indices} (total: {num_layers})")
    
    def set_deepstack_embeds(self, deepstack_embeds: Tuple[torch.Tensor], num_visual_tokens: int):
        """
        Store deepstack embeddings for injection during forward pass.
        
        Args:
            deepstack_embeds: List of (B, N, hidden_dim) tensors to inject
            num_visual_tokens: Number of visual tokens at the beginning of sequence
        """
        self.deepstack_buffer = deepstack_embeds
        self.num_visual_tokens = num_visual_tokens
    
    def clear_deepstack_embeds(self):
        """Clear stored deepstack embeddings."""
        self.deepstack_buffer = None
        self.num_visual_tokens = 0

    def set_image_pad_token_index(self, image_pad_token_index: Tuple[torch.Tensor, torch.Tensor]):
        """
        Store image pad token index for correct injection position.
        
        Args:
            image_pad_token_index: Tuple of tensors indicating positions
            batch_size: Batch size
        """
        self.image_pad_token_index = image_pad_token_index
        
    def _create_injection_hook(self, level: int) -> Callable:
        """
        Create a forward hook that injects deepstack features at a specific layer.
        
        Args:
            level: Which deepstack level to inject (0, 1, 2, ...)
        
        Returns:
            Hook function
        """
        def hook(module, input, output):
            # output is a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()
            
            if self.deepstack_buffer is not None and level < len(self.deepstack_buffer):
                deepstack_embed = self.deepstack_buffer[level]
                
                batch_size, seq_len, hidden_dim = hidden_states.shape
                num_vis = self.num_visual_tokens
                
                if num_vis > 0 and num_vis <= seq_len:
                    deepstack_embed = deepstack_embed.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
                    
                    # Add deepstack features via residual connection (as in Qwen3-VL)
                    image_pad_start = self.image_pad_token_index[1][0]
                    image_pad_end = image_pad_start + num_vis
                    hidden_states[:, image_pad_start:image_pad_end, :] += deepstack_embed
            
            # same structure as input
            if rest:
                return (hidden_states,) + rest
            else:
                return hidden_states
        
        return hook
    
    def register_hooks(self):
        """Register forward hooks on LLM layers for injection."""
        self.remove_hooks()
        
        for level, layer_idx in enumerate(self.injection_layer_indices):
            if level < self.num_injection_points:
                layer = self.llm_layers[layer_idx]
                hook = self._create_injection_hook(level)
                handle = layer.register_forward_hook(hook)
                self.hook_handles.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class SimpleDeepStackProjection(nn.Module):
    """
    Simplified DeepStack projection - just uses final + one intermediate layer.
    
    Lighter weight alternative for initial experiments.
    """
    
    def __init__(
        self,
        point_dim: int,
        qwen_visual_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project intermediate and final
        self.intermediate_proj = nn.Sequential(
            nn.Linear(point_dim, qwen_visual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(qwen_visual_dim)
        )
        
        self.final_proj = nn.Sequential(
            nn.Linear(point_dim, qwen_visual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(qwen_visual_dim)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(qwen_visual_dim * 2, qwen_visual_dim),
            nn.GELU(),
            nn.LayerNorm(qwen_visual_dim)
        )
    
    def forward(
        self,
        final_features: torch.Tensor,
        intermediate_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Simple fusion of final + middle layer.
        
        Args:
            final_features: Final layer output (B, N, point_dim)
            intermediate_features: List of intermediate outputs
        
        Returns:
            Fused visual tokens (B, N, qwen_visual_dim)
        """
        # Use middle intermediate layer
        mid_idx = len(intermediate_features) // 2
        mid_features = intermediate_features[mid_idx]
        
        # Project both
        mid_proj = self.intermediate_proj(mid_features)
        final_proj = self.final_proj(final_features)
        
        # Concatenate and fuse
        concatenated = torch.cat([mid_proj, final_proj], dim=-1)
        fused = self.fusion(concatenated)
        
        return fused
