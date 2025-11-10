"""
Main model combining PointEncoder with Qwen3-VL.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration

from .point_encoder import PointEncoder
from .projection import PointToQwenProjection
from .deepstack_projection import DeepStackProjection, DeepStackInjector


class PointQwen(nn.Module):
    """
    PointQwen: Qwen3-VL adapted for 3D point cloud understanding.
    
    Replaces the image encoder with a PointTransformer while keeping
    the language model frozen.
    """
    
    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        point_encoder_config: Optional[Dict[str, Any]] = None,
        projection_type: str = "deepstack",  # simple, adaptive, perceiver, deepstack, simple_deepstack
        projection_config: Optional[Dict[str, Any]] = None,
        freeze_qwen: bool = True,
        freeze_point_encoder: bool = False,
        pretrained_point_encoder: Optional[str] = None,
        use_deepstack: bool = False,  # Auto-set based on projection_type
        use_gradient_checkpointing: bool = False,  # Enable gradient checkpointing for memory savings
        low_cpu_mem_usage: bool = True,  # Load model with low CPU memory
        use_flash_attention: bool = True,  # Use flash attention if available
    ):
        """
        Args:
            qwen_model_name: Hugging Face model name for Qwen
            point_encoder_config: Configuration for PointEncoder
            projection_type: Type of projection layer ('simple', 'adaptive', 'perceiver', 'deepstack', 'simple_deepstack')
            projection_config: Configuration for projection layer
            freeze_qwen: Whether to freeze Qwen parameters
            freeze_point_encoder: Whether to freeze point encoder
            pretrained_point_encoder: Path to pretrained point encoder weights
            use_deepstack: Whether to use multi-level features (auto-set for deepstack projections)
        """
        super().__init__()
        
       
        print(f"Loading Qwen model: {qwen_model_name}")
        print(f"  Memory optimizations: gradient_checkpointing={use_gradient_checkpointing}, "
              f"flash_attention={use_flash_attention}")
        
        model_class = Qwen3VLForConditionalGeneration
        
        # Prepare loading arguments
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        
        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        try:
            self.qwen = model_class.from_pretrained(qwen_model_name, **load_kwargs)
        except Exception as e:
            if use_flash_attention:
                print(f"  Warning: Flash attention failed ({e}), falling back to standard attention")
                load_kwargs["attn_implementation"] = "sdpa"  # Use SDPA instead
                self.qwen = model_class.from_pretrained(qwen_model_name, **load_kwargs)
            else:
                raise
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable()
            print("  Enabled gradient checkpointing for Qwen")
        
        # Load processor for tokenization
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        
        # Get Qwen's visual token dimension
        # This needs to be extracted from Qwen's architecture
        self.qwen_visual_dim = self._get_qwen_visual_dim()
        
        # Initialize point encoder
        if point_encoder_config is None:
            raise ValueError("point_encoder_config must be provided")
        
        self.point_encoder = PointEncoder(**point_encoder_config)
        self._use_gradient_checkpointing = use_gradient_checkpointing
        if pretrained_point_encoder is not None:
            self.point_encoder.load_pretrained(pretrained_point_encoder)

        if projection_type in ["deepstack", "simple_deepstack"]:
            use_deepstack = True
        
        self.use_deepstack = use_deepstack
        
        point_dim = point_encoder_config.get("trans_dim", 384)
        
        if projection_config is None:
            projection_config = {}
        
        if projection_type == "simple":
            self.projection = PointToQwenProjection(
                point_dim=point_dim,
                qwen_visual_dim=self.qwen_visual_dim,
                **projection_config
            )
        elif projection_type == "deepstack":
            self.projection = DeepStackProjection(
                point_dim=point_dim,
                qwen_visual_dim=self.qwen_visual_dim,
                **projection_config
            )
            injection_indices = projection_config.get('injection_layer_indices', None)
            self.deepstack_injector = DeepStackInjector(
                qwen_model=self.qwen,
                injection_layer_indices=injection_indices
            )
            self.deepstack_injector.register_hooks()
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        
        self.projection_type = projection_type
        
        # Initialize frozen state tracker
        self.point_encoder_frozen = False
        
        # Freeze/unfreeze components
        if freeze_qwen:
            self._freeze_qwen()
        
        if freeze_point_encoder:
            self._freeze_point_encoder()
        
        # Move point encoder and projection to same device as Qwen
        qwen_device = next(self.qwen.parameters()).device
        self.point_encoder = self.point_encoder.to(qwen_device)
        self.projection = self.projection.to(qwen_device)
        
        print("\n" + "="*70)
        print("✓ PointQwen initialized successfully!")
        print(f"  Model: {qwen_model_name}")
        print(f"  Point encoder: {point_encoder_config.get('num_group', 512)} tokens, depth={point_encoder_config.get('depth', 12)}")
        print(f"  Projection: {projection_type}" + (" (DeepStack multi-level)" if self.use_deepstack else ""))
        print(f"  Qwen frozen: {freeze_qwen}")
        print(f"  Point encoder frozen: {freeze_point_encoder}")
        print("="*70 + "\n")
    
    def _get_qwen_visual_dim(self) -> int:
        """
        Extract Qwen's visual token dimension from the model.
        
        Note: This should match the language model's hidden size, not the vision encoder's,
        because visual tokens are projected to the same space as text embeddings.
        """
        try:
            # For Qwen3-VL and Qwen2-VL, use text_config.hidden_size
            if hasattr(self.qwen.config, 'text_config'):
                dim = self.qwen.config.text_config.hidden_size
                print(f"  Using text hidden size: {dim}")
                return dim
            # Fallback: try vision_config (older models)
            elif hasattr(self.qwen.config, 'vision_config'):
                dim = self.qwen.config.vision_config.hidden_size
                print(f"  Using vision hidden size: {dim}")
                return dim
            # Fallback: try general hidden_size
            elif hasattr(self.qwen.config, 'hidden_size'):
                dim = self.qwen.config.hidden_size
                print(f"  Using general hidden size: {dim}")
                return dim
            else:
                # Default fallback
                print("Warning: Could not determine visual dimension, using default 2560")
                return 2560
        except Exception as e:
            print(f"Warning: Error getting visual dim: {e}. Using default 2560")
            return 2560
    
    def _freeze_qwen(self):
        """Freeze all Qwen parameters."""
        for param in self.qwen.parameters():
            param.requires_grad = False
        self.qwen.eval()
        print("Froze Qwen parameters")
    
    def _freeze_point_encoder(self):
        """Freeze point encoder parameters."""
        for param in self.point_encoder.parameters():
            param.requires_grad = False
        self.point_encoder_frozen = True
        print("Froze PointEncoder parameters")
    
    def freeze_point_encoder(self):
        """Freeze point encoder parameters (public method)."""
        self._freeze_point_encoder()
    
    def unfreeze_point_encoder(self):
        """Unfreeze point encoder parameters."""
        for param in self.point_encoder.parameters():
            param.requires_grad = True
        self.point_encoder_frozen = False
        print("Unfroze PointEncoder parameters")
    
    @property
    def device(self):
        """Get the device where the model is located."""
        return next(self.qwen.parameters()).device
    
    def encode_point_cloud(self, point_cloud: torch.Tensor):
        """
        Encode point cloud to visual tokens.
        
        For DeepStack projection with layer-wise injection, returns both main tokens
        and deepstack embeddings. For other projections, returns only visual tokens.
        
        Args:
            point_cloud: (B, N, 3) or (B, N, C) tensor
        
        Returns:
            If projection_type == "deepstack":
                main_visual_tokens, deepstack_embeds
            Else:
                visual_tokens
        """
        if self.use_deepstack:
            # Use DeepStack: extract multi-level features
            final_tokens, intermediate_tokens, _ = self.point_encoder(
                point_cloud, 
                return_deepstack=True
            )
            
            # Project with DeepStack projection
            if self.projection_type == "deepstack":
                # Returns (main_tokens, deepstack_embeds) for layer-wise injection
                main_visual_tokens, deepstack_embeds = self.projection(final_tokens, intermediate_tokens)
                return main_visual_tokens, deepstack_embeds
            else:
                # simple_deepstack: fuses at input level
                visual_tokens = self.projection(final_tokens, intermediate_tokens)
                return visual_tokens
        else:
            # Standard single-level encoding
            point_tokens, _ = self.point_encoder(point_cloud)
            visual_tokens = self.projection(point_tokens)
            return visual_tokens
    
    def forward(
        self,
        point_cloud: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with proper multimodal integration.
        
        Args:
            point_cloud: (B, N, 3) point cloud tensor
            input_ids: (B, L) text token ids
            attention_mask: (B, L) attention mask
            labels: (B, L) labels for language modeling loss
        
        Returns:
            CausalLMOutputWithPast with loss and logits
        """
        batch_size = point_cloud.shape[0]
        
        encoded_output = self.encode_point_cloud(point_cloud)
        
        image_pad_token_id = 151655
        image_pad_token_index = (input_ids == image_pad_token_id).nonzero(as_tuple=True)
        
        if self.projection_type == "deepstack" and isinstance(encoded_output, tuple):
            visual_tokens, deepstack_embeds = encoded_output
            num_visual_tokens = visual_tokens.shape[1]
            if self.deepstack_injector is not None:
                self.deepstack_injector.set_deepstack_embeds(deepstack_embeds, num_visual_tokens)
                self.deepstack_injector.set_image_pad_token_index(image_pad_token_index)
        else:
            visual_tokens = encoded_output
            num_visual_tokens = visual_tokens.shape[1]
        
        embed_tokens = self.qwen.model.language_model.embed_tokens
        text_embeds = embed_tokens(input_ids)  # (B, L, hidden_dim)
        
        visual_tokens = visual_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
        
        inputs_embeds = text_embeds.clone()
        
        # Slow?????
        new_embeds_list = []
        for batch_idx in range(batch_size):
            batch_mask = image_pad_token_index[0] == batch_idx
            if batch_mask.any():
                token_positions = image_pad_token_index[1][batch_mask]
                if len(token_positions) > 0:
                    pos = token_positions[0].item()  # Support only one occurrence
                    # text before | point encoder tokens | text after
                    new_embeds = torch.cat([
                        inputs_embeds[batch_idx, :pos, :],
                        visual_tokens[batch_idx],
                        inputs_embeds[batch_idx, pos+1:, :]
                    ], dim=0)
                    new_embeds_list.append(new_embeds)
                else:
                    new_embeds_list.append(inputs_embeds[batch_idx])
            else:
                new_embeds_list.append(inputs_embeds[batch_idx])
        inputs_embeds = torch.stack(new_embeds_list, dim=0)

        # Update attention mask
        if attention_mask is not None:
            new_attention_mask_list = []
            for batch_idx in range(batch_size):
                batch_mask = image_pad_token_index[0] == batch_idx
                if batch_mask.any():
                    token_positions = image_pad_token_index[1][batch_mask]
                    if len(token_positions) > 0:
                        pos = token_positions[0].item()
                        # Create attention mask for visual tokens
                        visual_attn = torch.ones(num_visual_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
                        # Concatenate: mask before | visual mask | mask after
                        new_attn = torch.cat([
                            attention_mask[batch_idx, :pos],
                            visual_attn,
                            attention_mask[batch_idx, pos+1:]
                        ], dim=0)
                        new_attention_mask_list.append(new_attn)
                    else:
                        new_attention_mask_list.append(attention_mask[batch_idx])
                else:
                    new_attention_mask_list.append(attention_mask[batch_idx])
            extended_attention_mask = torch.stack(new_attention_mask_list, dim=0)
        else:
            extended_attention_mask = None
        
        # Update labels similarly
        if labels is not None:
            new_labels_list = []
            for batch_idx in range(batch_size):
                batch_mask = image_pad_token_index[0] == batch_idx
                if batch_mask.any():
                    token_positions = image_pad_token_index[1][batch_mask]
                    if len(token_positions) > 0:
                        pos = token_positions[0].item()
                        visual_labels_batch = torch.full((num_visual_tokens,), fill_value=-100, dtype=labels.dtype, device=labels.device)
                        # Concatenate: labels before | visual labels | labels after
                        new_labels_batch = torch.cat([
                            labels[batch_idx, :pos],
                            visual_labels_batch,
                            labels[batch_idx, pos+1:]
                        ], dim=0)
                        new_labels_list.append(new_labels_batch)
                    else:
                        new_labels_list.append(labels[batch_idx])
                else:
                    new_labels_list.append(labels[batch_idx])
            extended_labels = torch.stack(new_labels_list, dim=0)
        else:
            extended_labels = None
        
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            **kwargs
        )
        
        # Clear deepstack buffer after forward pass
        if self.projection_type == "deepstack" and self.deepstack_injector is not None:
            self.deepstack_injector.clear_deepstack_embeds()
        
        return outputs
    """
    point_cloud=point_cloud,
    prompt=prompt,
    max_new_tokens=self.config['training'].get('val_max_new_tokens', 256),
    temperature=gen_config.get('temperature', 0.7),
    top_p=gen_config.get('top_p', 0.9),
    top_k=gen_config.get('top_k', 50),
    do_sample=True,
    repetition_penalty=gen_config.get('repetition_penalty', 1.2),
    no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
    """
    def generate(self,
        point_cloud: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 1280,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 1,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        num_beams: int = 4,
        **kwargs
    ):
        """
        Generate text conditioned on point cloud and prompt.
        
        Args:
            point_cloud: (B, N, 3) point cloud tensor
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeated tokens
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            **kwargs: Additional generation arguments
        Returns:
            Generated token ids tensor
        """
        batch_size = point_cloud.shape[0]
        
        encoded_output = self.encode_point_cloud(point_cloud)
        
        image_pad_token_id = 151655
        inputs = self.processor(
            text=[prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            skip_special_tokens=False
        ).to(self.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        image_pad_token_index = (input_ids == image_pad_token_id).nonzero(as_tuple=True)
        
        if self.projection_type == "deepstack" and isinstance(encoded_output, tuple):
            visual_tokens, deepstack_embeds = encoded_output
            num_visual_tokens = visual_tokens.shape[1]
            if self.deepstack_injector is not None:
                self.deepstack_injector.set_deepstack_embeds(deepstack_embeds, num_visual_tokens)
                self.deepstack_injector.set_image_pad_token_index(image_pad_token_index)
        else:
            visual_tokens = encoded_output
            num_visual_tokens = visual_tokens.shape[1]
        
        embed_tokens = self.qwen.model.language_model.embed_tokens
        text_embeds = embed_tokens(input_ids)  # (B, L, hidden_dim)
        
        visual_tokens = visual_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
        
        inputs_embeds = text_embeds.clone()
        
        new_embeds_list = []
        for batch_idx in range(batch_size):
            batch_mask = image_pad_token_index[0] == batch_idx
            if batch_mask.any():
                token_positions = image_pad_token_index[1][batch_mask]
                if len(token_positions) > 0:
                    pos = token_positions[0].item()  # Support only one occurrence
                    # text before | point encoder tokens | text after
                    new_embeds = torch.cat([
                        inputs_embeds[batch_idx, :pos, :],
                        visual_tokens[batch_idx],
                        inputs_embeds[batch_idx, pos+1:, :]
                    ], dim=0)
                    new_embeds_list.append(new_embeds)
                else:
                    new_embeds_list.append(inputs_embeds[batch_idx])
            else:
                new_embeds_list.append(inputs_embeds[batch_idx])
        inputs_embeds = torch.stack(new_embeds_list, dim=0)

        # Update attention mask
        if attention_mask is not None:
            new_attention_mask_list = []
            for batch_idx in range(batch_size):
                batch_mask = image_pad_token_index[0] == batch_idx
                if batch_mask.any():
                    token_positions = image_pad_token_index[1][batch_mask]
                    if len(token_positions) > 0:
                        pos = token_positions[0].item()
                        # Create attention mask for visual tokens
                        visual_attn = torch.ones(num_visual_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
                        # Concatenate: mask before | visual mask | mask after
                        new_attn = torch.cat([
                            attention_mask[batch_idx, :pos],
                            visual_attn,
                            attention_mask[batch_idx, pos+1:]
                        ], dim=0)
                        new_attention_mask_list.append(new_attn)
                    else:
                        new_attention_mask_list.append(attention_mask[batch_idx])
                else:
                    new_attention_mask_list.append(attention_mask[batch_idx])
            extended_attention_mask = torch.stack(new_attention_mask_list, dim=0)
        else:
            extended_attention_mask = None
            
        generated_ids = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            **kwargs
        )
        
        # Clear deepstack buffer after generation
        if self.projection_type == "deepstack" and self.deepstack_injector is not None:
            self.deepstack_injector.clear_deepstack_embeds()
        
        # convert to strings
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts
        
    
    def get_trainable_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """Get trainable parameters grouped by component."""
        params = {
            "point_encoder": [p for p in self.point_encoder.parameters() if p.requires_grad],
            "projection": [p for p in self.projection.parameters() if p.requires_grad],
            "qwen": [p for p in self.qwen.parameters() if p.requires_grad]
        }
        return params
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable_params = self.get_trainable_parameters()
        
        print("\n" + "=" * 60)
        print("TRAINABLE PARAMETERS")
        print("=" * 60)
        
        total = 0
        for name, params in trainable_params.items():
            num_params = sum(p.numel() for p in params)
            total += num_params
            print(f"{name:20} {num_params:>15,} params")
        
        print("-" * 60)
        print(f"{'TOTAL':20} {total:>15,} params")
        print("=" * 60 + "\n")
