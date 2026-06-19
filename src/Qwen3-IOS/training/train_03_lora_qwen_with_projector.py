"""
LoRA fine-tuning script for PointQwen using Unsloth.

This script implements Stage 2 training:
- PointEncoder: Frozen (pretrained)
- Projection: Trainable (pretrained weights loaded)
- Qwen3-VL: Frozen base model with LoRA adapters (trainable)
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import traceback
import gc
        
# Unsloth imports devono essere in cima
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import wandb
from tqdm import tqdm
from typing import Dict, Optional

from transformers import AutoProcessor, Trainer, TrainingArguments

sys.path.append(str(Path(__file__).parent.parent))

from models import PointEncoder
from models.projection import PointToQwenProjection
from data import IOSPointCloudDataset, IOSCollator
from validation import compute_all_metrics


class CustomTrainer(Trainer):
    """Custom Trainer that handles tied weights for Qwen models and component-specific learning rates."""
    
    def __init__(self, *args, lora_trainer_wrapper=None, projection_lr=None, projection_weight_decay=None, lora_weight_decay=None, **kwargs):
        """
        Initialize with reference to LoRATrainer for custom evaluation.
        
        Args:
            lora_trainer_wrapper: Reference to LoRATrainer for custom evaluation
            projection_lr: Separate learning rate for projection layer (optional)
            projection_weight_decay: Weight decay for projection layer (optional)
            lora_weight_decay: Weight decay for LoRA adapters (optional)
        """
        self.projection_lr = projection_lr
        self.projection_weight_decay = projection_weight_decay
        self.lora_weight_decay = lora_weight_decay
        super().__init__(*args, **kwargs)
        self.lora_trainer_wrapper = lora_trainer_wrapper
    
    def create_optimizer(self):
        """
        Create optimizer with component-specific learning rates and weight decay.
        
        Separates parameters into:
        - Projection layer: uses projection_lr and projection_weight_decay
        - LoRA adapters: uses lora_lr and lora_weight_decay
        """
        if self.optimizer is None:
            # Get component-specific hyperparameters
            projection_lr = self.projection_lr
            projection_wd = self.projection_weight_decay if self.projection_weight_decay is not None else self.args.weight_decay
            lora_lr = self.args.learning_rate
            lora_wd = self.lora_weight_decay if self.lora_weight_decay is not None else self.args.weight_decay
            
            # Separate parameters by component
            projection_params = []
            lora_params = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'projection' in name:
                        projection_params.append(param)
                    else:
                        # LoRA adapters and other trainable params
                        lora_params.append(param)
            
            # Create parameter groups with different learning rates and weight decay
            param_groups = []
            
            if projection_params and projection_lr is not None:
                param_groups.append({
                    'params': projection_params,
                    'lr': projection_lr,
                    'weight_decay': projection_wd,
                    'name': 'projection'
                })
                print(f"  Projection LR: {projection_lr}, Weight Decay: {projection_wd}")
            
            if lora_params:
                param_groups.append({
                    'params': lora_params,
                    'lr': lora_lr,
                    'weight_decay': lora_wd,
                    'name': 'lora'
                })
                print(f"  LoRA LR: {lora_lr}, Weight Decay: {lora_wd}")
            
            # If no component-specific LR, fall back to default
            if not param_groups:
                return super().create_optimizer()
            
            # Create optimizer with parameter groups
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            
            # Remove 'lr' and 'weight_decay' from optimizer_kwargs as we specify them per group
            if 'lr' in optimizer_kwargs:
                optimizer_kwargs.pop('lr')
            if 'weight_decay' in optimizer_kwargs:
                optimizer_kwargs.pop('weight_decay')
            
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            
            # Log initial LRs for verification
            print("\n" + "=" * 60)
            print("OPTIMIZER PARAMETER GROUPS CREATED")
            print("=" * 60)
            for i, group in enumerate(self.optimizer.param_groups):
                name = group.get('name', f'group_{i}')
                print(f"  {name}: LR={group['lr']}, WD={group['weight_decay']}, {len(group['params'])} params")
            print("=" * 60 + "\n")
            
        return self.optimizer
    
    def log_learning_rates(self, step):
        """Log learning rates for all parameter groups."""
        if self.optimizer is not None and step % self.args.logging_steps == 0:
            lr_dict = {}
            for i, group in enumerate(self.optimizer.param_groups):
                name = group.get('name', f'group_{i}')
                lr_dict[f'lr/{name}'] = group['lr']
            
            # Log to wandb if enabled
            if self.args.report_to and 'wandb' in self.args.report_to:
                import wandb
                wandb.log(lr_dict, step=self.state.global_step)
    
    def training_step(self, *args, **kwargs):
        """Override training_step to log detailed LR info."""
        result = super().training_step(*args, **kwargs)
        
        # Log LRs for all parameter groups
        if self.state.global_step % self.args.logging_steps == 0:
            self.log_learning_rates(self.state.global_step)
        
        return result
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to use ONLY custom evaluation with generation (skip parent's eval)."""
        # Clean up any existing GPU memory before evaluation
        gc.collect()
        torch.cuda.empty_cache()
        
        # Skip parent's evaluate to save memory - we only need generation-based eval
        metrics = super().evaluate(*args, **kwargs)
        
        # Run custom evaluation with generation if wrapper is available
        if self.lora_trainer_wrapper is not None:
            print("\n" + "=" * 60)
            print("RUNNING CUSTOM EVALUATION WITH GENERATION")
            print("=" * 60)
            self.lora_trainer_wrapper.evaluate()
        
        # Final cleanup before returning to training
        gc.collect()
        torch.cuda.empty_cache()
        
        # Return dummy metrics dict with eval_loss to satisfy Trainer's expectations
        # The actual metrics are logged inside custom evaluate()
        return metrics
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override save to handle tied weights issue."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # For our PointQwenLoRA wrapper, we need to save components separately
        if hasattr(self.model, 'qwen'):
            # Save LoRA adapters from Qwen (handles tied weights properly)
            self.model.qwen.save_pretrained(
                output_dir,
                safe_serialization=True,
            )
            
            # Save projection layer separately
            projection_path = os.path.join(output_dir, 'projection.pt')
            torch.save(self.model.projection.state_dict(), projection_path)
            
            # Save processor
            self.model.processor.save_pretrained(output_dir)
            
            # Save training config
            config_path = os.path.join(output_dir, 'training_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.model._config_dict, f)
                
            print(f"✓ Model saved to {output_dir}")
        else:
            # Fallback to default save
            super()._save(output_dir, state_dict)


class PointQwenLoRA(nn.Module):
    """
    PointQwen with LoRA adapters for fine-tuning Qwen while keeping
    PointEncoder frozen and Projection trainable.
    """
    
    def __init__(
        self,
        config: Dict,
        qwen_lora_model,
        processor,
    ):
        """
        Args:
            config: Configuration dictionary
            qwen_lora_model: Unsloth FastVisionModel with LoRA adapters
            processor: Tokenizer/processor for Qwen
        """
        super().__init__()
        
        self._config_dict = config
        self.qwen = qwen_lora_model
        self.processor = processor
        
        # Get Qwen's visual token dimension
        self.qwen_visual_dim = self._get_qwen_visual_dim()
        
        # Initialize point encoder (frozen)
        point_encoder_config = config['model']['point_encoder']
        self.point_encoder = PointEncoder(**point_encoder_config)
        
        # Load pretrained point encoder weights
        pretrained_encoder = config['model'].get('pretrained_point_encoder')
        if pretrained_encoder:
            print(f"\nLoading pretrained PointEncoder from: {pretrained_encoder}")
            self.point_encoder.load_pretrained(pretrained_encoder)
        
        # Freeze point encoder
        for param in self.point_encoder.parameters():
            param.requires_grad = False
        self.point_encoder.eval()
        print("✓ PointEncoder frozen")
        
        # Initialize projection layer
        point_dim = point_encoder_config.get("trans_dim", 384)
        projection_config = config['model'].get('projection', {})
        
        self.projection = PointToQwenProjection(
            point_dim=point_dim,
            qwen_visual_dim=self.qwen_visual_dim,
            **projection_config
        )
        
        # Load pretrained projection weights if provided
        pretrained_projector = config['model'].get('pretrained_projector')
        if pretrained_projector and os.path.exists(pretrained_projector):
            print(f"\nLoading pretrained Projection from: {pretrained_projector}")
            checkpoint = torch.load(pretrained_projector, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'projection' in checkpoint:
                state_dict = checkpoint['projection']
            elif 'model_state_dict' in checkpoint:
                # Extract only projection weights
                state_dict = {k.replace('projection.', ''): v 
                             for k, v in checkpoint['model_state_dict'].items() 
                             if k.startswith('projection.')}
            else:
                # Assume checkpoint is the state dict itself
                state_dict = checkpoint
            
            self.projection.load_state_dict(state_dict)
            print("✓ Projection weights loaded")
        
        # Move components to same device as Qwen
        qwen_device = next(self.qwen.parameters()).device
        self.point_encoder = self.point_encoder.to(qwen_device)
        self.projection = self.projection.to(qwen_device)
        
        print("\n" + "="*70)
        print("✓ PointQwenLoRA initialized successfully!")
        print(f"  PointEncoder: Frozen (pretrained)")
        print(f"  Projection: Trainable" + (f" (pretrained)" if pretrained_projector else ""))
        print(f"  Qwen: Base frozen + LoRA adapters trainable")
        print("="*70 + "\n")
    
    def _get_qwen_visual_dim(self) -> int:
        """Extract Qwen's visual token dimension."""
        try:
            if hasattr(self.qwen.config, 'text_config'):
                return self.qwen.config.text_config.hidden_size
            elif hasattr(self.qwen.config, 'vision_config'):
                return self.qwen.config.vision_config.hidden_size
            elif hasattr(self.qwen.config, 'hidden_size'):
                return self.qwen.config.hidden_size
            else:
                print("Warning: Could not determine visual dimension, using default 2560")
                return 2560
        except Exception as e:
            print(f"Warning: Error getting visual dim: {e}. Using default 2560")
            return 2560
    
    @property
    def device(self):
        """Get the device where the model is located."""
        return next(self.qwen.parameters()).device
    
    @property
    def config(self):
        """Get config from the underlying Qwen model."""
        return self.qwen.config
    
    # Required HuggingFace model methods for SFTTrainer
    def get_input_embeddings(self):
        """Get input embeddings from the underlying Qwen model."""
        embeddings = self.qwen.get_input_embeddings()
        # Add dtype attribute if missing (required by Unsloth)
        if not hasattr(embeddings, 'dtype'):
            embeddings.dtype = embeddings.weight.dtype
        return embeddings
    
    def get_output_embeddings(self):
        """Get output embeddings from the underlying Qwen model."""
        return self.qwen.get_output_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings in the underlying Qwen model."""
        self.qwen.set_input_embeddings(value)
    
    def set_output_embeddings(self, value):
        """Set output embeddings in the underlying Qwen model."""
        self.qwen.set_output_embeddings(value)
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """Resize token embeddings in the underlying Qwen model."""
        return self.qwen.resize_token_embeddings(new_num_tokens)
    
    def tie_weights(self):
        """Tie weights in the underlying Qwen model."""
        if hasattr(self.qwen, 'tie_weights'):
            self.qwen.tie_weights()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing in the underlying Qwen model."""
        if hasattr(self.qwen, 'gradient_checkpointing_enable'):
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing in the underlying Qwen model."""
        if hasattr(self.qwen, 'gradient_checkpointing_disable'):
            self.qwen.gradient_checkpointing_disable()
    
    def encode_point_cloud(self, point_cloud: torch.Tensor):
        """
        Encode point cloud to visual tokens.
        
        Args:
            point_cloud: (B, N, 3) or (B, N, C) tensor
        
        Returns:
            visual_tokens: (B, num_tokens, qwen_visual_dim)
        """
        # Point encoder is frozen, use no_grad
        with torch.no_grad():
            point_tokens, _ = self.point_encoder(point_cloud)
        
        # Project to Qwen visual space (projection is trainable)
        visual_tokens = self.projection(point_tokens)
        
        return visual_tokens
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        point_cloud: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[int] = None,
        **kwargs
    ):
        """
        Forward pass with custom embedding injection.
        
        This method handles the injection of point cloud embeddings into
        the text token sequence at the image pad token position.
        
        Args:
            input_ids: (B, L) text token ids
            attention_mask: (B, L) attention mask
            labels: (B, L) labels for language modeling loss
            point_clouds: (B, N, 3) point cloud tensor (from collator)
            point_cloud: (B, N, 3) point cloud tensor (alternative name)
            num_items_in_batch: Number of items in batch (for Unsloth gradient accumulation)
        
        Returns:
            Model outputs with loss and logits
        """
        # Accept both 'point_clouds' (from collator) and 'point_cloud' (legacy)
        if point_clouds is not None:
            point_cloud = point_clouds
        
        if point_cloud is None:
            raise ValueError("point_cloud or point_clouds must be provided")
        
        batch_size = point_cloud.shape[0]
        
        # Encode point cloud to visual tokens
        visual_tokens = self.encode_point_cloud(point_cloud)
        num_visual_tokens = visual_tokens.shape[1]
        
        # Find image pad token positions (Qwen uses token ID 151655)
        image_pad_token_id = 151655
        image_pad_token_index = (input_ids == image_pad_token_id).nonzero(as_tuple=True)
        
        # Get text embeddings
        embed_tokens = self.qwen.model.language_model.embed_tokens
        text_embeds = embed_tokens(input_ids)  # (B, L, hidden_dim)
        
        # Ensure visual tokens match text embedding dtype and device
        visual_tokens = visual_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
        
        # Replace image pad tokens with point cloud embeddings
        inputs_embeds = text_embeds.clone()
        
        new_embeds_list = []
        for batch_idx in range(batch_size):
            batch_mask = image_pad_token_index[0] == batch_idx
            if batch_mask.any():
                token_positions = image_pad_token_index[1][batch_mask]
                if len(token_positions) > 0:
                    pos = token_positions[0].item()
                    # Concatenate: text before | visual tokens | text after
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
                        visual_attn = torch.ones(num_visual_tokens, dtype=attention_mask.dtype, 
                                                device=attention_mask.device)
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
        
        # Update labels
        if labels is not None:
            new_labels_list = []
            for batch_idx in range(batch_size):
                batch_mask = image_pad_token_index[0] == batch_idx
                if batch_mask.any():
                    token_positions = image_pad_token_index[1][batch_mask]
                    if len(token_positions) > 0:
                        pos = token_positions[0].item()
                        visual_labels = torch.full((num_visual_tokens,), fill_value=-100, 
                                                   dtype=labels.dtype, device=labels.device)
                        new_labels_batch = torch.cat([
                            labels[batch_idx, :pos],
                            visual_labels,
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
        
        # Forward through Qwen with custom embeddings
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            **kwargs
        )
        
        # Apply Unsloth's gradient accumulation scaling if num_items_in_batch is provided
        # This improves gradient accuracy when using gradient accumulation
        if num_items_in_batch is not None and outputs.loss is not None:
            # Scale loss by the ratio of actual batch size to accumulated batch size
            # This ensures gradients are properly weighted across accumulation steps
            outputs.loss = outputs.loss * (batch_size / num_items_in_batch)
        
        return outputs
    
    def generate(
        self,
        point_cloud: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 512,
        **kwargs
    ):
        """
        Generate text conditioned on point cloud and prompt.
        
        Args:
            point_cloud: (B, N, 3) point cloud tensor
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation arguments
        
        Returns:
            Generated text strings
        """
        batch_size = point_cloud.shape[0]
        
        # Encode point cloud
        visual_tokens = self.encode_point_cloud(point_cloud)
        num_visual_tokens = visual_tokens.shape[1]
        
        # Tokenize prompt
        image_pad_token_id = 151655
        inputs = self.processor(
            text=[prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Find image pad token positions
        image_pad_token_index = (input_ids == image_pad_token_id).nonzero(as_tuple=True)
        
        # Get text embeddings
        embed_tokens = self.qwen.model.language_model.embed_tokens
        text_embeds = embed_tokens(input_ids)
        
        # Match dtype
        visual_tokens = visual_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
        
        # Inject visual tokens
        inputs_embeds = text_embeds.clone()
        
        new_embeds_list = []
        for batch_idx in range(batch_size):
            batch_mask = image_pad_token_index[0] == batch_idx
            if batch_mask.any():
                token_positions = image_pad_token_index[1][batch_mask]
                if len(token_positions) > 0:
                    pos = token_positions[0].item()
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
                        visual_attn = torch.ones(num_visual_tokens, dtype=attention_mask.dtype,
                                                device=attention_mask.device)
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
        
        # Generate - bypass Unsloth's broken generate wrapper
        # Use the base model directly to avoid Unsloth's generate bugs
        try:
            # Try to access the underlying base model (without Unsloth wrapper)
            if hasattr(self.qwen, 'model'):
                # This is the actual model under PEFT/Unsloth wrappers
                base_model = self.qwen.model
            elif hasattr(self.qwen, 'base_model'):
                base_model = self.qwen.base_model.model
            else:
                base_model = self.qwen
            
            generated_ids = base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                temperature=kwargs.get('temperature', 0.1) if kwargs.get('do_sample', False) else None,
                do_sample=kwargs.get('do_sample', False),
                top_p=kwargs.get('top_p', 0.9) if kwargs.get('do_sample', False) else None,
                top_k=kwargs.get('top_k', 50) if kwargs.get('do_sample', False) else None,
                repetition_penalty=kwargs.get('repetition_penalty', 1.0),
                no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 0),
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
        
        # Decode
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Clean up intermediate tensors to prevent memory leaks
        del inputs_embeds, extended_attention_mask, visual_tokens, text_embeds, input_ids, attention_mask
        del new_embeds_list, inputs, generated_ids
        if 'new_attention_mask_list' in locals():
            del new_attention_mask_list
        
        return generated_texts
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        print("\n" + "=" * 60)
        print("TRAINABLE PARAMETERS")
        print("=" * 60)
        
        # Point encoder
        point_encoder_params = sum(p.numel() for p in self.point_encoder.parameters() if p.requires_grad)
        print(f"{'PointEncoder':20} {point_encoder_params:>15,} params (frozen)")
        
        # Projection
        projection_params = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
        print(f"{'Projection':20} {projection_params:>15,} params")
        
        # Qwen + LoRA
        qwen_trainable = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        qwen_total = sum(p.numel() for p in self.qwen.parameters())
        print(f"{'Qwen (LoRA)':20} {qwen_trainable:>15,} params")
        print(f"{'Qwen (total)':20} {qwen_total:>15,} params")
        
        total = point_encoder_params + projection_params + qwen_trainable
        print("-" * 60)
        print(f"{'TRAINABLE TOTAL':20} {total:>15,} params")
        print("=" * 60 + "\n")


class LoRATrainer:
    """Trainer wrapper for PointQwenLoRA using SFTTrainer."""
    
    def __init__(
        self,
        config: Dict,
        model: PointQwenLoRA,
        train_dataset,
        val_dataset: Optional = None,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            # Initialize wandb first to get the run name
            wandb.init(
                project="pointqwen",
                config=config,
                name=config.get('run_name', 'pointqwen_lora_training'),
                tags=["LoRA", "Scans"]
            )
            
            # Update output_dir to use wandb run name
            self.wandb_run_name = wandb.run.name
            self._update_output_dir_with_wandb_name()
        
        # Create SFT trainer
        self.trainer = self._create_trainer()
        
        print("\n" + "=" * 60)
        print("TRAINER INITIALIZED")
        print("=" * 60)
        self.model.print_trainable_parameters()
    
    def _update_output_dir_with_wandb_name(self):
        """Update output directory to use wandb run name, handle conflicts."""
        base_output_dir = self.config['training'].get('output_dir', 'outputs/pointqwen_lora')
        
        # Extract the parent directory and create new path with wandb run name
        parent_dir = str(Path(base_output_dir).parent)
        new_output_dir = os.path.join(parent_dir, self.wandb_run_name)
        
        # Handle conflicts: if directory exists, add suffix
        if os.path.exists(new_output_dir):
            counter = 1
            while os.path.exists(f"{new_output_dir}_{counter}"):
                counter += 1
            new_output_dir = f"{new_output_dir}_{counter}"
            print(f"\n⚠️  Output directory conflict resolved: using '{new_output_dir}'")
        
        # Update config
        self.config['training']['output_dir'] = new_output_dir
        
        print(f"\n📁 Output directory set to: {new_output_dir}")
        print(f"   (based on wandb run: {self.wandb_run_name})\n")
    
    def _create_trainer(self) -> CustomTrainer:
        """Create CustomTrainer with proper configuration."""
        train_config = self.config['training']
        
        # Get component-specific learning rates
        lora_lr = train_config.get('lora_lr', train_config['learning_rate'])
        projection_lr = train_config.get('projection_lr', None)
        
        # Get component-specific weight decay (defaults: projection=0.01, lora=0.0)
        projection_weight_decay = train_config.get('projection_weight_decay', 0.01)
        lora_weight_decay = train_config.get('lora_weight_decay', 0.0)
        
        # Print optimization configuration
        print("\n" + "=" * 60)
        print("OPTIMIZATION CONFIGURATION")
        print("=" * 60)
        print(f"  LoRA adapters:")
        print(f"    - Learning Rate:  {lora_lr}")
        print(f"    - Weight Decay:   {lora_weight_decay}")
        if projection_lr is not None:
            print(f"  Projection layer:")
            print(f"    - Learning Rate:  {projection_lr}")
            print(f"    - Weight Decay:   {projection_weight_decay}")
        else:
            print(f"  Projection layer:")
            print(f"    - Learning Rate:  {lora_lr} (same as LoRA)")
            print(f"    - Weight Decay:   {projection_weight_decay}")
        print("=" * 60 + "\n")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=train_config['output_dir'],
            num_train_epochs=train_config['num_epochs'],
            per_device_train_batch_size=train_config['batch_size'],
            per_device_eval_batch_size=train_config['batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            
            # Learning rate and optimization (this will be used for LoRA by default)
            learning_rate=lora_lr,
            weight_decay=lora_weight_decay,  # Default weight decay (will be overridden per component)
            warmup_steps=train_config['warmup_steps'],
            max_grad_norm=train_config['max_grad_norm'],
            
            # Optimizer
            optim=train_config.get('optimizer', 'adamw_8bit'),
            lr_scheduler_type=train_config.get('lr_scheduler', 'linear'),
            
            # Mixed precision
            bf16=train_config.get('use_bf16', True),
            fp16=train_config.get('use_fp16', False),
            
            # Logging
            logging_steps=train_config['logging_steps'],
            logging_dir=os.path.join(train_config['output_dir'], 'logs'),
            report_to="wandb" if self.use_wandb else "none",
            
            # Evaluation
            eval_strategy="steps" if self.val_dataset else "no",
            eval_steps=train_config.get('eval_steps', 5),
            save_strategy="steps" if self.val_dataset else "no",
            save_steps=train_config.get('save_steps', 25),
            save_total_limit=train_config.get('save_total_limit', 3),
            load_best_model_at_end=True if self.val_dataset else False,
            
            # Don't remove custom columns
            remove_unused_columns=False,
            
            # Misc
            seed=self.config.get('seed', 42),
        )
        
        # Create data collator
        collator = IOSCollator(self.model.processor)
        
        # Create trainer with custom collator and component-specific hyperparameters
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=collator,
            lora_trainer_wrapper=self,  # Pass reference for custom evaluation
            projection_lr=projection_lr,  # Component-specific learning rate
            projection_weight_decay=projection_weight_decay,  # Component-specific weight decay
            lora_weight_decay=lora_weight_decay,  # Component-specific weight decay
        )
        
        return trainer
    
    def train(self):
        """Run training."""
        print("\n" + "=" * 60)
        print("STARTING LORA TRAINING")
        print("=" * 60)
        
        # Train
        self.trainer.train()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        # Evaluate on validation set if available
        if self.val_dataset:
            print("\nRunning final validation...")
            self.evaluate()
        
        if self.use_wandb:
            wandb.finish()
    
    def evaluate(self):
        """Run comprehensive evaluation."""
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)
        
        self.model.eval()
        
        # Use smaller batch size for evaluation to avoid OOM during generation
        eval_batch_size = self.config['training'].get('eval_batch_size', 1)
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 during evaluation to avoid memory issues
            collate_fn=IOSCollator(self.model.processor),
        )
        
        all_predictions = []
        all_references = []
        debug_samples = []  # Store samples for detailed logging
        
        max_samples = self.config['training'].get('max_samples_for_metrics', 50)
        num_debug_samples = min(1, max_samples) if max_samples else 3  # Show first 3 samples in detail
        
        # Clear cache before evaluation
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
                if max_samples and len(all_predictions) >= max_samples:
                    break
                
                point_clouds = batch['point_clouds'].to(self.model.device)
                batch_size = point_clouds.shape[0]
                
                # Generate for each sample
                for j in range(batch_size):
                    if max_samples and len(all_predictions) >= max_samples:
                        break
                    
                    # Create a copy to avoid keeping reference to the parent tensor
                    point_cloud = point_clouds[j:j+1].clone()
                    prompt = batch['instructions'][j]
                    ground_truth = batch['descriptions'][j]
                    patient_id = batch['patient_ids'][j]
                    
                    try:
                        gen_config = self.config['training'].get('generation', {})
                        
                        generated_texts = self.model.generate(
                            point_cloud=point_cloud,
                            prompt=prompt,
                            max_new_tokens=self.config['training'].get('val_max_new_tokens', 512),
                            temperature=gen_config.get('temperature', 0.1),
                            top_p=gen_config.get('top_p', 0.9),
                            top_k=gen_config.get('top_k', 1),
                            do_sample=gen_config.get('do_sample', False),
                            repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                            no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
                        )
                        
                        generated_text = generated_texts[0]
                        
                        # Extract generated part
                        if prompt in generated_text:
                            generated_description = generated_text.split(prompt)[-1].strip()
                        else:
                            generated_description = generated_text.strip()
                        
                        clean_ground_truth = ground_truth.replace("<|im_end|>", "").strip()
                        
                        all_predictions.append(generated_description)
                        all_references.append(clean_ground_truth)
                        
                        # Store debug samples
                        if len(debug_samples) < num_debug_samples:
                            debug_samples.append({
                                'patient_id': patient_id,
                                'prompt': prompt,
                                'ground_truth': clean_ground_truth,
                                'generated': generated_description,
                                'full_output': generated_text
                            })
                        
                    except Exception as e:
                        print(f"\nWarning: Error during generation: {e}")
                        traceback.print_exc()
                    finally:
                        # Clean up GPU memory after each generation
                        del point_cloud
                        torch.cuda.empty_cache()
                
                # Clean up batch tensors
                del point_clouds, batch
                torch.cuda.empty_cache()
        
        # Print debug samples
        if debug_samples:
            print("\n" + "=" * 80)
            print("DEBUG: SAMPLE GENERATIONS")
            print("=" * 80)
            
            for idx, sample in enumerate(debug_samples, 1):
                print(f"\n{'─' * 80}")
                print(f"SAMPLE {idx}: Patient {sample['patient_id']}")
                print(f"{'─' * 80}")
                
                print(f"\n📝 PROMPT (INSTRUCTION):")
                print(f"{'─' * 80}")
                print(sample['prompt'])
                
                print(f"\n✅ GROUND TRUTH:")
                print(f"{'─' * 80}")
                print(sample['ground_truth'])
                
                print(f"\n🤖 GENERATED OUTPUT:")
                print(f"{'─' * 80}")
                print(sample['generated'])
                
                print(f"\n📄 FULL MODEL OUTPUT (with prompt):")
                print(f"{'─' * 80}")
                print(sample['full_output'][:500] + "..." if len(sample['full_output']) > 500 else sample['full_output'])
                print(f"{'─' * 80}")
            
            print("\n" + "=" * 80)
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb_table_data = []
                for sample in debug_samples:
                    wandb_table_data.append([
                        sample['patient_id'],
                        sample['prompt'],
                        sample['ground_truth'],
                        sample['generated']
                    ])
                
                wandb.log({
                    "validation_samples": wandb.Table(
                        columns=["Patient ID", "Prompt", "Ground Truth", "Generated"],
                        data=wandb_table_data
                    )
                })
        
        # Compute metrics
        if len(all_predictions) > 0:
            print(f"\nComputing metrics on {len(all_predictions)} samples...")
            
            try:
                metrics_dict = compute_all_metrics(
                    predictions=all_predictions,
                    references=all_references,
                    include_sbert=self.config['training'].get('compute_sbert', True),
                    sbert_device=str(self.model.device),
                    return_per_field=self.config['training'].get('return_per_field_accuracy', True)
                )
                
                # Print metrics
                print(f"\n{'Metric':<25} {'Score':>10}")
                print(f"{'-'*35}")
                print(f"{'Field Accuracy':<25} {metrics_dict.get('accuracy', 0.0):>10.3f}")
                print(f"{'Field Coverage':<25} {metrics_dict.get('coverage', 0.0):>10.3f}")
                print(f"{'BLEU-1':<25} {metrics_dict.get('bleu-1', 0.0):>10.3f}")
                print(f"{'ROUGE-L (F1)':<25} {metrics_dict.get('rouge-l-f', 0.0):>10.3f}")
                print(f"{'METEOR':<25} {metrics_dict.get('meteor', 0.0):>10.3f}")
                if 'sbert-sim' in metrics_dict:
                    print(f"{'Sentence-BERT':<25} {metrics_dict.get('sbert-sim', 0.0):>10.3f}")
                
                if 'per_field' in metrics_dict:
                    print(f"\n{'Per-Field Accuracy':^35}")
                    print(f"{'-'*35}")
                    for field, score in sorted(metrics_dict['per_field'].items()):
                        field_name = field.replace('_', ' ').title()
                        print(f"  {field_name:<23} {score:>10.3f}")
                
                print(f"{'='*35}\n")
                
                # Log to wandb
                if self.use_wandb:
                    log_dict = {}
                    for key, value in metrics_dict.items():
                        if key not in ['per_field'] and isinstance(value, (int, float)):
                            log_dict[f'eval/{key}'] = value
                    
                    if 'per_field' in metrics_dict:
                        for field, score in metrics_dict['per_field'].items():
                            log_dict[f'eval/field_{field}'] = score
                    
                    wandb.log(log_dict)
                
            except Exception as e:
                print(f"\nWarning: Error computing metrics: {e}")
                traceback.print_exc()
        
        # Clean up all evaluation data
        del all_predictions, all_references, debug_samples
        if 'val_loader' in locals():
            del val_loader
        
        # Force garbage collection and clear CUDA cache multiple times
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        # Put model back in training mode
        self.model.train()
        
        # Clear any cached states in the model (KV cache, etc.)
        if hasattr(self.model, 'qwen'):
            # Clear any generation-related caches
            for module in self.model.qwen.modules():
                if hasattr(module, '_cache'):
                    try:
                        del module._cache
                    except:
                        pass
                if hasattr(module, 'past_key_values'):
                    try:
                        del module.past_key_values
                    except:
                        pass
        
        # Additional cleanup after switching to training mode
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n✅ Evaluation complete, memory cleaned, resuming training...")
        # return average accuracy
        return metrics_dict.get('accuracy', 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train PointQwen with LoRA")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for wandb')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['use_wandb'] = args.use_wandb
    if args.run_name:
        config['run_name'] = args.run_name
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print("\n" + "="*70)
    print("INITIALIZING POINTQWEN WITH LORA")
    print("="*70)
    
    # Initialize Qwen with LoRA using Unsloth
    print("\nLoading Qwen3-VL with LoRA...")
    lora_config = config['model']['lora']
    
    model, tokenizer = FastVisionModel.from_pretrained(
        config['model']['qwen_model_name'],
        load_in_4bit=lora_config.get('use_4bit', False),
        load_in_8bit=False,
        use_gradient_checkpointing=config['model'].get('use_gradient_checkpointing', True),
        max_seq_length=config['training'].get('max_seq_length', 2048),
        dtype=None,  # Auto
        device_map="auto",
    )
    
    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config.get('finetune_vision_layers', False),
        finetune_language_layers=lora_config.get('finetune_language_layers', True),
        finetune_attention_modules=lora_config.get('finetune_attention_modules', True),
        finetune_mlp_modules=lora_config.get('finetune_mlp_modules', True),
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0.0),
        bias=lora_config.get('bias', 'none'),
        random_state=config.get('seed', 3407),
        use_rslora=lora_config.get('use_rslora', False),
    )
    
    # Enable training mode
    FastVisionModel.for_training(model)
    
    # Create PointQwenLoRA wrapper
    print("\nInitializing PointQwenLoRA...")
    processor = AutoProcessor.from_pretrained(config['model']['qwen_model_name'])
    pointqwen_lora = PointQwenLoRA(
        config=config,
        qwen_lora_model=model,
        processor=processor,
    )
    
    # Load data
    print("\nLoading data...")
    
    # Create base dataset
    base_dataset = IOSPointCloudDataset(
        point_cloud_dir=config['data']['point_cloud_dir'],
        captions_dir=config['data']['captions_dir'],
        num_points=config['data']['num_points'],
        normalize=config['data']['normalize'],
        augment=False,
        instruction_template_path=config['data'].get('instruction_template_path')
    )
    
    # Split dataset
    val_split = config['training'].get('validation_split', 0.1)
    total_size = len(base_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    np.random.seed(config['seed'])
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    
    # Create train dataset with augmentation
    train_dataset_full = IOSPointCloudDataset(
        point_cloud_dir=config['data']['point_cloud_dir'],
        captions_dir=config['data']['captions_dir'],
        num_points=config['data']['num_points'],
        normalize=config['data']['normalize'],
        augment=config['data']['augment'],
        instruction_template_path=config['data'].get('instruction_template_path')
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(base_dataset, val_indices)
    
    del base_dataset  # Free memory
    
    print(f"Split dataset: {train_size} train, {val_size} validation samples")
    
    # Create trainer
    trainer = LoRATrainer(
        config=config,
        model=pointqwen_lora,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
