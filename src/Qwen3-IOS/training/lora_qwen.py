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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import wandb
from transformers import AutoProcessor
from tqdm import tqdm
from typing import Dict, Optional

# Unsloth imports
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

sys.path.append(str(Path(__file__).parent.parent))

from models import PointEncoder
from models.projection import PointToQwenProjection
from data import IOSPointCloudDataset
from validation import compute_all_metrics


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
        super()</init>()
        
        self.config = config
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
        point_cloud: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with custom embedding injection.
        
        This method handles the injection of point cloud embeddings into
        the text token sequence at the image pad token position.
        
        Args:
            point_cloud: (B, N, 3) point cloud tensor
            input_ids: (B, L) text token ids
            attention_mask: (B, L) attention mask
            labels: (B, L) labels for language modeling loss
        
        Returns:
            Model outputs with loss and logits
        """
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
        
        # Generate
        generated_ids = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        # Decode
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
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
            wandb.init(
                project="pointqwen-lora",
                config=config,
                name=config.get('run_name', 'pointqwen_lora_training'),
                tags=["LoRA", "Scans"]
            )
        
        # Create SFT trainer
        self.trainer = self._create_trainer()
        
        print("\n" + "=" * 60)
        print("TRAINER INITIALIZED")
        print("=" * 60)
        self.model.print_trainable_parameters()
    
    def _create_trainer(self) -> SFTTrainer:
        """Create SFTTrainer with proper configuration."""
        train_config = self.config['training']
        
        # Create training arguments
        training_args = SFTConfig(
            output_dir=train_config['output_dir'],
            num_train_epochs=train_config['num_epochs'],
            per_device_train_batch_size=train_config['batch_size'],
            per_device_eval_batch_size=train_config['batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            
            # Learning rate and optimization
            learning_rate=train_config.get('lora_lr', train_config['learning_rate']),
            weight_decay=train_config['weight_decay'],
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
            evaluation_strategy="epoch" if self.val_dataset else "no",
            save_strategy="epoch",
            save_total_limit=train_config.get('save_total_limit', 3),
            load_best_model_at_end=True if self.val_dataset else False,
            
            # Required for vision models
            remove_unused_columns=train_config.get('remove_unused_columns', False),
            dataset_text_field=train_config.get('dataset_text_field', ""),
            dataset_kwargs=train_config.get('dataset_kwargs', {"skip_prepare_dataset": True}),
            max_seq_length=train_config.get('max_seq_length', 2048),
            
            # Misc
            seed=self.config.get('seed', 3407),
        )
        
        # Create data collator
        # We'll use a custom collator since UnslothVisionDataCollator expects specific format
        from data import IOSCollator
        collator = IOSCollator(self.model.processor)
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=collator,
            tokenizer=self.model.processor,
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
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            collate_fn=IOSCollator(self.model.processor),
        )
        
        all_predictions = []
        all_references = []
        
        max_samples = self.config['training'].get('max_samples_for_metrics', 50)
        
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
                    
                    point_cloud = point_clouds[j:j+1]
                    prompt = batch['instructions'][j]
                    ground_truth = batch['descriptions'][j]
                    
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
                        
                    except Exception as e:
                        print(f"\nWarning: Error during generation: {e}")
                        traceback.print_exc()
        
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
