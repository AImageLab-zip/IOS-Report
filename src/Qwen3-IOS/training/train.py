import os
import sys
import yaml
import argparse
from pathlib import Path
import math
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import Subset

import wandb
from transformers import get_linear_schedule_with_warmup, AutoProcessor
from tqdm import tqdm
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from models import PointQwen
from data import IOSPointCloudDataset, IOSCollator
from validation import compute_all_metrics
from losses import SemanticLoss


class Trainer:
    """Trainer for PointQwen."""
    
    def __init__(
        self,
        config: Dict,
        model: PointQwen,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = self.model.device
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize semantic loss if enabled
        self.use_semantic_loss = config['training'].get('use_semantic_loss', False)
        if self.use_semantic_loss:
            self.semantic_loss_weight = config['training'].get('semantic_loss_weight', 0.5)
            semantic_model_name = config['training'].get('semantic_loss_model', 'sentence-transformers/all-MiniLM-L6-v2')
            # Compute semantic loss only every N steps to reduce overhead
            self.semantic_loss_frequency = config['training'].get('semantic_loss_frequency', 10)
            print(f"\nInitializing Semantic Loss (weight: {self.semantic_loss_weight}, frequency: every {self.semantic_loss_frequency} steps)...")
            self.semantic_loss = SemanticLoss(
                model_name=semantic_model_name,
                device=str(self.device),
                freeze=True
            )
            # Generation config for semantic loss
            self.semantic_gen_config = config['training'].get('semantic_generation', {
                'max_new_tokens': 256,
                'temperature': 0.1,
                'top_p': 0.9,
                'top_k': 1,
                'do_sample': False,
                'repetition_penalty': 1.1,
                'no_repeat_ngram_size': 3
            })
        else:
            self.semantic_loss = None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="pointqwen",
                config=config,
                name=config.get('run_name', 'pointqwen_training'),
                tags=["Scans"]
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with component-specific learning rates."""
        train_config = self.config['training']
        
        # Get trainable parameters grouped by component
        trainable_params = self.model.get_trainable_parameters()
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        # Point encoder parameters
        if trainable_params['point_encoder']:
            point_encoder_lr = train_config.get('point_encoder_lr', train_config['learning_rate'])
            param_groups.append({
                'params': trainable_params['point_encoder'],
                'lr': point_encoder_lr,
                'name': 'point_encoder'
            })
            print(f"  Point encoder LR: {point_encoder_lr}")
        
        # Projection parameters
        if trainable_params['projection']:
            projection_lr = train_config.get('projection_lr', train_config['learning_rate'])
            param_groups.append({
                'params': trainable_params['projection'],
                'lr': projection_lr,
                'name': 'projection'
            })
            print(f"  Projection LR: {projection_lr}")
        
        # Qwen parameters (if unfrozen)
        if trainable_params['qwen']:
            qwen_lr = train_config.get('qwen_lr', train_config['learning_rate'])
            param_groups.append({
                'params': trainable_params['qwen'],
                'lr': qwen_lr,
                'name': 'qwen'
            })
            print(f"  Qwen LR: {qwen_lr}")
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        if train_config['optimizer'] == 'adamw':
            optimizer = AdamW(
                param_groups,
                betas=(train_config['adam_beta1'], train_config['adam_beta2']),
                eps=train_config['adam_epsilon'],
                weight_decay=train_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        train_config = self.config['training']
        
        num_training_steps = len(self.train_loader) * train_config['num_epochs']
        num_training_steps //= train_config.get('gradient_accumulation_steps', 1)
        warmup_steps = train_config.get('warmup_steps', 0)
        
        if train_config['lr_scheduler'] == 'cosine':
            # Cosine annealing with linear warmup
           
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                    return max(0.1, 0.5 * (1.0 + math.cos(progress * math.pi)))
            
            scheduler = LambdaLR(self.optimizer, lr_lambda)
            
        elif train_config['lr_scheduler'] == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _print_learning_rate_config(self):
        """Print learning rate configuration."""
        train_config = self.config['training']
        
        print("\nLearning Rate Configuration:")
        print("-" * 60)
        print(f"  Point Encoder LR: {train_config.get('point_encoder_lr', train_config['learning_rate']):.2e}")
        print(f"  Projection LR:    {train_config.get('projection_lr', train_config['learning_rate']):.2e}")
        print(f"  Qwen LR:          {train_config.get('qwen_lr', train_config['learning_rate']):.2e}")
        print(f"  Weight Decay:     {train_config['weight_decay']}")
        print(f"  Warmup Steps:     {train_config['warmup_steps']}")
        print("-" * 60)
    
    def _handle_point_encoder_freezing(self):
        """
        Handle freezing/unfreezing of point encoder based on epoch.
        
        Strategy:
        - Epoch 0 (first epoch): Freeze point encoder, train only projection
        - Epoch 1+: Unfreeze point encoder if pretrained weights were loaded
        """
        pretrained_encoder = self.config['model'].get('pretrained_point_encoder')
        
        # Only apply this logic if pretrained encoder was loaded
        if not pretrained_encoder:
            return
        
        if self.current_epoch == -1:
            # First epoch: freeze point encoder
            if not self.model.point_encoder_frozen:
                print("\n" + "="*80)
                print("EPOCH 0: Freezing point encoder (training projection only)")
                print("="*80)
                self.model.freeze_point_encoder()
                
                # Recreate optimizer with only trainable parameters
                print("\nRecreating optimizer with component-specific learning rates:")
                self.optimizer = self._create_optimizer()
                self.scheduler = self._create_scheduler()
                
                print("\nTrainable parameters after freezing:")
                self.model.print_trainable_parameters()
                print("="*80 + "\n")
        
        elif self.current_epoch == -1:
            # Second epoch onwards: unfreeze point encoder
            if self.model.point_encoder_frozen:
                print("\n" + "="*80)
                print("EPOCH 1: Unfreezing point encoder (fine-tuning all components)")
                print("="*80)
                self.model.unfreeze_point_encoder()
                
                # Recreate optimizer to include newly unfrozen parameters
                print("\nRecreating optimizer with component-specific learning rates:")
                self.optimizer = self._create_optimizer()
                self.scheduler = self._create_scheduler()
                
                print("\nTrainable parameters after unfreezing:")
                self.model.print_trainable_parameters()
                print("="*80 + "\n")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        # Handle point encoder freezing/unfreezing based on epoch
        self._handle_point_encoder_freezing()
        
        self.model.train()
        
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_semantic_loss = 0.0
        train_config = self.config['training']
        grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        self.optimizer.zero_grad()
        
        debug_first_batch = (self.current_epoch == 0)
        
        for step, batch in enumerate(progress_bar):

            if debug_first_batch and step == 0:
                self._debug_batch(batch)
                debug_first_batch = False

            point_clouds = batch['point_clouds'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                point_cloud=point_clouds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            ce_loss = outputs.loss
            total_loss = ce_loss
            semantic_loss_value = torch.tensor(0.0, device=self.device)
            
            # Compute semantic loss if enabled and at the right frequency
            if self.use_semantic_loss and (step % self.semantic_loss_frequency == 0):
                try:
                    # Set model to eval mode for generation and clear cache
                    self.model.eval()
                    torch.cuda.empty_cache()
                    
                    # Generate predictions for semantic loss
                    with torch.no_grad():
                        predictions = []
                        for i in range(point_clouds.shape[0]):
                            point_cloud = point_clouds[i:i+1].detach().clone()
                            prompt = batch['instructions'][i]
                            
                            generated_texts = self.model.generate(
                                point_cloud=point_cloud,
                                prompt=prompt,
                                **self.semantic_gen_config
                            )
                            
                            generated_text = generated_texts[0]
                            
                            # Extract only the generated part
                            if prompt in generated_text:
                                generated_description = generated_text.split(prompt)[-1].strip()
                            else:
                                generated_description = generated_text.strip()
                            
                            predictions.append(generated_description)
                    
                    # Set model back to train mode
                    self.model.train()
                    
                    # Get ground truth descriptions
                    targets = [desc.replace("<|im_end|>", "").strip() for desc in batch['descriptions']]
                    
                    # Compute semantic loss
                    semantic_loss_value = self.semantic_loss(predictions, targets)
                    total_loss = ce_loss + self.semantic_loss_weight * semantic_loss_value
                    
                except Exception as e:
                    # If semantic loss computation fails, fall back to CE loss only
                    print(f"\nWarning: Semantic loss computation failed at step {step}: {e}")
                    total_loss = ce_loss
                    # Set model back to train mode in case of error
                    self.model.train()

            # Normalize loss for gradient accumulation
            loss = total_loss / grad_accum_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_config['max_grad_norm'])

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            if self.use_semantic_loss and (step % self.semantic_loss_frequency == 0):
                epoch_semantic_loss += semantic_loss_value.item()
            
            # Update progress bar
            if self.use_semantic_loss:
                progress_bar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'ce': f'{ce_loss.item():.4f}',
                    'sem': f'{semantic_loss_value.item():.4f}'
                })
            else:
                progress_bar.set_postfix(loss=ce_loss.item())
            
            # Log to wandb
            if self.use_wandb and step % train_config['logging_steps'] == 0:
                log_dict = {
                    'train/total_loss': total_loss.item(),
                    'train/ce_loss': ce_loss.item(),
                    'train/epoch': self.current_epoch,
                    'train/step': self.global_step
                }
                
                if self.use_semantic_loss and (step % self.semantic_loss_frequency == 0):
                    log_dict['train/semantic_loss'] = semantic_loss_value.item()
                
                # Log learning rates for each parameter group
                for i, param_group in enumerate(self.optimizer.param_groups):
                    group_name = param_group.get('name', f'group_{i}')
                    log_dict[f'train/lr_{group_name}'] = param_group['lr']
                
                wandb.log(log_dict)
        
        avg_total_loss = epoch_loss / len(self.train_loader)
        avg_ce_loss = epoch_ce_loss / len(self.train_loader)
        
        if self.use_semantic_loss:
            # Count how many times semantic loss was computed
            num_semantic_steps = (len(self.train_loader) + self.semantic_loss_frequency - 1) // self.semantic_loss_frequency
            avg_semantic_loss = epoch_semantic_loss / max(num_semantic_steps, 1)
            return avg_total_loss, avg_ce_loss, avg_semantic_loss
        else:
            return avg_total_loss
    
    @torch.no_grad()
    def validate(self, generate_sample: Optional[bool] = None, compute_metrics: Optional[bool] = None) -> float:
        """Run validation with comprehensive metrics.
        
        Args:
            generate_sample: If True, generate text for first sample. If None, uses config value.
            compute_metrics: If True, compute generation metrics (BLEU, ROUGE, etc.). If None, uses config value.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        
        val_loss = 0.0
        
        # Get settings from config if not specified
        if generate_sample is None:
            generate_sample = self.config['training'].get('generate_sample_during_val', False)
        if compute_metrics is None:
            compute_metrics = self.config['training'].get('compute_generation_metrics', False)
        
        # Collect predictions and references for metrics
        all_predictions = []
        all_references = []
        
        # Control how many samples to generate for metrics (can be expensive)
        max_samples_for_metrics = self.config['training'].get('max_samples_for_metrics', None)
        
        for i, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            # Move batch to device
            point_clouds = batch['point_clouds'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass for loss computation
            outputs = self.model(
                point_cloud=point_clouds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
            
            # Generate text for metrics computation
            if compute_metrics:
                # Check if we've reached the limit
                if max_samples_for_metrics is not None and len(all_predictions) >= max_samples_for_metrics:
                    continue
                
                # Generate for all samples in batch (or just first few if limited)
                batch_size = point_clouds.shape[0]
                samples_to_generate = min(batch_size, max_samples_for_metrics - len(all_predictions)) if max_samples_for_metrics else batch_size
                
                for j in range(samples_to_generate):
                    point_cloud = point_clouds[j:j+1]  # Keep batch dim
                    prompt = batch['instructions'][j]
                    ground_truth = batch['descriptions'][j]
                    
                    try:
                        # Get generation config with defaults
                        gen_config = self.config['training'].get('generation', {})
                        
                        generated_texts = self.model.generate(
                            point_cloud=point_cloud,
                            prompt=prompt,
                            max_new_tokens=self.config['training'].get('val_max_new_tokens', 512),
                            temperature=gen_config.get('temperature', 0.1),
                            top_p=gen_config.get('top_p', 0.9),
                            top_k=gen_config.get('top_k', 1),
                            do_sample=gen_config.get('do_sample', False),
                            repetition_penalty=gen_config.get('repetition_penalty', 1.0),
                            no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
                        )
                        
                        generated_text = generated_texts[0]
                        
                        # Extract only the generated part (after the prompt)
                        if prompt in generated_text:
                            generated_description = generated_text.split(prompt)[-1].strip()
                        else:
                            generated_description = generated_text.strip()
                        
                        # Clean up the ground truth (remove end token if present)
                        clean_ground_truth = ground_truth.replace("<|im_end|>", "").strip()
                        
                        all_predictions.append(generated_description)
                        all_references.append(clean_ground_truth)
                        
                    except Exception as e:
                        print(f"\nWarning: Error during generation for sample {i}-{j}: {e}")
            
            # Generate and display first sample
            if i == 0 and generate_sample:
                self._generate_and_display_sample(batch)
        
        val_loss /= len(self.val_loader)
        
        print(f"\nValidation Loss (NLL): {val_loss:.4f}")
        
        # Compute generation metrics if requested
        metrics_dict = {}
        if compute_metrics and len(all_predictions) > 0:
            print(f"\n{'='*80}")
            print(f"COMPUTING GENERATION METRICS ({len(all_predictions)} samples)")
            print(f"{'='*80}")
            
            try:
                # Compute all metrics
                # Set include_sbert based on config (can be slow)
                include_sbert = self.config['training'].get('compute_sbert', True)
                return_per_field = self.config['training'].get('return_per_field_accuracy', True)
                
                metrics_dict = compute_all_metrics(
                    predictions=all_predictions,
                    references=all_references,
                    include_sbert=include_sbert,
                    sbert_device=str(self.device),
                    return_per_field=return_per_field
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
                
                # Print per-field accuracy if available
                if 'per_field' in metrics_dict:
                    print(f"\n{'Per-Field Accuracy':^35}")
                    print(f"{'-'*35}")
                    for field, score in sorted(metrics_dict['per_field'].items()):
                        field_name = field.replace('_', ' ').title()
                        print(f"  {field_name:<23} {score:>10.3f}")
                
                print(f"{'='*80}\n")
                
            except Exception as e:
                print(f"\nWarning: Error computing metrics: {e}")
                traceback.print_exc()
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {
                'val/loss': val_loss,
                'val/step': self.global_step
            }
            
            # Add generation metrics to wandb
            if metrics_dict:
                for key, value in metrics_dict.items():
                    if key not in ['per_field'] and isinstance(value, (int, float)):
                        log_dict[f'val/{key}'] = value
                
                # Log per-field metrics separately
                if 'per_field' in metrics_dict:
                    for field, score in metrics_dict['per_field'].items():
                        log_dict[f'val/field_{field}'] = score
            
            wandb.log(log_dict)
        
        return val_loss
    
    def _generate_and_display_sample(self, batch):
        """Generate and display text for a sample."""
        print("\n" + "="*80)
        print("GENERATED SAMPLE OUTPUT")
        print("="*80)
        
        # Take first sample from batch
        point_cloud = batch['point_clouds'][0:1].to(self.device)  # Keep batch dim
        prompt = batch['instructions'][0] if 'instructions' in batch else ""
        ground_truth = batch['descriptions'][0] if 'descriptions' in batch else ""
        patient_id = batch['patient_ids'][0] if 'patient_ids' in batch else "Unknown"
        
        print(f"\nPatient ID: {patient_id}")
        print(f"\nPrompt:\n{'-'*80}\n{prompt}")
        
        # Generate
        try:
            # Get generation config with defaults
            gen_config = self.config['training'].get('generation', {})
            
            generated_texts = self.model.generate(
                point_cloud=point_cloud,
                prompt=prompt,
                max_new_tokens=self.config['training'].get('val_max_new_tokens', 256),
                temperature=gen_config.get('temperature', 0.7),
                top_p=gen_config.get('top_p', 0.9),
                top_k=gen_config.get('top_k', 50),
                do_sample=True,
                repetition_penalty=gen_config.get('repetition_penalty', 1.2),
                no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
            )
            
            generated_text = generated_texts[0]
            
            # Extract only the generated part (after the prompt)
            if prompt in generated_text:
                generated_description = generated_text.split(prompt)[-1].strip()
            else:
                generated_description = generated_text.strip()
            
            print(f"\nGround Truth:\n{'-'*80}\n{ground_truth}")
            print(f"\nGenerated:\n{'-'*80}\n{generated_description}")
            
        except Exception as e:
            print(f"\nError during generation: {e}")
        
        print("="*80 + "\n")
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config['training']['num_epochs']
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        self._print_learning_rate_config()
        self.model.print_trainable_parameters()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            # Train epoch
            train_result = self.train_epoch()
            
            # Handle different return formats
            if self.use_semantic_loss:
                train_loss, ce_loss, semantic_loss = train_result
                print(f"\nEpoch {epoch} - Total Loss: {train_loss:.4f} (CE: {ce_loss:.4f}, Semantic: {semantic_loss:.4f})")
            else:
                train_loss = train_result
                print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            run_epoch_validation = self.config['training'].get('validate_per_epoch', False)
            if self.val_loader is not None and run_epoch_validation:
                val_loss = self.validate()
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model')
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), checkpoint_path / 'model.pt')
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }, checkpoint_path / 'training_state.pt')
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _debug_batch(self, batch):
        """Debug batch to verify input and loss masking."""
        print("\n" + "="*80)
        print("DEBUG: First Training Batch")
        print("="*80)
        
        sample_idx = 0
        input_ids = batch['input_ids'][sample_idx]
        labels = batch['labels'][sample_idx]
        patient_id = batch['patient_ids'][sample_idx]
        
        print(f"\nPatient ID: {patient_id}")
        print(f"Point cloud shape: {batch['point_clouds'][sample_idx].shape}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Decode full text
        processor = AutoProcessor.from_pretrained(self.config['model']['qwen_model_name'])
        full_text = processor.decode(input_ids, skip_special_tokens=False)
        
        print(f"\n{'Full Input Text (first 500 chars)':=^80}")
        print(full_text[:500] + "...")
        
        # Loss masking stats
        loss_computed_mask = labels != -100
        num_loss_tokens = loss_computed_mask.sum().item()
        num_masked_tokens = (labels == -100).sum().item()
        total_tokens = len(labels)
        
        print(f"\n{'Loss Masking Stats':=^80}")
        print(f"Total tokens: {total_tokens}")
        print(f"Loss computed on: {num_loss_tokens} tokens ({100*num_loss_tokens/total_tokens:.1f}%)")
        print(f"Masked tokens: {num_masked_tokens} tokens ({100*num_masked_tokens/total_tokens:.1f}%)")
        
        # Show masked vs unmasked portions
        masked_positions = (labels == -100).nonzero(as_tuple=True)[0]
        if len(masked_positions) > 0:
            masked_tokens = input_ids[masked_positions]
            masked_text = processor.decode(masked_tokens, skip_special_tokens=False)
            print(f"\n{'Masked Portion':=^80}")
            print(masked_text + "...")
        
        unmasked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if len(unmasked_positions) > 0:
            unmasked_tokens = input_ids[unmasked_positions]
            unmasked_text = processor.decode(unmasked_tokens, skip_special_tokens=False)
            print(f"\n{'Unmasked Portion (loss computed)':=^80}")
            print(unmasked_text)
        
        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train PointQwen")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for wandb')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['use_wandb'] = args.use_wandb
    if args.run_name:
        config['run_name'] = args.run_name

    torch.manual_seed(config['seed'])
    
    print("Initializing model...")
    model = PointQwen(
        qwen_model_name=config['model']['qwen_model_name'],
        point_encoder_config=config['model']['point_encoder'],
        projection_type=config['model']['projection_type'],
        projection_config=config['model'].get('projection', {}),
        freeze_qwen=config['model']['freeze_qwen'],
        freeze_point_encoder=config['model']['freeze_point_encoder'],
        pretrained_point_encoder=config['model'].get('pretrained_point_encoder'),
        use_gradient_checkpointing=config['model'].get('use_gradient_checkpointing', False),
        low_cpu_mem_usage=config['model'].get('low_cpu_mem_usage', True),
        use_flash_attention=config['model'].get('use_flash_attention', False)
    )
    
    print("Loading data...")
    processor = AutoProcessor.from_pretrained(config['model']['qwen_model_name'])
    
    # Create base dataset without augmentation to identify all samples
    base_dataset = IOSPointCloudDataset(
        point_cloud_dir=config['data']['point_cloud_dir'],
        captions_dir=config['data']['captions_dir'],
        num_points=config['data']['num_points'],
        normalize=config['data']['normalize'],
        augment=False,
        instruction_template_path=config['data'].get('instruction_template_path')
    )
    
    # Split dataset into train and validation
    val_split = config['training'].get('validation_split', 0.1)
    total_size = len(base_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Generate random indices for splitting
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
        augment=config['data']['augment'],  # Enable augmentation for training
        instruction_template_path=config['data'].get('instruction_template_path')
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    
    # Create validation dataset without augmentation
    val_dataset = Subset(base_dataset, val_indices)
    
    # Create dataloaders
    collator = IOSCollator(processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collator,
        pin_memory=True
    )
    
    print(f"Split dataset: {train_size} train, {val_size} validation samples")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
