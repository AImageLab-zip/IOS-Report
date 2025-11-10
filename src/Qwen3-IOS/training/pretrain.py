"""
Pretraining script for PointEncoder to match Qwen visual embeddings.

This script trains the PointEncoder to predict the concatenated Qwen visual embeddings
from the five intraoral photo views, using only the 3D point cloud as input.

Training objective: point cloud → embedding matching (supervised learning)
Loss function: MSE loss between predicted and target embeddings
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from typing import Dict, Optional, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.point_encoder import PointEncoder
from data.embedding_dataset import IOSEmbeddingDataset


class EmbeddingPretrainer:
    """Pretrainer for PointEncoder to match Qwen embeddings."""
    
    def __init__(
        self,
        config: Dict,
        model: PointEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # View order for concatenation: [center, up, down, left, right]
        self.view_order = ["center", "up", "down", "left", "right"]
        
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            run_name = f"pretrain_pointencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="pointqwen-pretrain",
                config=config,
                name=run_name
            )
        
        print(f"Pretrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"  Validation samples: {len(self.val_loader.dataset)}")
        print(f"  Output directory: {self.output_dir}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        train_config = self.config['training']
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            betas=(train_config['adam_beta1'], train_config['adam_beta2']),
            eps=train_config['adam_epsilon'],
            weight_decay=train_config['weight_decay']
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        train_config = self.config['training']
        
        num_epochs = train_config['num_epochs']
        num_training_steps = len(self.train_loader) * num_epochs
        
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=train_config['learning_rate'] * 0.01
        )
        
        return scheduler
    
    def concatenate_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate embeddings in the specified order.
        
        Args:
            embeddings: Dict with keys for each view, values are (B, 256, 2560) tensors
        
        Returns:
            Concatenated tensor of shape (B, 1280, 2560)
        """
        batch_size = None
        embedding_list = []
        
        for view_name in self.view_order:
            if view_name in embeddings and embeddings[view_name] is not None:
                emb = embeddings[view_name]
                if batch_size is None:
                    batch_size = emb.shape[0]
                embedding_list.append(emb)
            else:
                # If view is missing, create zero tensor
                print(f"  Warning: Missing embedding for view '{view_name}', using zeros.")
                if batch_size is not None and len(embedding_list) > 0:
                    zero_emb = torch.zeros_like(embedding_list[0])
                    embedding_list.append(zero_emb)
        
        if len(embedding_list) == 0:
            raise ValueError("No embeddings found in batch")
        
        # Concatenate along sequence dimension: (B, 256, 2560) * 5 -> (B, 1280, 2560)
        concatenated = torch.cat(embedding_list, dim=1)
        
        return concatenated
    
    def compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss between predicted and target embeddings.
        
        Args:
            predicted: (B, 1280, 2560) predicted embeddings
            target: (B, 1280, 2560) target embeddings
        
        Returns:
            Dict with 'loss' and additional metrics
        """
        # Primary loss: MSE loss
        # This preserves both direction AND magnitude of embeddings
        mse_loss = F.mse_loss(predicted, target)
        
        # Compute cosine similarity as a monitoring metric (not used for optimization)
        with torch.no_grad():
            # Normalize along the feature dimension
            pred_norm = F.normalize(predicted, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Compute cosine similarity
            cos_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        
        # Use MSE loss only to preserve magnitude information
        total_loss = mse_loss
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'cosine_similarity': cos_sim
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_cos_sim = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get point clouds and embeddings
            point_clouds = batch['point_cloud'].to(self.device)  # (B, N, 3)
            embeddings = batch['embeddings']  # Dict of (B, 256, 2560) tensors
            
            # Convert point clouds to model dtype (bfloat16 if model is bfloat16)
            model_dtype = next(self.model.parameters()).dtype
            if model_dtype != torch.float32:
                point_clouds = point_clouds.to(dtype=model_dtype)
            
            # Move embeddings to device and concatenate
            embeddings_device = {
                k: v.to(self.device) if v is not None else None
                for k, v in embeddings.items()
            }
            target_embeddings = self.concatenate_embeddings(embeddings_device)  # (B, 1280, 2560)
            
            # Forward pass
            predicted_embeddings, _ = self.model(point_clouds, return_deepstack=False)
            # predicted_embeddings: (B, 1280, output_dim=2560)
            
            # Compute loss
            loss_dict = self.compute_loss(predicted_embeddings, target_embeddings)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mse += loss_dict['mse_loss'].item()
            total_cos_sim += loss_dict['cosine_similarity'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cos_sim': f"{loss_dict['cosine_similarity'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config['training']['logging_steps'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/mse_loss': loss_dict['mse_loss'].item(),
                    'train/cosine_similarity': loss_dict['cosine_similarity'].item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/epoch': self.current_epoch,
                    'train/step': self.global_step
                })
        
        # Compute average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'mse_loss': total_mse / num_batches,
            'cosine_similarity': total_cos_sim / num_batches
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        total_cos_sim = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            # Get point clouds and embeddings
            point_clouds = batch['point_cloud'].to(self.device)
            embeddings = batch['embeddings']
            
            # Convert point clouds to model dtype (bfloat16 if model is bfloat16)
            model_dtype = next(self.model.parameters()).dtype
            if model_dtype != torch.float32:
                point_clouds = point_clouds.to(dtype=model_dtype)
            
            # Move embeddings to device and concatenate
            embeddings_device = {
                k: v.to(self.device) if v is not None else None
                for k, v in embeddings.items()
            }
            target_embeddings = self.concatenate_embeddings(embeddings_device)
            
            # Forward pass
            predicted_embeddings, _ = self.model(point_clouds, return_deepstack=False)
            
            # Compute loss
            loss_dict = self.compute_loss(predicted_embeddings, target_embeddings)
            
            # Update metrics
            total_loss += loss_dict['loss'].item()
            total_mse += loss_dict['mse_loss'].item()
            total_cos_sim += loss_dict['cosine_similarity'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'cos_sim': f"{loss_dict['cosine_similarity'].item():.4f}"
            })
        
        # Compute average metrics
        avg_metrics = {
            'val_loss': total_loss / num_batches,
            'val_mse_loss': total_mse / num_batches,
            'val_cosine_similarity': total_cos_sim / num_batches
        }
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        
        # Save only the PointEncoder weights
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_path = self.output_dir / "best_point_encoder.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        save_steps = self.config['training']['save_steps']
        
        print("\n" + "="*80)
        print("Starting pretraining...")
        print("="*80 + "\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train MSE: {train_metrics['mse_loss']:.4f}")
            print(f"  Train Cosine Similarity: {train_metrics['cosine_similarity']:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val MSE: {val_metrics['val_mse_loss']:.4f}")
                print(f"  Val Cosine Similarity: {val_metrics['val_cosine_similarity']:.4f}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics
                    })
                
                # Save best model
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                    print(f"  New best validation loss: {self.best_val_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % (num_epochs // 10) == 0 or is_best:
                    self.save_checkpoint(
                        f"checkpoint_epoch_{epoch+1}.pth",
                        is_best=is_best
                    )
            else:
                # No validation set
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics
                    })
                
                # Save checkpoint periodically
                if (epoch + 1) % (num_epochs // 10) == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
            
            print()
        
        # Save final model
        self.save_checkpoint("final_point_encoder.pth")
        
        print("\n" + "="*80)
        print("Pretraining completed!")
        print("="*80 + "\n")
        
        if self.use_wandb:
            wandb.finish()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict):
    """Create train and validation dataloaders."""
    data_config = config['data']
    train_config = config['training']
    
    # Create full dataset
    full_dataset = IOSEmbeddingDataset(
        point_cloud_dir=data_config['point_cloud_dir'],
        captions_dir=data_config['captions_dir'],
        embeddings_dir=config['data'].get(
            'embeddings_dir',
            "/work/grana_maxillo/IOS-DraftReport/_data/iop_qwen_embeddings"
        ),
        num_points=data_config['num_points'],
        normalize=data_config['normalize'],
        augment=data_config['augment'],
        require_all_views=True
    )
    
    # Split into train and validation
    val_split = train_config.get('validation_split', 0.1)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config: Dict) -> PointEncoder:
    """Create PointEncoder model with projection to match embeddings."""
    point_encoder_config = config['model']['point_encoder']
    
    model = PointEncoder(
        trans_dim=point_encoder_config['trans_dim'],
        depth=point_encoder_config['depth'],
        num_heads=point_encoder_config['num_heads'],
        encoder_dims=point_encoder_config['encoder_dims'],
        group_size=point_encoder_config['group_size'],
        num_group=point_encoder_config['num_group'],
        drop_path_rate=point_encoder_config['drop_path_rate'],
        output_dim=2560,  # Match Qwen embedding dimension
        projection_dropout=point_encoder_config.get('projection_dropout', 0.1)
    )
    
    # Load pretrained Point-BERT weights if specified
    if 'pretrained_point_encoder' in config['model']:
        pretrained_path = config['model']['pretrained_point_encoder']
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading pretrained Point-BERT weights from {pretrained_path}")
            model.load_pretrained(pretrained_path)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Pretrain PointEncoder to match Qwen embeddings")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pretrain_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.no_wandb:
        config['use_wandb'] = False
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Convert model to bfloat16 to match embedding dtype
    # Embeddings are saved as bfloat16 from Qwen
    if torch.cuda.is_available() and config['training'].get('use_bf16', True):
        print("Converting model to bfloat16...")
        model = model.to(dtype=torch.bfloat16)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = EmbeddingPretrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
