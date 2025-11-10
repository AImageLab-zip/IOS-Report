
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "src" / "Qwen3-IOS"))

from models.point_encoder import PointEncoder
from dataset.ios_classification_dataset import IOSClassificationDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionPooling(nn.Module):
    """Learnable attention pooling that learns which tokens are important"""
    def __init__(self, dim):
        super().__init__()
        self.attention_weights = nn.Linear(dim, 1)
    
    def forward(self, x):
        # x: [B, num_tokens, dim]
        attn_scores = self.attention_weights(x)  # [B, num_tokens, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize across tokens
        pooled = (x * attn_weights).sum(dim=1)  # Weighted sum -> [B, dim]
        return pooled

class MultiTaskClassificationHead(nn.Module):
    def __init__(self, input_dim, tasks_config):
        super().__init__()
        self.tasks = list(tasks_config.keys())
        self.heads = nn.ModuleDict()
        
        # Attention pooling layer (shared across all tasks)
        self.pooling = AttentionPooling(input_dim)
        
        h_dim_1 = 512
        h_dim_2 = 256
        
        for task, num_classes in tasks_config.items():
            self.heads[task] = nn.Sequential(
                nn.Linear(input_dim, h_dim_1),
                nn.BatchNorm1d(h_dim_1),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(h_dim_1, h_dim_2),
                nn.BatchNorm1d(h_dim_2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(h_dim_2, num_classes)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Use attention pooling instead of mean pooling
        x_pooled = self.pooling(x)  # [B, num_tokens, dim] -> [B, dim]
        outputs = {}
        for task in self.tasks:
            outputs[task] = self.heads[task](x_pooled)
        return outputs


class PointClassificationModel(nn.Module):
    def __init__(self, encoder, classification_head):
        super().__init__()
        self.encoder = encoder
        self.classification_head = classification_head
    
    def forward(self, points):
        embeddings, _ = self.encoder(points)
        outputs = self.classification_head(embeddings)
        return outputs

    
class Trainer:
    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = config.get('use_wandb', False)
        self.global_step = 0
        self.best_val_accuracy = 0.0
        
        checkpoint_base = Path("/work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS/outputs")
        wandb_project = config.get('wandb_project', 'ios-classification')
        wandb_run = config.get('run_name', 'default_run')
        self.checkpoint_dir = checkpoint_base / wandb_project / wandb_run
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        total_epochs = config['training']['num_epochs']
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            total_steps=total_epochs * len(train_loader),
            pct_start=warmup_epochs / total_epochs,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.max_grad_norm = config['training'].get('max_grad_norm', 1.0)
        self.focal_gamma = config['training'].get('focal_gamma', 0.0)
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {config['training']['batch_size'] * self.gradient_accumulation_steps}")
        
        if self.use_wandb:
            wandb.watch(self.model, log='all', log_freq=100)

    def compute_accuracy(self, outputs, labels, threshold=0.5):
        task_accuracies = {}
        task_class_accuracies = {}
        
        for task in outputs.keys():
            pred = outputs[task]
            target = labels[task]
            
            mask = (target >= 0).any(dim=1)
            
            if mask.sum() > 0:
                pred_masked = pred[mask]
                target_masked = target[mask]
                
                pred_binary = (torch.sigmoid(pred_masked) > threshold).float()
                
                correct = (pred_binary == target_masked).float()
                accuracy = correct.mean().item()
                
                task_accuracies[task] = accuracy
                
                per_class_acc = correct.mean(dim=0).cpu().numpy()
                task_class_accuracies[task] = per_class_acc
        
        return task_accuracies, task_class_accuracies
    
    def compute_loss(self, outputs, labels):
        total_loss = None
        task_losses = {}
        num_valid_tasks = 0
        
        for task in outputs.keys():
            pred = outputs[task]
            target = labels[task]
            
            mask = (target >= 0).any(dim=1)
            
            if mask.sum() > 0:
                pred_masked = pred[mask]
                target_masked = target[mask]
                
                bce_loss = self.criterion(pred_masked, target_masked)
                
                if self.focal_gamma > 0:
                    pt = torch.exp(-bce_loss)
                    focal_weight = (1 - pt) ** self.focal_gamma
                    loss = (focal_weight * bce_loss).mean()
                else:
                    loss = bce_loss.mean()
                
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss
                
                task_losses[task] = loss.item()
                num_valid_tasks += 1
        
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        elif num_valid_tasks > 0:
            total_loss = total_loss / num_valid_tasks
        
        return total_loss, task_losses
    
    def save_checkpoint(self, epoch, val_loss, val_accuracy, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'best_val_accuracy': self.best_val_accuracy,
            'global_step': self.global_step,
            'config': self.config
        }
        
        last_path = self.checkpoint_dir / "last.pth"
        torch.save(checkpoint, last_path)
        print(f"Saved last checkpoint to {last_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            epoch_loss = 0.0
            epoch_task_losses = {task: 0.0 for task in self.train_loader.dataset.tasks}
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                points, labels, patient_id = batch
                points = points.to(self.device)
                labels = {task: label.to(self.device) for task, label in labels.items()}
                
                # Forward pass
                outputs = self.model(points)
                loss, task_losses = self.compute_loss(outputs, labels)
                
                # Scale loss by accumulation steps
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Only step optimizer every gradient_accumulation_steps batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps  # Scale back for logging
                num_batches += 1
                
                for task in self.train_loader.dataset.tasks:
                    if task in task_losses:
                        epoch_task_losses[task] += task_losses[task]
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item() * self.gradient_accumulation_steps:.4f}")
                    
                    if self.use_wandb:
                        log_dict = {
                            'train/batch_loss': loss.item() * self.gradient_accumulation_steps,
                            'train/epoch': epoch,
                            'train/batch': batch_idx,
                            'train/learning_rate': self.optimizer.param_groups[0]['lr']
                        }
                        for task, task_loss in task_losses.items():
                            log_dict[f'train/batch_{task}_loss'] = task_loss
                        wandb.log(log_dict, step=self.global_step)
                
                self.global_step += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_task_losses = {task: loss / max(num_batches, 1) for task, loss in epoch_task_losses.items()}
            
            print(f"\nEpoch {epoch} Train Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            for task, loss in avg_task_losses.items():
                if loss > 0:
                    print(f"  {task}: {loss:.4f}")
            
            if self.use_wandb:
                log_dict = {
                    'train/epoch_loss': avg_loss,
                    'epoch': epoch
                }
                for task, loss in avg_task_losses.items():
                    log_dict[f'train/epoch_{task}_loss'] = loss
                wandb.log(log_dict, step=self.global_step)
            
            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_task_losses = {task: 0.0 for task in self.val_loader.dataset.tasks}
        val_task_accuracies = {task: 0.0 for task in self.val_loader.dataset.tasks}
        val_task_class_accuracies = {task: [] for task in self.val_loader.dataset.tasks}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                points, labels, patient_id = batch
                points = points.to(self.device)
                labels = {task: label.to(self.device) for task, label in labels.items()}
                
                outputs = self.model(points)
                loss, task_losses = self.compute_loss(outputs, labels)
                accuracies, class_accuracies = self.compute_accuracy(outputs, labels)
                
                val_loss += loss.item()
                num_batches += 1
                
                for task in self.val_loader.dataset.tasks:
                    if task in task_losses:
                        val_task_losses[task] += task_losses[task]
                    if task in accuracies:
                        val_task_accuracies[task] += accuracies[task]
                        val_task_class_accuracies[task].append(class_accuracies[task])
        
        avg_val_loss = val_loss / max(num_batches, 1)
        avg_val_task_losses = {task: loss / max(num_batches, 1) for task, loss in val_task_losses.items()}
        avg_val_task_accuracies = {task: acc / max(num_batches, 1) for task, acc in val_task_accuracies.items()}
        
        avg_val_class_accuracies = {}
        for task, accs_list in val_task_class_accuracies.items():
            if accs_list:
                import numpy as np
                avg_val_class_accuracies[task] = np.mean(accs_list, axis=0)
        
        print(f"\nEpoch {epoch} Validation Summary:")
        print(f"  Average Loss: {avg_val_loss:.4f}")
        
        overall_acc = []
        for task in self.val_loader.dataset.tasks:
            if avg_val_task_losses[task] > 0:
                task_acc = avg_val_task_accuracies.get(task, 0.0)
                print(f"  {task}:")
                print(f"    Loss: {avg_val_task_losses[task]:.4f}, Accuracy: {task_acc:.4f}")
                
                if task in avg_val_class_accuracies:
                    class_accs = avg_val_class_accuracies[task]
                    for class_idx, class_acc in enumerate(class_accs):
                        if class_acc > 0:
                            print(f"      Class {class_idx}: {class_acc:.4f}")
                
                if task_acc > 0:
                    overall_acc.append(task_acc)
        
        if overall_acc:
            mean_accuracy = sum(overall_acc) / len(overall_acc)
            print(f"\n  Overall Average Accuracy: {mean_accuracy:.4f}")
        print()
        
        if self.use_wandb:
            log_dict = {
                'val/epoch_loss': avg_val_loss,
                'epoch': epoch
            }
            
            for task, loss in avg_val_task_losses.items():
                log_dict[f'val/epoch_{task}_loss'] = loss
            
            for task, acc in avg_val_task_accuracies.items():
                if acc > 0:
                    log_dict[f'val/epoch_{task}_accuracy'] = acc
            
            if overall_acc:
                log_dict['val/epoch_mean_accuracy'] = mean_accuracy
            
            for task, class_accs in avg_val_class_accuracies.items():
                for class_idx, class_acc in enumerate(class_accs):
                    if class_acc > 0:
                        log_dict[f'val/{task}_class_{class_idx}_accuracy'] = class_acc
            
            wandb.log(log_dict, step=self.global_step)
        
        # Save checkpoints based on validation accuracy
        if overall_acc:
            mean_accuracy = sum(overall_acc) / len(overall_acc)
            is_best = mean_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = mean_accuracy
                print(f"\n🏆 New best validation accuracy: {mean_accuracy:.4f}")
            self.save_checkpoint(epoch, avg_val_loss, mean_accuracy, is_best)
        else:
            # Fallback to loss-based saving if no accuracy available
            is_best = False
            print(f"\n⚠️ No accuracy computed, saving checkpoint without 'best' designation")
            self.save_checkpoint(epoch, avg_val_loss, 0.0, is_best)


    
def main():
    parser = argparse.ArgumentParser(description="Train Point Cloud Multi-Task Classification")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for wandb')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation (0-4)')
    parser.add_argument('--num_folds', type=int, default=5, help='Total number of folds for cross-validation')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['use_wandb'] = args.use_wandb
    if args.run_name:
        config['run_name'] = args.run_name
    else:
        # Add fold suffix to default run name
        base_name = config.get('run_name', 'default_run')
        config['run_name'] = f"{base_name}_fold{args.fold}"

    torch.manual_seed(config['seed'])
    
    if config['use_wandb']:
        wandb.init(
            project=config.get('wandb_project', 'ios-classification'),
            name=config.get('run_name', None),
            config=config
        )
        print("Wandb initialized")
    
    print(f"\n{'='*60}")
    print(f"Training with {args.num_folds}-fold cross-validation")
    print(f"Current fold: {args.fold}/{args.num_folds - 1}")
    print(f"{'='*60}\n")
    
    print("Loading datasets...")
    train_dataset = IOSClassificationDataset(
        dataset_path=config['data']['dataset_path'],
        classification_path=config['data']['classification_path'],
        schema_path=config['data']['schema_path'],
        augment=True,
        num_points=config['data'].get('num_points', 32768),
        split='train',
        fold=args.fold,
        num_folds=args.num_folds
    )
    
    val_dataset = IOSClassificationDataset(
        dataset_path=config['data']['dataset_path'],
        classification_path=config['data']['classification_path'],
        schema_path=config['data']['schema_path'],
        augment=False,
        num_points=config['data'].get('num_points', 32768),
        split='val',
        fold=args.fold,
        num_folds=args.num_folds
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    tasks_config = train_dataset.num_classes_per_task
    print(f"\nTasks and classes detected:")
    for task, num_classes in tasks_config.items():
        print(f"  {task}: {num_classes} classes")
    
    print("\nInitializing model...")
    encoder = PointEncoder(**config['model']['encoder'])
    
    encoder_output_dim = config['model']['encoder'].get('output_dim', config['model']['encoder']['trans_dim'])
    
    classification_head = MultiTaskClassificationHead(
        input_dim=encoder_output_dim,
        tasks_config=tasks_config
    )
    
    model = PointClassificationModel(encoder, classification_head)
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    print("\nStarting training...")
    trainer.train()
    
    if config['use_wandb']:
        wandb.finish()


if __name__ == '__main__':
    main()