"""
Training pipeline for fine-tuning Qwen3 VL with intraoral photos.

This script supports three modes via CLI flags (combinable in a single run):
  --train_vision   Fine-tune the vision encoder only.
                   Best practice for limited data (~5k images): only the last
                   portion of ViT blocks are unfrozen to avoid over-fitting while
                   still adapting to unseen dental image statistics.
  --train_language Fine-tune the language model via LoRA adapters.
                   Parameter-efficient (no catastrophic forgetting), learns the
                   structured clinical report format and dental terminology.
  Both flags       Joint single-step fine-tuning with component-specific LRs.
                   Recommended for best end-to-end performance.

Config overrides (optional, all have sensible defaults):
  vision_finetune:
    freeze_first_n_blocks: 16   # freeze early ViT blocks (limited-data regularisation)
    lr: 1.0e-5
  language_finetune:
    lr: 2.0e-5
    lora:
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules: [q_proj, k_proj, v_proj, o_proj]
"""

import os
import sys

# Set tokenizers parallelism before importing transformers to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import argparse
from pathlib import Path
import math
import traceback
import json
import random

import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image

import wandb
from transformers import get_linear_schedule_with_warmup, AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from typing import Dict, Optional, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from validation import compute_all_metrics


class IntraoralPhotosDataset(Dataset):
    """
    Dataset for intraoral photos with clinical descriptions.
    
    Loads 5 intraoral photos per patient and pairs them with auto-generated captions.
    """
    
    def __init__(
        self,
        dataset_dir: str,
        captions_dir: str,
        image_size: tuple = (448, 448),  # Qwen3 VL default
        instruction_template_path: Optional[str] = None,
    ):
        """
        Args:
            dataset_dir: Directory containing patient folders with intraoral-photos/
            captions_dir: Directory containing JSON files with patient descriptions
            image_size: Target image size (width, height)
            instruction_template_path: Path to instruction template file
        """
        self.dataset_dir = Path(dataset_dir)
        self.captions_dir = Path(captions_dir)
        self.image_size = image_size
        
        # Load instruction template
        if instruction_template_path is None:
            # Default template
            self.instruction_template = {
                "system": "You are a dental expert analyzing intraoral photos. Describe the patient's occlusion in detail.",
                "user": "Attached is an intra-oral scan image.\n<|image_pad|>\nPlease describe it accurately using only these fields in the given order:\n<|fields_name|>\n\nDo not provide any other information outside the requested fields.",
                "assistant": ""
            }
        else:
            # Load JSON template
            with open(instruction_template_path, 'r') as f:
                template_content = f.read()
                try:
                    self.instruction_template = json.loads(template_content)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text (backward compatibility)
                    self.instruction_template = {
                        "system": "You are a dental expert analyzing intraoral photos.",
                        "user": template_content.replace("<|fields_name|>", "<|fields_name|>"),
                        "assistant": ""
                    }
        
        # Find all valid samples
        self.samples = self._prepare_samples()
        
        print(f"Loaded {len(self.samples)} samples with intraoral photos")
    
    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare list of valid samples with paths.
        
        Scans the captions directory for JSON files and matches them
        with patient directories containing intraoral photos.
        """
        samples = []
        
        # Get all JSON files from captions directory
        json_files = list(self.captions_dir.glob("*.json"))
        
        for json_file in json_files:
            patient_id = json_file.stem  # Filename without .json extension
            
            # Load description from JSON
            try:
                with open(json_file, 'r') as f:
                    caption_data = json.load(f)
                    
                    # Try to get intraoral-photo description first, fallback to ios
                    if 'intraoral-photo' in caption_data:
                        desc_field = 'intraoral-photo'
                    elif 'ios' in caption_data:
                        desc_field = 'ios'
                    else:
                        print(f"Warning: No caption field found in {json_file}")
                        continue
                    
                    description = caption_data.get(desc_field).get('description', '').strip()
                    
                    if not description:
                        print(f"Warning: Empty description in {json_file}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
                continue
            
            # Look for intraoral photos directory
            patient_dir = self.dataset_dir / patient_id
            photos_dir = patient_dir / "intraoral-photos"
            
            if not photos_dir.exists():
                # Try alternative patient ID formats (some have spaces or special chars)
                continue
            
            # Check for required photo files (at least some should exist)
            photo_names = ["left", "center", "right", "upper", "lower", "a", "b", "c", "d", "e"]
            photo_paths = []
            for name in photo_names:
                # Try different extensions
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    photo_path = photos_dir / f"{name}{ext}"
                    if photo_path.exists():
                        photo_paths.append(photo_path)
                        break
            
            # Require at least 3 photos (some patients may be missing 1-2)
            if len(photo_paths) >= 3:
                samples.append({
                    'patient_id': patient_id,
                    'photo_paths': photo_paths,
                    'description': description
                })
            else:
                print(f"Warning: Not enough photos for {patient_id} at {photos_dir} (found {len(photo_paths)})")
        
        return samples
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and resize an image."""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size, Image.BILINEAR)
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - images: List of PIL Images (up to 5)
                - description: str
                - patient_id: str
                - instruction: str
        """
        sample = self.samples[idx]
        
        # Load all available images for this patient
        images = []
        for photo_path in sample['photo_paths']:
            try:
                img = self._load_image(photo_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {photo_path}: {e}")
        
        # Ensure we have at least one image
        if len(images) == 0:
            raise ValueError(f"No valid images for patient {sample['patient_id']}")
        
        return {
            'images': images,
            'description': sample['description'],
            'patient_id': sample['patient_id'],
            'template': self.instruction_template  # Changed from 'instruction' to 'template'
        }


class IntraoralPhotosCollator:
    """
    Collate function for batching intraoral photo samples.
    
    Handles multi-image input for Qwen3 VL.
    """
    
    def __init__(self, processor):
        """
        Args:
            processor: Hugging Face processor for Qwen3 VL
        """
        self.processor = processor
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.
        
        For Qwen3 VL, we need to create a messages format with multiple images.
        
        Args:
            batch: List of dicts from IntraoralPhotosDataset
        
        Returns:
            Batched dict with processed inputs
        """
        # Prepare multimodal messages for each sample
        all_messages = []
        instructions = []
        descriptions = []
        patient_ids = []
        
        for item in batch:
            # Get template (dict with system, user, assistant)
            template = item['template']
            
            # Parse and shuffle description fields
            desc = item['description'].strip()
            
            # Remove header if present
            if desc.startswith("Patient Intra-Oral Description:"):
                desc = desc.replace("Patient Intra-Oral Description:", "", 1).strip()
            
            # Parse into field: value pairs
            field_value_pairs = []
            for line in desc.split('\n'):
                line = line.strip()
                if ':' in line and line:
                    field_value_pairs.append(line)
            
            # Shuffle for data augmentation
            random.shuffle(field_value_pairs)
            
            # Get field names
            field_names = [x.split(":")[0].strip() for x in field_value_pairs]
            field_names_list = ', '.join(field_names) + "."
            
            # Format description
            formatted_desc = '\n'.join(field_value_pairs)
            
            # Build user prompt with field names substituted
            user_text = template['user'].replace("<|fields_name|>", field_names_list)
            
            # Build Qwen3 VL message format with system, user (with images), assistant
            messages = []
            
            # System message
            messages.append({
                "role": "system",
                "content": template['system']
            })
            
            # User message with images
            user_content = []
            for img in item['images']:
                user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": user_text})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            all_messages.append(messages)
            instructions.append(user_text)
            descriptions.append(formatted_desc + "<|im_end|>")
            patient_ids.append(item['patient_id'])
        
        try:
            # Convert messages to text using chat template
            # The processor's text parameter expects strings, not message dictionaries
            text_prompts = []
            for messages in all_messages:
                # Use apply_chat_template to convert messages to text
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                text_prompts.append(text)
            
            images_batch = [item['images'] for item in batch]
            
            inputs = self.processor(
                text=text_prompts,
                images=images_batch,
                return_tensors="pt",
                padding=True
            )
            
            # If the processor doesn't handle images, fall back to text-only
            if 'pixel_values' not in inputs:
                print("Warning: Processor did not return pixel_values, processing images separately")
                # Extract all images from batch
                all_images = []
                for item in batch:
                    all_images.extend(item['images'])
                
                # Process images through image processor
                if hasattr(self.processor, 'image_processor'):
                    image_inputs = self.processor.image_processor(
                        images=all_images,
                        return_tensors="pt"
                    )
                    inputs['pixel_values'] = image_inputs['pixel_values']
                    if 'image_grid_thw' in image_inputs:
                        inputs['image_grid_thw'] = image_inputs['image_grid_thw']
                
                # Process text separately
                text_inputs = self.processor.tokenizer(
                    text=instructions,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=True
                )
                inputs['input_ids'] = text_inputs['input_ids']
                inputs['attention_mask'] = text_inputs['attention_mask']
                
        except Exception as e:
            print(f"Error in processor: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: process images and text separately
            print("Falling back to separate image and text processing")
            
            # Process all images
            all_images = []
            for item in batch:
                all_images.extend(item['images'])
            
            if hasattr(self.processor, 'image_processor'):
                image_inputs = self.processor.image_processor(
                    images=all_images,
                    return_tensors="pt"
                )
            else:
                # If no image_processor, create dummy pixel_values
                print("Warning: No image_processor found, using fallback")
                image_inputs = {}
            
            # Process text
            text_inputs = self.processor.tokenizer(
                text=instructions,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            )
            
            # Combine
            inputs = {
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask'],
            }
            if 'pixel_values' in image_inputs:
                inputs['pixel_values'] = image_inputs['pixel_values']
            if 'image_grid_thw' in image_inputs:
                inputs['image_grid_thw'] = image_inputs['image_grid_thw']
        
        # Tokenize responses separately
        response_tokenized = self.processor(
            text=descriptions,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True
        )
        
        # Concatenate input_ids and attention_mask
        input_ids = torch.cat([inputs['input_ids'], response_tokenized['input_ids']], dim=1)
        attention_mask = torch.cat([inputs['attention_mask'], response_tokenized['attention_mask']], dim=1)
        
        # Create labels (mask instruction, keep description)
        labels = input_ids.clone()
        instr_len = inputs['input_ids'].shape[1]
        for i in range(len(batch)):
            labels[i, :instr_len] = -100  # Mask instruction tokens
        
        # Prepare output
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'patient_ids': patient_ids,
            'instructions': instructions,
            'descriptions': descriptions,
            'images': [item['images'] for item in batch]  # Keep raw images for generation
        }
        
        # Add pixel_values and image_grid_thw if available (for vision encoder)
        if 'pixel_values' in inputs:
            output['pixel_values'] = inputs['pixel_values']
        if 'image_grid_thw' in inputs:
            output['image_grid_thw'] = inputs['image_grid_thw']
        
        return output


class Trainer:
    """Trainer for Qwen3 VL vision encoder fine-tuning."""
    
    def __init__(
        self,
        config: Dict,
        model: Qwen3VLForConditionalGeneration,
        processor: AutoProcessor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ):
        self.config = config
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training mode flags (stored on config by main())
        self.train_vision = config.get('_train_vision', False)
        self.train_language = config.get('_train_language', False)
        self.use_lora = config.get('_use_lora', False)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            mode_tag = []
            if self.train_vision:
                mode_tag.append('vision')
            if self.train_language:
                mode_tag.append('lora')
            wandb.init(
                project="pointqwen",
                config=config,
                name=config.get('run_name', 'qwen3vl_photos_' + '+'.join(mode_tag or ['unknown'])),
                tags=["Photos"] + mode_tag
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with component-specific learning rates."""
        train_config = self.config['training']
        
        vision_cfg = self.config.get('vision_finetune', {})
        lang_cfg = self.config.get('language_finetune', {})
        
        vision_lr = vision_cfg.get('lr', 1e-5)
        lora_lr = lang_cfg.get('lr', 2e-5)
        
        # Bucket trainable params by component
        vision_params = []
        lora_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'visual' in name:
                vision_params.append(param)
            else:
                lora_params.append(param)  # LoRA adapter weights
        
        if not vision_params and not lora_params:
            raise ValueError(
                "No trainable parameters found. "
                "Pass --train_vision and/or --train_language."
            )
        
        param_groups = []
        print("\nOptimizer Configuration:")
        
        if vision_params:
            param_groups.append({
                'params': vision_params,
                'lr': vision_lr,
                'name': 'vision_encoder'
            })
            print(f"  Vision encoder  LR: {vision_lr:.2e}  "
                  f"({sum(p.numel() for p in vision_params):,} params)")
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': lora_lr,
                'name': 'language_lora'
            })
            print(f"  Language LoRA   LR: {lora_lr:.2e}  "
                  f"({sum(p.numel() for p in lora_params):,} params)")
        
        print(f"  Weight Decay: {train_config['weight_decay']}")
        
        optimizer = AdamW(
            param_groups,
            betas=(train_config['adam_beta1'], train_config['adam_beta2']),
            eps=train_config['adam_epsilon'],
            weight_decay=train_config['weight_decay']
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        train_config = self.config['training']
        
        num_training_steps = len(self.train_loader) * train_config['num_epochs']
        num_training_steps //= train_config.get('gradient_accumulation_steps', 1)
        warmup_steps = train_config.get('warmup_steps', 0)
        
        if train_config['lr_scheduler'] == 'cosine':
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
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        train_config = self.config['training']
        grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # DEBUG: Print tokens for first iteration of first epoch
            if self.current_epoch == 0 and step == 0:
                print("\n" + "=" * 80)
                print("DEBUG: First Iteration Token Inspection")
                print("=" * 80)
                
                # Get the first sample in the batch
                first_input_ids = batch['input_ids'][0].cpu().tolist()
                first_labels = batch['labels'][0].cpu().tolist()
                
                # Decode the full sequence with special tokens
                full_text = self.processor.tokenizer.decode(
                    first_input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                
                # Replace vision tokens (between <|vision_start|> and <|vision_end|>) with [IMAGE_TOKENS]
                vision_pattern = r'<\|vision_start\|>.*?<\|vision_end\|>'
                full_text_with_placeholder = re.sub(vision_pattern, '[IMAGE_TOKENS]', full_text, flags=re.DOTALL)
                
                print("\n--- FULL INPUT PROMPT (with special tokens) ---")
                print(full_text_with_placeholder)
                print("\n--- TOKEN IDs (first 100 tokens) ---")
                print(first_input_ids[:100])
                
                print("\n--- LABELS (first 100, -100 = masked) ---")
                print(first_labels[:100])
                
                # Count tokens
                num_total_tokens = len(first_input_ids)
                num_masked_tokens = sum(1 for x in first_labels if x == -100)
                num_training_tokens = num_total_tokens - num_masked_tokens
                
                print("\n--- TOKEN STATISTICS ---")
                print(f"Total tokens: {num_total_tokens}")
                print(f"Masked tokens (instruction): {num_masked_tokens}")
                print(f"Training tokens (description): {num_training_tokens}")
                
                # Show where training starts (first non-masked token)
                first_training_idx = next((i for i, x in enumerate(first_labels) if x != -100), None)
                if first_training_idx:
                    print(f"\nTraining starts at token index: {first_training_idx}")
                    training_start_text = self.processor.tokenizer.decode(
                        first_input_ids[first_training_idx:first_training_idx+20],
                        skip_special_tokens=False
                    )
                    print(f"First 20 training tokens: {training_start_text}")
                
                print("\n" + "=" * 80 + "\n")
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Move vision inputs if present
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            
            image_grid_thw = batch.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )
            
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_config['max_grad_norm'])
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            epoch_loss += loss.item() * grad_accum_steps
            progress_bar.set_postfix(loss=loss.item() * grad_accum_steps)
            
            # Log to wandb
            if self.use_wandb and step % train_config['logging_steps'] == 0:
                wandb.log({
                    'train/loss': loss.item() * grad_accum_steps,
                    'train/epoch': self.current_epoch,
                    'train/step': self.global_step,
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        return epoch_loss / len(self.train_loader)
    
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
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            
            image_grid_thw = batch.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            
            # Forward pass for loss computation
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
            
            # Generate text for metrics computation
            if compute_metrics:
                # Check if we've reached the limit
                if max_samples_for_metrics is not None and len(all_predictions) >= max_samples_for_metrics:
                    continue
                
                # Generate for all samples in batch (or just first few if limited)
                batch_size = input_ids.shape[0]
                samples_to_generate = min(batch_size, max_samples_for_metrics - len(all_predictions)) if max_samples_for_metrics else batch_size
                
                for j in range(samples_to_generate):
                    # Get the prompt (instruction part before description)
                    instruction_text = batch['instructions'][j]
                    ground_truth = batch['descriptions'][j]
                    
                    # Get images for this sample
                    sample_images = batch.get('images', [[]])[j] if 'images' in batch else []
                    
                    try:
                        # Build messages format with images (same as _generate_and_display_sample)
                        messages = [
                            {
                                "role": "system",
                                "content": "You are a dental expert analyzing intraoral photos. Describe the patient's occlusion in detail."
                            },
                            {
                                "role": "user",
                                "content": []
                            }
                        ]
                        
                        # Add images to user content
                        for img in sample_images:
                            messages[1]["content"].append({"type": "image", "image": img})
                        
                        # Add text instruction
                        messages[1]["content"].append({"type": "text", "text": instruction_text})
                        
                        # Convert messages to text using chat template
                        generation_prompt = self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Get generation config with defaults
                        gen_config = self.config['training'].get('generation', {})
                        
                        # Process text and images together
                        text_inputs = self.processor(
                            text=[generation_prompt],
                            images=[sample_images],
                            return_tensors="pt",
                            padding=True
                        )
                        
                        generated_ids = self.model.generate(
                            input_ids=text_inputs['input_ids'].to(self.device),
                            attention_mask=text_inputs['attention_mask'].to(self.device),
                            pixel_values=text_inputs.get('pixel_values').to(self.device) if 'pixel_values' in text_inputs else None,
                            image_grid_thw=text_inputs.get('image_grid_thw').to(self.device) if 'image_grid_thw' in text_inputs else None,
                            max_new_tokens=self.config['training'].get('val_max_new_tokens', 512),
                            temperature=gen_config.get('temperature', 0.1),
                            top_p=gen_config.get('top_p', 0.9),
                            top_k=gen_config.get('top_k', 1),
                            do_sample=gen_config.get('do_sample', False),
                            repetition_penalty=gen_config.get('repetition_penalty', 1.0),
                            no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
                        )
                        
                        # Extract only the newly generated tokens (not the input prompt)
                        input_length = text_inputs['input_ids'].shape[1]
                        new_tokens = generated_ids[:, input_length:]
                        
                        generated_text = self.processor.batch_decode(
                            new_tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )[0]
                        
                        generated_description = generated_text.strip()
                        
                        # Clean up the ground truth (remove end token if present)
                        clean_ground_truth = ground_truth.replace("<|im_end|>", "").strip()
                        
                        all_predictions.append(generated_description)
                        all_references.append(clean_ground_truth)
                        
                    except Exception as e:
                        print(f"\nWarning: Error during generation for sample {i}-{j}: {e}")
                        import traceback
                        traceback.print_exc()
            
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
        instruction = batch['instructions'][0] if 'instructions' in batch else ""
        ground_truth = batch['descriptions'][0] if 'descriptions' in batch else ""
        patient_id = batch['patient_ids'][0] if 'patient_ids' in batch else "Unknown"
        
        print(f"\nPatient ID: {patient_id}")
        print(f"\nPrompt:\n{'-'*80}\n{instruction}")
        
        # Generate
        try:
            # Get generation config with defaults
            gen_config = self.config['training'].get('generation', {})
            
            # Get first sample's raw images
            first_sample_images = batch.get('images', [[]])[0] if 'images' in batch else []
            
            # Build messages format similar to training
            messages = [
                {
                    "role": "system",
                    "content": "You are a dental expert analyzing intraoral photos. Describe the patient's occlusion in detail."
                },
                {
                    "role": "user", 
                    "content": []
                }
            ]
            
            # Add images to user content
            for img in first_sample_images:
                messages[1]["content"].append({"type": "image", "image": img})
            
            # Add text instruction
            messages[1]["content"].append({"type": "text", "text": instruction})
            
            # Convert messages to text using chat template
            generation_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process text and images together
            inputs = self.processor(
                text=[generation_prompt],
                images=[first_sample_images],
                return_tensors="pt",
                padding=True
            )
            
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                pixel_values=inputs.get('pixel_values').to(self.device) if 'pixel_values' in inputs else None,
                image_grid_thw=inputs.get('image_grid_thw').to(self.device) if 'image_grid_thw' in inputs else None,
                max_new_tokens=self.config['training'].get('val_max_new_tokens', 256),
                temperature=gen_config.get('temperature', 0.7),
                top_p=gen_config.get('top_p', 0.9),
                top_k=gen_config.get('top_k', 50),
                do_sample=True,
                repetition_penalty=gen_config.get('repetition_penalty', 1.2),
                no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
            )
            
            # Extract only the newly generated tokens (not the input prompt)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            generated_text = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            generated_description = generated_text.strip()
            
            print(f"\nGround Truth:\n{'-'*80}\n{ground_truth}")
            print(f"\nGenerated:\n{'-'*80}\n{generated_description}")
            
        except Exception as e:
            print(f"\nError during generation: {e}")
            traceback.print_exc()
        
        print("="*80 + "\n")
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config['training']['num_epochs']
        
        mode_desc = ' + '.join(
            filter(None, [
                'ViT encoder' if self.train_vision else None,
                'LLM LoRA'    if self.train_language else None,
            ])
        ) or 'unknown'
        print("\n" + "=" * 60)
        print(f"STARTING TRAINING: Qwen3 VL  [{mode_desc}]")
        print("=" * 60)
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Training epochs: {num_epochs}")
        print("=" * 60 + "\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Validate with metrics
            run_epoch_validation = self.config['training'].get('validate_per_epoch', False)
            if self.val_loader is not None and run_epoch_validation:
                # Determine if we should generate samples and compute metrics
                # Only do full metrics computation occasionally to save time
                compute_full_metrics = (epoch + 1) % self.config['training'].get('metrics_every_n_epochs', 5) == 0
                generate_sample = compute_full_metrics or (epoch == 0)  # Always show sample on first epoch
                
                val_loss = self.validate(
                    generate_sample=generate_sample,
                    compute_metrics=compute_full_metrics
                )
                
                # Save best model based on validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model')
                    print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config['training'].get('save_every_n_epochs', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint.
        
        Saving strategy:
        - No LoRA (vision only): full model via save_pretrained.
        - LoRA only: adapter weights via save_pretrained (PeftModel).
        - LoRA + vision: adapter weights + separate vision_encoder_state.pt.

        Loading back:
          # vision only
          model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint_path)
          # LoRA only
          from peft import PeftModel
          base = Qwen3VLForConditionalGeneration.from_pretrained(base_name)
          model = PeftModel.from_pretrained(base, checkpoint_path)
          # Both
          base = Qwen3VLForConditionalGeneration.from_pretrained(base_name)
          model = PeftModel.from_pretrained(base, checkpoint_path)
          vision_sd = torch.load(checkpoint_path / 'vision_encoder_state.pt')
          model.load_state_dict(vision_sd, strict=False)
        """
        checkpoint_path = self.output_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        if self.use_lora:
            # PeftModel.save_pretrained saves the LoRA adapter weights only
            self.model.save_pretrained(checkpoint_path)
            
            if self.train_vision:
                # Also persist the fine-tuned vision encoder weights separately
                vision_sd = {
                    k: v.detach().cpu()
                    for k, v in self.model.named_parameters()
                    if 'visual' in k
                }
                torch.save(vision_sd, checkpoint_path / 'vision_encoder_state.pt')
                print(f"  ↳ LoRA adapter + vision encoder weights saved")
            else:
                print(f"  ↳ LoRA adapter weights saved")
        else:
            # No LoRA – full model (vision encoder fine-tuned in place)
            self.model.save_pretrained(checkpoint_path)
            print(f"  ↳ Full model saved")
        
        self.processor.save_pretrained(checkpoint_path)
        
        # Save training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_vision': self.train_vision,
            'train_language': self.train_language,
        }, checkpoint_path / 'training_state.pt')
        
        print(f"Saved checkpoint: {checkpoint_path}")


def setup_trainable_parameters(
    model: Qwen3VLForConditionalGeneration,
    train_vision: bool,
    train_language: bool,
    config: Dict,
) -> Qwen3VLForConditionalGeneration:
    """
    Set up which parameters are trainable based on CLI flags.

    Limited-data best practices applied automatically:
    - Vision encoder: freeze the first N ViT blocks (early blocks capture low-level
      features that transfer well; only deeper blocks need adaptation for a new visual
      domain with ~5k images).  Default: freeze the first 60 % of blocks.
    - Language model: LoRA adapters (rank 16) on attention projections only.
      Low rank + attention-only = minimal param count, low overfitting risk.
    """
    if not train_vision and not train_language:
        raise ValueError(
            "At least one of --train_vision or --train_language must be specified."
        )
    
    print("\nSetting up trainable parameters...")
    
    # ── 1. Freeze everything ──────────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False
    
    # ── 2. Language model: LoRA adapters (applied first so PEFT can wrap) ─────
    if train_language:
        lang_cfg = config.get('language_finetune', {})
        lora_cfg = lang_cfg.get('lora', {})
        
        # q/k/v/o_proj are LLM-specific names in Qwen3-VL;
        # the ViT uses combined qkv projections with different naming.
        target_modules = lora_cfg.get(
            'target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        lora_config = LoraConfig(
            r=lora_cfg.get('r', 16),
            lora_alpha=lora_cfg.get('lora_alpha', 32),
            lora_dropout=lora_cfg.get('lora_dropout', 0.05),
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("✓ Applied LoRA to language model")
        model.print_trainable_parameters()
    
    # ── 3. Vision encoder: selectively unfreeze deeper ViT blocks ────────────
    if train_vision:
        vision_cfg = config.get('vision_finetune', {})
        
        # Count total ViT blocks so we can compute 60 % dynamically
        total_blocks = sum(
            1 for n, _ in model.named_modules()
            if re.fullmatch(r'.*visual\.blocks\.\d+', n)
        )
        default_freeze = max(0, int(total_blocks * 0.6)) if total_blocks > 0 else 0
        freeze_first_n = vision_cfg.get('freeze_first_n_blocks', default_freeze)
        
        vision_trainable = 0
        for name, param in model.named_parameters():
            if 'visual' not in name:
                continue
            if freeze_first_n > 0:
                m = re.search(r'blocks\.(\d+)', name)
                if m and int(m.group(1)) < freeze_first_n:
                    continue  # keep early block frozen
            param.requires_grad = True
            vision_trainable += param.numel()
        
        if total_blocks > 0:
            print(
                f"✓ Unfrozen vision encoder: {vision_trainable:,} params  "
                f"(blocks {freeze_first_n}–{total_blocks - 1} of {total_blocks}; "
                f"first {freeze_first_n} frozen for limited-data regularisation)"
            )
        else:
            print(f"✓ Unfrozen vision encoder: {vision_trainable:,} params")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 VL with intraoral photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # fine-tune vision encoder only (limited-data safe: freezes first 60% of ViT blocks)
  python train_intraoralphotos.py --config cfg.yaml --train_vision

  # fine-tune language model via LoRA only
  python train_intraoralphotos.py --config cfg.yaml --train_language

  # joint single-step fine-tuning (recommended for best performance)
  python train_intraoralphotos.py --config cfg.yaml --train_vision --train_language
"""
    )
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for wandb')
    parser.add_argument(
        '--train_vision', action='store_true',
        help='Fine-tune the vision encoder (deeper ViT blocks only for limited data).'
    )
    parser.add_argument(
        '--train_language', action='store_true',
        help='Fine-tune the language model with LoRA adapters.'
    )
    args = parser.parse_args()
    
    if not args.train_vision and not args.train_language:
        parser.error("Specify at least one of --train_vision or --train_language.")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['use_wandb'] = args.use_wandb
    if args.run_name:
        config['run_name'] = args.run_name
    
    # Store training-mode flags on config so Trainer can read them
    config['_train_vision'] = args.train_vision
    config['_train_language'] = args.train_language
    config['_use_lora'] = args.train_language  # LoRA is always used for language fine-tuning
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    # Initialize model and processor
    print("Loading Qwen3 VL model...")
    model_name = config['model']['qwen_model_name']
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config['model'].get('attn_implementation', 'sdpa')
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Configure trainable parameters according to --train_vision / --train_language
    model = setup_trainable_parameters(
        model,
        train_vision=args.train_vision,
        train_language=args.train_language,
        config=config,
    )
    
    # Create datasets
    print("\nLoading data...")
    
    # Create base dataset
    base_dataset = IntraoralPhotosDataset(
        dataset_dir=config['data']['dataset_dir'],
        captions_dir=config['data']['captions_dir'],
        image_size=tuple(config['data'].get('image_size', [448, 448])),
        instruction_template_path=config['data'].get('instruction_template_path')
    )
    
    # Split into train/val
    val_split = config['training'].get('validation_split', 0.1)
    total_size = len(base_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Random split
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    
    train_dataset = Subset(base_dataset, train_indices)
    val_dataset = Subset(base_dataset, val_indices)
    
    print(f"Split dataset: {train_size} train, {val_size} validation samples")
    
    # Create dataloaders
    collator = IntraoralPhotosCollator(processor)
    
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
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        config=config,
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
