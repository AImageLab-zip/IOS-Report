"""
Evaluation script for trained PointQwen models.

This script loads a trained model checkpoint and evaluates it on a validation dataset,
computing all metrics and generating sample predictions.

The model now predicts each field individually (field-by-field approach) instead of 
generating the entire description at once. This allows for shorter generation with
max_new_tokens=32 per field.

Usage:
    python validate.py --checkpoint <path_to_checkpoint> \\
                       --config <path_to_config> \\
                       --data_dir <path_to_data> \\
                       --captions_dir <path_to_captions> \\
                       [--num_samples N] [--output_dir DIR] [--max_new_tokens_per_field 32]
"""
from unsloth import FastVisionModel

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from transformers import AutoProcessor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import PointEncoder
from models.projection import PointToQwenProjection
from data import IOSPointCloudDataset, IOSCollator
from validation import compute_all_metrics


class PointQwenLoRA(nn.Module):
    """
    PointQwen with LoRA adapters - identical to training version.
    This is needed to load the trained model checkpoint.
    """
    
    def __init__(
        self,
        config: Dict,
        qwen_lora_model,
        processor,
    ):
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
        
        # Initialize projection layer
        point_dim = point_encoder_config.get("trans_dim", 384)
        projection_config = config['model'].get('projection', {})
        
        self.projection = PointToQwenProjection(
            point_dim=point_dim,
            qwen_visual_dim=self.qwen_visual_dim,
            **projection_config
        )
        
        # Move components to same device as Qwen
        qwen_device = next(self.qwen.parameters()).device
        self.point_encoder = self.point_encoder.to(qwen_device)
        self.projection = self.projection.to(qwen_device)
    
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
                return 2560
        except:
            return 2560
    
    @property
    def device(self):
        """Get the device where the model is located."""
        return next(self.qwen.parameters()).device
    
    def to(self, *args, **kwargs):
        """Override to ensure all submodules are moved correctly."""
        super().to(*args, **kwargs)
        self.point_encoder.to(*args, **kwargs)
        self.projection.to(*args, **kwargs)
        self.qwen.to(*args, **kwargs)
        return self
    
    def encode_point_cloud(self, point_cloud: torch.Tensor):
        """Encode point cloud to visual tokens."""
        # Ensure input is on correct device and dtype
        point_cloud = point_cloud.to(device=self.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            point_tokens, _ = self.point_encoder(point_cloud)
        
        # Convert point_tokens to the same dtype as projection layer
        # This is necessary because point_encoder (frozen) outputs float32
        projection_dtype = next(self.projection.parameters()).dtype
        point_tokens = point_tokens.to(dtype=projection_dtype)
        
        visual_tokens = self.projection(point_tokens)
        return visual_tokens
    
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
        
        # CRITICAL: Ensure everything is in bfloat16 to match model's dtype
        target_dtype = torch.bfloat16
        text_embeds = text_embeds.to(dtype=target_dtype)
        visual_tokens = visual_tokens.to(device=text_embeds.device, dtype=target_dtype)
        
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
        
        # Generate using base model to avoid Unsloth wrapper issues
        try:
            if hasattr(self.qwen, 'model'):
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
        
        return generated_texts


def create_random_model(config: Dict) -> PointQwenLoRA:
    """
    Create a randomly initialized PointQwen model (for baseline evaluation).
    
    - Qwen weights: pretrained from HuggingFace
    - LoRA adapters: initialized to zero (no adaptation)
    - Projection layer: randomly initialized
    - Point encoder: pretrained from checkpoint
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Randomly initialized PointQwenLoRA model
    """
    print("\n" + "="*70)
    print("CREATING RANDOMLY INITIALIZED MODEL")
    print("="*70)
    
    # Verify point encoder path exists
    point_encoder_path = config['model'].get('pretrained_point_encoder')
    if point_encoder_path:
        if os.path.exists(point_encoder_path):
            print(f"✓ Point encoder path verified: {point_encoder_path}")
        else:
            print(f"⚠️  WARNING: Point encoder path does not exist: {point_encoder_path}")
            raise ValueError(f"Point encoder path does not exist: {point_encoder_path}")
    else:
        raise ValueError("pretrained_point_encoder must be specified in config for random initialization")
    
    # Load base Qwen model with LoRA (LoRA will be initialized to zero)
    print("\nLoading Qwen3-VL with zero-initialized LoRA adapters...")
    lora_config = config['model']['lora']
    
    # Get base model name from config
    base_model_name = config['model'].get('base_model', 'Qwen/Qwen2-VL-2B-Instruct')
    
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model_name,
        load_in_4bit=lora_config.get('use_4bit', False),
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=config['training'].get('max_seq_length', 2048),
        dtype=None,  # Auto
        device_map="auto",
    )
    
    # Apply LoRA (this initializes LoRA weights to zero by default)
    print("\nApplying LoRA configuration (zero-initialized adapters)...")
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0.0),
        target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
        use_rslora=lora_config.get('use_rslora', False),
        use_gradient_checkpointing="unsloth",
    )
    
    # Set to inference mode
    FastVisionModel.for_inference(model)
    
    # Create PointQwenLoRA wrapper
    print("\nInitializing PointQwenLoRA wrapper with random projection...")
    processor = AutoProcessor.from_pretrained(base_model_name)
    pointqwen_lora = PointQwenLoRA(
        config=config,
        qwen_lora_model=model,
        processor=processor,
    )
    
    # Projection layer is already randomly initialized in __init__
    print("✓ Projection layer randomly initialized")
    
    # Convert to bfloat16
    target_dtype = pointqwen_lora.qwen.model.language_model.embed_tokens.weight.dtype
    print(f"Converting projection layer to dtype: {target_dtype}")
    pointqwen_lora.projection = pointqwen_lora.projection.to(dtype=target_dtype)
    
    # Set model to eval mode
    pointqwen_lora.eval()
    
    print("\n✓ Random model created successfully!")
    print("="*70 + "\n")
    
    pointqwen_lora = pointqwen_lora.to(dtype=torch.bfloat16)
    
    return pointqwen_lora


def load_model_from_checkpoint_non_lora(checkpoint_path: str, config: Dict) -> PointQwenLoRA:
    """
    Load a trained PointQwen model from non-LoRA checkpoint (full model weights).
    
    This is for checkpoints saved during joint training of encoder+projector with frozen LLM,
    where the full model state dict is saved as model.pt.
    
    Args:
        checkpoint_path: Path to the checkpoint directory (contains model.pt)
        config: Configuration dictionary
    
    Returns:
        Loaded PointQwenLoRA model with weights from checkpoint
    """
    print("\n" + "="*70)
    print("LOADING NON-LORA MODEL FROM CHECKPOINT")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load base Qwen model WITHOUT LoRA
    print("\nLoading base Qwen3-VL model...")
    model_name = config['model']['qwen_model_name']
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=False,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=config['training'].get('max_seq_length', 2048),
        dtype=None,  # Auto
        device_map="auto",
    )
    
    # Set to inference mode
    FastVisionModel.for_inference(model)
    
    # Create processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Create PointQwenLoRA wrapper (even though it's not using LoRA)
    print("\nInitializing PointQwenLoRA wrapper...")
    pointqwen_lora = PointQwenLoRA(
        config=config,
        qwen_lora_model=model,
        processor=processor,
    )
    
    # Load full model checkpoint
    model_checkpoint_path = os.path.join(checkpoint_path, 'model.pt')
    if os.path.exists(model_checkpoint_path):
        print(f"\nLoading model weights from: {model_checkpoint_path}")
        state_dict = torch.load(model_checkpoint_path, map_location='cpu')
        
        # Load the state dict into the model
        missing_keys, unexpected_keys = pointqwen_lora.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
            if len(missing_keys) < 10:
                for key in missing_keys:
                    print(f"    - {key}")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) < 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
        
        print(f"✓ Model weights loaded successfully")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at: {model_checkpoint_path}")
    
    # Convert to target dtype
    target_dtype = torch.bfloat16
    print(f"\nConverting model to dtype: {target_dtype}")
    pointqwen_lora = pointqwen_lora.to(dtype=target_dtype)
    
    # Set model to eval mode
    pointqwen_lora.eval()
    
    print("\n✓ Non-LoRA model loaded successfully!")
    print("="*70 + "\n")
    
    return pointqwen_lora


def load_model_from_checkpoint(checkpoint_path: str, config: Dict) -> PointQwenLoRA:
    """
    Load a trained PointQwen model from checkpoint.
    
    Priority for loading:
    1. If training_config.yaml exists in checkpoint, use it (overrides passed config)
    2. Otherwise use the passed config
    
    Detects checkpoint type:
    - If model.pt exists: Non-LoRA checkpoint (full weights)
    - Otherwise: LoRA checkpoint (adapter weights)
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        config: Configuration dictionary (used as fallback)
    
    Returns:
        Loaded PointQwenLoRA model
    """
    print("\n" + "="*70)
    print("LOADING MODEL FROM CHECKPOINT")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Check if checkpoint has its own training config
    checkpoint_config_path = os.path.join(checkpoint_path, 'training_config.yaml')
    if os.path.exists(checkpoint_config_path):
        print(f"\n✓ Found training_config.yaml in checkpoint, loading configuration from checkpoint...")
        with open(checkpoint_config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  - Using config from: {checkpoint_config_path}")
    else:
        print(f"\n⚠️  No training_config.yaml found in checkpoint, using provided config")
    
    # Verify point encoder path exists
    point_encoder_path = config['model'].get('pretrained_point_encoder')
    if point_encoder_path:
        if os.path.exists(point_encoder_path):
            print(f"✓ Point encoder path verified: {point_encoder_path}")
        else:
            print(f"⚠️  WARNING: Point encoder path does not exist: {point_encoder_path}")
    else:
        print(f"⚠️  WARNING: No pretrained_point_encoder specified in config")
    
    # Detect checkpoint type
    model_pt_path = os.path.join(checkpoint_path, 'model.pt')
    adapter_model_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
    
    if os.path.exists(model_pt_path):
        print(f"\n✓ Detected non-LoRA checkpoint (model.pt found)")
        return load_model_from_checkpoint_non_lora(checkpoint_path, config)
    elif os.path.exists(adapter_model_path):
        print(f"\n✓ Detected LoRA checkpoint (adapter_model.safetensors found)")
    else:
        print(f"\n⚠️  WARNING: Could not detect checkpoint type, assuming LoRA")
    
    # Load base Qwen model with LoRA adapters
    print("\nLoading Qwen3-VL with LoRA adapters...")
    lora_config = config['model']['lora']
    
    model, tokenizer = FastVisionModel.from_pretrained(
        checkpoint_path,  # Load from checkpoint which contains LoRA adapters
        load_in_4bit=lora_config.get('use_4bit', False),
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",  # Use string value
        max_seq_length=config['training'].get('max_seq_length', 2048),
        dtype=None,  # Auto
        device_map="auto",
    )
    
    # Set to inference mode
    FastVisionModel.for_inference(model)
    
    # Create PointQwenLoRA wrapper
    print("\nInitializing PointQwenLoRA wrapper...")
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    pointqwen_lora = PointQwenLoRA(
        config=config,
        qwen_lora_model=model,
        processor=processor,
    )
    
    # Load projection layer weights
    projection_path = os.path.join(checkpoint_path, 'projection.pt')
    if os.path.exists(projection_path):
        print(f"\nLoading projection weights from: {projection_path}")
        projection_state = torch.load(projection_path, map_location='cpu')
        
        # Handle missing 'alpha' parameter (for backward compatibility)
        if 'alpha' not in projection_state:
            print("⚠️  Warning: 'alpha' parameter not found in checkpoint, setting to default value of 1.0")
            projection_state['alpha'] = torch.tensor(1.0)
        
        pointqwen_lora.projection.load_state_dict(projection_state)
        
        # CRITICAL: Convert projection layer to model's dtype (bfloat16)
        # This prevents dtype mismatches during inference
        target_dtype = pointqwen_lora.qwen.model.language_model.embed_tokens.weight.dtype
        print(f"Converting projection layer to dtype: {target_dtype}")
        
        # Convert all parameters and buffers in the projection module to target dtype
        pointqwen_lora.projection = pointqwen_lora.projection.to(dtype=target_dtype)
        
        # Verify conversion
        first_param_dtype = next(pointqwen_lora.projection.parameters()).dtype
        print(f"✓ Projection layer converted successfully")
        print(f"  - First parameter dtype: {first_param_dtype}")
        print(f"  - Target dtype: {target_dtype}")
        
        if first_param_dtype != target_dtype:
            raise RuntimeError(f"Dtype conversion failed! Expected {target_dtype}, got {first_param_dtype}")
    else:
        print(f"\n⚠️  Warning: No projection.pt found at {projection_path}")
        print("   Using initialized projection weights (this may give poor results)")
    
    # Set model to eval mode
    pointqwen_lora.eval()
    
    print("\n✓ Model loaded successfully!")
    print("="*70 + "\n")
    
    pointqwen_lora = pointqwen_lora.to(dtype=torch.bfloat16)
    
    return pointqwen_lora


def predict_single_field(
    model: PointQwenLoRA,
    point_cloud: torch.Tensor,
    base_prompt: str,
    field_name: str,
    max_new_tokens: int = 64,
    **gen_kwargs
) -> str:
    """
    Predict the value for a single field.
    
    Args:
        model: Trained PointQwenLoRA model
        point_cloud: (1, N, 3) point cloud tensor
        base_prompt: Template prompt with <|field_to_predict|> placeholder
        field_name: Name of the field to predict (e.g., "Overbite")
        max_new_tokens: Maximum tokens to generate (default: 32 for single field)
        **gen_kwargs: Additional generation parameters
    
    Returns:
        Predicted field value as string
    """
    # Replace placeholder with actual field name
    prompt = base_prompt.replace("<|field_to_predict|>", field_name)
    
    # Generate prediction
    generated_texts = model.generate(
        point_cloud=point_cloud,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        **gen_kwargs
    )
    
    generated_text = generated_texts[0]
    
    # Extract generated part (remove prompt)
    if prompt in generated_text:
        generated_value = generated_text.split(prompt)[-1].strip()
    else:
        generated_value = generated_text.strip()
    
    # Remove any trailing special tokens
    generated_value = generated_value.replace("<|im_end|>", "").strip()
    
    return generated_value


def predict_all_fields(
    model: PointQwenLoRA,
    point_cloud: torch.Tensor,
    base_prompt: str,
    field_names: List[str],
    max_new_tokens: int = 32,
    verbose: bool = False,
    **gen_kwargs
) -> str:
    """
    Predict all fields one at a time and merge into complete description.
    
    Args:
        model: Trained PointQwenLoRA model
        point_cloud: (1, N, 3) point cloud tensor
        base_prompt: Template prompt with <|field_to_predict|> placeholder
        field_names: List of field names to predict
        max_new_tokens: Maximum tokens per field (default: 32)
        verbose: Print progress for each field
        **gen_kwargs: Additional generation parameters
    
    Returns:
        Complete description with all fields merged
    """
    predicted_fields = []
    
    for i, field_name in enumerate(field_names):
        if verbose:
            print(f"  [{i+1}/{len(field_names)}] Predicting: {field_name}")
        
        try:
            field_value = predict_single_field(
                model=model,
                point_cloud=point_cloud,
                base_prompt=base_prompt,
                field_name=field_name,
                max_new_tokens=max_new_tokens,
                **gen_kwargs
            )
            predicted_fields.append((field_name, field_value))
            
            if verbose:
                print(f"       → {field_value}")
        except Exception as e:
            print(f"⚠️  Error predicting field '{field_name}': {e}")
            predicted_fields.append((field_name, "Unknown"))
    
    # Format as "Field: Value" lines
    formatted_description = "\n".join([f"{field}: {value}" for field, value in predicted_fields])
    
    return formatted_description


def evaluate_model(
    model: PointQwenLoRA,
    dataset,
    config: Dict,
    num_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_predictions: bool = True,
    field_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate model on dataset and compute metrics.
    
    Args:
        model: Trained PointQwenLoRA model
        dataset: Dataset to evaluate on
        config: Configuration dictionary
        num_samples: Number of samples to evaluate (None = all)
        output_dir: Directory to save results
        save_predictions: Whether to save predictions to file
        field_names: List of field names to predict (if None, uses default fields)
    
    Returns:
        Dictionary of computed metrics
    """
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    print(f"Dataset size: {len(dataset)}")
    print(f"Evaluating on: {num_samples if num_samples else 'all'} samples")
    print(f"Decoding method: Field-by-field prediction")
    
    model.eval()
    
    # Define default field names if not provided
    if field_names is None:
        field_names = [
            "Overbite",
            "Crowding",
            "Molar occlusion - Right side",
            "Molar occlusion - Left side",
            "Canine occlusion - Right side",
            "Canine occlusion - Left side",
            "Curve of Spee",
            "Curve of Wilson",
            "Midlines",
            "Transverse relationship",
        ]
    
    print(f"Fields to predict: {len(field_names)}")
    for field in field_names:
        print(f"  - {field}")
    
    # Create subset if num_samples specified
    # Otherwise use the full dataset passed (which could be val set or full dataset)
    if num_samples and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
        eval_dataset = Subset(dataset, indices)
    else:
        # Use all samples in the dataset (respects validation split if --use_validation_set was specified)
        eval_dataset = dataset
    
    # DON'T use dataloader with collator - we need original data, not training-modified data
    # The collator picks random fields for training, but we need all fields for evaluation
    
    all_predictions = []
    all_references = []
    all_patient_ids = []
    
    # Generation parameters from config
    gen_config = config['training'].get('generation', {})
    max_new_tokens_per_field = 32  # Low token count for single field prediction
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
            # Get original dataset item (not collated)
            if isinstance(eval_dataset, Subset):
                original_idx = eval_dataset.indices[idx]
                item = eval_dataset.dataset[original_idx]
            else:
                item = eval_dataset[idx]
            
            # Extract data from item
            point_cloud = item['point_cloud'].unsqueeze(0).to(model.device)  # Add batch dimension
            prompt_template = item['instruction']  # Original instruction with placeholder
            ground_truth_full = item['description']  # Full description with all fields
            patient_id = item['patient_id']
            
            # Clean ground truth
            clean_ground_truth = ground_truth_full.replace("<|im_end|>", "").strip()
            if clean_ground_truth.startswith("Patient Intra-Oral Description:"):
                clean_ground_truth = clean_ground_truth.replace("Patient Intra-Oral Description:", "", 1).strip()
            
            # Remove "Missing teeth" field from ground truth (same as training)
            lines = []
            for line in clean_ground_truth.split('\n'):
                line = line.strip()
                if line and 'missing teeth' not in line.lower():
                    lines.append(line)
            clean_ground_truth = '\n'.join(lines)
            
            try:
                # Predict all fields using the new field-by-field approach
                generated_description = predict_all_fields(
                    model=model,
                    point_cloud=point_cloud,
                    base_prompt=prompt_template,
                    field_names=field_names,
                    max_new_tokens=max_new_tokens_per_field,
                    temperature=gen_config.get('temperature', 0.1),
                    top_p=gen_config.get('top_p', 0.9),
                    top_k=gen_config.get('top_k', 1),
                    do_sample=gen_config.get('do_sample', False),
                    repetition_penalty=gen_config.get('repetition_penalty', 1.0),
                    no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 0),
                )
                
                all_predictions.append(generated_description)
                all_references.append(clean_ground_truth)
                all_patient_ids.append(patient_id)
                
            except Exception as e:
                print(f"\n⚠️  Error generating for patient {patient_id}: {e}")
                traceback.print_exc()
                # Add empty prediction to maintain alignment
                all_predictions.append("")
                all_references.append(clean_ground_truth)
                all_patient_ids.append(patient_id)
    
    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)
    
    metrics_dict_rounded = {}
    
    try:
        metrics_dict = compute_all_metrics(
            predictions=all_predictions,
            references=all_references,
            include_sbert=config['training'].get('compute_sbert', True),
            sbert_device=str(model.device),
            return_per_field=config['training'].get('return_per_field_accuracy', True)
        )
        
        # Round all metrics to 2 decimal places
        def round_metrics(d):
            """Recursively round all float values in dict to 2 decimal places."""
            if isinstance(d, dict):
                return {k: round_metrics(v) for k, v in d.items()}
            elif isinstance(d, float):
                return round(d, 2)
            else:
                return d
        
        metrics_dict_rounded = round_metrics(metrics_dict)
        
        # Print metrics
        print(f"\n{'Metric':<30} {'Score':>10}")
        print(f"{'-'*40}")
        print(f"{'Field Accuracy':<30} {metrics_dict_rounded.get('accuracy', 0.0):>10.2f}")
        print(f"{'Field Coverage':<30} {metrics_dict_rounded.get('coverage', 0.0):>10.2f}")
        print(f"{'Precision':<30} {metrics_dict_rounded.get('precision', 0.0):>10.2f}")
        print(f"{'Recall':<30} {metrics_dict_rounded.get('recall', 0.0):>10.2f}")
        print(f"{'F1-Score':<30} {metrics_dict_rounded.get('f1', 0.0):>10.2f}")
        print(f"{'BLEU-1':<30} {metrics_dict_rounded.get('bleu-1', 0.0):>10.2f}")
        print(f"{'ROUGE-L (F1)':<30} {metrics_dict_rounded.get('rouge-l-f', 0.0):>10.2f}")
        print(f"{'METEOR':<30} {metrics_dict_rounded.get('meteor', 0.0):>10.2f}")
        if 'sbert-sim' in metrics_dict_rounded:
            print(f"{'Sentence-BERT Similarity':<30} {metrics_dict_rounded.get('sbert-sim', 0.0):>10.2f}")
        
        if 'per_field' in metrics_dict_rounded and isinstance(metrics_dict_rounded['per_field'], dict):
            print(f"\n{'Per-Field Accuracy':^40}")
            print(f"{'-'*40}")
            for field, score in sorted(metrics_dict_rounded['per_field'].items()):
                field_name = field.replace('_', ' ').title()
                print(f"  {field_name:<28} {score:>10.2f}")
        
        print(f"{'='*40}\n")
        
    except Exception as e:
        print(f"\n⚠️  Error computing metrics: {e}")
        traceback.print_exc()
        metrics_dict_rounded = {}
    
    # Print sample predictions
    num_samples_to_show = min(5, len(all_predictions))
    if num_samples_to_show > 0:
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS")
        print("="*70)
        
        for i in range(num_samples_to_show):
            print(f"\n{'─'*70}")
            print(f"SAMPLE {i+1}: Patient {all_patient_ids[i]}")
            print(f"{'─'*70}")
            
            print(f"\n✅ GROUND TRUTH:")
            print(f"{'─'*70}")
            print(all_references[i])
            
            print(f"\n🤖 PREDICTION:")
            print(f"{'─'*70}")
            print(all_predictions[i])
            print(f"{'─'*70}")
        
        print("\n" + "="*70)
    
    # Save results to file if output_dir specified
    if output_dir and save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics (rounded to 2 decimal places)
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict_rounded, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_file}")
        
        # Save predictions (without prompts, as requested)
        predictions_file = os.path.join(output_dir, 'predictions.json')
        predictions_data = [
            {
                'patient_id': patient_id,
                'ground_truth': ref,
                'prediction': pred,
            }
            for patient_id, ref, pred in zip(
                all_patient_ids, all_references, all_predictions
            )
        ]
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"✓ Predictions saved to: {predictions_file}")
    
    return metrics_dict_rounded


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PointQwen model")
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=False, default=None,
                        help='Path to model checkpoint directory (if not provided, uses random initialization)')
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='Path to config file (required if --checkpoint not provided)')
    
    # Data arguments (can override config)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to point cloud data directory (overrides config)')
    parser.add_argument('--captions_dir', type=str, default=None,
                        help='Path to captions directory (overrides config)')
    parser.add_argument('--instruction_template', type=str, default=None,
                        help='Path to instruction template file (overrides config)')
    
    # Evaluation arguments
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: checkpoint_dir/eval_results)')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save predictions to file')
    parser.add_argument('--use_validation_set', action='store_true',
                        help='Use validation split from training (default: use full dataset)')
    
    # Generation arguments (can override config)
    parser.add_argument('--max_new_tokens_per_field', type=int, default=32,
                        help='Maximum new tokens to generate per field (default: 32)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Generation temperature (overrides config)')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use sampling for generation')
    
    # Metrics arguments
    parser.add_argument('--compute_sbert', action='store_true',
                        help='Compute Sentence-BERT similarity (slower)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    if args.checkpoint:
        # Load config from checkpoint (if exists) or from --config
        checkpoint_config_path = os.path.join(args.checkpoint, 'training_config.yaml')
        
        if os.path.exists(checkpoint_config_path):
            print(f"\nLoading config from checkpoint: {checkpoint_config_path}")
            with open(checkpoint_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # If user also provided a config, warn them it will be ignored (except for overrides)
            if args.config:
                print(f"⚠️  Note: Checkpoint has training_config.yaml, ignoring --config {args.config}")
                print(f"         (Command-line overrides will still be applied)")
        elif args.config:
            print(f"\nLoading config from: {args.config}")
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(
                "No config file found! Either:\n"
                "  1. Checkpoint must contain training_config.yaml, OR\n"
                "  2. You must provide --config argument"
            )
    else:
        # No checkpoint - random initialization mode
        if not args.config:
            raise ValueError(
                "When --checkpoint is not provided, you must specify --config for random initialization"
            )
        print(f"\nLoading config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['point_cloud_dir'] = args.data_dir
    if args.captions_dir:
        config['data']['captions_dir'] = args.captions_dir
    if args.instruction_template:
        config['data']['instruction_template_path'] = args.instruction_template
    if args.temperature is not None:
        config['training']['generation']['temperature'] = args.temperature
    if args.do_sample:
        config['training']['generation']['do_sample'] = True
    if args.compute_sbert:
        config['training']['compute_sbert'] = True
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if args.checkpoint:
            output_dir = os.path.join(args.checkpoint, 'eval_results')
        else:
            output_dir = 'eval_results_random'
    
    # Load or create model
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint, config)
    else:
        model = create_random_model(config)
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    print(f"Point cloud dir: {config['data']['point_cloud_dir']}")
    print(f"Captions dir: {config['data']['captions_dir']}")
    
    full_dataset = IOSPointCloudDataset(
        point_cloud_dir=config['data']['point_cloud_dir'],
        captions_dir=config['data']['captions_dir'],
        num_points=config['data']['num_points'],
        normalize=config['data']['normalize'],
        augment=False,  # No augmentation for evaluation
        instruction_template_path=config['data'].get('instruction_template_path')
    )
    
    print(f"✓ Full dataset loaded: {len(full_dataset)} samples")
    
    # Create validation split if requested
    if args.use_validation_set:
        validation_split = config['training'].get('validation_split', 0.05)
        print(f"\n📊 Creating validation split (split={validation_split})...")
        
        # Use same seed as training for reproducibility
        seed = config['training'].get('seed', args.seed)
        
        # Split dataset
        total_size = len(full_dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        # Use same random split as training
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        dataset = val_dataset
        print(f"✓ Using validation split: {len(dataset)} samples (train: {train_size}, val: {val_size})")
    else:
        dataset = full_dataset
        print(f"✓ Using full dataset: {len(dataset)} samples")
    
    print("="*70)
    
    # Run evaluation
    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        config=config,
        num_samples=args.num_samples,
        output_dir=output_dir,
        save_predictions=not args.no_save,
        field_names=None,  # Use default fields
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
