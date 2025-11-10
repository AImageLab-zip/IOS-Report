# PointQwen LoRA Fine-tuning Implementation

## Overview

This implementation adds LoRA (Low-Rank Adaptation) fine-tuning capability to PointQwen for Stage 2 training, where:
- **PointEncoder**: Frozen (uses pretrained weights)
- **Projection Layer**: Trainable (loads pretrained weights from Stage 1)
- **Qwen3-VL-4B**: Base model frozen + LoRA adapters trainable

## Files Created

### 1. Configuration File
**Location**: `src/Qwen3-IOS/configs/finetune_lora_config.yaml`

Key configuration sections:
- **Model Configuration**:
  - `train_lora: true` - Enables LoRA training
  - `pretrained_projector: /path/to/weights.pt` - Path to load pretrained projection weights
  - LoRA parameters (r=16, alpha=16, dropout=0.0, etc.)
  
- **Training Configuration**:
  - `lora_lr: 2.0e-4` - Separate learning rate for LoRA adapters
  - Uses `adamw_8bit` optimizer for memory efficiency
  - Same batch size, gradient accumulation, and validation settings as original

### 2. Training Script
**Location**: `src/Qwen3-IOS/training/lora_qwen.py`

Main components:

#### `PointQwenLoRA` Class
Custom model wrapper that combines:
- Frozen PointEncoder (loads pretrained weights)
- Trainable Projection layer (loads pretrained weights)
- Qwen with LoRA adapters (applied via Unsloth)

Key features:
- **Custom embedding injection**: Maintains the existing approach of replacing image pad tokens with point cloud embeddings
- **`forward()` method**: Handles the full multimodal forward pass with custom embedding injection
- **`generate()` method**: Text generation conditioned on point clouds
- **`encode_point_cloud()`**: Encodes point clouds through frozen encoder and trainable projection

#### `LoRATrainer` Class
Wrapper around Hugging Face's `SFTTrainer` that:
- Uses your existing `IOSCollator` for data collation
- Maintains same validation and metrics computation as original trainer
- Supports WandB logging
- Runs comprehensive evaluation with BLEU, ROUGE, METEOR, Sentence-BERT metrics

## Key Design Decisions

### 1. Unsloth Integration
- Uses `FastVisionModel` from Unsloth (which is just an alias to `FastModel`)
- Unsloth's LoRA implementation supports `inputs_embeds` in forward pass ✓
- Our custom embedding injection works seamlessly with Unsloth's optimizations

### 2. Model Architecture
```python
PointQwenLoRA(
    qwen_lora_model,      # Unsloth FastVisionModel with LoRA
    point_encoder,        # Frozen, pretrained
    projection,           # Trainable, pretrained
)
```

### 3. Forward Pass Flow
```
point_cloud → point_encoder (frozen) → projection (trainable) 
    → visual_tokens → inject into text embeddings 
    → qwen (base frozen + LoRA trainable) → output
```

### 4. Data Handling
- Uses existing `IOSPointCloudDataset` and `IOSCollator`
- No changes needed to dataset format
- SFTTrainer configured with:
  - `remove_unused_columns=False`
  - `dataset_text_field=""`
  - `skip_prepare_dataset=True`

### 5. Validation & Metrics
- Same comprehensive metrics as original trainer:
  - Field accuracy and coverage
  - BLEU, ROUGE, METEOR
  - Sentence-BERT similarity
  - Per-field accuracy breakdown
- Generates sample outputs during validation

## Usage

### 1. Update Configuration
Edit `configs/finetune_lora_config.yaml`:
```yaml
model:
  pretrained_point_encoder: /path/to/stage1/point_encoder/best.pth
  pretrained_projector: /path/to/stage1/projection/weights.pt
  
  lora:
    r: 16                    # LoRA rank
    lora_alpha: 16          # LoRA alpha
    use_4bit: false         # Set true to save memory

training:
  lora_lr: 2.0e-4          # Learning rate for LoRA
  projection_lr: 1.0e-4    # Learning rate for projection
  num_epochs: 10
```

### 2. Run Training
```bash
python src/Qwen3-IOS/training/lora_qwen.py \
    --config src/Qwen3-IOS/configs/finetune_lora_config.yaml \
    --use_wandb \
    --run_name pointqwen_lora_experiment1
```

### 3. Monitor Training
- WandB dashboard will show:
  - Training loss
  - Validation metrics (accuracy, BLEU, ROUGE, etc.)
  - Per-field accuracy breakdown
  - Sample generations

## Loading Pretrained Projection Weights

The script handles multiple checkpoint formats:

### Option 1: Full model checkpoint (from train.py)
```python
{
    'model_state_dict': {
        'projection.layer1.weight': ...,
        'projection.layer1.bias': ...,
        ...
    },
    ...
}
```

### Option 2: Projection-only checkpoint
```python
{
    'projection': {
        'layer1.weight': ...,
        'layer1.bias': ...,
        ...
    }
}
```

### Option 3: Direct state dict
```python
{
    'layer1.weight': ...,
    'layer1.bias': ...,
    ...
}
```

The script automatically detects the format and loads appropriately.

## Advantages of This Approach

1. **Memory Efficient**: LoRA dramatically reduces trainable parameters
2. **Faster Training**: Only trains projection + LoRA adapters
3. **Preserves Pretrained Knowledge**: Frozen PointEncoder maintains learned features
4. **Flexible**: Can adjust LoRA rank to trade off between capacity and efficiency
5. **Compatible**: Uses existing datasets, metrics, and validation code

## Dependencies

Add to `requirements.txt`:
```
unsloth>=2025.1
trl>=0.7.0
```

Install:
```bash
pip install unsloth trl
```

## Comparison with Original Training

| Component | Original (train.py) | LoRA (lora_qwen.py) |
|-----------|-------------------|---------------------|
| PointEncoder | Trainable/Frozen | **Frozen** (pretrained) |
| Projection | Trainable | **Trainable** (pretrained) |
| Qwen | Frozen | **LoRA adapters** (trainable) |
| Semantic Loss | Supported | **Not included** (simpler) |
| Trainer | Custom loop | **SFTTrainer** (HF) |
| Validation | Full metrics | **Same metrics** |
| Dataset | IOSPointCloudDataset | **Same** |

## Notes

1. **Epoch-based freezing removed**: The logic that froze/unfroze PointEncoder based on epoch number has been removed as requested.

2. **SFTTrainer benefits**:
   - Built-in distributed training
   - Better integration with HF ecosystem
   - Automatic mixed precision
   - Checkpoint management

3. **Custom embedding injection preserved**: The critical feature of injecting point cloud embeddings at image pad token positions is maintained exactly as in the original implementation.

4. **4-bit quantization option**: Can enable `use_4bit: true` in config to further reduce memory if needed.

## Troubleshooting

### If projection weights don't load:
Check the checkpoint structure:
```python
import torch
ckpt = torch.load('path/to/checkpoint.pt')
print(ckpt.keys())
```

### If out of memory:
1. Enable 4-bit quantization: `use_4bit: true`
2. Reduce batch size
3. Increase gradient accumulation steps
4. Reduce LoRA rank

### If training is slow:
1. Check if flash attention is enabled
2. Verify gradient checkpointing is on
3. Use `adamw_8bit` optimizer

## Future Enhancements

Potential improvements:
1. Add DeepStack projection support for LoRA training
2. Implement checkpoint resuming from previous LoRA runs
3. Add GGUF export for inference optimization
4. Support multi-GPU training with FSDP
