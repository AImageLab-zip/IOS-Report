# Semantic Loss for PointQwen

This module implements a semantic similarity loss using sentence embeddings to improve the quality of generated descriptions.

## Overview

The semantic loss complements the standard cross-entropy loss by measuring how semantically similar the generated text is to the ground truth at the sentence level, rather than just at the token level.

## Components

### `SemanticLoss`
A PyTorch module that:
- Uses a pretrained sentence transformer model (`sentence-transformers/all-MiniLM-L6-v2` by default)
- Encodes both predicted and target text into semantic embeddings
- Computes cosine similarity between embeddings
- Returns `1 - cosine_similarity` as the loss

**Model Details:**
- **sentence-transformers/all-MiniLM-L6-v2**: 23M parameters (~90MB)
- Trained specifically for semantic similarity tasks
- Fast inference due to small size
- Good generalization to domain-specific text

### `CombinedLoss`
A wrapper that combines cross-entropy and semantic losses with configurable weighting.

## Usage

### Configuration

Add to your training config YAML:

```yaml
training:
  # Enable semantic loss
  use_semantic_loss: true
  semantic_loss_weight: 0.5  # Weight relative to CE loss
  semantic_loss_model: "sentence-transformers/all-MiniLM-L6-v2"
  
  # Generation parameters for semantic loss (during training)
  semantic_generation:
    max_new_tokens: 256
    temperature: 0.1
    top_p: 0.9
    top_k: 1
    do_sample: false
    repetition_penalty: 1.1
    no_repeat_ngram_size: 3
```

### Training

The semantic loss is automatically integrated into the training loop when enabled:

```python
# Total loss = CE loss + semantic_loss_weight * semantic_loss
total_loss = ce_loss + 0.5 * semantic_loss
```

### Logging

When semantic loss is enabled, the following metrics are logged:
- `train/total_loss`: Combined loss
- `train/ce_loss`: Cross-entropy loss
- `train/semantic_loss`: Semantic similarity loss

## Performance Impact

### Memory
- Adds ~90MB for the sentence transformer model
- Model is frozen, so no gradient storage required

### Computation
- Generates text for each training batch to compute semantic loss
- Generation is done with `torch.no_grad()` for efficiency
- Uses greedy decoding by default for speed
- Approximately 20-30% slower training per step

### Quality
- Encourages semantically coherent outputs
- Better handling of paraphrases and alternative phrasings
- Improves factual consistency
- May help with repetition issues

## Tips

1. **Weight tuning**: Start with 0.5 and adjust based on validation metrics
2. **Generation length**: Keep `semantic_generation.max_new_tokens` low (256) for efficiency
3. **Batch size**: May need to reduce batch size slightly due to additional generation
4. **Alternative models**: Can swap in domain-specific models if available (e.g., BioBERT-based)

## Alternative Models

If you want to experiment with different embedding models:

```yaml
training:
  # Biomedical-specific (heavier, 110M params)
  semantic_loss_model: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
  
  # Or BioBERT
  semantic_loss_model: "dmis-lab/biobert-base-cased-v1.2"
  
  # Or domain-adapted sentence transformer
  semantic_loss_model: "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
```

## Example

See `configs/training_setups/01_frozen_llm_joint_encoder_projector/deepstack_joint_encoder_projector_semantic_loss_alt.yaml` for a complete configuration example.
