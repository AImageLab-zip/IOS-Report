# External Model Evaluation

This module provides evaluation scripts for comparing external API-based models (OpenAI GPT5.2, Google Gemini3, DeepSeek) with IOS-Qwen on intraoral photo analysis tasks.

## Features

- **Unified API Interface**: Common base class for all external models
- **Multi-Model Support**: OpenAI, Google Gemini, and DeepSeek
- **Same Metrics**: Uses the same evaluation metrics as IOS-Qwen (Field Accuracy, BLEU, ROUGE, METEOR, Sentence-BERT)
- **Batch Evaluation**: Evaluate and compare multiple models at once
- **Visualization**: Automatic generation of comparison charts and heatmaps

## Installation

### 1. Install Dependencies

```bash
cd /work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS
pip install -r requirements_external.txt
```

### 2. Set Up API Keys

Set environment variables for the APIs you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

Or add them directly to the config file (see Configuration section).

## Configuration

Edit [`configs/evaluation/external_model_baselines.yaml`](../configs/evaluation/external_model_baselines.yaml) to configure:

- **Dataset paths**: Location of intraoral photos and captions
- **Model parameters**: Temperature, max tokens, rate limits
- **API credentials**: Can be set here or via environment variables
- **Prompts**: System and user prompts for each model

Example:

```yaml
models:
  openai:
    model_name: "gpt-4-vision-preview"  # Update to "gpt-5.2" when available
    api_key: null  # Or set directly: "sk-..."
    temperature: 0.7
    max_tokens: 512
    system_prompt: "You are a dental expert..."
    user_prompt_template: "Analyze these intraoral photos..."
```

## Usage

### Evaluate a Single Model

```bash
python scripts/evaluate_external_models.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --model openai \
    --num_samples 50 \
    --output_dir outputs/openai_eval
```

**Arguments:**
- `--config`: Path to configuration file
- `--model`: Model to evaluate (`openai`, `gemini`, or `deepseek`)
- `--num_samples`: Number of samples to evaluate (optional, defaults to all)
- `--output_dir`: Output directory (optional)

### Evaluate All Models and Compare

```bash
python scripts/evaluate_all_external.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --models openai gemini deepseek \
    --num_samples 100 \
    --output_dir outputs/external_models_comparison
```

**Arguments:**
- `--config`: Path to configuration file
- `--models`: Space-separated list of models to evaluate (optional, defaults to all)
- `--num_samples`: Number of samples to evaluate (optional, defaults to all)
- `--output_dir`: Output directory for comparison results

### Quick Test

Test with a small subset to verify API credentials:

```bash
# Test OpenAI
python scripts/evaluate_external_models.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --model openai \
    --num_samples 5

# Test Gemini
python scripts/evaluate_external_models.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --model gemini \
    --num_samples 5
```

## Output Structure

```
outputs/
├── openai_eval/
│   ├── results_openai.json          # Full results with predictions
│   ├── metrics_openai.txt           # Metrics summary
│   └── {patient_id}_prediction.txt  # Individual predictions
├── gemini_eval/
│   └── ...
├── deepseek_eval/
│   └── ...
└── external_models_comparison/
    ├── model_comparison.csv         # Comparison table
    ├── model_comparison.png         # Bar chart comparison
    ├── per_field_comparison.csv     # Per-field accuracy table
    └── per_field_heatmap.png        # Heatmap visualization
```

## Metrics

The evaluation computes the following metrics:

1. **Field-level Accuracy**: Percentage of correctly predicted field values (exact match)
2. **BLEU-1**: Unigram precision
3. **ROUGE-L**: Longest common subsequence F1 score
4. **METEOR**: Morphology-aware matching score
5. **Sentence-BERT**: Semantic similarity using sentence embeddings

All metrics are computed using the same code as IOS-Qwen evaluation for fair comparison.

## Customization

### Adding a New Model

1. Create a new client in `external_models/`:

```python
# external_models/newmodel_client.py
from .base import BaseExternalModel

class NewModel(BaseExternalModel):
    def __init__(self, api_key, model_name, ...):
        super().__init__(...)
        # Initialize API client
    
    def generate(self, image_paths, field_names=None):
        # Implement generation logic
        return generated_text
```

2. Add to `external_models/__init__.py`:

```python
from .newmodel_client import NewModel
__all__ = [..., 'NewModel']
```

3. Update evaluation scripts to support the new model.

### Customizing Prompts

Edit the `system_prompt` and `user_prompt_template` in the config file:

```yaml
models:
  openai:
    system_prompt: "Your custom system prompt..."
    user_prompt_template: |
      Your custom user prompt with {fields} placeholder...
```

The `{fields}` placeholder will be replaced with the list of fields extracted from the reference captions.

## Troubleshooting

### API Rate Limits

If you hit rate limits, adjust `rate_limit_delay` in the config:

```yaml
models:
  openai:
    rate_limit_delay: 2.0  # Wait 2 seconds between calls
```

### Model Name Not Found

Update the `model_name` in the config to match the actual API model identifier:

```yaml
models:
  openai:
    model_name: "gpt-4-vision-preview"  # Check OpenAI docs for correct name
```

### API Errors

Check the error messages and verify:
- API keys are correct and have sufficient credits
- Model names are correct
- API endpoints are accessible (check firewall/proxy settings)

### Memory Issues

If processing many images causes memory issues, reduce batch size by using `--num_samples`:

```bash
# Process in chunks
python scripts/evaluate_external_models.py ... --num_samples 50
```

## Notes

- **Cost**: External API calls have costs associated with them. Monitor your usage!
- **Rate Limits**: Most APIs have rate limits. The scripts include delays to avoid hitting them.
- **Privacy**: Intraoral photos contain medical data. Ensure compliance with data privacy regulations when using external APIs.
- **Model Availability**: Update model names in config as new versions become available (e.g., when GPT-5.2 is released).

## Example Results

After running the comparison script, you'll see output like:

```
================================================================================
MODEL COMPARISON
================================================================================
    Model  Field Accuracy  BLEU-1  ROUGE-L  METEOR  SBERT
  OPENAI            72.5    45.3     68.2    52.1   85.3
  GEMINI            68.9    42.1     65.7    49.8   82.6
DEEPSEEK            65.2    39.8     62.3    47.2   79.8
================================================================================
```

With visualizations saved to the output directory.

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{ios-qwen-external-eval,
  title={External Model Evaluation Framework for Intraoral Photo Analysis},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/...}
}
```
