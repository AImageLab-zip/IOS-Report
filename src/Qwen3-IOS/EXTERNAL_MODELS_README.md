# External Model Evaluation for IOS-DraftReport

Evaluation framework for comparing external API-based models (OpenAI GPT5.2, Google Gemini3, DeepSeek) with IOS-Qwen using **intraoral photos only**.

## ✨ Key Features

- **Same Prompts**: Uses identical prompts as IOS-Qwen for fair comparison
- **Same Metrics**: Field Accuracy, BLEU-1, ROUGE-L, METEOR, Sentence-BERT
- **Easy Setup**: Simple configuration and command-line interface
- **Batch Evaluation**: Compare multiple models at once with visualizations

## 📋 Quick Start

### 1. Install Dependencies

```bash
cd /work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS
pip install -r requirements_external.txt
```

### 2. Set API Keys

```bash
# OpenAI (required for GPT-4/GPT-5.2)
export OPENAI_API_KEY="sk-your-key-here"

# Google (required for Gemini)
export GOOGLE_API_KEY="your-key-here"

# DeepSeek (required for DeepSeek)
export DEEPSEEK_API_KEY="your-key-here"
```

### 3. Run Quick Test

```bash
# Test with OpenAI (5 samples)
./scripts/evaluation/external_models/quick_eval_external_models.sh openai

# Test with Gemini (5 samples)
./scripts/evaluation/external_models/quick_eval_external_models.sh gemini

# Test with DeepSeek (5 samples)
./scripts/evaluation/external_models/quick_eval_external_models.sh deepseek
```

## 🚀 Usage

### Evaluate Single Model

```bash
python scripts/evaluate_external_models.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --model openai \
    --num_samples 100 \
    --output_dir outputs/openai_eval
```

**Options:**
- `--model`: Choose `openai`, `gemini`, or `deepseek`
- `--num_samples`: Number of patients to evaluate (omit for all)
- `--output_dir`: Where to save results (optional)

### Compare All Models

```bash
python scripts/evaluate_all_external.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --models openai gemini deepseek \
    --num_samples 100
```

This will:
- Evaluate all specified models
- Generate comparison charts
- Create per-field accuracy heatmaps
- Save all results to `outputs/external_models_comparison/`

## 📊 Output Structure

```
outputs/
├── openai_eval/
│   ├── results_openai.json          # Full results
│   ├── metrics_openai.txt           # Metrics summary
│   └── {patient_id}_prediction.txt  # Individual predictions
├── gemini_eval/
│   └── ...
└── external_models_comparison/
    ├── model_comparison.csv         # Comparison table
    ├── model_comparison.png         # Bar charts
    ├── per_field_comparison.csv     # Per-field accuracy
    └── per_field_heatmap.png        # Heatmap visualization
```

## ⚙️ Configuration

Edit [`configs/evaluation/external_model_baselines.yaml`](configs/evaluation/external_model_baselines.yaml):

```yaml
# Dataset paths
data:
  dataset_dir: "/work/grana_maxillo/IOS-DraftReport/_data/Dataset_FerraraDump_400"
  captions_dir: "/work/grana_maxillo/IOS-DraftReport/src/auto-captioning/real_captions"

# Model settings (same prompts as IOS-Qwen)
models:
  openai:
    model_name: "gpt-4-vision-preview"  # Update when GPT-5.2 available
    temperature: 0.7
    max_tokens: 512
```

## 📈 Metrics Explained

All metrics use the same code as IOS-Qwen evaluation:

1. **Field Accuracy**: % of exactly matched field values
2. **BLEU-1**: Unigram precision score
3. **ROUGE-L**: Longest common subsequence F1
4. **METEOR**: Morphology-aware matching
5. **Sentence-BERT**: Semantic similarity (0-1)

## 🔧 Troubleshooting

### API Rate Limits

Increase delay in config:
```yaml
models:
  openai:
    rate_limit_delay: 2.0  # Wait 2 seconds between calls
```

### Model Not Found

Update model name in config:
```yaml
models:
  openai:
    model_name: "gpt-4-vision-preview"  # Check OpenAI docs
```

### Cost Management

Use `--num_samples` to limit evaluation:
```bash
python scripts/evaluate_external_models.py ... --num_samples 10
```

## 📝 Important Notes

⚠️ **Privacy**: Intraoral photos contain medical data. Ensure GDPR/HIPAA compliance when using external APIs.

💰 **Cost**: External API calls have associated costs. Monitor your usage!

🔄 **Prompts**: All models use the **exact same prompts** as IOS-Qwen for fair comparison. See `configs/evaluation/external_model_baselines.yaml`.

## 🎯 Example Output

```
================================================================================
EVALUATION RESULTS
================================================================================

Field-level Accuracy: 68.50%
BLEU-1: 0.4234
ROUGE-L: 0.6512
METEOR: 0.4987
Sentence-BERT: 0.8234

Per-Field Accuracy:
  overbite: 85.00%
  crowding: 72.00%
  molar_right: 65.00%
  molar_left: 63.00%
  ...
```

## 📚 Files Overview

```
external_models/
├── __init__.py              # Package initialization
├── base.py                  # Base class for all models
├── openai_client.py         # OpenAI/GPT client
├── gemini_client.py         # Google Gemini client
├── deepseek_client.py       # DeepSeek client
├── test_external_setup.py   # Setup verification
└── README.md                # This file

scripts/
├── evaluate_external_models.py   # Single model evaluation
├── evaluate_all_external.py      # Multi-model comparison
└── evaluation/external_models/quick_eval_external_models.sh        # Quick start script

configs/
└── external_models_eval.yaml     # Configuration file
```

## 🤝 Support

Issues? Check:
1. API keys are set correctly
2. Model names match API documentation
3. Dataset paths are correct in config
4. Run test script: `python external_models/test_external_setup.py`

---

**Ready to evaluate!** Start with `./scripts/evaluation/external_models/quick_eval_external_models.sh openai` to test your setup.
