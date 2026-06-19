# Summary: External Model Evaluation Setup

## ✅ What Was Created

### 1. **API Client Modules** (`external_models/`)
   - `base.py` - Base class with common functionality (image loading, encoding, field extraction)
   - `openai_client.py` - OpenAI GPT-4/GPT-5.2 client
   - `gemini_client.py` - Google Gemini client (fixed deprecation warning)
   - `deepseek_client.py` - DeepSeek client
   - `__init__.py` - Package exports

### 2. **Evaluation Scripts** (`scripts/`)
   - `evaluate_external_models.py` - Single model evaluation with metrics
   - `evaluate_all_external.py` - Batch evaluation and comparison with visualizations
   - `scripts/evaluation/external_models/quick_eval_external_models.sh` - Quick start bash script
   - `test_external_setup.py` - Setup verification test

### 3. **Configuration**
   - `configs/evaluation/external_model_baselines.yaml` - **Uses EXACT same prompts as IOS-Qwen**
   - `requirements_external.txt` - Additional dependencies

### 4. **Documentation**
   - `EXTERNAL_MODELS_README.md` - Comprehensive usage guide
   - `external_models/README.md` - Detailed technical documentation

## 🎯 Key Features

✅ **Same Prompts**: All models use identical prompts from `qwen-template-iop.json`
✅ **Same Metrics**: Uses existing `validation/metrics.py` for fair comparison
✅ **Multi-Model**: Evaluate OpenAI, Gemini, and DeepSeek
✅ **Batch Processing**: Compare all models at once
✅ **Visualizations**: Automatic charts and heatmaps

## 🚀 How to Use

### Quick Test (5 samples)
```bash
cd /work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS

# Set API key
export OPENAI_API_KEY="your-key"

# Run quick test
./scripts/evaluation/external_models/quick_eval_external_models.sh openai
```

### Full Evaluation
```bash
python scripts/evaluate_external_models.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --model openai \
    --num_samples 100
```

### Compare All Models
```bash
# Set all API keys
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"

# Run comparison
python scripts/evaluate_all_external.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --models openai gemini deepseek \
    --num_samples 100
```

## 📁 File Structure

```
src/Qwen3-IOS/
├── external_models/           # API clients
│   ├── __init__.py
│   ├── base.py
│   ├── openai_client.py
│   ├── gemini_client.py
│   ├── deepseek_client.py
│   ├── test_external_setup.py
│   └── README.md
│
├── scripts/                   # Evaluation scripts
│   ├── evaluate_external_models.py
│   ├── evaluate_all_external.py
│   └── evaluation/external_models/quick_eval_external_models.sh
│
├── configs/                   # Configuration
│   └── evaluation/external_model_baselines.yaml
│
├── requirements_external.txt  # Dependencies
├── EXTERNAL_MODELS_README.md  # Main documentation
└── validation/               # (existing) Metrics module
    └── metrics.py
```

## 📊 Output Example

After running evaluation, you get:

```
outputs/
├── openai_eval/
│   ├── results_openai.json          # Full results with predictions
│   ├── metrics_openai.txt           # Summary metrics
│   └── {patient}_prediction.txt     # Individual predictions
│
└── external_models_comparison/      # (when comparing multiple models)
    ├── model_comparison.csv         # Table with all metrics
    ├── model_comparison.png         # Bar chart comparison
    ├── per_field_comparison.csv     # Per-field accuracy table
    └── per_field_heatmap.png        # Heatmap visualization
```

## 🔍 Verification

Test setup (no API calls):
```bash
python external_models/test_external_setup.py
```

Expected output:
```
✓ BaseExternalModel tests passed!
✓ OpenAI client initialized
✓ Gemini client initialized
✓ DeepSeek client initialized
✓ Config loaded successfully
✓ Loaded N samples
```

## ⚙️ Configuration Highlights

The config file (`configs/evaluation/external_model_baselines.yaml`) contains:

1. **Dataset Paths**
   - Points to existing Dataset_FerraraDump_400
   - Uses existing auto-captioning/real_captions

2. **Model Settings**
   - Temperature: 0.7
   - Max tokens: 512
   - Rate limit delays to avoid API throttling

3. **Prompts** - EXACT SAME AS IOS-QWEN
   - System prompt from `qwen-template-iop.json`
   - User prompt from `qwen-template-iop.json`
   - All field specifications included

## 💡 Important Notes

1. **API Costs**: External APIs charge per request. Start with small samples!
2. **Rate Limits**: Built-in delays prevent hitting API rate limits
3. **Privacy**: Medical data - ensure GDPR/HIPAA compliance
4. **Model Names**: Update when newer models (GPT-5.2, Gemini-3) are released

## 🔄 Next Steps

1. **Install dependencies**:
   ```bash
   pip install openai google-generativeai requests pandas matplotlib seaborn
   ```

2. **Set API keys**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   export DEEPSEEK_API_KEY="your-key"
   ```

3. **Run test**:
   ```bash
   ./scripts/evaluation/external_models/quick_eval_external_models.sh openai
   ```

4. **Full evaluation**:
   ```bash
   python scripts/evaluate_external_models.py \
       --config configs/evaluation/external_model_baselines.yaml \
       --model openai
   ```

## ✨ Highlights

- ✅ All models use **exact same prompts** as IOS-Qwen
- ✅ All models use **exact same metrics** via `validation/metrics.py`
- ✅ Easy to add new models by extending `BaseExternalModel`
- ✅ Comprehensive error handling and retry logic
- ✅ Automatic visualization generation
- ✅ Individual prediction files for analysis

---

**Everything is ready to go!** Just set your API keys and run the quick test.
