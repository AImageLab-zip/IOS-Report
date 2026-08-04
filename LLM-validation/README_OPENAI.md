# OpenAI API Report Generator
This module provides an alternative to the local model-based report generator, using OpenAI's API instead.
## Features
- Uses OpenAI's vision models (GPT-4o, GPT-4 Turbo, etc.) via API
- Processes intraoral scan images to generate orthodontic reports
- Same dataset and prompt structure as local models
- Faster inference (no local GPU needed)
- Pay-per-use pricing model
## Setup
### 1. Install Dependencies
```bash
pip install -r requirements_openai.txt
```
### 2. Set API Key
Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```
Alternatively, you can add it to your `.bashrc` or `.zshrc`:
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```
## Usage
### Basic Usage
Process the dataset with default OpenAI models (gpt-4o, gpt-4-turbo, gpt-4o-mini):
```bash
python src/main_openai.py
```
### Specify Custom Models
Use specific OpenAI models:
```bash
python src/main_openai.py --models gpt-4o gpt-4-turbo
```
### Full Options
```bash
python src/main_openai.py \
    --root_dir _data/Dataset_FerraraDump_400 \
    --captions_dir _data/captions \
    --output_dir output \
    --models gpt-4o gpt-4o-mini \
    --system_prompt_path templates/system.txt \
    --user_prompt_path templates/user.txt \
    --skip_existing
```
### Command-line Arguments
- `--root_dir`: Root directory of the dataset (default: `_data/Dataset_FerraraDump_400`)
- `--captions_dir`: Directory containing caption JSON files (default: `_data/captions`)
- `--output_dir`: Directory to save generated reports (default: `output`)
- `--models`: List of OpenAI model names to test (default: `gpt-4o gpt-4-turbo gpt-4o-mini`)
- `--system_prompt_path`: Path to system prompt template (default: `templates/system.txt`)
- `--user_prompt_path`: Path to user prompt template (default: `templates/user.txt`)
- `--skip_existing`: Skip models whose output directories already exist
- `--image_size`: Optional image resize (tuple, e.g., `(512, 512)`)
## Available Models
Common OpenAI vision models:
- `gpt-4o` - Latest GPT-4 with vision (recommended)
- `gpt-4o-mini` - Smaller, faster, cheaper GPT-4 variant
- `gpt-4-turbo` - GPT-4 Turbo with vision
- `gpt-4-vision-preview` - Earlier GPT-4 vision preview
For the most up-to-date model list, see: https://platform.openai.com/docs/models
## Output Structure
Generated reports are saved in:
```
output/
├── gpt-4o/
│   ├── 201_1979.txt
│   ├── 202_1980.txt
│   └── ...
├── gpt-4-turbo/
│   ├── 201_1979.txt
│   └── ...
└── ...
```
Each file contains the generated orthodontic report for one patient.
## Cost Estimation
OpenAI API pricing (as of 2025):
- **GPT-4o**: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens
- **GPT-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- Images: ~$0.00765 per image (1024x1024 high detail)
Approximate cost per patient (5 images + prompts):
- GPT-4o: ~$0.05-0.10
- GPT-4o-mini: ~$0.005-0.01
For 400 patients:
- GPT-4o: ~$20-40
- GPT-4o-mini: ~$2-4
## Differences from Local Models
### Advantages
- No GPU required
- Faster inference (typically 5-10 seconds per patient)
- Access to latest models
- Scalable without hardware constraints
### Disadvantages
- Requires internet connection
- Pay-per-use (cost per request)
- Data sent to OpenAI servers
- Rate limits apply (see: https://platform.openai.com/docs/guides/rate-limits)
## Rate Limits
OpenAI enforces rate limits. If you hit rate limits:
1. Add delays between requests (use `time.sleep()`)
2. Implement exponential backoff on errors
3. Request higher rate limits from OpenAI
4. Use batch processing with smaller batches
## Troubleshooting
### "OPENAI_API_KEY not set" error
Set your API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```
### Rate limit errors
Add a delay in the code or reduce batch size. The generator already handles basic retries.
### "Model not found" error
Make sure you're using a valid model name. Check available models at:
https://platform.openai.com/docs/models
### Image size errors
If images are too large, use `--image_size` to resize them:
```bash
python src/main_openai.py --image_size "(512, 512)"
```
## API Reference
See OpenAI's official documentation:
- Chat Completions API: https://platform.openai.com/docs/api-reference/chat
- Vision Guide: https://platform.openai.com/docs/guides/vision
- Rate Limits: https://platform.openai.com/docs/guides/rate-limits
## Security Notes
- Never commit your API key to version control
- Use environment variables or secure secret management
- Monitor your API usage and costs in the OpenAI dashboard
- Consider using API key restrictions (IP whitelisting, usage limits)
## Example Output
```
Model: gpt-4o
Patient ID: 201_1979
Number of images: 5
Ground Truth Report:
--------------------------------------------------------------------------------
Overbite: Normal
Crowding: Mild in the upper arch
Molar occlusion - Right side: Class I
...
--------------------------------------------------------------------------------
Generated Report:
--------------------------------------------------------------------------------
Overbite: Normal
Crowding: Mild in the upper arch
Molar occlusion - Right side: Class I
...
--------------------------------------------------------------------------------
Saved to: output/gpt-4o/201_1979.txt
```
## License
Same as the main project.
