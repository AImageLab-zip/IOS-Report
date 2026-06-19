#!/bin/bash
# Quick start script for evaluating external models
# This demonstrates how to run evaluations with the exact same setup as IOS-Qwen

# Set your working directory
cd /work/grana_maxillo/IOS-DraftReport/src/Qwen3-IOS

echo "=========================================="
echo "External Model Evaluation - Quick Start"
echo "=========================================="
echo ""

# Step 1: Set API keys (you need to do this first!)
echo "Step 1: Setting API Keys"
echo "Make sure you have set your API keys:"
echo "  export OPENAI_API_KEY='your-key-here'"
echo "  export GOOGLE_API_KEY='your-key-here'"
echo "  export DEEPSEEK_API_KEY='your-key-here'"
echo ""

# Check if at least one API key is set
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ] && [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  Warning: No API keys detected!"
    echo "Please set at least one API key before running evaluations."
    echo ""
fi

# Step 2: Test with a small sample
echo "Step 2: Testing with 5 samples"
echo ""

# Choose which model to test
MODEL=${1:-openai}  # Default to openai if no argument provided

echo "Testing model: $MODEL"
echo ""

python scripts/evaluate_external_models.py \
    --config configs/evaluation/external_model_baselines.yaml \
    --model $MODEL \
    --num_samples 5 \
    --output_dir outputs/${MODEL}_test

echo ""
echo "=========================================="
echo "Test complete! Check outputs/${MODEL}_test/ for results"
echo ""
echo "To run full evaluation:"
echo "  python scripts/evaluate_external_models.py --config configs/evaluation/external_model_baselines.yaml --model $MODEL"
echo ""
echo "To compare all models:"
echo "  python scripts/evaluate_all_external.py --config configs/evaluation/external_model_baselines.yaml"
echo "=========================================="
