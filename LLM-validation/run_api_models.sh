#!/bin/bash

# Script to run API model evaluations with batch processing
# This uses concurrent API calls for much faster generation

# Activate virtual environment
source /homes/llumetti/ios_draftreport_venv/bin/activate

cd /work/grana_maxillo/CVPR2026

echo "========================================="
echo "Running API Models with Batch Processing"
echo "========================================="
echo ""
echo "Models: gpt-5.2, gemini3, deepseek3.2"
echo "Mode: Batch processing (concurrent API calls)"
echo ""

# Run with batch mode enabled for parallel processing
python src/main_api.py \
    --models gpt-5.2 gemini3 deepseek3.2 \
    --batch_mode \
    --max_workers 10

echo ""
echo "========================================="
echo "Generation Complete!"
echo "========================================="
echo ""
echo "Running validation metrics..."
echo ""

# Run validation
python src/validation.py \
    --output_dir output \
    --models gpt-5_2 gemini3 \
    2>&1 | tee logs/api_validation.log

echo ""
echo "========================================="
echo "All Done!"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  - output/gpt-5_2/"
echo "  - output/gemini3/"
echo "  - validation_results.csv"
echo ""
