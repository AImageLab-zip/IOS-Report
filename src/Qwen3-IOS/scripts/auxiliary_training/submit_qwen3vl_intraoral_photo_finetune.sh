#!/bin/bash
#SBATCH --job-name=Qwen3VL-IntraoralPhotos
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen3vl-intraoral_%j.out
#SBATCH --error=logs/qwen3vl-intraoral_%j.err
#SBATCH --account=grana_maxillo

# Qwen3 VL Vision Encoder Fine-tuning with Intraoral Photos
# This script trains ONLY the vision encoder while keeping the LLM frozen
# Uses 5 intraoral photos per patient as input

echo "=========================================="
echo "Qwen3 VL Intraoral Photos Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time: $(date)"
echo "=========================================="

# Activate virtual environment
source /homes/llumetti/ios_draftreport_venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
CONFIG_FILE="${1:-configs/auxiliary_training/qwen3vl_intraoral_photo_finetune.yaml}"
RUN_NAME="${3:-qwen3vl_intraoral_$(date +%Y%m%d_%H%M%S)}"

echo "Configuration:"
echo "  Config: $CONFIG_FILE"
echo "  Run name: $RUN_NAME"
echo "=========================================="
echo ""

# Run training
python training/train_intraoralphotos.py \
    --config "$CONFIG_FILE" \
    --use_wandb \
    --run_name "$RUN_NAME"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
