#!/bin/bash
#SBATCH --job-name=PointQwen-LoRA-Train
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/pointqwen-lora-train_%j.out
#SBATCH --error=logs/pointqwen-lora-train_%j.err
#SBATCH --account=grana_maxillo

source /homes/llumetti/ios_draftreport_venv/bin/activate

python training/train_03_lora_qwen_with_projector.py \
    --config configs/training_setups/03_lora_qwen_projector_finetune/qwen_lora_with_trainable_projector.yaml \
    --use_wandb
