#!/bin/bash
#SBATCH --job-name=PointQwen-Train
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --output=logs/pointqwen-train_%j.out
#SBATCH --error=logs/pointqwen-train_%j.err
#SBATCH --account=grana_maxillo
#SBATCH --array=0-4

source /homes/llumetti/ios_draftreport_venv/bin/activate

# Run training for the fold specified by SLURM_ARRAY_TASK_ID
python training/train_02a_pointtransformer_field_classification.py \
    --config configs/training_setups/02_classification_pretrain_then_projector/01_pointtransformer_field_classification.yaml \
    --use_wandb \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --num_folds 5
