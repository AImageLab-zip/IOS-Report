#!/bin/bash
#SBATCH --job-name=PointQwen-Train
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=logs/pointqwen-train_%j.out
#SBATCH --error=logs/pointqwen-train_%j.err
#SBATCH --account=grana_maxillo

source /homes/llumetti/ios_draftreport_venv/bin/activate

python training/train.py --config configs/base_config.yaml --use_wandb
