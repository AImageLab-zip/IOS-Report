#!/bin/bash
#SBATCH --job-name=compile_flashattn
#SBATCH --partition=all_usr_prod
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --account=grana_maxillo

source /homes/llumetti/ios_draftreport_venv/bin/activate

python training/pretrain.py --config configs/pretrain_config.yaml
