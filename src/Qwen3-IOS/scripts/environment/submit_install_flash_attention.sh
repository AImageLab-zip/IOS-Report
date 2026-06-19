#!/bin/bash
#SBATCH --job-name=compile_flashattn
#SBATCH --partition=all_usr_prod
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_48G|gpu_RTX6000_24G|gpu_A40_48G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/install_flashattn_%j.out
#SBATCH --error=logs/install_flashattn_%j.err
#SBATCH --account=grana_maxillo

source /homes/llumetti/ios_draftreport_venv/bin/activate

python -m pip install --upgrade pip wheel setuptools ninja
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9"
MAX_JOBS=32 python -m pip -v install flash-attn --no-build-isolation
echo "FlashAttention installation completed."
