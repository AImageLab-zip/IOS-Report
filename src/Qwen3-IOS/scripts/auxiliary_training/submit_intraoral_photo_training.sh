#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job script — Qwen3 VL intraoral photos fine-tuning
#
# Usage (from the repo root or any directory):
#   sbatch scripts/auxiliary_training/submit_intraoral_photo_training.sh [--train_vision] [--train_language] [extra args]
#
# Examples:
#   sbatch scripts/auxiliary_training/submit_intraoral_photo_training.sh --train_vision --train_language
#   sbatch --partition=boost_usr_prod scripts/auxiliary_training/submit_intraoral_photo_training.sh --train_vision
#   sbatch --nodelist=nico scripts/auxiliary_training/submit_intraoral_photo_training.sh --train_vision --train_language
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=ios-qwen-photos
#SBATCH --partition=all_usr_prod        # change to boost_usr_prod for priority nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                    # request 1 GPU (works on any node)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00                 # max walltime (boost_usr_prod = 1 day)
#SBATCH --output=logs/ios_qwen_%j.out
#SBATCH --error=logs/ios_qwen_%j.err
# Optional: send email on job events
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@your.institute

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/work/grana_maxillo/IOS-DraftReport
TRAIN_SCRIPT=$REPO/src/Qwen3-IOS/training/train_intraoralphotos.py
CONFIG=$REPO/src/Qwen3-IOS/configs/auxiliary_training/qwen3vl_intraoral_photo_finetune.yaml   # edit if needed
CONDA_ENV=/homes/llumetti/llumetti_venvs/iosqwen

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p $REPO/logs

# Activate conda (works whether the login shell sources conda or not)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Python     : $(which python)"
echo "Torch      : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU info   : $(python -c 'import torch; [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]')"
echo "========================================"

# ── Launch ────────────────────────────────────────────────────────────────────
cd $REPO/src/Qwen3-IOS

# All CLI arguments passed to sbatch are forwarded to the Python script.
# If none are given, default to joint training (both vision + language).
EXTRA_ARGS="$@"
if [ -z "$EXTRA_ARGS" ]; then
    EXTRA_ARGS="--train_vision --train_language"
fi

echo "Running: python $TRAIN_SCRIPT --config $CONFIG $EXTRA_ARGS"
python $TRAIN_SCRIPT \
    --config $CONFIG \
    $EXTRA_ARGS

echo "Job finished with exit code $?"
