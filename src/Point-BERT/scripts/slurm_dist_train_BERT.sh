#!/bin/bash
#SBATCH --job-name=pointbert_train
#SBATCH --account=grana_maxillo
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"

# Set environment variables
export PYTHONPATH=/work/grana_maxillo/IOS-DraftReport:$PYTHONPATH
export OMP_NUM_THREADS=32
export NCCL_DEBUG=INFO

# Navigate to the Point-BERT directory
cd /work/grana_maxillo/IOS-DraftReport/src/Point-BERT

# Get the number of GPUs allocated
NGPUS=$SLURM_GPUS_ON_NODE
if [ -z "$NGPUS" ]; then
    NGPUS=4
fi

# Set a random port for distributed training
PORT=$(( 10000 + RANDOM % 20000 ))

# Training arguments - modify these as needed
CONFIG="cfgs/Mixup_models/Point-BERT.yaml"
EXP_NAME="pointBERT_pretrain_dist_4GPU_64BS_100E"
VAL_FREQ=1

# Print training configuration
echo "Configuration file: $CONFIG"
echo "Experiment name: $EXP_NAME"
echo "Number of GPUs: $NGPUS"
echo "Master port: $PORT"
scp aimagelab@figaroa:/home/aimagelab/IOS_DraftReport/src/Point-BERT/experiments/dvae/PreprocessedIOS_models/figaroa/ckpt-best.pth /work/grana_maxillo/IOS-DraftReport/src/Point-BERT/experiments/dvae/PreprocessedIOS_models/figaroa/ckpt-best.pth 

# Run distributed training
python -m torch.distributed.launch \
    --master_port=${PORT} \
    --nproc_per_node=${NGPUS} \
    main_BERT.py \
    --launcher pytorch \
    --sync_bn \
    --config ${CONFIG} \
    --exp_name ${EXP_NAME} \
    --val_freq ${VAL_FREQ} \
    --num_workers 32

echo "End time: $(date)"
