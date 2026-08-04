#!/bin/bash

# MODELS=(
#     "google/medgemma-4b-it"
#     "llava-hf/llava-v1.6-mistral-7b-hf"
#     "Qwen/Qwen3-VL-2B-Instruct"
#     "Qwen/Qwen3-VL-4B-Instruct"
#     "Qwen/Qwen3-VL-8B-Instruct"
#     "Qwen/Qwen2.5-VL-7B-Instruct"
# )


MODELS=(
    "google/medgemma-27b-it"
    # "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # "Qwen/Qwen3-VL-8B-Instruct"
)
mkdir -p logs

for model in "${MODELS[@]}"; do
    job_name=$(echo "$model" | sed 's/\//_/g')
    
    echo "Submitting job for model: $model"
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=report_${job_name}
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/report_${job_name}_%j.out
#SBATCH --error=logs/report_${job_name}_%j.err
#SBATCH --account=grana_maxillo

source /homes/llumetti/cvpr2026/bin/activate
source /work/grana_maxillo/CVPR2026/set_hf_folder.sh

python src/main.py --models $model
EOF

    echo "Job submitted for $model"
    echo "---"
done

echo "All jobs submitted!"
